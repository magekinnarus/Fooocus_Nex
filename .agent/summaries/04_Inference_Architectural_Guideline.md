# Inference Architecture Standards

This document defines the architectural contracts for the `Fooocus_Nex` backend. It explicitly outlines the responsibilities of the runner (`app.py`) versus the backend modules, ensuring that critical runtime requirements (memory, precision, hardware awareness) are not left to implicit knowledge.

## 1. The Inference Lifecycle
Every inference pipeline (SD1.5, SDXL, Flux) must adhere to this 4-stage lifecycle to ensure hardware compatibility.

### Stage 1: Setup (CPU)
- **Responsibility**: `backend.loader`
- **Goal**: Load model weights from disk to CPU (or meta-device).
- **Contract**: 
    - MUST return a `ModelPatcher` or container.
    - MUST NOT load weights to GPU (unless explicitly requested).
    - **CRITICAL**: Use `move` or `pop` semantics when extracting from state dicts; never use `.clone()` on full weights as it doubles RAM usage during the loading phase.
    - MUST clean up system RAM (garbage collection) immediately after loading.

### Stage 2: Encoding (Conditioning)
- **Responsibility**: `app.py` (Orchestrator) calling `backend.conditioning`
- **Goal**: Convert text/image prompts into embeddings.
- **Contract**:
    - **Memory**: MUST request `resources.load_models_gpu([clip])` before execution.
    - **Context**: MUST run inside `torch.inference_mode()` (CRITICAL: prevents graph building).
    - **Precision**: MUST run inside `torch.autocast()` (unless SD1.5 FP32 fallback is needed).
    - **Metadata Expansion**: MUST execute `process_conds` to expand base embeddings with required metadata pairs.
    - **Output**: Returns CPU-resident tensors (to allow CLIP offloading).

### Stage 3: Sampling (Latent Generation)
- **Responsibility**: `app.py` calling `backend.sampling`
- **Goal**: Denoise latents using the UNet/DiT.
- **Contract**:
    - **Memory**: MUST request `resources.load_models_gpu([unet])` before execution.
    - **Context**: MUST run inside `torch.inference_mode()`.
    - **Precision**: MUST run inside `torch.autocast()`.
    - **Hardware Awareness**: For cards with < 4GB VRAM, the backend must prioritize preserving the "Working Set" over the "Resident Set" to prevent driver-side swapping.
    - **Conditioning Strictness**: The `sampler` WILL FAIL (`AssertionError: must specify y`) if class-conditional models (SDXL/GGUF) do not receive the ADM vector (`y`). This vector must be explicitly generated during Stage 2 via `process_conds`.

### Stage 4: Decoding (Pixel Generation)
- **Responsibility**: `app.py` calling `backend.decode`
- **Goal**: Convert latents to pixels.
- **Contract**:
    - **Memory**: MUST request `resources.load_models_gpu([vae])` before execution.
    - **Context**: MUST run inside `torch.inference_mode()`.
    - **UI Compatibility**: For live previews via progress callbacks, latents MUST be decoded to RGB numpy arrays (`get_previewer` pipeline wrapper). Raw PyTorch tensors will crash the interface.
    - **Output**: Returns HWC numpy-ready tensors to `app.py`.

## 2. Memory Management Strategy ("The Contract")
The backend relies on **Stage-based Cycling** to support all hardware tiers with a single code path.

| Tier | VRAM | Strategy | Behavior |
| :--- | :--- | :--- | :--- |
| **Edge** | < 4GB | **Strict Cycling** | `load_models_gpu` forces previous models to CPU to make room for current stage. |
| **Consumer** | 8-12GB | **Lazy Cycling** | Models stay in VRAM until pressure is detected. |
| **Cloud** | > 16GB | **Resident** | `force_high_vram` flag prevents any offloading. All models stay hot. |

## 3. Critical Implementation Details
*Failure to adhere to these leads to performance regressions (e.g., 12s -> 45s).*

1.  **Global Inference Mode**: Use `torch.inference_mode()` for ALL model forward passes.
    - *Why*: PyTorch defaults to training mode. Without this, it builds massive computational graphs (GBs of VRAM) that are useless for inference.
2.  **Explicit Offloading**: Never manually call `.to('cpu')`. Use `resources.load_models_gpu()` to declare *what you need now*, and let the Memory Manager decide *what to evict*.
    - *Driver Swapping Prevention*: On Windows, if VRAM is exceeded, the driver invisible swaps weights to RAM ("Shared GPU Memory"). This pegs the CPU (99%) and destroys performance. The backend MUST proactively offload models to keep total allocated VRAM under the physical limit.
3.  **Garbage Collection**: Heavy loaders (SDXL) must explicitly call `gc.collect()` after moving weights to clear the "staging" copies from RAM.
4.  **Component Isolation (GGUF)**: Base models distributed as `.gguf` are *UNet-only*. The loading pipeline MUST implement explicit parallel resolution for VAE and CLIP components; standard checkpoint loaders will silently fail on these components.
5.  **Unified Input Casting (Standard)**: Always cast ALL inputs (`x`, `timesteps`, `context`, `y`, `control`) to Model precision at the start of the `forward` pass.
    - *Observation*: Relying on PyTorch's internal layer-wise casting causes a 3-4x slowdown on mixed-hardware (GTX/RTX) environments.
6.  **Native Patching over Monkey-patches**: Avoid `os.getpid()` based global configuration lookup. Patch models once at load-time using closures or `model_options` injection.
    - *Why*: Eliminates system calls and dictionary overhead during the sampling loop.
