# Mission Brief: 002 ??Targeted Refactoring
**Phase:** 1.5
**Date Issued:** 2026-02-12
**Status:** Not Started

## Required Reading
Read these files BEFORE starting any work:
- `.agent/archive/summaries/DEPRECATED_Nex_project_summary01.md` ??Project state and architectural decisions
- `.agent/archive/summaries/DEPRECATED_02_Phase_Roadmap.md` ??Phase overview and current position
- `.agent/rules/01_Global_Context_Rules.md` ??Consensus protocol and core philosophies
- `.agent/missions/completed/001_ldm_foundation_mission_report.md` ??Phase 1 findings (especially "Architectural Insights" section)

## Background
Phase 1 modernized the `ldm_patched` core by porting ComfyUI 0.3.47 components. During this work, a systemic problem was identified: **Fooocus monkey-patches 17+ functions/methods** across `modules/patch.py`, `modules/patch_clip.py`, and `modules/patch_precision.py` that override ComfyUI module internals. Each override creates implicit coupling ??when the underlying module evolves, the patch may silently break.

Phase 1's most time-consuming bugs all traced to this coupling:
1. **LoRA Regression** ??`calculate_weight_patched` (in `patch.py`) assumed old device placement logic; broke when `ModelPatcher` was modernized
2. **Double-Patching OOM** ??Precision forcing in `stochastic_rounding` collided with the patched weight calculation, upcasting to FP32
3. **AttributeError Cascade** ??`lora.py` references to model classes that didn't exist in the legacy `model_base.py`

This mission addresses the coupling **before Phase 2 (GGUF)** introduces additional complexity.

## Objective
Reduce the coupling between Fooocus's monkey-patching layer and the ported ComfyUI modules through three targeted refactorings. The goal is NOT to rewrite anything ??it is to create explicit boundaries where implicit coupling currently exists.

## Scope

### In Scope

- **1.5.A ??Patch Registry** (Low effort)
  - Create a manifest that documents all monkey-patches: what is patched, why, what version of the target it was written against
  - This is documentation/housekeeping, not a code change
  - Deliverable: a single file (e.g., `modules/PATCH_MANIFEST.md` or a dict in `patch.py`) that serves as a checklist for future upgrades

- **1.5.B ??Absorb `calculate_weight` into `model_patcher.py`** (Medium effort, highest priority)
  - The function `calculate_weight_patched` in `modules/patch.py` (lines 53??86) currently overrides `ldm_patched.modules.lora.calculate_weight` via monkey-patch
  - Move this logic into the Nex copy of `ldm_patched/modules/model_patcher.py` as a proper method or module-level function
  - This eliminates the most dangerous implicit coupling ??the #1 source of Phase 1 bugs
  - **Key files:**
    - `modules/patch.py` ??current location of `calculate_weight_patched` (134 lines)
    - `ldm_patched/modules/model_patcher.py` ??target location (1232 lines, class `ModelPatcher`)
    - `ldm_patched/modules/lora.py` ??has the original `calculate_weight` that gets overridden
  - **Must preserve:** Fooocus's custom patch types (`"fooocus"`, `"lokr"`, `"loha"`) and precision casting logic
  - **Must preserve:** The `intermediate_dtype` parameter support added in Phase 1

- **1.5.C ??`nex_loader.py` Adapter** (Medium effort)
  - Create `modules/nex_loader.py` ??a thin adapter that provides component-wise model loading
  - Currently, `ldm_patched/modules/sd.py` ??`load_checkpoint_guess_config()` loads everything from a single checkpoint file
  - Phase 2 (GGUF) needs to load UNet from `.gguf`, CLIP separately, VAE separately
  - Instead of patching `load_checkpoint_guess_config`, create a parallel loading path that:
    1. Delegates to existing pipeline for standard `.safetensors` checkpoints (backward compat)
    2. Provides a new path for component-wise loading (GGUF UNet + separate CLIP + separate VAE)
    3. Returns the same `ModelPatcher`/`CLIP`/`VAE` objects the rest of the system expects
  - **Key files to study:**
    - `ldm_patched/modules/sd.py` ??`load_checkpoint_guess_config()` (lines 431??94), `CLIP` class, `VAE` class
    - `ldm_patched/modules/model_detection.py` ??model architecture detection
    - `ldm_patched/modules/model_management.py` ??device/memory management
    - `modules/default_pipeline.py` ??where models are currently loaded in the Fooocus pipeline
  - **Reference for GGUF loading:**
    - `ComfyUI_reference/custom_nodes/ComfyUI-GGUF/` ??the GGUF implementation we'll adapt in Phase 2

### Out of Scope
- GGUF loader implementation (Phase 2 ??this mission only creates the adapter interface)
- Sampler/ControlNet/hooks upgrades (Phase 3)
- UI changes (Phase 4)
- Refactoring `patch_clip.py` or `patch_precision.py` (these are stable and well-motivated)
- Refactoring ComfyUI reference modules themselves

## Current Monkey-Patch Inventory
For reference, here is the complete list of overrides as of Phase 1 completion:

### `modules/patch.py` ??`patch_all()`
| # | Target | Replacement | Purpose |
|---|--------|-------------|---------|
| 1 | `ldm_patched.modules.lora.calculate_weight` | `calculate_weight_patched` | Custom weight calc with Fooocus types, precision control |
| 2 | `ldm_patched.ldm...openaimodel.UNetModel.forward` | `patched_unet_forward` | Unified Precision Pass, timed ADM, anisotropic sharpness |
| 3 | `ldm_patched.modules.samplers.sampling_function` | `patched_sampling_function` | Adaptive CFG, global diffusion progress tracking |
| 4 | `ldm_patched.modules.model_management.load_models_gpu` | `patched_load_models_gpu` | Timer logging around model loading |
| 5 | `ldm_patched.modules.model_base.SDXL.encode_adm` | `sdxl_encode_adm_patched` | Custom ADM scaling (positive/negative scales, aspect ratio) |
| 6 | `ldm_patched.controlnet.cldm.ControlNet.forward` | `patched_cldm_forward` | ControlNet softness, timed ADM |
| 7 | `ldm_patched.modules.samplers.KSamplerX0Inpaint.forward` | `patched_KSamplerX0Inpaint_forward` | Inpaint-specific denoising |
| 8 | `ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler` | `BrownianTreeNoiseSamplerPatched` | Noise sampler with global init |
| 9 | `safetensors.torch.load_file` | wrapped via `build_loaded` | Corrupted-file detection and recovery |
| 10 | `torch.load` | wrapped via `build_loaded` | Corrupted-file detection and recovery |

### `modules/patch_clip.py` ??`patch_all_clip()`
| # | Target | Replacement | Purpose |
|---|--------|-------------|---------|
| 11 | `sd1_clip.ClipTokenWeightEncoder.encode_token_weights` | `patched_encode_token_weights` | Token weight handling |
| 12 | `sd1_clip.SDClipModel.__init__` | `patched_SDClipModel__init__` | Use HuggingFace `CLIPTextModel` directly |
| 13 | `sd1_clip.SDClipModel.forward` | `patched_SDClipModel_forward` | HF Transformers forward pass |
| 14 | `clip_vision.ClipVisionModel.__init__` | `patched_ClipVisionModel__init__` | Use HF `CLIPVisionModelWithProjection` |
| 15 | `clip_vision.ClipVisionModel.encode_image` | `patched_ClipVisionModel_encode_image` | HF-based image encoding |

### `modules/patch_precision.py` ??`patch_all_precision()`
| # | Target | Replacement | Purpose |
|---|--------|-------------|---------|
| 16 | `openaimodel.timestep_embedding` | `patched_timestep_embedding` | Kohya-consistent timestep embedding |
| 17 | `model_sampling.ModelSamplingDiscrete._register_schedule` | `patched_register_schedule` | Kohya-consistent schedule registration |

## Constraints
- **Consensus Protocol:** No file writes without Director-approved plan
- **Surgical approach:** Each sub-task should be independently testable
- **Backward compatibility:** Existing Colab workflow must continue to work ??standard checkpoint loading must be unaffected
- **No ComfyUI reference modifications:** Only modify files in `Fooocus_Nex/`

## Deliverables
- [ ] 1.5.A: Patch manifest document
- [ ] 1.5.B: `calculate_weight` absorbed into `model_patcher.py`, monkey-patch removed from `patch.py`
- [ ] 1.5.C: `nex_loader.py` with component-wise loading interface (standard checkpoint passthrough + GGUF stub)
- [ ] Verification: All existing generation workflows still work on Colab after changes

## Success Criteria
- Standard SDXL checkpoint loading works identically to current behavior
- LoRA application works correctly with the absorbed `calculate_weight`
- `nex_loader.py` can load a standard checkpoint via delegation to existing pipeline
- Performance unchanged: <1.1s/it on T4, ~9.9GB VRAM with 2 LoRAs

## Suggested Approach
1. **Start with 1.5.A** (Patch Registry) ??Low risk, immediate documentation value
2. **Then 1.5.B** (Absorb calculate_weight) ??Highest impact, removes biggest source of coupling
3. **Then 1.5.C** (nex_loader.py) ??Creates the boundary needed for Phase 2

For each sub-task:
1. Read the relevant source files listed above
2. Draft an implementation plan
3. Get Director approval
4. Implement and verify

## Project Structure Reference
```
Fooocus_Nex/
?œâ??€ modules/                    # Fooocus application layer
??  ?œâ??€ patch.py                # Main monkey-patching (10 overrides)
??  ?œâ??€ patch_clip.py           # CLIP monkey-patching (5 overrides)
??  ?œâ??€ patch_precision.py      # Precision monkey-patching (2 overrides)
??  ?œâ??€ default_pipeline.py     # Model loading and pipeline orchestration
??  ?œâ??€ async_worker.py         # Generation worker
??  ?”â??€ ops.py                  # Fooocus-specific ops utilities
?œâ??€ ldm_patched/modules/        # Ported ComfyUI modules (our copy)
??  ?œâ??€ model_management.py     # 1505 lines ??device/VRAM management
??  ?œâ??€ model_patcher.py        # 1232 lines ??weight patching engine
??  ?œâ??€ sd.py                   # 551 lines ??checkpoint loading
??  ?œâ??€ lora.py                 # 19KB ??LoRA weight parsing
??  ?œâ??€ model_base.py           # 19KB ??model class definitions
??  ?œâ??€ model_detection.py      # 18KB ??model architecture detection
??  ?œâ??€ ops.py                  # 18KB ??operations layer
??  ?œâ??€ weight_adapter/         # 9 adapter types (LoKr, LoHa, BOFT, etc.)
??  ?”â??€ ...
?œâ??€ ComfyUI_reference/comfy/    # Frozen reference (v0.3.47, DO NOT MODIFY)
?”â??€ .agent/                     # Project management
    ?œâ??€ summaries/              # Persistent project state
    ?œâ??€ missions/active/        # Current mission briefs
    ?œâ??€ missions/completed/     # Mission reports
    ?”â??€ rules/                  # Global context rules
```
