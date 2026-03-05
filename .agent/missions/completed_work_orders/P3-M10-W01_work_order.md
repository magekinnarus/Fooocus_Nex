# Work Order: P3-M10-W01 — Route Pipeline Through Backend
**ID:** P3-M10-W01
**Mission:** P3-M10
**Status:** Completed (with known issues)
**Depends On:** —

## Mandatory Reading
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `app.py` (backend calling-sequence reference — lines 39–261)
- `Fooocus_Nex/modules/default_pipeline.py` (current pipeline — 326 lines)
- `Fooocus_Nex/modules/core.py` (current model wrapper — 352 lines)
- `Fooocus_Nex/backend/loader.py`
- `Fooocus_Nex/backend/resources.py`
- `Fooocus_Nex/backend/sampling.py`
- `Fooocus_Nex/backend/conditioning.py`
- `Fooocus_Nex/backend/decode.py`

## Objective
Rewire `modules/default_pipeline.py` and `modules/core.py` to call `backend/` modules for model loading, CLIP encoding, diffusion sampling, and VAE decoding — replacing their current `ldm_patched` calls. This establishes the plumbing that W02's monkey-patch removal will rely on.

## Scope

### Modify `modules/default_pipeline.py`

#### `refresh_base_model()` (lines 56–79)
- **Current**: Calls `core.load_model()` → `modules/nex_loader.py` → `ldm_patched` loaders
- **Change**: Route through `backend/loader.py` → `loader.load_sdxl_checkpoint()` or `loader.load_sd15_checkpoint()` based on model type detection
- **Reference**: `app.py` lines 39–63 shows the correct backend loading pattern

#### `refresh_loras()` (lines 84–94)
- **Current**: Calls `model_base.refresh_loras()` which uses `ldm_patched.modules.utils.load_torch_file()`
- **Change**: Use `backend.utils.load_torch_file()` and `backend.lora.model_lora_keys_unet/clip()` + `backend.lora.load_lora()`
- **Reference**: `app.py` lines 71–105 shows the LoRA pattern

#### `clip_encode()` (lines 132–153)
- **Current**: Calls `clip_encode_single()` → `final_clip.encode_from_tokens()` (works through `ldm_patched` CLIP patcher)
- **Note**: This may already work if the CLIP patcher is correctly initialized. Test before changing. `backend/clip.py` has `encode_text_sdxl()` but the Fooocus style system concatenates multiple prompts — the existing `clip_encode()` approach of encoding each text separately and `torch.cat`-ting might be simpler to preserve.

#### `process_diffusion()` (lines 282–325)
- **Current**: Calls `modules.patch.BrownianTreeNoiseSamplerPatched.global_init()` then `core.ksampler()` → `ldm_patched.modules.sample.sample()`
- **Change**: Route through `backend/sampling.py` → `sampling.sample_sdxl()` (or a unified entry point)
- **Reference**: `app.py` lines 156–215 shows the backend sampling pattern
- **Key difference**: `process_diffusion()` receives `positive_cond`/`negative_cond` already encoded; the backend sampler expects the same format. The `model_options` dict should carry quality settings.
- **Decode**: Replace `core.decode_vae()` with `backend.decode.decode_latent()`

#### `calculate_sigmas()` / `calculate_sigmas_all()` (lines 247–273)
- **Current**: Uses `ldm_patched.modules.samplers` sigma calculation
- **Change**: Route through `backend/schedulers.py` if available, or keep if backend sampling handles sigmas internally

### Modify `modules/core.py`

#### `load_model()` (lines 144–162)
- **Current**: Calls `modules.nex_loader.load_checkpoint()` → returns `StableDiffusionModel`
- **Change**: Use `backend/loader.py` equivalents. The `StableDiffusionModel` wrapper class should still exist for compatibility with `async_worker.py` but internally hold backend model objects.

#### `StableDiffusionModel.refresh_loras()` (lines 60–122)
- **Current**: Uses `ldm_patched.modules.utils.load_torch_file()` and `match_lora()`
- **Change**: Use `backend.utils.load_torch_file()` and `backend.lora` functions

#### `ksampler()` (lines 276–335)
- **Current**: Calls `ldm_patched.modules.sample.sample()` with monkey-patched sampling function
- **Change**: Route through `backend/sampling.py`. The callback pattern should be preserved for UI progress reporting.

#### `decode_vae()` / `encode_vae()` (lines 171–186)
- **Current**: Uses `ldm_patched` VAE operations
- **Change**: Route `decode_vae()` through `backend/decode.py`. Keep `encode_vae()` using `ldm_patched` for now (only needed for inpainting/img2img, which are out of scope).

### Fix `app.py` (Minor)
| Line | Current Import | Change To |
|------|---------------|-----------|
| 76 | `import ldm_patched.modules.utils` | `from backend import utils as backend_utils` |
| 86 | `ldm_patched.modules.utils.load_torch_file(lora_path)` | `backend_utils.load_torch_file(lora_path)` |

## Verification
1. Fooocus launches without errors (`python launch.py --preset default`)
2. Model loading completes successfully through the UI (SD1.5, SDXL, and GGUF)
3. Basic txt2img generation produces a valid image
4. `app.py` still works with both SD1.5 and SDXL test configs
5. No new `ldm_patched` imports added to `default_pipeline.py` or `core.py`
6. Live previews render correctly in the Gradio UI.

## Success Criteria
- `default_pipeline.py` routes model loading and diffusion through `backend/`
- `core.py` LoRA loading uses `backend.utils.load_torch_file()`
- `app.py` has zero `ldm_patched` imports
- Fooocus UI can load a model and generate a basic image

## Supplemental Work (Debugging & Integration)
During the execution of this work order, several critical integration issues were identified and resolved to ensure the UI and backend functioned together:
- **UI Launch Hang**: Fixed by wrapping module-scope model loading in `default_pipeline.py` with try/except blocks and ensuring a valid default model in `config.txt`.
- **GGUF Component Loading**: Implemented explicit manual CLIP and VAE loading paths in `core.py` for UNet-only `.gguf` files. Fixed `paths_clips` typo.
- **Device Placement**: Fixed device mismatches between noise/latents and the model by explicitly calling `resources.load_models_gpu([unet])` before sampling.
- **UI Preview Error**: Fixed `ValueError: Cannot process this value as an Image` by ensuring the latent tensor passed to the callback is decoded to a NumPy array via `get_previewer()`.
- **Callback Signature Mismatch**: Aligned the lambda callback signature in `sampling.sample_sdxl` with the 5 arguments expected by the UI handler.
- **SDXL/GGUF ADM Vector Assertion**: Fixed `AssertionError: must specify y` in `patched_unet_forward` by injecting the missing `process_conds()` call into `CFGGuider.sample` to generate the required ADM conditioning vectors.
