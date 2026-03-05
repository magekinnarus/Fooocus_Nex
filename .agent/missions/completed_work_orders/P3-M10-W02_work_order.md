# Work Order: P3-M10-W02 — Replace Monkey-Patches with Backend Calls
**ID:** P3-M10-W02
**Mission:** P3-M10
**Status:** Ready
**Depends On:** P3-M10-W01

## Mandatory Reading
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `Fooocus_Nex/modules/patch.py` (current monkey-patches — 397 lines)
- `Fooocus_Nex/backend/sampling.py`
- `Fooocus_Nex/backend/conditioning.py`
- `Fooocus_Nex/backend/resources.py`
- `Fooocus_Nex/backend/k_diffusion.py`
- `Fooocus_Nex/modules/anisotropic.py`

## Objective
Replace the 5 txt2img-relevant monkey-patches in `modules/patch.py` with direct calls to `backend/` modules, eliminating `ldm_patched` overrides for the core inference path. **Extended Objective**: Address CPU/RAM spikes on low-end hardware (3GB VRAM) by optimizing loader memory usage and driver-side swapping behavior.

## Scope

### Patch #1: `patched_sampling_function` (lines 87–117)
**Current behavior:**
- Overrides `ldm_patched.modules.samplers.sampling_function`
- Implements: anisotropic sharpness, adaptive CFG (`compute_cfg`), eps recording
- Reads from `PatchSettings` per-PID dict for `sharpness`, `adaptive_cfg`, `global_diffusion_progress`

**Migration strategy:**
- The backend's `sampling.py` already absorbed these quality features during M09-W02
- Quality settings should flow through `model_options["quality"]` dict (same pattern as `app.py` line 212)
- Remove the monkey-patch assignment in `patch_all()` line 389
- Verify the backend's sampling function handles: (a) CFG=1.0 optimization, (b) anisotropic filter, (c) adaptive CFG blending, (d) eps recording

### Patch #2: `sdxl_encode_adm_patched` (lines 129–158)
**Current behavior:**
- Overrides `ldm_patched.modules.model_base.SDXL.encode_adm`
- Applies ADM scaling (positive/negative) from `PatchSettings`
- Concatenates emphasized + consistent ADM embeddings

**Migration strategy:**
- `backend/conditioning.py` → `get_adm_embeddings_sdxl()` already implements this
- The monkey-patch reads `positive_adm_scale`, `negative_adm_scale` from `PatchSettings` via PID
- Route these values through `model_options` or call `conditioning.get_adm_embeddings_sdxl()` directly from the pipeline instead of relying on the monkey-patch
- Remove the override in `patch_all()` line 386

### Patch #3: `patched_load_models_gpu` (lines 324–330)
**Current behavior:**
- Wraps `ldm_patched.modules.model_management.load_models_gpu` with timing
- Calls the original function underneath

**Migration strategy:**
- `backend/resources.py` → `load_models_gpu()` already provides this functionality
- Replace with direct `resources.load_models_gpu()` calls from `default_pipeline.py`
- Remove the override in `patch_all()` lines 380–383
- Preserve the timing log (add to `resources.load_models_gpu()` if not already present)

### Patch #4: `BrownianTreeNoiseSamplerPatched` (lines 47–70)
**Current behavior:**
- Replaces `ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler`
- Uses class-level `global_init()` pattern for seed-controlled noise generation

**Migration strategy:**
- `backend/k_diffusion.py` already has this sampler
- The `process_diffusion()` function (modified in W01) should call the backend's noise sampler directly
- Remove the override in `patch_all()` line 388

### Patch #5: `patched_unet_forward` (lines 241–321) — **FULLY REMOVABLE**

**Current behavior (3 additions over original UNet forward):**
1. **Precision casting** (lines 242–253): Casts inputs (x, context, y, control) to `weight_dtype` — prevents per-layer upcasting slowness
2. **Diffusion progress tracking** (lines 255–256): Computes `1.0 - timesteps / 999.0`
3. **Timed ADM** (line 258): Masks ADM embeddings based on timestep

**Backend status:**
| Behavior | Backend Location | Status |
|----------|-----------------|--------|
| Diffusion progress | `sampling.py` line 268 | ✅ Already there (M09-W02) |
| Anisotropic sharpness | `sampling.py` lines 287–297 | ✅ Already there (M09-W02) |
| Adaptive CFG | `sampling.py` lines 227–238 | ✅ Already there (M09-W02) |
| Timed ADM | `loader.py` lines 157–176 | ✅ Already there (M09-W02) |
| **Precision casting** | Not in backend | ❌ **Migrate in this WO** |

**Migration strategy:**
- Create `backend/precision.py` [NEW] with a `cast_unet_inputs()` function (~12 lines):
  - Detect `weight_dtype` from model's first layer
  - Cast `x`, `context`, `y`, `control` tensors to that dtype
- Apply via `model_options["unet_input_cast"]` callback, or monkey-patch the forward in `loader.patch_unet_for_quality()` alongside timed ADM
- Remove the override in `patch_all()` line 385
- The structural UNet forward (lines 260–321) is a verbatim copy of the original — zero risk to remove

**Risk: LOW** — each migrated piece can be tested independently, and 4 of 5 behaviors are already proven in production via `app.py`.

### Migrate `PatchSettings` → `model_options`
- **Current pattern**: `PatchSettings` per-PID dict read from `async_worker.py`'s `apply_patch_settings()`
- **Target pattern**: Quality settings flow through `model_options["quality"]` dict, matching `app.py`'s approach
- Settings to migrate: `sharpness`, `adaptive_cfg`, `adm_scaler_end`, `positive_adm_scale`, `negative_adm_scale`, `controlnet_softness`

### Update `patch_all()`
Remove these overrides from `patch_all()`:
```python
# REMOVE these lines:
ldm_patched.modules.model_management.load_models_gpu = patched_load_models_gpu  # line 383
ldm_patched.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = patched_unet_forward  # line 385
ldm_patched.modules.model_base.SDXL.encode_adm = sdxl_encode_adm_patched       # line 386
ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSamplerPatched  # line 388
ldm_patched.modules.samplers.sampling_function = patched_sampling_function      # line 389

### Bonus Objective: Resource Optimization
- Resolve 99% CPU spike on 3GB VRAM cards (GTX 1050).
- Reduce RAM footprint of SDXL GGUF from 10GB to < 2GB.
- Implement "Weight Diet" in `loader.py` (remove clone/copy operations).
- Tune `resources.py` thresholds (extra_reserved_memory, minimum_inference_memory) for 3GB cards.
```

## Verification (Local: GTX 1050, SD1.5 full + SDXL GGUF)
1. Fooocus txt2img with **SD1.5** — generation succeeds, quality features visible
2. Fooocus txt2img with **SDXL GGUF** — ADM scaling, adaptive CFG function correctly
3. LoRA applied through Fooocus UI — visible style change
4. Anisotropic sharpness at 0.0 vs 2.0 produces visibly different outputs
5. Inference time within ±10% of pre-migration baseline (same model, same seed, same steps)
6. No VRAM OOM on GTX 1050 with either model type
7. Precision casting verified: ensure no dtype mismatch warnings in console

## Success Criteria
- All 5 monkey-patches (#1–5) removed from `patch_all()`
- Precision casting logic lives in `backend/` (new `precision.py` or in `loader.py`)
- `PatchSettings` per-PID dict eliminated — quality settings flow through `model_options`
- txt2img + LoRA produce correct results through Fooocus UI (SD1.5 + SDXL GGUF)
- No regression in inference time or VRAM usage

