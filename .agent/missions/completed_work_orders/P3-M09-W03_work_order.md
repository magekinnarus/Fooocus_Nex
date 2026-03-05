# Work Order: P3-M09-W03 — Extract NexModelPatcher

**Status:** Pending (depends on W01)
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-22

## Objective
Extract the core of `ldm_patched/modules/model_patcher.py` into a self-contained `backend/patching.py` module. Focus on the weight patching (LoRA application) and device management logic that `loader.py` currently depends on.

## Context
Ref: `.agent/missions/active/P3-M09_mission_brief.md`
Ref: `Fooocus_Nex/ldm_patched/modules/model_patcher.py` — source (1398 lines)
Ref: `Fooocus_Nex/ldm_patched/modules/weight_adapter/` — already extracted, can import

## Tasks

### 1. Create `backend/patching.py` [NEW]
- [ ] Extract `calculate_weight()` (lines 125–289 of `model_patcher.py`):
  - Core LoRA weight-delta computation
  - Handles ~12 weight adapter types via dispatch
  - Can import adapter type definitions from `weight_adapter/`
- [ ] Build `NexModelPatcher` class with essential methods:
  - `__init__`, `model_size`, `loaded_size`
  - `clone`, `is_clone`, `clone_has_same_weights`
  - `add_patches` — register LoRA patches by key
  - `patch_model` / `unpatch_model` — apply/remove weight deltas
  - `load` / `partially_load` / `partially_unload` / `detach` — device management
  - `model_patches_to` — move patches to device
  - `get_dtype` / `model_dtype`
  - `set_model_sampler_cfg_function`, `set_model_sampler_post_cfg_function`
  - `memory_required`, `current_loaded_device`
- [ ] Include `LowVramPatch` class for low-VRAM weight streaming.
- [ ] Include `get_key_weight` utility.

### 2. Handle GGUF Compatibility
- [ ] `GGUFModelPatcher` (in `modules/gguf/patcher.py`) extends `ModelPatcher`.
- [ ] Option A: make `NexModelPatcher` accept a `patcher_class` override.
- [ ] Option B: make `GGUFModelPatcher` extend `NexModelPatcher` instead.
- [ ] Choose whichever requires fewer changes to existing GGUF code.

### 3. Update `backend/loader.py`
- [ ] Replace `from ldm_patched.modules import model_patcher` with `from . import patching`.
- [ ] Replace `model_patcher.ModelPatcher(...)` with `patching.NexModelPatcher(...)`.
- [ ] Verify CLIP, VAE, and UNet containers all work with `NexModelPatcher`.
- [ ] Goal: `loader.py` should have zero imports from `ldm_patched/modules/model_patcher.py`.

### 4. Update `backend/resources.py`
- [ ] Verify `LoadedModel` class works with `NexModelPatcher` (it currently expects `model.load_device`, `model.model_size()`, etc.).
- [ ] No changes expected if `NexModelPatcher` implements the same interface, but verify.

### 5. Verify
- [ ] Run `app.py` with `test_sd15_config.json` — works with `NexModelPatcher`.
- [ ] Run `app.py` with `test_sdxl_config.json` (GGUF) — works with `GGUFModelPatcher`.
- [ ] No regression in inference quality or performance.

## Deliverables
- `backend/patching.py` [NEW]
- Updated `backend/loader.py`
- Possibly updated `modules/gguf/patcher.py`

## Notes
- `model_patcher.py` also contains `AutoPatcherEjector`, `MemoryCounter`, and extensive hook/callback code. These are NOT needed for M09. Only extract what's required for LoRA + device management.
- The remaining `ldm_patched` imports in `loader.py` after this work order are: `model_base` (SD15/SDXL UNet architecture), `latent_formats`, `supported_models_base`, `AutoencoderKL`. These are model architecture definitions, not runtime logic, and can be addressed in a future mission.
