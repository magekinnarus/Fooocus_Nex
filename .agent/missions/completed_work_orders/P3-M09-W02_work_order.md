# Work Order: P3-M09-W02 — Fooocus Quality Features

**Status:** Completed
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-22

## Objective
Absorb four quality-of-life features from Fooocus's `modules/patch.py` into the backend natively. All features must be configurable via JSON config and default to Fooocus's proven defaults.

## Context
Ref: `.agent/missions/active/P3-M09_mission_brief.md`
Ref: `Fooocus_Nex/modules/patch.py` — source of all features
Ref: `Fooocus_Nex/modules/anisotropic.py` — bilateral blur implementation

## Tasks

### 1. Copy `backend/anisotropic.py` [NEW]
- [x] Copy `Fooocus_Nex/modules/anisotropic.py` to `Fooocus_Nex/backend/anisotropic.py`.
- [x] Verify zero external dependencies (it only uses `torch`).
- [x] The key function is `adaptive_anisotropic_filter(x, g=None)`.

### 2. Add Sharpness to `backend/sampling.py`
- [x] Modify `sampling_function()` (or `cfg_function()`) to apply the anisotropic sharpness filter.
- [ ] Logic (from `patched_sampling_function` lines 87–117 of `patch.py`):
  - Compute `alpha = 0.001 * sharpness * diffusion_progress`
  - If `alpha >= 0.01`: blend `anisotropic.adaptive_anisotropic_filter(positive_eps, positive_x0)` with `positive_eps` by `alpha`
  - Apply before CFG combination
- [x] Add `sharpness` parameter (default: `2.0`), passable through sampling config.

### 3. Add Adaptive CFG to `backend/sampling.py`
- [x] Modify `cfg_function()` to support adaptive CFG blending.
- [ ] Logic (from `compute_cfg` lines 73–84 of `patch.py`):
  - If `cfg_scale > adaptive_cfg`: compute `mimicked_eps` at `adaptive_cfg` scale, blend with real eps by `diffusion_progress`
  - Otherwise: use standard CFG
- [x] Add `adaptive_cfg` parameter (default: `7.0`).

### 4. Add ADM Scaling to `backend/conditioning.py`
- [x] Modify `get_adm_embeddings_sdxl()` to accept `adm_scale_positive` and `adm_scale_negative`.
- [ ] Logic (from `sdxl_encode_adm_patched` lines 129–158 of `patch.py`):
  - Scale width/height by `positive_adm_scale` for positive embeddings
  - Scale width/height by `negative_adm_scale` for negative embeddings
  - Generate both "emphasized" (scaled) and "consistent" (original) ADM vectors
  - Concatenate: `[clip_pooled, adm_emphasized, clip_pooled, adm_consistent]`
- [x] Default: `adm_scale_positive=1.5`, `adm_scale_negative=0.8`.

### 5. Add Timed ADM to `backend/conditioning.py` or `backend/sampling.py`
- [x] Implement `timed_adm(y, timesteps, adm_scaler_end)` function.
- [ ] Logic (from `timed_adm` lines 194–200 of `patch.py`):
  - If `y` tensor has dim 5632: swap from "emphasized" ADM to "consistent" ADM after `adm_scaler_end` fraction of diffusion
  - Split at index 2816
- [x] Default: `adm_scaler_end=0.3`.

### 6. Track Diffusion Progress
- [x] Expose `diffusion_progress` (0.0→1.0) through the sampling pipeline.
- [ ] Formula: `progress = 1.0 - timestep / 999.0`
- [ ] This value is needed by sharpness (step 2) and adaptive CFG (step 3).

### 7. Update `app.py` Config
- [x] Add quality feature parameters to JSON config schema:
  ```json
  "quality": {
      "sharpness": 2.0,
      "adaptive_cfg": 7.0,
      "adm_scale_positive": 1.5,
      "adm_scale_negative": 0.8,
      "adm_scaler_end": 0.3
  }
  ```
- [x] Pass these through to sampling/conditioning functions.

### 8. Verify
- [x] Generate image WITHOUT quality features (all set to neutral: `sharpness=0`, `adaptive_cfg=999`, `adm_scale=1.0`).
- [x] Generate image WITH quality features (Fooocus defaults).
- [x] Confirm visible quality improvement with features enabled.
- [x] No regression in SD1.5 mode (quality features are SDXL-specific, SD1.5 should be unaffected).

## Deliverables
- `backend/anisotropic.py` [NEW]
- Updated `backend/sampling.py`
- Updated `backend/conditioning.py`
- Updated `app.py`
