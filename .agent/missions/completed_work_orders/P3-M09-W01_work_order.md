# Work Order: P3-M09-W01 — Decompose sampling.py

**Status:** Ready
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-22

## Objective
Extract condition processing logic from `backend/sampling.py` (626 lines) into a new `backend/cond_utils.py`. This is a pure refactor — no behavior change, only file reorganization.

## Context
Ref: `.agent/missions/active/P3-M09_mission_brief.md`
Ref: `.agent/summaries/04_Inference_Architectural_Guideline.md`

## Tasks

### 1. Create `backend/cond_utils.py`
- [ ] Move the following functions from `sampling.py`:
  - `add_area_dims`
  - `get_area_and_mult`
  - `cond_equal_size`
  - `can_concat_cond`
  - `cond_cat`
  - `calc_cond_batch`
  - `resolve_areas_and_cond_masks_multidim`
  - `calculate_start_end_timesteps`
  - `encode_model_conds`
  - `process_conds`
- [ ] Move necessary imports with them.

### 2. Update `backend/sampling.py`
- [ ] Replace moved functions with `from .cond_utils import ...` imports.
- [ ] Verify that `sampling_function`, `cfg_function`, `CFGGuider`, and `sample_sdxl` still work with the imported functions.
- [ ] `sampling.py` should be ~320 lines after this.

### 3. Verify
- [ ] Run `app.py` with `test_sd15_config.json` — same output as before.
- [ ] Run `app.py` with `test_sdxl_config.json` — same output as before.
- [ ] No import errors, no behavior changes.

## Deliverables
- `backend/cond_utils.py` [NEW]
- Updated `backend/sampling.py`
