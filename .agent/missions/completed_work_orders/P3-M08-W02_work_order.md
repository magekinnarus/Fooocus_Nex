# Work Order: P3-M08-W02 — SDXL CLIP Extraction

**Status:** Ready
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-18

## Objective
Extend the `backend/clip.py` module to support SDXL's CLIP-G encoder. Implement the necessary key transformations to load OpenCLIP weights into the `NexClipEncoder` structure (which uses HF naming). Update `backend/loader.py` to support SDXL CLIP loading and verify parity.

## Context
Ref: `.agent/missions/active/P3-M08_mission_brief.md`
Ref: `.agent/reference/P3_ldm_patched_clip_extraction_map.md`

## Tasks

### 1. Update `backend/clip.py`
- [ ] Implement `normalize_clip_g_keys(sd)`:
    - Handle OpenCLIP -> HF key mapping (e.g., `resblocks.` -> `encoder.layers.`).
    - Handle prefix stripping (`conditioner.embedders.1.model.`, `clip_g.`).
    - Ensure it returns clean HF-style keys compatible with `NexClipEncoder`.
- [ ] Extend `NexClipEncoder` (if needed) or verify existing flexibility:
    - Confirm `use_projection` flag correctly enables/disables `text_projection` and `logit_scale`.
    - Ensure generic config handling supports CLIP-G's different hidden sizes/layer counts.

### 2. Update `backend/loader.py`
- [ ] Update `load_sdxl_clip` to use `NexClipEncoder`:
    - Instantiate `CLIP-L` (using `normalize_clip_l_keys`).
    - Instantiate `CLIP-G` (using `normalize_clip_g_keys` + `use_projection=True`).
    - Wrap both in the `CLIP` container (backend/loader.py `CLIP` class likely needs adjustment to hold both, or `SDXLClipModel` wrapper needs to use `NexClipEncoder`).
    - **Crucial:** Ensure the `CLIP` container creates a valid `SDXLClipModel` equivalent that the `app.py` or samplers expect. *Wait, `loader.py` defines `CLIP` wrapper. We need to replace `sdxl_clip.SDXLClipModel` with a new `NexSDXLClipModel` composed of two `NexClipEncoders`.*

### 3. Verification
- [ ] Create `tests/test_p3m8_w02_sdxl_clip_parity.py`.
    - Load SDXL checkpoint.
    - Run inference with `ldm_patched` CLIP-G.
    - Run inference with `backend/clip.py` CLIP-G.
    - Assert hidden states and pooled outputs match (atol < 1e-4, FP16 might be slightly noisier).

## Deliverables
- Updated `backend/clip.py`
- Updated `backend/loader.py`
- `tests/test_p3m8_w02_sdxl_clip_parity.py`
