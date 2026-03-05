# Work Order: P3-M08-W03 — Integration & Verification

**Status:** Ready
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-18

## Objective
Integrate the newly verified `backend/clip.py` into the main application logic (`app.py`), enabling full SD1.5 and SDXL inference using the extracted CLIP module. This is the final verification step to confirm that the extraction works end-to-end, producing valid images for both architectures.

## Context
Ref: `.agent/missions/active/P3-M08_mission_brief.md`
Ref: `.agent/reference/P3_ldm_patched_clip_extraction_map.md`

## Tasks

### 1. Update `app.py`
- [ ] Modify `app.py` to import `backend.loader` and `backend.clip` instead of `ldm_patched`.
- [ ] Ensure `app.py` uses recent updates in `backend/loader.py` (which now uses `NexClipEncoder`).
- [ ] Verify that model loading paths in `app.py` correctly utilize the unified key normalization from W01/W02.

### 2. Verify SD1.5 Inference
- [ ] Run `app.py` with an SD1.5 checkpoint (e.g., `realistic_vision`).
- [ ] Generate an image.
- [ ] Verify success criteria:
    - Image is recognizable and coherent (not noise/black).
    - No NaNs in console output.
    - Performance is comparable to previous implementation.

### 3. Verify SDXL Inference
- [ ] Run `app.py` with an SDXL checkpoint (e.g., `JuggernautXL` or `animagine`).
- [ ] Generate an image.
- [ ] Verify success criteria:
    - Image is recognizable and high quality.
    - CLIP-G/L concatenation works correctly (implied by good prompt adherence).

## Deliverables
- Updated `app.py`
- Successful SD1.5 generation (screenshot/log)
- Successful SDXL generation (screenshot/log)
- Final Mission Report `P3-M08_mission_report.md`
