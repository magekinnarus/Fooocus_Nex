# Mission Work List: P3-M08

**Mission:** CLIP Pipeline Extraction (SD1.5 → SDXL)
**Status:** In Progress
**Depends On:** P3-M07

## Work Orders

| ID | Status | Owner | Description |
|---|---|---|---|
| **P3-M08-W01** | **Done** | Role 2 | Extract SD1.5 CLIP pipeline into `backend/clip.py`. Implement unified normalization. Fix `backend/loader.py` for SD1.5. Parity verification. |
| **P3-M08-W02** | **Done** | Role 2 | Extend `backend/clip.py` for SDXL (CLIP-G, OpenCLIP keys, projection). Parity verification for SDXL. |
| **P3-M08-W03** | **Done** | Role 2 | Integration: Update `app.py`, full inference validation (SD1.5 & SDXL). |

## Clarification Notes

- **W01:** `backend/clip.py` must contain internal `NexLinear` and `NexLayerNorm` classes to handle auto-casting (simulating `ops.cast_bias_weight` without the `model_management` dependency).
- **W01:** `NexTokenizer` should wrap `transformers.CLIPTokenizer` but handle the token weighting logic from `sd1_clip.py`.
- **General:** `backend/resources.py` will be used for device management, but not for casting.
