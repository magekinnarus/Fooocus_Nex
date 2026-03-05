# Work Order: P3-M08-W01 — SD1.5 CLIP Extraction

**Status:** Ready
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-18

## Objective
Extract the SD1.5 CLIP text encoding pipeline from `ldm_patched` into a self-contained `backend/clip.py` module. Remove all `ldm_patched` dependencies to fix precision and normalization issues. Ensure the extracted module produces identical outputs to the original.

## Context
Ref: `.agent/missions/active/P3-M08_mission_brief.md`
Ref: `.agent/reference/P3_ldm_patched_clip_extraction_map.md`

## Tasks

### 1. Create `backend/clip.py`
- [ ] Define `NexTokenizer` class (wrapping `transformers.CLIPTokenizer`).
    - Implement `tokenize_with_weights` logic from `sd1_clip.py`.
- [ ] Define internal `NexLinear`, `NexLayerNorm`, `NexEmbedding` classes.
    - Mimic `ops.manual_cast` behavior: cast weights to input dtype during forward pass.
    - Do NOT import `ldm_patched`.
- [ ] Define `NexClipEncoder` class.
    - Port `CLIPTextModel_`, `CLIPEncoder`, `CLIPLayer`, `CLIPAttention`, `CLIPMLP` from `clip_model.py`.
    - Inline `optimized_attention` (simple PyTorch SDP with FP32 guard).
    - Ensure `embeddings` are always FP32.
    - Implement `text_projection` loading logic (only if `use_projection=True`).

### 2. Implement Unified Key Normalization (in `backend/clip.py`)
- [ ] Implement `normalize_clip_l_keys(sd)`:
    - Strip prefixes: `clip_l.`, `cond_stage_model.transformer.`, etc.
    - Return clean HF keys.
- [ ] Implement `load_sd` in `NexClipEncoder` to accept normalized keys.

### 3. Update `backend/loader.py`
- [ ] Implement `load_sd15_clip` using `NexClipEncoder`.
    - Use `normalize_clip_l_keys` on the state dict.
    - Initialize `NexClipEncoder` with `use_projection=False` for SD1.5.

### 4. Verification
- [ ] Create `tests/test_p3m8_w01_clip_parity.py`.
    - Load SD1.5 checkpoint.
    - Run inference with `ldm_patched` CLIP (current path).
    - Run inference with `backend/clip.py` CLIP (new path).
    - Assert specific hidden states match (atol < 1e-5).
    - Assert pooled outputs match (if applicable, SD1.5 doesn't use pooled).

## Deliverables
- `backend/clip.py`
- Updated `backend/loader.py`
- `tests/test_p3m8_w01_clip_parity.py`
