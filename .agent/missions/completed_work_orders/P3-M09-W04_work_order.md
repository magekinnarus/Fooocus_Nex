# Work Order: P3-M09-W04 — LoRA Support in app.py

**Status:** Pending (depends on W03)
**Owner:** Role 2 (Implementor)
**Date:** 2026-02-22

## Objective
Add LoRA loading and application to `app.py` via JSON config. Verify that LoRA patching produces visible style changes and that the patch/unpatch cycle works cleanly.

## Context
Ref: `.agent/missions/active/P3-M09_mission_brief.md`
Ref: `Fooocus_Nex/ldm_patched/modules/lora.py` — LoRA file parsing (to be extracted to `backend/lora.py`)

## CM Role 1 Review & Mandatory References
> [!IMPORTANT]
> **Mandatory Reading for Role 2:**
> - [.agent/summaries/05_Local_Environment_Guidelines.md](file:///d:/AI/Fooocus_revision/.agent/summaries/05_Local_Environment_Guidelines.md) (Hardware constraints & Git rules)
> - [.agent/summaries/04_Inference_Architectural_Guideline.md](file:///d:/AI/Fooocus_revision/.agent/summaries/04_Inference_Architectural_Guideline.md) (Memory management contracts)
> - [.agent/missions/active/P3-M09-W03_work_report.md](file:///d:/AI/Fooocus_revision/.agent/missions/active/P3-M09-W03_work_report.md) (NexModelPatcher details)

**Role 1 Technical Notes:**
- Extract `load_lora`, `model_lora_keys_clip`, and `model_lora_keys_unet` from `ldm_patched.modules.lora` into `backend/lora.py` to maintain backend independence.
- LoRA application must happen on the CPU (Stage 1) to avoid VRAM fragmentation.
- Use `sd_xl_offset_example-lora_1.0.safetensors` for SDXL verification.
- Ensure `calculate_weight` in `backend/patching.py` or `backend/weight_ops.py` is called correctly by the patcher.
- ZERO automated `git push` allowed.

## Tasks

### 1. Add LoRA Config to JSON
- [ ] Extend config schema:
  ```json
  "loras": [
      {"path": "path/to/lora.safetensors", "weight": 0.8},
      {"path": "path/to/another_lora.safetensors", "weight": 0.6}
  ]
  ```
- [ ] Support 0–5 LoRAs (matching Fooocus default max).

### 2. Implement LoRA Loading in `app.py`
- [ ] After model loading (Stage 1), parse LoRA files.
- [ ] Use `ldm_patched.modules.lora.load_lora` (or extract the parsing logic) to get LoRA state dicts.
- [ ] Apply LoRA patches to UNet via `NexModelPatcher.add_patches()`.
- [ ] Optionally apply LoRA patches to CLIP if the LoRA contains CLIP keys.

### 3. Verify Patch/Unpatch Cycle
- [ ] Generate image WITH LoRA applied.
- [ ] Call `unpatch_model()` after generation.
- [ ] Generate image WITHOUT LoRA (same prompt/seed).
- [ ] Confirm the two images are different (LoRA had effect) and the second matches no-LoRA baseline.

### 4. Verify with Real LoRA Files
- [ ] Use at least one SDXL LoRA (from `Fooocus_Nex/models/loras/` if available).
- [ ] Confirm visible style/quality change.
- [ ] Test with LoRA weight = 0.0 (should be identical to no LoRA).

### 5. Document Memory Behavior
- [ ] Note if LoRA application causes unexpected VRAM/RAM spikes.
- [ ] Compare memory signature with and without LoRA (if measurably different, document findings).

## Deliverables
- Updated `app.py` with LoRA support
- Updated config JSON files with LoRA examples
- Verification images (with/without LoRA)
- Memory notes (if anomalies found)
