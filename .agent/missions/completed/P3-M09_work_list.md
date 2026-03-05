# Mission Work List: P3-M09 — Patching, Quality Features & Backend Refinement

**Mission ID:** P3-M09
**Mission Brief:** `P3-M09_mission_brief.md`

## Work Orders

| ID | Status | Description | Assignee |
| :--- | :--- | :--- | :--- |
| **P3-M09-W01** | Completed | **Decompose `sampling.py`:** Extract condition processing into `backend/cond_utils.py`. Pure refactor, no behavior change. | Role 2 |
| **P3-M09-W02** | Completed | **Fooocus Quality Features:** Copy `anisotropic.py` to backend. Add sharpness, adaptive CFG, ADM scaling, timed ADM. Update `app.py` config. | Role 2 |
| **P3-M09-W03** | Completed | **Extract NexModelPatcher:** Build `backend/patching.py` with `calculate_weight` and core ModelPatcher. Update `loader.py`. | Role 2 |
| **P3-M09-W04** | Completed | **LoRA in app.py:** Add LoRA config to JSON, load and apply LoRAs, verify with real LoRA files. | Role 2 |

## Execution Order
- W01 → W02 (sequential: decompose first, then add features to the clean structure)
- W03 → W04 (sequential: extract patcher first, then wire LoRA into app.py)
- W01/W02 and W03/W04 can run in parallel tracks if needed

## Notes
- W01 changes only file locations, not function signatures. All imports that reference `sampling.X` for moved functions should be updated to `cond_utils.X`.
- W02 requires exposing `diffusion_progress` (0.0→1.0) through the sampling pipeline — currently this is only available inside the UNet forward monkey-patch.
- W03 must handle `GGUFModelPatcher` compatibility (extends `ModelPatcher`).
- If memory signature anomalies appear during W04 LoRA testing, document them.
