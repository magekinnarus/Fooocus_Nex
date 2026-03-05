# Mission Brief: P3-M06 — End-to-End Validation

**ID:** P3-M06
**Phase:** 3
**Date Issued:** 2026-02-16
**Status:** Ready
**Depends On:** P3-M05
**Work List:** `.agent/missions/active/P3-M06_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`

## Objective
Validate the complete Phase 3 backend extraction by generating an image end-to-end using a standalone Python script that relies **only** on the newly implemented `backend/` modules. This proves that the core SDXL pipeline (load -> condition -> sample -> decode) is fully decoupled from the legacy `ldm_patched` codebase where intended.

## Scope

### In Scope
- **Verification Script:** Create `scripts/validate_p3.py` (or similar).
- **Pipeline Segments:**
  - **Loader:** Use `backend.loader` to load SDXL (Safetensors or GGUF).
  - **Conditioning:** Use `backend.conditioning` for CLIP text encoding (L/G) and ADM.
  - **Sampling:** Use `backend.sampling` for K-Diffusion/Scheduler execution.
  - **Decode:** Use `backend.decode` for VAE decoding (tiled/standard).
- **Resource Management:** Ensure `backend.resources` correctly manages VRAM transitions between stages.
- **Dependency Isolation:** Verify clean separation from `ldm_patched`.

### Out of Scope
- **UI Integration:** No Gradio or Frontend work.
- **Advanced Features:** No ControlNet, IP-Adapter, or complex patching (deferred to M07/Phase 4).
- **Performance Optimization:** Focus on correctness.

## Reference Files
- `backend/loader.py`
- `backend/conditioning.py`
- `backend/sampling.py`
- `backend/decode.py`
- `run.py` (reference)

## Constraints
- **Environment:** Local development environment (GTX 1050 4GB target).
- **Dependencies:** The script must NOT import `ldm_patched` directly. All calls must go through `backend/` interfaces.
- **Models:** Support standard SDXL checkpoints and GGUF variants.

## Deliverables
- [ ] `scripts/validate_p3.py`: The standalone generation script.
- [ ] `completed_image.png`: Evidence of successful generation.
- [ ] `P3-M06_mission_report.md`: Summary of findings, any lingering coupling discovered.

## Success Criteria
1. **Execution:** The script runs from start to finish without crashing on a 4GB VRAM target.
2. **Quality:** The output image matches the prompt and is coherent.
3. **Isolation:** Source code analysis of the script confirms no direct imports of legacy modules.
4. **Resource Check:** Setup properly frees memory between stages.

## Work Orders
Registered in mission work-list (`P3-M06_work_list.md`):
- `[P3-M06-W01]` Create Validation Script
