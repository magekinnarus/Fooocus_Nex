# Work Order: P3-M04-W03 — Verification & Inventory

**Mission:** P3-M04
**Work Order ID:** P3-M04-W03
**Status:** Completed
**Depends On:** P3-M04-W02

## Objective
Verify the `Fooocus_Nex/backend/sampling.py` module with a standalone test suite and document all external dependencies.

## Requirements

1.  **Create Test Suite:**
    -   File: `tests/test_backend_sampling.py`
    -   **Test 1: Sampler/Scheduler registry:** Verify all 36 samplers and 9 schedulers are present and callable.
    -   **Test 2: Mock Sampling Loop:**
        -   Mock the `ModelPatcher` and `UNet`.
        -   Run `sample_sdxl` for 1-2 steps.
        -   Verify output shape (should be `(B, 4, H, W)`).
        -   Verify that `CFG` scales change the output (run with scale 1.0 vs 7.0 and assert result difference).
    -   **Test 3: Dependency Check:**
        -   Assert NO imports from `ldm_patched`.
        -   Verify imports are only from `Fooocus_Nex.backend`, `torch`, `numpy`, `tqdm` (if used), `safetensors` etc.

2.  **Dependency Inventory:**
    -   Create a markdown table in `P3-M04_work_report.md` listing all imports used in `sampling.py`.
    -   Flag any "risk" imports (e.g., if you had to import something from `ComfyUI_reference` directly instead of copying, or if `ModelPatcher` caused issues).

3.  **Documentation:**
    -   Add docstrings to public API (`sample_sdxl`, `CFGGuider`).
    -   Ensure type hints are complete.

## Acceptance Criteria
-   `pytest tests/test_backend_sampling.py` passes.
-   Code is fully typed.
-   Work report contains the dependency inventory.
