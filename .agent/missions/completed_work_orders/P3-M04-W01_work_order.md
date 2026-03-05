# Work Order: P3-M04-W01 — Core Sampling Logic

**Mission:** P3-M04
**Work Order ID:** P3-M04-W01
**Status:** Ready
**Depends On:** None

## Objective
Extract the core sampling logic (`CFGGuider`, `sampling_function`) from `ComfyUI_reference` into `Fooocus_Nex/backend/sampling.py`.

## Context
The `CFGGuider` class orchestrates the sampling process, managing the condition/uncond inputs and applying the CFG scale. The `sampling_function` is the inner loop called by the k-diffusion sampler at every step.

## Requirements

1.  **Create Module:** `Fooocus_Nex/backend/sampling.py`.
2.  **Extract `CFGGuider`:**
    -   Source: `ComfyUI_reference/comfy/samplers.py`
    -   Keep dependencies minimal.
    -   It will need `ModelPatcher` (ideally typed as `Any` or imported if clean).
3.  **Extract `sampling_function`:**
    -   Source: `ComfyUI_reference/comfy/samplers.py`
    -   This function calculates `cond_pred` and `uncond_pred`.
4.  **Extract `cfg_function`:**
    -   Source: `ComfyUI_reference/comfy/samplers.py` (inside or near `sampling_function`).
    -   Logic: `uncond + (cond - uncond) * scale`.
5.  **Clean Imports:**
    -   **NO** imports from `ldm_patched`.
    -   Use `Fooocus_Nex.backend.conditioning` for typing/definitions where appropriate.
    -   Use `Fooocus_Nex.backend.resources` if device management is needed (though `ModelPatcher` handles most of it).

## Verification
-   Run `pytest tests/test_backend_sampling.py` (created in W03, but you can create a stub to verify import).
-   Ensure class instantiates without errors.

## Notes
-   The `process_conds` logic called inside `sampling_function` might need to be stubbed or imported from `ComfyUI_reference.comfy.sampler_helpers` or extracted. **Decision:** Extract it if it's small, or use a helper. The mission brief mentions `process_conds` as part of W02, but `sampling_function` calls it. You might need to add a placeholder or a minimal implementation in W01.
