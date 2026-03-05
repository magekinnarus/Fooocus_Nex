# Mission Work List: P3-M04 — Sampling Engine

**Mission ID:** P3-M04
**Status:** Completed
**Current Phase:** Closed

## Work Orders

The implementation should be split into **two distinct sessions** to manage complexity.

### Session 1: Core Logic Extraction
| ID | Objective | Status | Assignee |
|----|-----------|--------|----------|
| **P3-M04-W01** | Extract CFGGuider + sampling_function + cfg_function | [x] Done | Antigravity |

### Session 2: API & Verification
| ID | Objective | Status | Assignee |
|----|-----------|--------|----------|
| **P3-M04-W02** | Extract schedulers + KSAMPLER + integrate into `sample_sdxl()` API | [x] Done | Antigravity |
| **P3-M04-W03** | Verification & dependency inventory | [x] Done | Antigravity |

## Implementation Plan

1.  **W01: Core Sampling Logic**
    -   Create `Fooocus_Nex/backend/sampling.py`.
    -   Extract `CFGGuider` class from `ComfyUI_reference/comfy/samplers.py`.
    -   Extract `sampling_function` (inner sampling loop).
    -   Extract `cfg_function` (guidance calculation).
    -   Ensure no `ldm_patched` imports.

2.  **W02: Samplers & Schedulers**
    -   Extract all 36 sampler names/mappings from `KSAMPLER_NAMES`.
    -   Extract 9 scheduler functions from `SCHEDULER_HANDLERS`.
    -   Implement `sample_sdxl()` top-level function.
    -   Connect `process_conds` (SDXL conditioning resolution).

3.  **W03: Verification**
    -   Create `tests/test_backend_sampling.py`.
    -   Verify standard SDXL sampling flow.
    -   Verify CFG guidance.
    -   Document dependencies.

## Key References for Implementors
**(Ensure you read these before starting)**

-   **Mission Brief:** `.agent/missions/active/P3-M04_mission_brief.md` (Scope & Constraints)
-   **Reference Trace:** `.agent/reference/P3-M01_reference_trace.md` (Stage 3: Sampling)
-   **Anti-Pattern Guide:** `.agent/reference/ldm_patched_analysis.md` (What NOT to do)
-   **Source Code:** `ComfyUI_reference/comfy/samplers.py` (Primary Source)

## Blockers / Risks
-   **Dependency Depth:** `CFGGuider` chain is deep. Need to simplify without breaking.
-   **ModelPatcher:** Explicit dependency on `ModelPatcher` is required. Ensure it comes from a clean source (`ComfyUI_reference` preferred if possible, or `Fooocus_Nex.backend.loader` if wrapped there).
-   **Future Memory Scaling:** Current simplified batching is sufficient for single-pass generation. Future Inpainting/ControlNet missions may require re-implementing memory estimation (`model.memory_required`) to prevent OOMs with large conditioning batches.
