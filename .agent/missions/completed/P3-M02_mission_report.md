# Mission Report: P3-M02 Component-First Loader

**ID:** P3-M02
**Status:** Complete
**Date:** 2026-02-14
**Philosophy:** Clean Slate / Definition-Driven Backend

## 1. Executive Summary
Mission P3-M02 successfully established a "Clean Slate" loader in `Fooocus_Nex/backend/`. This loader is logically isolated from the legacy `modules/` package and uses a definition-driven architecture (`backend/defs/sdxl.py`).

However, the implementation process revealed a critical strategic insight: **The underlying `ldm_patched` runtime is more entangled than anticipated.** What should have been a simple task (instantiating a model) required navigating a maze of implicit dependencies, unwritten interface contracts, and circular imports within the ComfyUI core.

## 2. Deliverables
- **New Backend Package**: `Fooocus_Nex/backend/`
- **Definition-Driven Architecture**: `backend/defs/sdxl.py` (Data) vs `backend/loader.py` (Logic).
- **Atomic Extraction**: `extract_sdxl_components` correctly splits Checkpoints and Bundled CLIPs into 4 distinct buckets.
- **Deep Extraction Loading**: `load_sdxl_unet/clip/vae` instantiate models directly, bypassing legacy `sd.py` wrappers.
- **Native GGUF**: Integrated GGUF loading without legacy adapters.
- **Verification**: `tests/p3_m02_clean_loader_test.py` proves execution without the full UI.

## 3. Critical Findings: The Entanglement Trap
The team encountered significant friction performing what should be atomic operations.

### The "Organic Growth" Legacy
The `ldm_patched` modules (derived from ComfyUI) are not structured as a library of primitives. They are structured as a monolithic application state:
1.  **Implicit Configs**: Classes like `ModelBase` expect configuration dictionaries with specific, undocumented keys (e.g., `manual_cast_dtype`, `adm_in_channels`) that are usually constructed by 500+ line detection functions in `sd.py`.
2.  **Circular Dependencies**: Utilities in `model_sampling.py`, `utils.py`, and `model_management.py` import each other freely, making it nearly impossible to extract *just* the sampling logic or *just* the device management logic.
3.  **Hidden Global State**: methods often imply the existence of a global `model_management` state for VRAM estimation.

### Strategic Implication
We succeeded in P3-M02 by creating a **"Cleaned" Adapter**—a nice interface that hides the mess. But we did NOT fix the mess.
**Risk**: Future missions (Sampling, Memory) will face exponentially higher friction if we continue to build on top of `ldm_patched` without refactoring it.

## 4. Recommendations for Next Steps
1.  **Refactor vs. Wrap**: We must decide immediately whether to:
    *   **A) Contain**: Treat `ldm_patched` as a "Radioactive Core". Do not touch it. Wrap it in clean `backend/` interfaces (as done here).
    *   **B) Decontaminate**: Begin a campaign to rewrite `ldm_patched` modules into true primitives (e.g., `nex_core.py`).
    
    *Recommendation*: Given the velocity hit we took on P3-M02, **Option B (Decontaminate)** is likely necessary for the long-term health of "Assemble-Core". We cannot build a high-performance engine on a foundation we are afraid to touch.

## 5. Artifacts
- **Code**: `Fooocus_Nex/backend/`
- **Tests**: `tests/p3_m02_clean_loader_test.py`
