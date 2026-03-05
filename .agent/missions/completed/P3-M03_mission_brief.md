# Mission Brief: P3-M03 (Backend Foundation: Resources & Conditioning)
**ID:** P3-M03
**Phase:** 3
**Date Created:** 2026-02-14
**Status:** Completed
**Mission Lead:** Antigravity
**Timeline:** 2026-02-15
**Design Reference:** `.agent/decisions/nex_module_design_principles.md`

## Objective
Build the **Backend Foundation**: `backend/resources.py` (Clean Device/Memory Resource Manager) and `backend/conditioning.py` (Clean CLIP/ADM Logic).
This mission establishes the "Clean Architecture" pattern by implementing the resource management system first (a prerequisite for loading models) and then the conditioning system (embedding prompts).

## The Strategy: "Clean Extraction" & "Standalone Verification"
> [!IMPORTANT]
> **Source of Truth:** All logic MUST be extracted from `ComfyUI_reference`.
> **Forbidden:** Do NOT import from `ldm_patched` under any circumstances.
> **Verification:** Since the full engine is not ready, every new module must be verified with a standalone script in `tests/`.

We are building a new resource management system (`backend.resources`) that is:
1.  **Clean:** Derived from `ComfyUI_reference/comfy/model_management.py`.
2.  **Modern:** Stripped of "Hybrid" legacy classes and ancient argument mappings.
3.  **Configurable:** Re-implements environment fixes (Colab, DirectML) as explicit configurations, not hardcoded hacks.
4.  **Verified:** Tested in isolation using `tests/test_backend_resources.py`.

## Core Task Sequence
1.  **Analyze & Extract**: Read `ComfyUI_reference/comfy/model_management.py`.
2.  **Implement `backend/resources.py`**:
    *   Copy the `ModelManagement` / `LoadedModel` logic.
    *   **Remove** the `legacy_mode` checks in `LoadedModel`.
    *   **Port** the `load_models_gpu` logic but remove `PatchSettings` injection (we will handle that in `modules.py` later).
    *   **Add** a configuration check for "Force High VRAM" (The Colab Fix).
3.  **Verify**: Create `tests/test_backend_resources.py` to:
    *   Import `backend.resources`.
    *   Mock a Model object.
    *   Verify `load_model_gpu` allocates memory and updates counters.

## Scope

### In Scope
1. **`timestep_embedding` extraction** — Copy the sinusoidal embedding function (20 lines of pure math) into `backend/` so it's ours. Zero `ldm_patched` dependency.
2. **`encode_sdxl_adm` function** — Standalone function implementing SDXL ADM conditioning (pooled CLIP + resolution embedding). Replaces the monkey-patched `sdxl_encode_adm_patched` from `modules/patch.py`. Uses our extracted `timestep_embedding`.
3. **`encode_sdxl_prompt` function** — Takes the CLIP container (from `backend/loader.py`) and a text string, returns `(cond_embedding, pooled_output)`. Clean function wrapping the tokenize → encode flow.
4. **Typed return values** — Use dataclass or named tuple per DP6.
5. **Add SDXL conditioning constants to `defs/sdxl.py`** — Timestep dim (256), ADM channel count (2816), default resolution, etc.

### Out of Scope
- CLIP model loading (already in `backend/loader.py`)
- Modifying `modules/patch.py` or `modules/patch_clip.py` (legacy path stays as-is)
- CLIP model internals / transformer architecture (stays in `ldm_patched`)
- Sampling, VAE, or any other pipeline stage

## Constraints
- Follow all 7 design principles from DR-001
- New code in `backend/conditioning.py` and `backend/defs/sdxl.py` only
- `conditioning.py` may import the CLIP container from `backend/loader.py` but must NOT import `model_management`, `conds`, or `ops`
- The `encode_sdxl_adm` function must match the behavior of `sdxl_encode_adm_patched` in `modules/patch.py` (the Fooocus version, not the ComfyUI original)
- Include the Dependency Inventory table per DR-001

## Deliverables
1. `backend/conditioning.py` — the module
2. Updated `backend/defs/sdxl.py` — new constants 
3. Verification script — proves conditioning output matches Fooocus's monkey-patched version
4. Work report documenting dependency inventory

## Success Criteria
1. `encode_sdxl_prompt` produces identical `(cond, pooled)` tensors as the current Fooocus path
2. `encode_sdxl_adm` produces identical ADM tensor as `sdxl_encode_adm_patched`
3. `conditioning.py` has zero imports from `ldm_patched.modules.model_management`, `ldm_patched.modules.conds`, or `ldm_patched.modules.ops`
4. `timestep_embedding` has zero `ldm_patched` imports whatsoever
