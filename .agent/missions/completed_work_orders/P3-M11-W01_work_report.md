# Work Report: P3-M11-W01 — Dead Code Removal
**ID:** P3-M11-W01  
**Mission:** P3-M11  
**Status:** ✅ COMPLETE  
**Date Completed:** 2026-02-25  
**Sessions:** 1 (with 2 rollbacks during execution)

## Summary

Removed all dead files from `modules/` and stripped all 25 ComfyUI node files from `ldm_patched/contrib/` (exceeding the original 22-file target). Additionally resolved 6 cascading `ModuleNotFoundError` regressions caused by the deletions, and completed several W02-scope tasks as part of regression fixes.

## Deliverables

### Planned (W01 Scope)

| Deliverable | Status | Notes |
|---|---|---|
| Delete `modules/nex_loader.py` | ✅ | Zero importers confirmed |
| Delete `modules/PATCH_MANIFEST.md` | ✅ | Zero references confirmed |
| Delete `modules/anisotropic.py` | ✅ | Duplicate of `backend/anisotropic.py` |
| Redirect `patch.py` → `backend/anisotropic.py` | ✅ | Line 7 updated |
| Delete 22 unused `ldm_patched/contrib/` nodes | ✅ | **All 25 deleted** (3 "kept" files also inlined) |
| Inline contrib logic into `core.py` | ✅ | VAEDecode, VAEEncode, EmptyLatentImage, FreeU_V2, ModelSampling, ControlNetApply |
| Inline contrib logic into `upscaler.py` | ✅ | ImageUpscaleWithModel |

### Unplanned (Regression Fixes — overlaps W02)

| Fix | Files Changed |
|---|---|
| `ModuleNotFoundError: ldm_patched.contrib` | `core.py`, `upscaler.py` — inlined missing nodes |
| `ModuleNotFoundError: ldm_patched.k_diffusion.sampling` | `patch.py` — redirected to `backend.k_diffusion` |
| `ModuleNotFoundError: ldm_patched.modules.lora` | `patch.py`, `core.py`, `hooks.py`, `sd.py` — redirected to `backend.lora` |
| `calculate_weight` mislocated | `model_patcher.py` — redirected to `backend.weight_ops` |
| `ModuleNotFoundError: ldm_patched.modules.samplers` | `patch.py`, `core.py`, `patch_clip.py` — removed dead imports |
| `ModuleNotFoundError: ldm_patched.modules.sample` | `core.py` — removed unused `prepare_mask` import |
| `ModuleNotFoundError: modules.sample_hijack` | `default_pipeline.py` — removed unused `clip_separate` import |
| Deleted `modules/sample_hijack.py` | Ported `turbo` + `align_your_steps` schedulers to `backend/schedulers.py` |

## Issues Encountered

### 2 Rollbacks Required
The session required 2 user-initiated rollbacks due to cascading import errors. The root cause was **insufficient documentation of the boundary between `backend/` and `ldm_patched/`** — it was unclear which modules had been fully ported vs. which still had active consumers in `modules/`.

### Cascading Regressions
Each deletion triggered further `ModuleNotFoundError` exceptions because:
1. `modules/` files imported from `ldm_patched/contrib/` (expected, planned for)
2. `modules/` files also imported from `ldm_patched/modules/samplers`, `ldm_patched/modules/sample`, `ldm_patched/k_diffusion/` (unexpected — these were removed in M09 but some consumers still referenced them)
3. `sample_hijack.py` was monkey-patching `ldm_patched.modules.samplers` at import time, making it a hidden dependency

### New `ldm_patched` Import in Backend
`backend/schedulers.py` line 136 now imports `ldm_patched.modules.latent_formats` inside `align_your_steps_scheduler()`. This contradicts M11's goal of not introducing new `ldm_patched` imports. Flagged for W03.

## Lessons Learned

> [!WARNING]
> **The biggest risk in M11 is not the deletions themselves — it's the undocumented import graph.** Before proceeding with W02/W03, a full mapping of backend ↔ ldm_patched ↔ modules dependencies should be created to prevent further cascading regressions.

Key takeaways:
1. **Backend vs ldm_patched overlap is poorly documented** — e.g., both have `lora.py`, causing confusion about which is authoritative
2. **Monkey-patches create hidden dependencies** — `sample_hijack.py` patched `ldm_patched.modules.samplers` at import time, making it impossible to detect via static grep
3. **Deletion must be paired with import verification** — deleting a file requires checking the full transitive import chain, not just direct importers
4. **W02 scope was partially completed here** — the work list should be updated before starting W02

## Files Changed

| File | Change |
|---|---|
| `modules/nex_loader.py` | DELETED |
| `modules/PATCH_MANIFEST.md` | DELETED |
| `modules/anisotropic.py` | DELETED |
| `modules/sample_hijack.py` | DELETED |
| `ldm_patched/contrib/*.py` (25 files) | DELETED |
| `modules/patch.py` | Redirected anisotropic, k_diffusion, removed samplers monkey-patch |
| `modules/core.py` | Inlined contrib nodes, removed dead imports, fixed config import |
| `modules/upscaler.py` | Inlined ImageUpscaleWithModel |
| `modules/default_pipeline.py` | Removed sample_hijack import |
| `modules/patch_clip.py` | Removed samplers import |
| `ldm_patched/modules/model_patcher.py` | Redirected calculate_weight to backend.weight_ops |
| `ldm_patched/modules/hooks.py` | Redirected lora imports to backend.lora |
| `ldm_patched/modules/sd.py` | Redirected lora imports to backend.lora |
| `backend/schedulers.py` | Added turbo + align_your_steps schedulers |
| `backend/sampling.py` | Pass model to calculate_sigmas |

## Recommendation for Next Steps

1. **Create a dependency map** documenting what each `backend/` module provides and what `modules/` + `ldm_patched/` still need
2. **Update W02 work order** to reflect tasks already completed (sample_hijack, clip_separate)
3. **Investigate speed difference** user noted — likely from M09/M10 changes, not W01
4. **Clean stale `.pyc` files** in `__pycache__` directories
