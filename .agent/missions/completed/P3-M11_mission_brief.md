# Mission Brief: P3-M11 — Codebase Cleanup & Dead Code Removal
**ID:** P3-M11
**Phase:** 3
**Date Issued:** 2026-02-25
**Status:** COMPLETED
**Depends On:** P3-M10 (completed)
**Work List:** `.agent/missions/active/P3-M11_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/06_Implementation_Observations_and_Insights.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`

## Objective

Eliminate dead code, consolidate duplicate/bridge files, and reduce `ldm_patched` coupling across the codebase. This mission removes confusion-causing remnants from Phase 1–3 extractions, moves shared dependencies (GGUF) to their proper layer, and strips ~200 KB of unused ComfyUI contrib nodes. The result is a leaner, more navigable codebase ready for structural refactoring (M12) and Phase 3.5 feature work.

## Scope

### In Scope

#### Dead Code Removal (W01)
- **`modules/nex_loader.py`** [DELETE] — Zero importers, superseded by `backend/loader.py`
- **`modules/PATCH_MANIFEST.md`** [DELETE] — Zero references, obsolete tracking doc
- **`modules/anisotropic.py`** [DELETE] — Exact duplicate of `backend/anisotropic.py`; redirect `patch.py` import
- **22 unused `ldm_patched/contrib/` node files** [DELETE] — Only 3 of 25 are actively imported by Fooocus code

#### Bridge File Consolidation (W02)
- **`modules/gguf/`** [MOVE] — Move to `backend/gguf/`; update imports in `backend/loader.py`
- **`modules/sample_hijack.py`** [MODIFY/DELETE] — Absorb `clip_separate()` into `default_pipeline.py`; remove `sample_hacked` and `calculate_sigmas_scheduler_hacked` monkey-patches (backend already handles sampling/scheduling)
- **`modules/lora.py`** [MODIFY/DELETE] — Consolidate `match_lora()` into `backend/lora.py`; update `core.py` import
- **`modules/ops.py`** [EVALUATE] — Determine if `use_patched_ops` can be inlined into its 2 consumers

#### Backend `ldm_patched` Import Reduction (W03)
- **`backend/lora.py`** [MODIFY] — Remove `ldm_patched.modules.model_base` type-checking for unsupported architectures (StableCascade, HunyuanDiT, GenmoMochi, HunyuanVideo, HiDream, ACEStep)
- **`backend/utils.py`** [MODIFY] — Inline `ldm_patched.modules.checkpoint_pickle` (7-line safe unpickler) to remove the import
- **`backend/weight_ops.py`** [EVALUATE] — Assess `ldm_patched.modules.weight_adapter` dependency

#### UI Cleanup & Backend Wiring (W04)
- **`launch.py`** [MODIFY] — Fix double console load display by guarding module-scope side-effects
- **`webui.py`** [MODIFY] — Remove dead Enhance tab, Describe tab, FreeU tab, SAM/GroundingDINO options
- **`modules/flags.py`** [MODIFY] — Wire sampler/scheduler lists from `backend/sampling.py` and `backend/schedulers.py`; remove dead feature constants
- **`args_manager.py`** [MODIFY] — Remove dead enhance/describe CLI arguments
- **`modules/async_worker.py`** [MODIFY] — Update `AsyncTask` arg unpacking to match reduced `ctrls` list

### Out of Scope
- **`modules/patch.py`** — Still needed for ControlNet/Inpainting patches (#6, #7) and `build_loaded()` corruption handler
- **`modules/patch_clip.py`** / **`modules/patch_precision.py`** — Kohya consistency patches, still needed for all inference
- **Structural refactoring of `async_worker.py` and `webui.py`** — Deferred to P3-M12
- **Deep `ldm_patched/modules/` extraction** (model_base, ops, sd.py) — Phase 4 territory
- **`ldm_patched/ldm/`** directory — Core model architecture definitions, still needed
- **`extras/` directory** — IP-Adapter, VAE interpose, censor; out of scope
- **New feature additions** (background removal, object removal, GAN upscale, SUPIR) — Deferred to P3-M13/M14

## Reference Files
- `Fooocus_Nex/modules/nex_loader.py` — dead file, zero importers
- `Fooocus_Nex/modules/anisotropic.py` — duplicate of `backend/anisotropic.py`
- `Fooocus_Nex/modules/sample_hijack.py` — bridge file with monkey-patches
- `Fooocus_Nex/modules/lora.py` — bridge `match_lora()` function
- `Fooocus_Nex/modules/gguf/` — GGUF loading (4 files, used by `backend/loader.py`)
- `Fooocus_Nex/backend/lora.py` — has unsupported model type checks
- `Fooocus_Nex/backend/utils.py` — has `checkpoint_pickle` import
- `Fooocus_Nex/ldm_patched/contrib/` — 25 ComfyUI node files, 22 unused

## Constraints
- **Incremental approach** — each work order must leave Fooocus in a runnable state
- **Test through UI** — validation is done by generating images through the Fooocus Gradio interface
- **Preserve ControlNet/Inpainting patches** — `patch.py` patches #6 and #7 must remain functional
- **Preserve `build_loaded()` corruption handler** — safety feature in `patch.py`
- All testing on local GTX 1050 with **SD1.5 full checkpoint** and **SDXL GGUF**

## Deliverables
- [ ] Dead files removed (`nex_loader.py`, `PATCH_MANIFEST.md`, `modules/anisotropic.py`, 22 contrib nodes)
- [ ] `modules/gguf/` moved to `backend/gguf/` with all imports updated
- [ ] `modules/sample_hijack.py` eliminated or reduced to only `clip_separate()`
- [ ] `modules/lora.py` `match_lora()` consolidated into `backend/lora.py`
- [ ] Backend `ldm_patched` imports reduced (unsupported model types removed, checkpoint_pickle inlined)
- [ ] Console startup info prints exactly once (no double-load)
- [ ] Enhance, Describe, FreeU, SAM/DINO UI elements removed
- [ ] Sampler/scheduler dropdowns populated from backend
- [ ] txt2img + LoRA still works through Fooocus UI (regression test)
- [ ] `app.py` headless tests still pass (SD1.5 + SDXL)

## Success Criteria
1. Zero dead/duplicate files in `modules/` directory
2. `modules/gguf/` no longer exists (moved to `backend/gguf/`)
3. `ldm_patched/contrib/` reduced from 25 to 3 active files
4. `backend/` has zero `ldm_patched.modules.model_base` imports (except mandatory structural ones)
5. Fooocus UI txt2img with LoRA works identically to post-M10 baseline
6. No new `ldm_patched` imports introduced anywhere
7. Console startup prints exactly once
8. UI has no dead/disabled feature tabs (Enhance, Describe, FreeU, SAM)

## Work Orders
Registered in `P3-M11_work_list.md`:
- `P3-M11-W01` — Dead Code Removal (modules + contrib) ✅
- `P3-M11-W02` — Bridge File Consolidation (gguf, lora, ops)
- `P3-M11-W03` — Backend `ldm_patched` Import Reduction
- `P3-M11-W04` — UI Cleanup & Backend Wiring (double-load, dead UI, sampler/scheduler)

## Notes
- The 3 actively used `ldm_patched/contrib/` files are: `external_align_your_steps.py`, `external_custom_sampler.py`, `external_upscale_model.py`. Only these 3 should be retained.
- `modules/ops.py` provides `use_patched_ops` used by `patch_clip.py` and `extras/ip_adapter.py`. It should be evaluated in W02 but may need to stay if inlining is disruptive.
- `backend/weight_ops.py` has a lazy import of `ldm_patched.modules.weight_adapter` — this is deep infrastructure shared with `model_patcher.py` and may not be safe to extract in this mission. Evaluate in W03 and defer if risky.
- `modules/sample_hijack.py` monkey-patches `ldm_patched.modules.samplers` at module scope (lines 158–159). These patches feed the ControlNet/Inpainting paths which still use `ldm_patched`. Removal must verify that the txt2img path (which uses `backend/sampling.py`) is unaffected.
