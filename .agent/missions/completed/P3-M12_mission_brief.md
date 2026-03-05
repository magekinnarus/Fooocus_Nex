# Mission Brief: P3-M12 — Modules Structural Refactoring
**ID:** P3-M12
**Phase:** 3
**Date Issued:** 2026-02-25
**Status:** Ready
**Depends On:** P3-M11 (codebase cleanup)
**Work List:** `.agent/missions/active/P3-M12_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/06_Implementation_Observations_and_Insights.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/temp/refactoring_assessment.md`

## Objective

Decompose the two largest monolithic files in the Fooocus `modules/` layer — `async_worker.py` (1128 lines) and `webui.py` (1016 lines) — into modular, testable components, and extract self-contained pipeline modules for **Inpainting** and **ControlNet**. This structural refactoring reduces debugging complexity, fixes the known inpaint resolution bug, and creates clean insertion points for Phase 3.5's ControlNet, Inpainting, and JS canvas features.

### Why This Must Precede Phase 3.5

M10 demonstrated that the current monolithic structure causes cascading debugging issues:
- **W01** required 6 supplemental integration fixes caused by implicit coupling in `async_worker.py`
- **W03** surfaced closure-scoping bugs (`UnboundLocalError` in `process_prompt`)
- **W04** was an unplanned work order created entirely for UI stability bugs (consecutive hangs, duplicate results, broken buttons)

Phase 3.5 will add ControlNet pipeline stages, inpainting logic, and JS canvas elements — all of which interact with `async_worker.py`'s handler flow and `webui.py`'s layout. Without decomposition, every new feature risks the same debugging multiplier effect.

## Scope

### In Scope

#### `async_worker.py` Decomposition (W01)
- **`modules/pipeline/`** [NEW DIR] — Extract pipeline stage functions from `worker()` closures
- **`modules/task_state.py`** [NEW] — Formal `TaskState` dataclass replacing implicit `AsyncTask` mutation
- **`async_worker.py`** [MODIFY] — Strip to a dumb orchestrator: sequence stages → catch exceptions → yield UI updates
- **Remove 64 `gr.State(False)` placeholder slots** in `webui.py`/`AsyncTask.__init__` for dead enhance/describe features

#### `webui.py` Component Modularization (W02)
- **`modules/ui_components/`** [NEW DIR] — Break UI into component functions by feature area
- **`modules/ui_controller.py`** [NEW] — Centralized event binding and handler registration
- **`webui.py`** [MODIFY] — Reduced to top-level Gradio app assembly

#### InpaintPipeline & ControlNetPipeline Extraction (W03)
- **`modules/pipeline/inpaint.py`** [NEW] — Self-contained InpaintPipeline: `prepare → encode → [diffusion] → stitch`. Replaces current `inpaint_worker.py` + scattered `apply_inpaint` logic in `async_worker.py`. No module-level global state (`inpaint_worker.current_task`).
- **`modules/pipeline/controlnet.py`** [NEW] — Self-contained ControlNetPipeline: `preprocess → load → apply`. Replaces `apply_control_nets` + `apply_image_input` CN logic.
- **Fix inpaint 256×256 resolution bug** — BB generates at native resolution but output is wrong size due to rescale/stitch issue in post-processing.

### Out of Scope
- **`modules/config.py` refactoring** — Lower priority; additive structure is manageable. Defer to Phase 3.5 if pain points emerge.
- **Phase 3.5 feature implementation** — This mission only creates the structural foundation
- **Registry/extension system for ControlNet** — Deferred to M13
- **`ldm_patched` import reduction** — Covered by M11

## Reference Files
- `Fooocus_Nex/modules/async_worker.py` — 1128 lines, 31 outline items, God Function pattern
- `Fooocus_Nex/webui.py` — 1016 lines, 93 outline items, monolithic UI layout
- `.agent/temp/refactoring_assessment.md` — detailed analysis of both files
- `.agent/missions/completed/P3-M10_integration_bugs.md` — evidence for debugging complexity

## Constraints
- **Incremental approach** — each work order must leave Fooocus in a runnable state
- **Preserve all existing behavior** — this is a pure structural refactoring, not a feature change
- **Test through UI** — validation is done by generating images through the Fooocus Gradio interface
- **Closure extraction must preserve state contracts** — extracted functions must have explicit input/output parameters, no implicit closure captures
- **Backward compatibility** — if any external code references `async_worker.AsyncTask`, the class must remain importable from the same location (or re-exported)
- All testing on local GTX 1050 with **SD1.5 full checkpoint** and **SDXL GGUF**

## Deliverables
- [ ] `modules/pipeline/` directory with extracted pipeline stage modules
- [ ] `modules/pipeline/inpaint.py` — self-contained InpaintPipeline (no globals)
- [ ] `modules/pipeline/controlnet.py` — self-contained ControlNetPipeline
- [ ] `modules/task_state.py` with formal `TaskState` dataclass
- [ ] `async_worker.py` reduced to < 300 lines (orchestrator + task queue only)
- [ ] `modules/ui_components/` directory with feature-grouped UI component functions
- [ ] `modules/ui_controller.py` with centralized event bindings
- [ ] `webui.py` reduced to < 200 lines (app assembly only)
- [ ] 64 `gr.State(False)` placeholder slots removed from `webui.py`/`AsyncTask`
- [ ] Inpaint 256×256 resolution bug fixed
- [ ] txt2img + LoRA still works through Fooocus UI (regression test)
- [ ] Inpainting returns correct resolution output
- [ ] Skip/Stop buttons still work correctly
- [ ] All existing UI features (tabs, settings, presets) function identically

## Success Criteria
1. `async_worker.py` is under 300 lines — all pipeline logic lives in `modules/pipeline/`
2. `webui.py` is under 200 lines — all component definitions live in `modules/ui_components/`
3. No closure-captured state mutation — all pipeline functions have explicit parameters
4. InpaintPipeline has explicit stage handoffs (`InpaintContext`), no module-level globals
5. ControlNetPipeline dispatches per-type without hardcoded if/else blocks
6. Fooocus UI txt2img with LoRA works identically to post-M11 baseline
7. Inpainting returns correct full-resolution output (256×256 bug fixed)
8. Skip/Stop/consecutive generation all function correctly
9. No behavioral difference — same images produced with same seeds

## Work Orders
Registered in `P3-M12_work_list.md`:
- [x] `P3-M12-W01` — `async_worker.py` decomposition (TaskState, pipeline stages, orchestrator)
- [ ] `P3-M12-W02` — `webui.py` component modularization (UI components, event controller)
- [ ] `P3-M12-W03` — InpaintPipeline & ControlNetPipeline extraction (self-contained pipeline modules, inpaint resolution fix)

## Notes
- **`async_worker.py` closure inventory** (31 outline items): `progressbar`, `yield_result`, `build_image_wall`, `process_task`, `save_and_log`, `apply_control_nets`, `apply_vary`, `apply_inpaint`, `apply_outpaint`, `apply_upscale`, `apply_overrides`, `process_prompt`, `apply_freeu`, `patch_discrete`, `patch_edm`, `patch_samplers`, `set_hyper_sd_defaults`, `set_lightning_defaults`, `set_lcm_defaults`, `apply_image_input`, `prepare_upscale`, `stop_processing`, `handler`, `callback`. All are currently defined as closures inside `worker()`.
- **Natural pipeline stage groupings** for `async_worker.py`:
  - Preprocessing: `process_prompt`, `apply_overrides`, `apply_freeu`, `patch_samplers`
  - Image input: `apply_image_input`, `apply_vary`, `apply_inpaint`, `apply_outpaint`, `prepare_upscale`, `apply_upscale`
  - Inference: `process_task`, `callback`
  - Output: `yield_result`, `build_image_wall`, `save_and_log`
  - Control: `stop_processing`, `handler`
- **InpaintPipeline target architecture** (replaces `inpaint_worker.py` + `apply_inpaint`):
  - `prepare(image, mask, k)` → `InpaintContext` (BB, crop, upscaled image, masks)
  - `encode(context, vae)` → latents, latent_mask
  - `patch_model(context, unet, head_path, latent, mask)` → patched UNet
  - `stitch(context, generated_image)` → final composited image
  - Known bug: 256×256 output from 1024×1024 input — BB generates correctly at native resolution (confirmed via preview) but rescale/stitch returns wrong size. Root cause is in `InpaintWorker.post_process`.
- **ControlNetPipeline target architecture** (replaces `apply_control_nets` + CN wiring in `apply_image_input`):
  - `preprocess(cn_type, image, params)` → preprocessed image
  - `apply(cn_tasks, unet, positive_cond, negative_cond)` → patched conditions
  - Dispatch per CN type without hardcoded if/else. Prepares for M13 registry system.
- **`webui.py` natural component groupings**:
  - Main generation panel (prompt, generate button, gallery)
  - Image input panel (upscale, image prompt, inpaint tabs)
  - Advanced settings panel (model selection, performance, resolution, etc.)
  - Style panel
  - Settings panel (output format, metadata, etc.)
- The `handler()` function (lines 905–1106) is the most critical extraction target — it contains the entire generation flow logic and should become the new "dumb orchestrator" body.
- **64 `gr.State(False)` placeholder slots** exist in `webui.py` and `AsyncTask.__init__` for removed enhance/describe features. These maintain index alignment but are pure tech debt — must be cleaned up as part of W01.
