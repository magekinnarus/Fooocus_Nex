# Work List: P3-M12 — Modules Structural Refactoring
**Mission:** P3-M12
**Date Created:** 2026-02-25
**Last Updated:** 2026-02-28 (W03 superseded by M12-1)

| # | Work Order | Description | Status | Depends On |
|---|-----------|-------------|--------|------------|
| 1 | P3-M12-W01 | `async_worker.py` decomposition & Metadata Refactor — TaskState, modular pipeline, meta_parser modularization, GGUF/CLIP fixes. | [x] | — |
| 2 | P3-M12-W02 | `webui.py` component modularization (v2 — phased) — 4 atomic phases: Settings → Styles+Models → Advanced+Control+Inpaint, each with commit gate. Event controller deferred. | [x] | W01 verified |
| 3 | P3-M12-W03 | ~~InpaintPipeline & ControlNetPipeline extraction~~ | Superseded | W01 verified |
| 3.1 | **P3-M12-1** | **Inpainting Architecture Pivot (denoise_mask)** — Sub-mission replacing W03's inpaint scope. 3 work orders: W01 (denoise_mask plumbing + pipeline core), W02 (outpaint + context mask), W03 (integration + cleanup). See `P3-M12-1_mission_brief.md` | [ ] | W01 verified |
| 4 | P3-M12-W04 | ControlNetPipeline extraction (deferred from W03 scope split) | Draft | M12-1 |

## Sequencing Rationale
W01 targets `async_worker.py` first because it is the higher-risk, higher-value refactoring. The God Function pattern poses the greatest debugging risk for Phase 3.5. Getting this right before touching the UI layer ensures the pipeline logic is stable.

W02 targets `webui.py` second because UI component extraction is more mechanical (moving DOM definitions into separate files) and lower-risk. It also benefits from a stable `async_worker.py` since event handlers in `webui.py` interact with the worker thread.

W03 was superseded by sub-mission **P3-M12-1** after discovering fundamental architectural flaws in Fooocus's inpainting (InpaintHead dependency, forced 1:1 squashing, no pixel freezing). The inpainting pivot became a full architectural rewrite using `denoise_mask`, warranting its own mission with 3 work orders. ControlNet extraction was split into a separate W04.

## Change Log
- 2026-02-28: W03 superseded by P3-M12-1 (inpainting architecture pivot). ControlNet extraction deferred to W04.

