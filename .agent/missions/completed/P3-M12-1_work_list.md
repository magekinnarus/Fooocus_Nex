# Work List: P3-M12-1 — Inpainting Architecture Pivot (denoise_mask)
**Mission:** P3-M12-1
**Date Created:** 2026-02-28
**Last Updated:** 2026-02-28
**Overall Status:** Ready
**Brief Reference:** `.agent/missions/active/P3-M12-1_mission_brief.md`

## Purpose
Track work orders for the inpainting architecture pivot from legacy InpaintHead/1:1-squashing to
denoise_mask-based native-AR pipeline.

## Work Orders
| # | Work Order | Description | Status | Depends On |
|---|-----------|-------------|--------|------------|
| 1 | P3-M12-1-W01 | denoise_mask Plumbing & InpaintPipeline Core Rewrite — thread `denoise_mask` through `core.ksampler()`, rewrite `InpaintPipeline` with SDXL-snapped native-AR bounding boxes, pixelation primer, VAE encode with denoise_mask, and morphological blend stitching | [x] | P3-M12-W01 |
| 2 | P3-M12-1-W02 | Outpaint Canvas Extension & Context Mask — single-direction outpaint with edge-copy + pixelation primer, optional context mask (blue brush) support, edge-overflow canvas expansion, color guidance preview | [x] | W01 |
| 3 | P3-M12-1-W03 | Orchestrator Integration, Legacy Cleanup & Verification — update `apply_inpaint()` / `apply_outpaint()` / `process_task()`, remove `InpaintHead` / `solve_abcd` / legacy globals, full regression testing | [x] | W02 |
| 5 | P3-M12-1-W005 | Streamlined VRAM Lifecycle Management — explicit stage-based model loading/offloading (`VRAMStageManager`), fix progressive degradation on subsequent generations, align headless and UI memory paths | [ ] | None |

## Sequencing Rationale
**W01** establishes the critical foundation: `denoise_mask` must flow through the sampling chain before any
inpainting can work. The InpaintPipeline core (`prepare`, `encode`, `stitch`) is rewritten here because
these methods must produce the `denoise_mask` tensor that W01's plumbing will consume. This work order is
self-verifiable with a hardcoded test (bypass UI, directly call pipeline methods).

**W02** builds on W01's working pipeline to add outpaint canvas extension and context mask support. These
are features that modify the *input* to the pipeline (expanding canvases, combining masks) but don't change
the core `denoise_mask` mechanics. Separated to keep W01 focused on getting the fundamental pipeline working.

**W03** is the integration and cleanup pass. It connects the new pipeline to the existing orchestrator
(`apply_inpaint`, `apply_outpaint`, `process_task`), purges all legacy code (`InpaintHead`, `solve_abcd`,
`inpaint_worker` globals), and runs full regression testing. Separated last because integration changes are
highest-risk and benefit from a known-good pipeline.

## Change Log
- 2026-02-28: Created work list. Supersedes P3-M12-W03 (original scope split due to architectural pivot).
