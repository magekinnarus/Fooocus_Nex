# Mission Report: P3-M12-1 — Inpainting Architecture Pivot (denoise_mask)
**ID:** P3-M12-1
**Phase:** 3
**Date Completed:** 2026-03-04
**Status:** COMPLETED (with deferred scope)

## Mission Executive Summary
Mission P3-M12-1 achieved its primary objective: replacing the legacy Fooocus inpainting architecture (`InpaintHead` monkey-patch + forced 1:1 square squashing) with a modern `denoise_mask`-based pipeline. The new architecture preserves native aspect ratios, mathematically freezes original pixels during sampling, and retains Fooocus's superior morphological blending for seamless stitching.

Over three core work orders, the mission delivered end-to-end `denoise_mask` plumbing through the sampling chain, a complete `InpaintPipeline` rewrite with SDXL resolution snapping, single-direction outpainting with 8×8 pixelation primer aligned to SDXL's VAE latent space, comprehensive UI modernization for the Advanced Masking panel, and full legacy code removal (`InpaintHead`, `inpaint_worker`, `solve_abcd`).

W005 (Streamlined VRAM Lifecycle Management) was intentionally deferred — the scope is better addressed at a later stage when the broader system architecture is more settled.

## Work Order Execution Breakdown

### P3-M12-1-W01: denoise_mask Plumbing & InpaintPipeline Core Rewrite
- **Status:** Completed
- **Outcomes:**
  - Threaded `denoise_mask` through `core.ksampler()` → `sample_sdxl()` using ComfyUI-standard `latent['noise_mask']` convention.
  - Rewrote `InpaintContext` dataclass with native-AR bounding box fields.
  - Implemented `snap_to_sdxl_resolution()` using the 26 SDXL resolution buckets from `flags.sdxl_aspect_ratios`.
  - Rewrote `prepare()`, `encode()`, and `stitch()` methods with SDXL-snapped bounding boxes, VAE encode with max-pooled binary `denoise_mask`, and preserved morphological blend stitching.
  - Removed all legacy methods: `InpaintHead`, `_solve_abcd`, `_fooocus_fill`, `_up255`, `patch_model`.
  - Fixed critical bug where `task_state.denoising_strength` defaulted to 1.0 for inpainting.
  - Eliminated full-image hallucination via proper low-denoise propagation masking.

### P3-M12-1-W02: Outpaint Canvas Extension & Context Mask Support
- **Status:** Completed (with scope adjustments)
- **Outcomes:**
  - Implemented `prepare_outpaint_canvas_only()` and `pixelate_mask_area()` for 2-step outpaint workflow.
  - Adopted 8×8 block-size pixelation primer aligned to SDXL VAE latent compression (8×8 pixels → 1×1 latent), eliminating "window grid" hallucinations from prior block sizing experiments.
  - Modernized Advanced Masking UI: auto-open by default, removed orphan Mask Extraction models, locked outpaint expansion to SDXL-friendly multiples of 32.
  - Added Mask Expansion Tool with directional auto-detection.
  - Fixed critical latent mask bug at 1.0 denoising strength (`max_denoise` bypass) that was destroying VAE primer patches with pure noise.
  - Fixed transparent mask upload logic for Inpaint Step 2.
- **Descoped:** Context Mask (blue brush) and Color Guidance Preview — deprioritized in favor of critical bug fixes. Outpaint functions comprehensively without them.

### P3-M12-1-W03: Orchestrator Integration, Legacy Cleanup & Verification
- **Status:** Completed
- **Outcomes:**
  - Completed full removal of `InpaintHead` network routing from `config.py`, `image_input.py`, `async_worker.py`, and `inference.py`.
  - Deprecated `inpaint_worker.py` with all import references removed.
  - Validated orchestrator flow: `apply_outpaint()` → `apply_inpaint()` → `process_task()`.
  - Full regression testing passed: txt2img, Variation, Upscale, ControlNet, consecutive generations.
  - Edge case testing passed: mask at image edge, tiny mask, full-image mask, no mask drawn.

### P3-M12-1-W005: Streamlined VRAM Lifecycle Management
- **Status:** Deferred (intentionally omitted)
- **Rationale:** System architecture is still evolving. VRAM lifecycle management will be addressed at a later stage when the broader pipeline structure is finalized.

## Descoped Elements (Across Mission)
| Item | Original Work Order | Reason |
|------|-------------------|--------|
| Context Mask (blue brush) | W02 | Deprioritized for critical bug fixes |
| Color Guidance Preview | W02 | Deprioritized for critical bug fixes |
| VRAM Lifecycle Management | W005 | Architecture still evolving; deferred to later stage |

## Risk Observations & Lessons Learned
1. **VAE Latent Alignment is Critical.** Early pixelation primer experiments (Gaussian blur, large block sizes) produced compounding artifacts because they didn't align with the VAE's 8×8 compression ratio. The 8×8 block size discovery was the key breakthrough for clean outpainting.
2. **Denoise Strength 1.0 Requires Special Handling.** At full denoise, the sampler's `max_denoise` bypass was replacing VAE-encoded primer patches with pure noise, defeating the purpose of the pixelation primer. This required a targeted `noise_scaling()` bypass in `KSAMPLER.sample` for masked areas.
3. **Transparent RGBA Mask Compositing.** Gradio's ImageMask component sends complex RGBA data that required a unified `combine_image_and_mask()` merge pass instead of naive dictionary unpacking.
4. **Scope Flexibility Pays Off.** Descoping the context mask and color guidance preview to focus on critical bugs was the right call — the pipeline works robustly without them, and they can be added incrementally.

## Archival & Next Steps
- **Archive:** Mission documents (Brief, Work List, Mission Report) move to `completed/`. Work orders and work reports (W01, W02, W03) move to `completed_work_orders/`.
- **Deferred Work:** W005 remains in backlog for future scheduling.
- **Next Mission:** P4-M01 (Backend API Server) has been briefed and is ready for execution.
