# Mission Brief: P3-M12-1 — Inpainting Architecture Pivot (denoise_mask)
**ID:** P3-M12-1
**Phase:** 3
**Date Issued:** 2026-02-28
**Status:** Ready
**Depends On:** P3-M12-W01 (async_worker decomposition), P3-M12-W02 (webui modularization)
**Supersedes:** P3-M12-W03 (InpaintPipeline & ControlNetPipeline Extraction)
**Work List:** `.agent/missions/active/P3-M12-1_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/08_inpaint_architecture.md` (**Critical** — documents the legacy flaws and the target architecture)
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `ComfyUI_reference/custom_nodes/ComfyUI-Inpaint-CropAndStitch/inpaint_cropandstitch.py` (Reference implementation)

## Context & Motivation

During execution of P3-M12-W03, we attempted to extract the inpaint logic into a self-contained pipeline
module. This surfaced a series of **fundamental architectural flaws** in Fooocus's inpainting methodology,
documented in `08_inpaint_architecture.md`:

1. **InpaintHead UNet Monkey-Patch**: Fooocus never used `denoise_mask` to protect original pixels. It
   relied on a custom `InpaintHead` model (320×5×3×3 convolution) to inject context features into the
   UNet's first input block. This patch conflicted with our backend loader quality improvements
   (`patch_unet_for_quality`), causing full-image hallucination.

2. **Forced 1:1 Square Squashing**: The legacy `solve_abcd` forcibly squashes every inpaint bounding box
   to a perfect square (1024×1024), regardless of the mask shape. This causes severe quality degradation
   on non-square images — the classic "blurry outpaint" complaint.

3. **No True Pixel Freezing**: Without `denoise_mask`, the entire BB area (including unmasked regions) is
   regenerated. The `InpaintHead` merely "guesses" what the original content looked like, requiring a
   complex color-correction blending pass to hide seams.

Since the backend sampling chain **already fully supports `denoise_mask`** (`sample_sdxl` → `KSampler` →
`CFGGuider` → `KSamplerX0Inpaint`), this mission pivots to a modern architecture inspired by the user's
proven GIMP workflow and ComfyUI's CropAndStitch methodology.

## Objective

Replace the legacy Fooocus inpainting architecture with a `denoise_mask`-based pipeline that:
- Preserves native aspect ratios (no 1:1 squashing)
- Mathematically freezes original pixels via `denoise_mask` during sampling
- Uses intelligent canvas extension with pixelation primer for outpainting
- Snaps bounding boxes to SDXL-native resolutions
- Retains Fooocus's superior morphological blending for seamless stitching
- Supports an optional context mask (blue brush) for directional context control
- Provides a color guidance preview for advanced users

## Scope

### In Scope
- **denoise_mask threading** through `core.ksampler()` into the backend sampling chain
- **InpaintPipeline rewrite** — native-AR bounding boxes, SDXL resolution snapping, pixelation primer, denoise_mask encode, morphological blend stitch
- **Single-direction outpaint** — automated canvas extension with edge-copy + pixelation, replacing the multi-direction `np.pad(mode='edge')` approach
- **Context mask support** — optional second-color brush for directional context control
- **Color guidance preview** — export primer image for external editing via `debugging_inpaint_preprocessor`
- **Legacy cleanup** — remove `InpaintHead`, `solve_abcd`, `fooocus_fill`, `_morphological_open` mask prep (keep the stitch blending version)
- **Orchestrator integration** — update `apply_inpaint()`, `apply_outpaint()`, and `process_task()` to use new pipeline

### Out of Scope
- **ControlNetPipeline extraction** — split into separate future work order P3-M12-W04
- **Multi-direction simultaneous outpaint** — single direction per pass only
- **Interactive in-UI color guidance editing** — external editing via export/reimport only
- **UI redesign of Inpaint tab** — only adding context mask color selector; layout stays unchanged
- **SD1.5 inpainting** — focus on SDXL only; SD1.5 may work via `vae_swap` but is not a verification target

## Reference Files
- `Fooocus_Nex/modules/pipeline/inpaint.py` — current InpaintPipeline (to be rewritten)
- `Fooocus_Nex/modules/pipeline/image_input.py` — `apply_inpaint()`, `apply_outpaint()`
- `Fooocus_Nex/modules/pipeline/inference.py` — `process_task()`
- `Fooocus_Nex/modules/core.py` — `ksampler()`, `encode_vae_inpaint()`
- `Fooocus_Nex/modules/default_pipeline.py` — `process_diffusion()`
- `Fooocus_Nex/backend/sampling.py` — `sample_sdxl()`, `KSamplerX0Inpaint`
- `Fooocus_Nex/modules/inpaint_worker.py` — legacy `InpaintWorker` (deprecation target)
- `Fooocus_Nex/modules/flags.py` — `sdxl_aspect_ratios` (26 SDXL resolution buckets)
- `ComfyUI_reference/custom_nodes/ComfyUI-Inpaint-CropAndStitch/inpaint_cropandstitch.py` — reference implementation

## Constraints
- **Incremental approach** — each work order must leave Fooocus in a runnable state
- **Preserve Fooocus morphological blending** — the `morphological_open` gradient + full-image alpha blend for seamless stitching must be preserved
- **BB must snap to SDXL resolutions** — use the 26 resolutions from `flags.sdxl_aspect_ratios`
- **Edge overflow handled via canvas expansion** — edge-replicate + pixelate, never clip
- **No `InpaintHead` dependency** — must work without `fooocus_inpaint_head.pth`
- Testing on local GTX 1050 with **SDXL GGUF** model
- **denoise_mask convention** — use `latent['noise_mask']` key (ComfyUI standard)

## Deliverables
- [ ] `denoise_mask` threaded from `core.ksampler()` through `sample_sdxl()`
- [ ] `modules/pipeline/inpaint.py` rewritten with native-AR BB, SDXL snapping, pixelation primer, denoise_mask
- [ ] Single-direction outpaint via `prepare_outpaint_canvas()`
- [ ] Morphological blend stitching preserved (no seam artifacts)
- [ ] `InpaintHead` class and related legacy code removed
- [ ] `apply_inpaint()` and `apply_outpaint()` updated to use new pipeline
- [ ] Context mask support (separate from inpaint mask)
- [ ] Color guidance preview via `debugging_inpaint_preprocessor`
- [ ] Inpainting works on non-square images without quality degradation
- [ ] txt2img / Variation / Upscale not regressed

## Success Criteria
1. Inpainting on a 832×1216 image produces sharp, native-resolution output (no 1:1 squash)
2. Unmasked pixels are pixel-perfect identical to original (verified by zooming in)
3. No visible seam at mask boundaries (Fooocus morphological blend quality)
4. Single-direction outpaint produces seamless canvas extension with pixelated primer
5. No `InpaintHead` model loaded during generation (verified via console logs)
6. txt2img generation works identically to current baseline
7. Variation/Upscale flows are unaffected
8. Color guidance preview exports primer image + mask when `debugging_inpaint_preprocessor` is enabled

## Work Orders
Registered in `P3-M12-1_work_list.md`:
- `P3-M12-1-W01` — denoise_mask Plumbing & InpaintPipeline Core Rewrite
- `P3-M12-1-W02` — Outpaint Canvas Extension & Context Mask Support
- `P3-M12-1-W03` — Orchestrator Integration, Legacy Cleanup & Verification

## Notes
- **Relationship to P3-M12-W03**: This mission supersedes and replaces the original W03 work order. The
  original W03 scope (extract InpaintPipeline + ControlNetPipeline) was split because the inpaint pivot
  became a fundamental architectural change, not just an extraction. ControlNet extraction is deferred to
  a separate work order (P3-M12-W04).
- **The user's proven GIMP workflow** (copy edge → pixelate → color guidance → inpaint → composite) is the
  gold standard this implementation automates. The key innovation is combining `denoise_mask` for pixel
  freezing with Fooocus's morphological blending for seamless stitching — best of both worlds.
- **`denoise_mask` mechanics**: In `KSamplerX0Inpaint.__call__()`, the mask is applied at each denoising
  step: `x = x * denoise_mask + scaled_original * (1 - denoise_mask)`. This mathematically prevents the
  UNet from modifying frozen pixels. The UNet's self-attention still reads the frozen pixels as context.
- **SDXL resolution snapping**: The 26 SDXL buckets in `flags.sdxl_aspect_ratios` range from 704×1408 to
  1728×576. BB snapping finds the closest aspect ratio match, then resizes the BB to that exact resolution.
  This eliminates any aspect ratio distortion.
- **VAE rounding artifacts**: Even with `denoise_mask`, VAE encode→decode introduces subtle color shifts.
  The morphological gradient blend (≈96px transition zone) smooths these out during stitching.
