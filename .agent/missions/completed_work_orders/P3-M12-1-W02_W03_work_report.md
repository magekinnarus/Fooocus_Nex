# Work Report: P3-M12-1-W02 & W03 — Outpaint Pipeline & Legacy Clean-Up Finalization
**IDs:** P3-M12-1-W02, P3-M12-1-W03
**Mission:** P3-M12-1
**Date:** 2026-03-04
**Status:** Completed (with scope adjustments)

## Executive Summary
This report combines the finalization of Work Orders W02 and W03, culminating in the complete replacement of the legacy `InpaintHead` architecture with a native `denoise_mask` pipeline. 

Major milestones achieved include the successful implementation of 2-step outpainting with SDXL-aligned pixelated color guidance, comprehensive UI modernization for Advanced Masking, the complete eradication of the obsolete `inpaint_worker.py` dependencies, and the resolution of deep-rooted sampler and bounding-box masking bugs. The pipeline is now highly robust, preventing UI context-bleed and grid hallucinations.

## Key Accomplishments

### 1. 2-Step Outpaint & 8x8 Pixelation Primer (W02)
- **Implementation:** Successfully implemented the `prepare_outpaint_canvas_only()` and `pixelate_mask_area()` workflows in `InpaintPipeline`.
- **SDXL VAE Latent Alignment:** Replaced standard block downscaling and Gaussian Blur experiments with a precise `8x8` block size using `INTER_NEAREST` interpolation. Since the SDXL VAE mathematically compresses 8x8 pixel blocks into exactly 1x1 latent pixels, this structure provides the UNet with flawless color-field guidance without producing compounding scaling artifacts or "window grid" hallucinations.
- **Drawn Structure Preservation:** The 8x8 block sizing prevents the total destruction of user-drawn structures in the outpaint canvas, acting seamlessly as a highly fine-grained img2img primer guide.

### 2. UI Refactoring & Polish (W02)
- **Advanced Masking Modernization:** Renamed "Enable Advanced Masking Features" to "Hide Advanced Masking Features". Auto-open is now the standard behavior.
- **Orphan Cleanup:** Removed out-of-scope Mask Extraction models from the Inpaint tab.
- **Parameter Streamlining:** Outpaint Expansion limits locked to SDXL-friendly multiples of 32 (384, 416, 448). Removed legacy fields.
- **Mask Expansion Tool:** Added an explicit UI button to "Expand Mask (32 pixels)". Automatically detects outpaint direction (`Left`, `Right`, `Top`, `Bottom`) and expands the uploaded mask against the direction.

### 3. Mask Compositing & Denoise Bug Fixes (W02/W03)
- **Latent Masking Eradication (1.0 Denoise):** Fixed a critical backend bug where Denoising Strength 1.0 explicit overwrite behavior was destroying the generated primer patches with pure noise. Implemented a `noise_scaling()` bypass in `KSAMPLER.sample` strictly within masked areas to maintain max denoise while preserving VAE primer outputs.
- **Transparent Mask Upload Logic:** Fixed Inpaint Step 2 logic which was incorrectly interpreting empty or transparent RGBA sketch layers as a full black override. Replaced raw dictionary unpacking with a unified `combine_image_and_mask(dict)` `np.maximum` merge pass safely discarding empty layers while properly composing user-uploaded BB references perfectly.

### 4. Legacy Cleanup & System Integration (W03)
- **`InpaintHead` Purge:** Completed the full removal of legacy `InpaintHead` network routing. All calls to `inpaint_head_model_path`, patching logic, and external weights downloads have been removed from `config.py`, `image_input.py`, `async_worker.py`, and `inference.py`. The system exclusively utilizes the unified V2.6 patch methodology.
- **`inpaint_worker` Deprecation:** Fully removed all import references and usage of the legacy `inpaint_worker.py` module context managers. Added explicit deprecation warnings to the file pending its final system purge.
- **Orchestrator Validation:** Verified `apply_inpaint`, `apply_outpaint`, and `process_task` correctly route bounding box masking and prompt processing. Resolved silent task failures where Outpaint step 2 incorrectly aborted without generating results due to flawed initialization prompts.

## Descoped Elements
- **Context Mask Support (Blue Brush) & Color Guidance Previews:** Descoped from W02 intentionally to prioritize critical bug squashing for Denoise constraints and UI Step 2 transparent mask failures. The Outpaint logic functions comprehensively without them.

## Next Steps
- P3-M12-1-W05 Streamlined VRAM Lifecycle Management (VRAMStageManager) to resolve subsequent generation progressive degradation and align headless/UI paths.
