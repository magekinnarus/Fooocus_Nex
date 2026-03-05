# Handoff Document: P3-M12-1 
**Date:** 2026-03-02
**Context:** Wrapping up a very long session dealing with the Inpaint/Outpaint core rewrite and UI refactoring.

## What Has Been Done

**1. The 2-Step Outpaint Pipeline**
- **Phase 1 (Canvas Expansion):** The engine now correctly generates padded canvases based on Outpaint directions (Left, Right, Top, Bottom) natively snapped to SDXL aspect ratios. It saves the composite image and returns it to the user.
- **Phase 2 (Primer Guide):** If the user checks "Outpaint 2nd Step generation" (formerly Pixelate Primer), the pipeline mathematically downsamples and upsamples the blank space to create a blocky color guide. This is VAE-encoded directly into the latent space to guide generation.

**2. UI Layout & Component Polish**
- **Advanced Mask Navigation:** The Advanced Masking tab is now **visible by default**. The checkbox has been inverted and renamed to "Hide Advanced Masking Features".
- **Mask Modifiers:** Added a highly requested `Expand Mask (32 pixels)` button. It correctly leverages NumPy to push the mask exactly 32 pixels inward from the outpaint edge, saving the result properly as a `.png` for gallery preview and download.
- **Cleanup:** Purged orphaned mask generating models, One-Click Outpaint toggles, and Respective Field parameters to streamline UX.

**3. Deep Backend Sampling Fixes**
- **The 1.0 Denoise Bug:** Discovered that ComfyUI's backend strictly overwrites initial latents with 100% pure noise when Denoising is 1.0 (`max_denoise=True`). This fundamentally broke our Pixelated Primer because the color blocks were erased before Step 1.
- **The Fix:** Modified `KSAMPLER.sample` in `backend/sampling.py` to identify when a `denoise_mask` is present and explicitly blend the masked areas back to standard `scaled_noise`, saving the primer's lifecycle.
- **The Mask Origin Bug:** Fixed an inverted UI toggle bug in `image_input.py` where the uploaded image mask was completely zeroed out because the renaming to "Hide Features" logically inverted `if checkbox:`.

## What Needs To Be Done Next (Current Work Order)

We are currently tracking against **P3-M12-1-W02** and impending **W03**.

**1. Context Mask Support (Deferred in W02)**
- The user originally requested a "Blue Brush" feature where users can paint over areas they want to provide *contextual influence* to the inpainter without regenerating those pixels.
- **Implementation path:** Needs bounding box union mechanics `_find_bb()` and `_union_bb()` inside `InpaintPipeline`, and color thresholding in `image_input.py` to extract Blue pixels vs White pixels.

**2. Legacy Cleanup & Integration (W03)**
- With the pipeline functioning beautifully via `pipeline.process_diffusion` + `denoise_mask`, the old `inpaint_worker.py` and `InpaintHead` architectures are fundamentally obsolete.
- **Implementation path:** We need to systematically execute `P3-M12-1-W03_work_order.md`, which involves purging all references to `inpaint_worker`, removing legacy `_solve_abcd` code, and cleaning up orchestrator files (`async_worker.py`, `inference.py`) to permanently cement the new architecture.

## How to Resume
1. Review this handoff document.
2. Check `walkthrough.md` for specific technical accomplishments.
3. Consult the user to decide whether to attack the Deferred W02 items (Context Masks) or proceed directly to W03 (Legacy Purge & Verification).
