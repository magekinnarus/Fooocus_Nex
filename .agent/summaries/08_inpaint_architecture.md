# Inpaint Architecture Analysis: Fooocus vs. Modern Pipelines

This document summarizes our investigation into why the refactored `InpaintPipeline` failed, how the original Fooocus inpainting process worked, and how we will pivot to a vastly superior methodology inspired by ComfyUI's `CropAndStitch`.

## 1. The Current Issues We Face

During the refactor, we discovered that inpainting and outpainting returned completely hallucinated images that ignored the original unmasked picture. The core issues are:
*   **The Lost Context:** Fooocus never used a `denoise_mask` in the sampler to freeze the original pixels. It relied entirely on a custom `InpaintHead` model to "patch" the UNet with context. During the backend loader refactoring (`patch_unet_for_quality`), this specific `InpaintHead` patch was either overwritten or ignored, rendering the UNet completely blind to what the unmasked area looked like.
*   **Morphological Masking Bug:** Our refactored mask smoothing algorithm (`_morphological_open`) corrupted the mask matrix values, destroying the final image stitching step.

## 2. How the Original Fooocus Process Works (and Why It's Flawed)

The original Fooocus inpaint logic relies on a series of aggressive mathematical hacks that limit quality and flexibility:

1.  **Forced 1:1 Aspect Ratios (The Squashing Bug):** Fooocus forces *every single* inpaint Bounding Box (BB) to be a perfect square, regardless of the prompt or original image size. If your BB is 832x1216, Fooocus squashes it into a 1024x1024 square before sending it to the Base SDXL model.
2.  **Resolution Degeneration:** After generating the 1024x1024 square, Fooocus stretches the image back to 832x1216 to stitch it in. This constant squashing and stretching drastically degrades image sharpness, causing the classic "blurry outpaint" complaint.
3.  **The `InpaintHead` Dependency:** Fooocus allows the Base SDXL to regenerate the *entire* 1024x1024 square, including the areas you didn't mask! It expects the custom `InpaintHead` model to successfully guess the context and force the UNet to draw the unmasked regions identically. This frequently fails, necessitating a complex, buggy "color correction" blending pass to hide the seams.
4.  **Zero Respective Field (Improve Detail):** When using `Improve Detail` (`k=0`), the BB perfectly tightly crops the mask. The `InpaintHead` has literally zero surrounding pixels to look at. It flies completely blind, resulting in aggressive color shifting and hallucinatory lighting changes.

## 3. How ComfyUI CropAndStitch Works (The Modern Approach)

Modern professional nodes like `ComfyUI-Inpaint-CropAndStitch` solve all of these problems by elegantly utilizing core SDXL mechanics instead of hacky UNet patches:

1.  **Native Aspect Ratios:** It dynamically calculates the aspect ratio of the Bounding Box. If the target area is wide (e.g., 1216x832), it mathematically expands the BB to perfectly match that exact ratio. **No squashing, no stretching, zero resolution loss.**
2.  **Explicit `denoise_mask` Freezing:** It passes a `denoise_mask` directly into the latent sampler. This mathematically freezes the exact latent noise of the unmasked area. The Base SDXL model cannot alter a single unmasked pixel.
3.  **No Context Injection needed:** Because the unmasked pixels are frozen directly in the latent space, there is no need for a buggy `InpaintHead` to "inject" context. The surrounding pixels are literally locked in place for the UNet's self-attention to read.
4.  **Context Masks:** If you explicitly want to include specific areas of the image in the generation context, you simply pass an `optional_context_mask`, and the node automatically expands the BB to envelop it.

## 4. The Path Forward (Automated Smart Outpainting)

We will abandon the legacy Fooocus `InpaintHead` and `fooocus_fill` logic. We will build a clean, extremely high-quality `InpaintPipeline` utilizing the `denoise_mask` capabilities we just discovered in the backend:

**The New Outpaint/Inpaint Algorithm:**
1.  **Target Calculation:** Determine the BB size and the required extension padding.
2.  **Smart Primer Extension:** Instead of blank padding, we take a slice of the original image bordering the extension, mirror/stretch it into the void, and apply a heavy pixelization/blur. This serves as a perfect color and structural "primer" for the Base SDXL.
3.  **Context Expansion:** Expand the BB backward into the original image by an explicitly calculated "context" margin to give the Base SDXL structural hints.
4.  **Native Generation:** Send the exact, unsquashed BB to the UNet, accompanied by a `denoise_mask` that perfectly freezes the original image pixels within the Context Area.
5.  **Clean Stitching:** Since the boundary was mathematically frozen during generation, we simply crop the generated patch and stitch it seamlessly onto the canvas. No color correction needed.
