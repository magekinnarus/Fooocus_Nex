# Inpaint BB Context and ControlNet Alignment

## Purpose
This document defines the two-step inpaint workflow used in the Gradio 5 migration baseline and explains why the Step 1 bounding-box context must be preserved for both final stitching and future ControlNet alignment.

## Step 1: Source Image + Context Mask
User inputs:
- Source image upload
- Single white context mask painted in source-image coordinates

Backend actions:
1. Load the source image.
2. Load the Step 1 context mask.
3. Compute the bounding box from the context mask.
4. Expand and snap the bounding box to the target SDXL resolution.
5. Produce the BB image used for Step 2.
6. Persist the full `InpaintContext` on the backend.

Important note:
- The Step 1 mask is not the final regeneration mask.
- Its job is to define the spatial context and the BB transform.

## Step 2: Edited BB Image + Optional BB Mask
User inputs:
- Edited BB image upload
- Optional BB mask upload

Backend actions:
1. Reuse the saved Step 1 `InpaintContext`.
2. Replace `ctx.bb_image` with the edited BB image.
3. Use the Step 2 BB mask if provided; otherwise fall back to the prepared BB mask.
4. Run inference in BB/native-resolution space.
5. Stitch the result back into the original image using the saved Step 1 geometry.
6. Apply Fooocus morphological blending in original-image coordinates.

## Backend-Owned Geometry
The frontend must not be treated as the source of truth for placement. The backend-owned `InpaintContext` is authoritative and must preserve at least:
- `original_image`
- `original_mask`
- `bb = (y1, y2, x1, x2)` in original-image coordinates
- `bb_image`
- `bb_mask`
- `target_w`, `target_h`
- `blend_mask`

## Why This Matters for ControlNet
Future ControlNet alignment should use the exact same Step 1 BB context.

That means:
- External edits for ControlNet must target the Step 1 BB image.
- The backend must continue to treat Step 1 `bb` coordinates as authoritative.
- Any ControlNet conditioning image intended to align with the inpainted region must be interpreted in the saved BB coordinate space from Step 1.

## Current Gradio 5 Baseline
The current implementation intentionally avoids Gradio `ImageEditor` because it caused severe browser CPU spikes when mounted.

Current baseline UI:
- Step 1 uses `gr.Image` plus a lightweight custom HTML/canvas overlay for the context mask.
- Step 2 uses `gr.Image` upload slots for BB image and optional BB mask.

This baseline exists to preserve the inpaint architecture while removing the unstable legacy Gradio 3 canvas hacks and the heavy Gradio 5 editor lifecycle.
