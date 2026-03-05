# Work Order: P3-M12-1-W01 — denoise_mask Plumbing & InpaintPipeline Core Rewrite
**ID:** P3-M12-1-W01
**Mission:** P3-M12-1
**Status:** Completed
**Depends On:** P3-M12-W01 (async_worker decomposition)

## Mandatory Reading
- `.agent/summaries/08_inpaint_architecture.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `Fooocus_Nex/modules/pipeline/inpaint.py` (current — to be rewritten)
- `Fooocus_Nex/modules/core.py` (specifically `ksampler()` and `encode_vae_inpaint()`)
- `Fooocus_Nex/backend/sampling.py` (specifically `sample_sdxl()` and `KSamplerX0Inpaint`)
- `Fooocus_Nex/modules/flags.py` (`sdxl_aspect_ratios` — 26 SDXL resolution buckets)

## Objective
Thread `denoise_mask` through the sampling chain and rewrite the `InpaintPipeline` core to use
native-aspect-ratio bounding boxes, SDXL resolution snapping, and morphological blend stitching.
After this work order, a standard inpaint operation (user draws mask on an image) works end-to-end
without `InpaintHead`.

## Scope

### 1. Thread `denoise_mask` Through `core.ksampler()`

**Current state**: `core.ksampler()` ignores `denoise_mask`. The backend (`sample_sdxl` → `KSampler` →
`CFGGuider` → `KSamplerX0Inpaint`) already fully supports it.

**The gap is 3 lines in `core.ksampler()`:**

```python
# In ksampler(), after `latent_image = latent["samples"].to(device)`:
denoise_mask = latent.get("noise_mask", None)
if denoise_mask is not None:
    denoise_mask = denoise_mask.to(device)

# In the sample_sdxl() call, add:
denoise_mask=denoise_mask,
```

**Files modified:**
- `modules/core.py` — add `denoise_mask` extraction and pass-through in `ksampler()`

**Convention:** The `denoise_mask` is stored in the latent dict under key `noise_mask` (ComfyUI standard).
This means `process_diffusion()` requires **no changes** — it already passes `latent=task_state.initial_latent`
to `ksampler()`, which will now automatically read `noise_mask` from the dict.

### 2. Rewrite `InpaintContext` Dataclass

Replace the current dataclass with the new fields needed for native-AR processing:

```python
@dataclass
class InpaintContext:
    """Carries all state between inpaint stages. No globals."""
    original_image: np.ndarray       # Full original image for final compositing
    original_mask: np.ndarray        # Full original mask (0=keep, 255=regenerate)
    bb: tuple                        # (y1, y2, x1, x2) bounding box in original image coords
    bb_image: np.ndarray             # Cropped region resized to SDXL resolution
    bb_mask: np.ndarray              # Cropped mask resized to SDXL resolution
    target_w: int                    # SDXL-snapped width
    target_h: int                    # SDXL-snapped height
    blend_mask: np.ndarray           # Full-image morphological gradient for stitching
```

**Removed fields:** `interested_fill`, `interested_image`, `interested_mask`, `context_mask`,
`latent_fill`, `latent_mask`, `latent_swap`, `inpaint_head_feature`

### 3. Rewrite `InpaintPipeline` Core Methods

#### `snap_to_sdxl_resolution(w, h) → (target_w, target_h)`
Find the SDXL resolution from `flags.sdxl_aspect_ratios` whose aspect ratio is closest to `w/h`.

```python
SDXL_RESOLUTIONS = [(int(s.split('*')[0]), int(s.split('*')[1])) for s in flags.sdxl_aspect_ratios]

def snap_to_sdxl_resolution(self, w, h):
    target_ratio = w / h
    best = min(self.SDXL_RESOLUTIONS, key=lambda r: abs(r[0]/r[1] - target_ratio))
    return best
```

#### `prepare(image, mask) → InpaintContext`

Algorithm:
1. Find tight bounding box around mask (`np.where(mask > 127)`)
2. Expand BB by `extend_factor` (default 1.2×) for context margin, clamped to image bounds
3. Compute BB aspect ratio → `snap_to_sdxl_resolution(bb_w, bb_h)` → get `(target_w, target_h)`
4. Grow BB dimensions to match the snapped aspect ratio (centered on mask center, clamped to image bounds)
5. Handle edge overflow: if BB exceeds image bounds after growth, the overflow region is filled with
   edge-replicated pixels + pixelation (handled in `_expand_canvas()`)
6. Crop `image[y1:y2, x1:x2]` and `mask[y1:y2, x1:x2]`
7. Resize both to `(target_w, target_h)` using bicubic interpolation
8. Generate `blend_mask` via `_morphological_open(original_mask)` for stitching
9. Return `InpaintContext`

**Key utility — `_expand_canvas(image, y1, y2, x1, x2) → expanded_image, adjusted_coords`:**
When BB extends beyond image bounds, create an expanded canvas:
- Copy original image into the center of the expanded canvas
- Fill overflow regions with edge-replicated pixels
- Apply pixelation to the overflow (block_size = overflow_width // 16, minimum 4)
- Return the expanded image and adjusted BB coordinates

#### `encode(context, vae) → latent_dict`

1. Convert `bb_image` to pytorch tensor (B, C, H, W with pixel range [0, 1])
2. VAE encode → latent samples
3. Convert `bb_mask` to latent-space resolution (H/8, W/8):
   - Downscale mask via max_pool with kernel_size=8
   - Threshold to binary (>0.5 → 1.0, else 0.0)
   - 1.0 = regenerate (masked area), 0.0 = freeze (original)
4. Pack as ComfyUI-standard latent dict:
   ```python
   return {'samples': latent, 'noise_mask': denoise_mask}
   ```

#### `stitch(context, generated_image) → np.ndarray`

Preserve Fooocus's superior blending:
1. Resize `generated_image` back to BB pixel dimensions `(y2-y1, x2-x1)`
2. Hard-paste into a copy of original image: `result[y1:y2, x1:x2] = content`
3. Apply morphological gradient blend at full-image resolution:
   ```python
   fg = result.astype(np.float32)
   bg = context.original_image.astype(np.float32)
   w = context.blend_mask[:, :, None].astype(np.float32) / 255.0
   y = fg * w + bg * (1 - w)
   return y.clip(0, 255).astype(np.uint8)
   ```

#### Preserved utility methods:
- `_morphological_open(mask)` — generates the wide gradient mask for stitching
- `_max_filter_opencv(x, ksize)` — used by `_morphological_open`
- `_box_blur(x, k)` — may be used for primer pixelation

#### Deleted legacy methods:
- `InpaintHead` class
- `_solve_abcd()`, `_compute_initial_abcd()`, `_regulate_abcd()`
- `_fooocus_fill()` — replaced by denoise_mask (no need to guess fill content)
- `_up255()` — replaced by threshold in `encode()`
- `patch_model()` — InpaintHead monkey-patch, no longer needed

## Implementation Steps

### Step 1: denoise_mask Plumbing
- [x] Add `denoise_mask` extraction and pass-through in `core.ksampler()`
- [x] **Verification**: Run txt2img generation. Confirm no regression (denoise_mask is `None` for txt2img, so the path should be a no-op)

### Step 2: InpaintPipeline Core Rewrite
- [x] Rewrite `InpaintContext` dataclass with new fields
- [x] Implement `snap_to_sdxl_resolution()`
- [x] Implement `_expand_canvas()` (edge-replicate + pixelate overflow)
- [x] Implement `prepare()` with native-AR bounding box algorithm
- [x] Implement `encode()` with VAE encode + denoise_mask generation
- [x] Implement `stitch()` with morphological blend (preserved from original Fooocus)
- [x] Remove `InpaintHead`, `_solve_abcd`, `_fooocus_fill`, `_up255`, `patch_model`
- [x] **Verification**: Write a minimal test script that calls `prepare()` with a sample image+mask, verifies the BB dimensions match an SDXL resolution, and verifies the denoise_mask shape is correct

### Step 3: Minimal Integration
- [x] Update `apply_inpaint()` in `image_input.py` to use new pipeline (no InpaintHead patching)
- [x] Set `task_state.initial_latent` to the latent dict from `encode()` (contains `noise_mask`)
- [x] **Verification**: Start UI, perform basic inpaint on a 832×1216 image. Verify that:
  - Generated BB is at an SDXL resolution (check console log)
  - Unmasked area is pixel-identical to original
  - No `InpaintHead` loading message in console
  - No seam artifacts at mask boundary

## Success Criteria
- `denoise_mask` flows from `latent['noise_mask']` through `ksampler()` → `sample_sdxl()`
- `InpaintPipeline.prepare()` returns BB dimensions matching an SDXL resolution
- `InpaintPipeline.encode()` returns `{'samples': ..., 'noise_mask': ...}`
- `InpaintPipeline.stitch()` produces seamless composites (Fooocus morphological blend quality)
- txt2img generation is unaffected (no regression)
- Basic inpainting works end-to-end without `InpaintHead`
