# Work Order: P3-M12-1-W02 — Outpaint Canvas Extension & Context Mask Support
**ID:** P3-M12-1-W02
**Mission:** P3-M12-1
**Status:** Completed (with scope adjustments)
**Depends On:** P3-M12-1-W01

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/summaries/08_inpaint_architecture.md`
- `P3-M12-1-W01_work_order.md` (completed — defines `InpaintContext`, `snap_to_sdxl_resolution`, `_expand_canvas`)
- `Fooocus_Nex/modules/pipeline/inpaint.py` (post-W01 rewrite)
- `Fooocus_Nex/modules/pipeline/image_input.py` (`apply_outpaint()`)
- `ComfyUI_reference/custom_nodes/ComfyUI-Inpaint-CropAndStitch/inpaint_cropandstitch.py` (lines 252-299: `extend_imm`, lines 353-381: `combinecontextmask_m`)

## Objective
Add single-direction outpaint with pixelation primer, optional context mask support (blue brush for
directional context control), and color guidance preview. After this work order, outpainting produces
seamless canvas extensions, and users can optionally specify which parts of the image provide context.

## Scope

### 1. Single-Direction Outpaint — `prepare_outpaint_canvas()`

**Add to `InpaintPipeline`:**

```python
def prepare_outpaint_canvas(self, image, mask, direction, expansion_ratio=0.3):
    """
    Expand canvas in a single direction with pixelated primer.

    Args:
        image: np.ndarray (H, W, 3) — original image
        mask:  np.ndarray (H, W) or None — existing mask (if any)
        direction: str — 'left', 'right', 'top', 'bottom'
        expansion_ratio: float — how much to expand relative to image dimension

    Returns:
        (expanded_image, expanded_mask)

    Algorithm:
        1. Calculate strip dimensions from the relevant edge
        2. Copy edge strip from the image
        3. Pixelate the strip (downscale → upscale with INTER_NEAREST)
        4. Attach pixelated strip to the specified side
        5. Generate mask: 0 for original region, 255 for new region
    """
```

**Pixelation algorithm:**
```python
strip_w = int(W * expansion_ratio)
block_size = max(strip_w // 16, 4)
small = cv2.resize(strip, (strip_w // block_size, H // block_size), interpolation=cv2.INTER_AREA)
pixelated = cv2.resize(small, (strip_w, H), interpolation=cv2.INTER_NEAREST)
```

**Direction handling:**
| Direction | Edge strip source | Attachment | Mask region |
|-----------|------------------|------------|-------------|
| `right`   | `image[:, -strip_w:, :]` | Concatenate right | `mask[:, W:]` = 255 |
| `left`    | `image[:, :strip_w, :]`  | Concatenate left  | `mask[:, :strip_w]` = 255 |
| `bottom`  | `image[-strip_h:, :, :]` | Concatenate below | `mask[H:, :]` = 255 |
| `top`     | `image[:strip_h, :, :]`  | Concatenate above | `mask[:strip_h, :]` = 255 |

### 2. Context Mask Support (Optional Blue Brush)

**Concept:** The user draws two types of brush strokes on the inpaint canvas:
- **White** = inpaint mask (regenerate this area)
- **Blue** (or other distinguishable color) = context mask (include in BB but freeze)

**Implementation approach:**

The context mask affects how the bounding box is calculated in `prepare()`:
1. If no context mask provided: BB = tight mask BB × `extend_factor`
2. If context mask provided: BB = union(mask BB, context_mask BB)
3. Then snap to SDXL resolution as before

**Add parameter to `prepare()`:**
```python
def prepare(self, image, mask, context_mask=None, extend_factor=1.2):
    # Find tight mask BB
    mask_bb = self._find_bb(mask)

    if context_mask is not None:
        # Find context mask BB, union with mask BB
        ctx_bb = self._find_bb(context_mask)
        combined_bb = self._union_bb(mask_bb, ctx_bb)
    else:
        combined_bb = mask_bb

    # Expand by extend_factor, snap to SDXL, etc.
    ...
```

**Utility methods:**
```python
def _find_bb(self, mask):
    """Find tight bounding box around non-zero mask pixels."""
    indices = np.where(mask > 127)
    if len(indices[0]) == 0:
        return None
    y1, y2 = np.min(indices[0]), np.max(indices[0]) + 1
    x1, x2 = np.min(indices[1]), np.max(indices[1]) + 1
    return (y1, y2, x1, x2)

def _union_bb(self, bb1, bb2):
    """Compute union bounding box of two BBs."""
    if bb1 is None: return bb2
    if bb2 is None: return bb1
    return (min(bb1[0], bb2[0]), max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]), max(bb1[3], bb2[3]))
```

**UI-level mask splitting** (in `apply_inpaint` or `apply_image_input`):
The Gradio inpaint component sends a single RGBA mask. We need to split it by color channel:
- White pixels (R>200, G>200, B>200) → inpaint mask
- Blue pixels (R<100, G<100, B>200) → context mask

> **Note:** The exact UI mechanism for selecting brush color depends on Gradio's ImageMask
> component capabilities. If Gradio doesn't support multi-color brushes natively, we may need
> to use the existing "inpaint mask color" selector in the Advanced/Inpaint tab to switch modes.
> The mask splitting logic in the pipeline should work regardless of how the UI captures the colors.

### 3. Color Guidance Preview

**Concept:** When `debugging_inpaint_preprocessor` is enabled, export the primer image and mask for
the user to inspect and optionally edit externally.

**Implementation in `apply_inpaint()`:**
```python
if task_state.debugging_inpaint_preprocessor:
    # After prepare() but before encode()
    # Show the user: (1) the BB image (primer), (2) the BB mask, (3) the full BB overlay on original
    if yield_result_callback:
        preview_overlay = context.original_image.copy()
        y1, y2, x1, x2 = context.bb
        cv2.rectangle(preview_overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        yield_result_callback(task_state, [context.bb_image, context.bb_mask, preview_overlay],
                             100, do_not_show_finished_images=True)
    raise EarlyReturnException
```

### 4. Update `apply_outpaint()` in `image_input.py`

Replace the multi-direction `np.pad(mode='edge')` approach:

```python
def apply_outpaint(task_state, inpaint_image, inpaint_mask):
    if len(task_state.outpaint_selections) > 0:
        from modules.pipeline.inpaint import InpaintPipeline
        inpaint = InpaintPipeline()
        direction = task_state.outpaint_selections[0].lower()  # Single direction only
        inpaint_image, inpaint_mask = inpaint.prepare_outpaint_canvas(
            inpaint_image, inpaint_mask, direction
        )
        task_state.inpaint_strength = 1.0
        task_state.inpaint_respective_field = 1.0
    return inpaint_image, inpaint_mask
```

## Implementation Steps

### Step 1: Outpaint Canvas Extension (COMPLETED)
- [x] Implement `prepare_outpaint_canvas()` in `InpaintPipeline`
- [x] Handle all 4 directions (left, right, top, bottom)
- [x] Implement pixelation algorithm (downscale → upscale INTER_NEAREST)
- [x] Update `apply_outpaint()` in `image_input.py`
- [x] **Verification**: Outpaint correctly generates 2-step process with correct primer and seamless blending.

### Step 2: Context Mask Support (DESCOPED)
- [ ] Implement `_find_bb()` and `_union_bb()` in `InpaintPipeline`
- [ ] Add `context_mask` parameter to `prepare()`
- [ ] Implement mask color splitting logic (white=inpaint, blue=context)
- [ ] Wire context mask through `apply_inpaint()` → `pipeline.prepare()`

### Step 3: Color Guidance Preview (DESCOPED)
- [ ] Implement preview export in `apply_inpaint()` under `debugging_inpaint_preprocessor` flag
- [ ] Show BB image, BB mask, and BB overlay on original

### Additional Features Delivered Outside Original Scope (COMPLETED)
- [x] Extensive UI Refactor of the Inpaint/Outpaint panel
- [x] Renamed "Enable Advanced Masking Features" to "Hide Advanced Masking Features" (Visible by default)
- [x] Mask Expansion Tool: Added UI button and logic to expand drawn mask by 32px based on outpaint direction
- [x] Fixed latent mask pure noise bug at 1.0 Denoising (`max_denoise` bypass)

## Success Criteria (Interim)
- [x] Single-direction outpaint produces expanded canvas with pixelated primer
- [x] Generated outpaint extension is seamless (no visible boundary)
- [x] Mask Expansion UI functioning correctly (downloads/gallery pipeline established)
- [x] Latent Mask bug fixed (1.0 denoise preserves VAE primer)
- [x] Context mask (blue brush) support (Descoped to prioritize bug fixes)
- [x] Color guidance preview (Descoped to prioritize bug fixes)
