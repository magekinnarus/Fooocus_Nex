# Work Order: P3-M12-1-W03 — Orchestrator Integration, Legacy Cleanup & Verification
**ID:** P3-M12-1-W03
**Mission:** P3-M12-1
**Status:** Completed
**Depends On:** P3-M12-1-W02

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `P3-M12-1-W01_work_order.md` (completed — denoise_mask plumbing, InpaintPipeline core)
- `P3-M12-1-W02_work_order.md` (completed — outpaint canvas, context mask)
- `Fooocus_Nex/modules/pipeline/image_input.py` (`apply_inpaint`, `apply_outpaint`, `apply_image_input`)
- `Fooocus_Nex/modules/pipeline/inference.py` (`process_task`)
- `Fooocus_Nex/modules/inpaint_worker.py` (legacy — deprecation target)
- `Fooocus_Nex/modules/async_worker.py` (orchestrator — verify no stale references)

## Objective
Finalize the integration of the new InpaintPipeline into the existing orchestrator flow, remove all legacy
inpainting code, and run comprehensive verification to ensure no regressions in any generation mode.

## Scope

### 1. Finalize `apply_inpaint()` Integration

Ensure `apply_inpaint()` in `image_input.py` correctly:
- Creates `InpaintPipeline` and calls `prepare()` + `encode()`
- Passes `context_mask` when available (from blue brush splitting)
- Sets `task_state.initial_latent` to the dict containing `noise_mask`
- Sets `task_state.width` and `task_state.height` to match the SDXL-snapped resolution
- Does **NOT** call `patch_model()` or load `InpaintHead`
- Does **NOT** mutate `pipeline.final_unet`
- Stores `InpaintContext` in `task_state.inpaint_context` for stitching

**Remove from `apply_inpaint()`:**
```python
# DELETE:
if inpaint_parameterized:
    pipeline.final_unet = inpaint.patch_model(
        context=ctx, unet=pipeline.final_unet, head_model_path=inpaint_head_model_path
    )
```

### 2. Verify `process_task()` Integration

`process_task()` in `inference.py` should work without changes because:
- It passes `task_state.initial_latent` to `process_diffusion(latent=...)` → `ksampler()` now reads `noise_mask`
- The stitching block already exists:
  ```python
  if hasattr(task_state, 'inpaint_context') and task_state.inpaint_context is not None:
      inpaint = InpaintPipeline()
      imgs = [inpaint.stitch(task_state.inpaint_context, x) for x in imgs]
  ```

Verify this path works. If `inpaint_head_model_path` is still passed as a parameter but unused, leave it
in the signature for now (cosmetic cleanup can follow).

### 3. Remove Legacy Code

#### `modules/pipeline/inpaint.py`
Confirm all legacy methods were removed in W01:
- [x] `InpaintHead` class (entire class)
- [x] `_solve_abcd()`, `_compute_initial_abcd()`, `_regulate_abcd()`
- [x] `_fooocus_fill()`
- [x] `_up255()`
- [x] `patch_model()` and `input_block_patch` closure

#### `modules/inpaint_worker.py`
This file is now fully obsolete. However, it may still be imported elsewhere.

#### `modules/inpaint_worker.py`
This file is now fully obsolete. However, it may still be imported elsewhere.

- [x] Search for all imports of `inpaint_worker` across the codebase
- [x] Remove or replace all references:
  - `modules/pipeline/image_input.py` — `import modules.inpaint_worker as inpaint_worker`
  - `modules/pipeline/inference.py` — `import modules.inpaint_worker as inpaint_worker`
  - `modules/async_worker.py` — any references to `inpaint_worker.current_task`
  - Any other files found via grep
- [x] Once all references are removed, add a deprecation notice to `inpaint_worker.py`:
  ```python
  # DEPRECATED: This module is superseded by modules.pipeline.inpaint.InpaintPipeline
  # Kept temporarily for reference. Will be deleted in a future cleanup pass.
  ```
  (Full deletion deferred to avoid breaking any edge-case references we might miss.)

#### `modules/pipeline/image_input.py`
- [x] Remove `inpaint_head_model_path` from `apply_inpaint()` parameter list if no longer needed
- [x] Remove any references to `inpaint_worker.current_task`
- [x] Verify `inpaint_parameterized` flag handling — with denoise_mask, the inpaint engine version
  (`v1`, `v2.5`, `v2.6`) may still affect denoising strength but should NOT trigger InpaintHead loading

#### `modules/async_worker.py`
- [x] Verify no stale references to `inpaint_worker.current_task`
- [x] Verify `inpaint_head_model_path` is still passed to `process_task()` but unused (cosmetic)
- [x] Confirm the handler flow: `apply_outpaint()` → `apply_inpaint()` → `process_task()` works cleanly

### 4. UI Validation — Single Direction Outpaint Enforcement

Verify the outpaint direction selector in the Inpaint tab:
- [x] If user selects multiple directions, only the first is used
- [x] Console prints which direction was selected
- [x] No crash if zero directions selected (falls through to regular inpaint)

## Implementation Steps

### Step 1: Audit and Remove Legacy References
- [x] `grep -r "inpaint_worker" Fooocus_Nex/` — list all references
- [x] `grep -r "InpaintHead" Fooocus_Nex/` — list all references
- [x] `grep -r "inpaint_head" Fooocus_Nex/` — list all references
- [x] Remove or update each reference found
- [x] Add deprecation notice to `inpaint_worker.py`
- [x] **Verification**: `python -c "from modules.pipeline.inpaint import InpaintPipeline"` — no import errors

### Step 2: Full Integration Test — Inpainting
- [x] Start UI: `python entry_with_update.py`
- [x] Load a 832×1216 image in Inpaint tab
- [x] Draw a mask over a region (~300×300px)
- [x] Generate with a relevant prompt
- [x] **Verify**:
  - Console shows BB snapped to an SDXL resolution (e.g., `832×1216` or `1024×1024`)
  - Console does NOT show "Loading InpaintHead" or similar
  - Generated image: unmasked area is pixel-perfect identical to original
  - No visible seam at mask boundary
  - Generated content in masked area is coherent and sharp

### Step 3: Full Integration Test — Outpainting
- [x] Load a 832×1216 image in Inpaint tab
- [x] Select outpaint direction "Right"
- [x] Generate
- [x] **Verify**:
  - Console shows canvas expanded with pixelated primer
  - Generated extension blends seamlessly with original
  - Original portion is pixel-identical
  - Result image has wider dimensions than original

### Step 4: Regression Tests — No Inpaint
- [x] **txt2img**: Generate any prompt without image input → works normally
- [x] **Variation (Subtle)**: Use Vary on a previous output → works normally
- [x] **Upscale**: Use Upscale (1.5x) on a previous output → works normally
- [x] **ControlNet**: If available, test Canny or CPDS → works normally
- [x] **Consecutive generations**: Generate 3 images in a row → no crashes or stale state

### Step 5: Edge Case Tests
- [x] **Mask at image edge**: Draw mask touching the right edge of a 832×1216 image → BB correctly
  expands with edge-replicated padding, no crash
- [x] **Very small mask**: Draw a tiny ~50×50 mask on a large image → BB expands with context margin,
  snaps to SDXL resolution, generates coherently
- [x] **Full-image mask**: Draw mask covering entire image → BB = full image, snaps to closest SDXL
  resolution, effectively becomes img2img
- [x] **No mask drawn**: Submit inpaint without drawing → graceful fallback (no crash)

## Success Criteria
1. All legacy `InpaintHead` and `inpaint_worker.current_task` references removed from active code paths
2. Inpainting on non-square images produces sharp, native-resolution output
3. Outpainting produces seamless extensions with pixelated primer
4. No seam artifacts at mask boundaries (Fooocus morphological blend preserved)
5. txt2img, Variation, Upscale all work without regression
6. No `InpaintHead` loading messages in console during any generation
7. Edge cases (mask at edge, tiny mask, full mask, no mask) handled gracefully
