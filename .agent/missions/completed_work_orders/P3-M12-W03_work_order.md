# Work Order: P3-M12-W03 — InpaintPipeline & ControlNetPipeline Extraction
**ID:** P3-M12-W03
**Mission:** P3-M12
**Status:** Superseded by P3-M12-1 (see `P3-M12-1_mission_brief.md`)
**Depends On:** P3-M12-W01

## Mandatory Reading
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/08_inpaint_architecture.md` (Crucial new architectural context)
- `Fooocus_Nex/modules/inpaint_worker.py` (To be heavily deprecated/removed)
- `Fooocus_Nex/modules/async_worker.py` (post-W01 — `apply_inpaint`, `apply_control_nets`, `apply_outpaint`)
- `Fooocus_Nex/modules/core.py` 
- `Fooocus_Nex/backend/sampling.py` (Specifically `denoise_mask` logic)

## Objective
Extract the inpainting and ControlNet logic into two self-contained pipeline modules. **Pivot the Inpaint architecture:** Abandon the legacy `InpaintHead` and forced 1:1 squashing approach. Implement a modern `denoise_mask` methodology (inspired by ComfyUI `CropAndStitch`) to natively support true aspect ratios, flawless context freezing, and eliminate blurry upscaling issues.

## Scope

### 1. Create `modules/pipeline/inpaint.py` — InpaintPipeline (The Modern Rewrite)

**Problem**: The original Fooocus inpainting process forces all Bounding Boxes (BB) to be 1:1 squares, leading to massive squashing and blurry resizing. It relies on a fragile `InpaintHead` monkey-patch to guess the context, which frequently breaks. We discovered during W03 execution that this patch conflicts with our new backend loader quality patches.

**Target Architecture (denoise_mask approach)**:
```python
class InpaintContext:
    """Carries all state between inpaint stages. No globals."""
    original_image: np.ndarray       # Full original image for final compositing
    bb_x: int                        # Crop coordinates
    bb_y: int
    bb_w: int
    bb_h: int
    latent_target: torch.Tensor      # Encoded native aspect-ratio BB
    latent_mask: torch.Tensor        # Downsampled denoise_mask for the sampler

class InpaintPipeline:
    def prepare_smart_outpaint(self, image, padding_directions) -> tuple:
        """Create target padding. Extend edge pixels natively and blur to create a structural primer instead of white noise."""

    def prepare_context_bb(self, image, target_mask) -> InpaintContext:
        """Calculate a Bounding Box that retains the native aspect ratio. Do NOT squash to 1:1. 
        Expand the BB by a context margin to include original pixels for the UNet to read."""
    
    def encode(self, context, vae) -> InpaintContext:
        """VAE encode the unsquashed BB. Generate the denoise_mask."""

    def stitch(self, context, generated_image) -> np.ndarray:
        """Crop the generated patch from the BB and stitch it perfectly onto the original image. No color correction needed."""
```

**Key Changes from Current Code**:
- **DEPRECATE `InpaintHead`:** We no longer monkey-patch the UNet input blocks.
- **DEPRECATE `solve_abcd` & Square Forcing:** The Bounding Box will maintain its native aspect ratio (no forced squashing).
- **NEW SDXL MECHANIC:** We pass `denoise_mask` down into `process_diffusion` -> `core.ksampler` -> `sampling.sample_sdxl`. This mathematically freezes the original pixels during generation, completely removing the need for `InpaintHead` guessing.
- `InpaintContext` replaces the module-level `inpaint_worker.current_task` global.

**Action**:
- Rewrite `modules/pipeline/inpaint.py` utilizing the `denoise_mask` approach.
- Update `backend/sampling.py` or `core.py` as needed to ensure `denoise_mask` is properly threaded from the UI pipeline down to the sampler.
- Update `async_worker.py` orchestrator to use the new `InpaintPipeline`.
- Remove `InpaintHead`, `InpaintWorker`, and `solve_abcd` entirely.

### 2. Create `modules/pipeline/controlnet.py` — ControlNetPipeline

*(Unchanged from original W03 Scope)*
**Target Architecture**:
```python
class ControlNetPipeline:
    def preprocess(self, cn_type, image, width, height, params=None) -> torch.Tensor:
        """Apply CN-type-specific preprocessing (canny, cpds, resize, etc.)."""
        # Dispatches to preprocessor by cn_type

    def apply(self, cn_tasks, positive_cond, negative_cond, pipeline) -> tuple:
        """Load CN models and apply to conditions."""
        # Iterates all CN task types
        # Returns (patched_positive, patched_negative)
```

**Action**:
- Create `modules/pipeline/controlnet.py` with `ControlNetPipeline`
- Extract per-type preprocessing from `apply_control_nets`
- Use a dispatch dict instead of if/else for CN types
- Move CN model downloading/loading triggers from `apply_image_input` into the pipeline

### 3. Update Orchestrator Integration

After pipeline extraction, the orchestrator in `handler()` should thread the `denoise_mask` correctly into diffusion.

## Verification
1. **Inpainting** works flawlessly at non-square aspect ratios (e.g., 832x1216).
2. **Outpainting** preserves sharpness and structurally blends extensions.
3. **ControlNet (Canny, CPDS, IP-Adapter)** works.
4. **No module-level globals** — `inpaint_worker.current_task` pattern eliminated.
5. **No `InpaintHead`** loading during generation.

## Success Criteria
- Native aspect-ratio inpainting achieved via `denoise_mask`.
- Legacy `inpaint_worker` and `InpaintHead` purged.
- ControlNet dispatch uses dict/registry pattern.
- Clean foundation for M13 registry/extension system.

## Implementation Steps (from Implementation Plan)

### Step 1: Create `InpaintPipeline` & Localize State (In Progress)
- [X] Create `modules/pipeline/inpaint.py` and `InpaintContext` dataclass.
- [ ] Pivot `prepare`, `encode`, and `stitch` methods to use native aspect-ratio bounding boxes and `denoise_mask`.
- [ ] Update `async_worker.py` and `inference.py` to pass the `denoise_mask` down to the diffusion sampler.
- [ ] **Verification 1**: Start UI, perform basic Inpainting and Outpainting. Verify that full-resolution stitching works without blurring or hallucinatory output.

### Step 2: Clean up `inpaint_worker.py`
- [ ] Remove `current_task` global from `inpaint_worker.py`.
- [ ] Delete `solve_abcd`, `InpaintWorker` class, and `InpaintHead` model as they are obsolete in the new architecture.
- [ ] **Verification 2**: Run an Inpainting task and ensure no crashes occur due to missing globals or missing `InpaintHead` references. 

### Step 3: Create `ControlNetPipeline`
- [ ] Create `modules/pipeline/controlnet.py`.
- [ ] Implement `ControlNetPipeline.preprocess()` with dispatch for Canny, CPDS, IP-Adapter, and Face IP-Adapter.
- [ ] Implement `ControlNetPipeline.apply()` to handle CN loading and model patching.
- [ ] Update `handler()` in `async_worker.py` and `process_task()` in `inference.py` to use `ControlNetPipeline`.
- [ ] **Verification 3**: Start UI. Test ControlNet Canny, CPDS, and IP-Adapter to ensure they function correctly.

### Step 4: Cleanup `image_input.py`
- [ ] Remove extracted legacy logic (`apply_control_nets`, `apply_inpaint`) from `modules/pipeline/image_input.py`.
- [ ] Ensure download triggers map directly to the active pipeline workflows cleanly.
- [ ] **Verification 4**: Final integration test. Test Inpaint + ControlNet combined. Ensure txt2img and Upscale/Variation flows remain unaffected.
