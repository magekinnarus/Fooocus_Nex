# Work Order: P3-M05-W01 — Implementation

**ID:** P3-M05-W01
**Mission:** P3-M05 — VAE Decode
**Status:** In Progress
**Depends On:** None

## Reference Material
- **Mission Brief:** `.agent/missions/active/P3-M05_mission_brief.md`
- **Source Code (Truth):**
  - `ComfyUI_reference/comfy/sd.py` (lines ~516-609)
  - `ComfyUI_reference/comfy/utils.py` (tiling utilities)
Extract VAE decode logic from `ComfyUI_reference` into a clean `backend/decode.py` module. Include tiled decoding capabilities for low-VRAM environments.

## Success Criteria
1. `backend/decode.py` exists and exports `decode_latent(vae, latent, tiled=False, tile_size=64)`.
2. `backend/utils.py` (or similar) contains the extracted `tiled_scale` utilities.
3. No imports from `ldm_patched` in the new modules.
4. Uses `backend/resources.py` for device management.
5. `decode_latent` returns `[B, H, W, 3]` float32 tensor in `[0, 1]`.

## Tasks
1. **Extract Utilities**:
   - Create `backend/utils.py` (if not exists) or adding to it.
   - Trace and extract `tiled_scale`, `tiled_scale_multidim`, `get_tiled_scale_steps` from `ComfyUI_reference/comfy/utils.py`.
   - Ensure these utilities use `backend/resources.py` or standard torch where applicable, removing any `comfy.model_management` dependencies if they exist within them (likely not, but check `model_management` usage in `utils.py`).

2. **Extract Decode Logic**:
   - Create `backend/decode.py`.
   - Implement `decode_latent` function that wraps the VAE's `decode` method.
   - Port `VAE.decode` and `VAE.decode_tiled_` logic from `ComfyUI_reference/comfy/sd.py`.
   - Adapt logic to use `backend/resources.py` for device handling (replacing `model_management.load_models_gpu`).
   - Ensure the code handles the `VAE` container from `backend/loader.py`.

3. **Refine & Clean**:
   - Type hints for all public functions.
   - Docstrings explaining inputs/outputs.
   - Remove any unused code or `comfy` specific artifacts (e.g. `pbar` if not strictly needed or replace with simple tqdm).

## Technical Notes
- **Source of Truth**: `ComfyUI_reference/comfy/sd.py` lines ~516-609 (decode methods) and `comfy/utils.py` ~900-1012 (tiled scale).
- **Memory Management**: The `VAE.decode` in Comfy checks memory usage. We should preserve this logic or simplify it using `backend/resources.py`.
- **Tiling**: The tiling logic is crucial for GTX 1050 (4GB) support. Ensure the `tile_size` default (64 latent pixels = 512 image pixels) is preserved or configurable.
