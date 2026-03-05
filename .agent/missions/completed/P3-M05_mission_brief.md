# Mission Brief: P3-M05 — VAE Decode
**ID:** P3-M05
**Phase:** 3
**Date Issued:** 2026-02-16
**Status:** Draft
**Depends On:** P3-M04 (Sampling Engine), P3-M02 (Loader — provides `VAE` container)
**Work List:** `.agent/missions/active/P3-M05_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/reference/P3-M01_reference_trace.md` (Stage 4: VAE Decode)
- `.agent/missions/completed/P3-M04_mission_report.md` (predecessor)

## Objective

Build `backend/decode.py` — a clean SDXL VAE decode module that takes denoised latent tensors (output of `backend/sampling.py`) and produces pixel-space images. This includes a tiled decode fallback for VRAM-limited environments (e.g., GTX 1050 with 4GB VRAM).

This is a **low-to-medium complexity extraction**. The VAE decode path is significantly simpler than the sampling engine (M04) — it's a single forward pass through the decoder network with normalization, plus a tiled fallback path.

## The Strategy: "Clean Extraction" & "Standalone Verification"
> [!IMPORTANT]
> **Source of Truth:** All logic MUST be extracted from `ComfyUI_reference`.
> **Forbidden:** Do NOT import from `ldm_patched` under any circumstances for new logic.
> **Verification:** Every new module must be verified with a standalone script in `tests/`.

## Scope

### In Scope

1. **VAE Decode Function** — Extract the core decode logic from `ComfyUI_reference/comfy/sd.py` → `VAE.decode()`:
   - Latent → pixel conversion via `first_stage_model.decode()`
   - Output normalization: `torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)`
   - Batch processing for multi-sample decode
   - VRAM-aware batching (simplified from ComfyUI's `memory_used_decode`)

2. **Tiled Decode Fallback** — Extract `VAE.decode_tiled_()` for VRAM-limited environments:
   - 3-pass averaged tiled decode (standard, wide, tall tiles) for artifact-free results
   - Tile sizes appropriate for SDXL's 8× spatial compression (default: 64×64 latent tiles)
   - Requires extracting `tiled_scale` / `tiled_scale_multidim` utilities from `ComfyUI_reference/comfy/utils.py`

3. **Image Post-Processing** — Convert decoded tensor to saveable image:
   - `.movedim(1, -1)` to convert CHW → HWC
   - Tensor → PIL Image / numpy array conversion
   - Optional: save to file utility function

4. **Clean API** — A top-level function like:
   ```python
   def decode_latent(
       vae: VAE,                    # From backend/loader.py
       latent: torch.Tensor,        # [B, 4, H, W] from backend/sampling.py
       tiled: bool = False,         # Force tiled decode
       tile_size: int = 64,         # Latent-space tile size
   ) -> torch.Tensor:               # [B, H*8, W*8, 3] pixel tensor, float32 [0,1]
   ```

### Out of Scope
- **VAE encode** (image→latent) — Not needed for text-to-image generation pipeline; can be added when img2img or inpainting is implemented
- **Video VAE** (1D, 3D decode paths) — SDXL is 2D only
- **TAESD** (Tiny AutoEncoder) — Preview decoder, not production quality
- **Diffusers format conversion** — Already handled by `loader.py`
- Modifying any existing `modules/` or `ldm_patched/` code

## Reference Files
- `ComfyUI_reference/comfy/sd.py` — `VAE` class: `decode()`, `decode_tiled_()`, `process_output()`
- `ComfyUI_reference/comfy/utils.py` — `tiled_scale()`, `tiled_scale_multidim()`, `get_tiled_scale_steps()`
- `Fooocus_Nex/backend/loader.py` — `VAE` container class, `load_sdxl_vae()`
- `Fooocus_Nex/backend/sampling.py` — Upstream module (produces latent tensors)

## Constraints
- Follow all design principles from DR-001
- New code in `backend/decode.py` only (+ utilities if needed)
- `decode.py` may import from `backend/resources.py` (device management) and `backend/loader.py` (VAE type)
- The `first_stage_model` (AutoencoderKL) comes from the `VAE` container loaded by `loader.py` — the decode module should **not** handle model loading
- The tiled_scale utility may be placed in `backend/decode.py` or a separate `backend/utils.py` — implementor's choice based on size
- Include the Dependency Inventory table per DR-001

## Deliverables
- [x] `backend/decode.py` — the main decode module
- [x] `tests/test_backend_decode.py` — standalone verification script
- [x] Dependency Inventory in work report

## Success Criteria
1. `decode_latent(vae, latent)` produces a pixel tensor of shape `[B, H*8, W*8, 3]` with values in `[0, 1]`
2. Tiled decode produces visually identical results (no visible tile boundaries)
3. `decode.py` has zero imports from `ldm_patched`
4. Standalone test passes without requiring the full Fooocus pipeline
5. The decode module connects cleanly to the `VAE` container from `loader.py`

## Work Orders
Registered in mission work-list (`P3-M05_work_list.md`):
- `P3-M05-W01` — Extract core decode logic + tiled fallback + `tiled_scale` utility
- `P3-M05-W02` — Verification & dependency inventory

## Notes
- **Complexity Assessment:** This is significantly simpler than M04 (Sampling). The decode path is a single forward pass — no multi-step loop, no guidance calculation, no condition processing. The main complexity is in the tiled fallback, which is well-isolated.
- **VAE Container:** The `VAE` class in `loader.py` already wraps `first_stage_model` and `patcher`. The decode module should accept this container directly.
- **VRAM on GTX 1050:** Full decode of a 1024×1024 SDXL image requires ~1.3GB. The tiled fallback should work for environments where this is too much. The 3-pass averaged approach (standard + wide + tall tiles) from ComfyUI prevents tile boundary artifacts.
- **`model_management` Dependency:** ComfyUI's `VAE.decode()` uses `model_management.load_models_gpu()` for device placement. Our `backend/resources.py` has equivalent functionality — use that instead.
- **Output Format:** The pixel tensor should follow the convention `[B, H, W, C]` (channels-last, float32, range [0,1]). This is what PIL/numpy expect for image saving.
