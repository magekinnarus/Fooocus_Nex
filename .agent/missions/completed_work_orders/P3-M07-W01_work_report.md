# Work Report: P3-M07-W01

**ID:** P3-M07-W01
**Date Completed:** 2026-02-17
**Status:** Complete
**Depends On:** None
**Reference Material:** [Mission Brief](P3-M07_mission_brief.md), [Work Order](P3-M07-W01_work_order.md)

## 1. Objective
Extract process-flow logic for SD 1.5 and SDXL inference to create lean backend modules, ensuring compatibility with 3GB VRAM constraints (simulated) and supporting split GGUF loading.

## 2. Changes Implemented
### Backend Definitions
- **Created `backend/defs/sd15.py`**: defined `UNET_CONFIG` and state dict `PREFIXES` for SD 1.5, mirroring the definitions used for SDXL.

### Backend Loader
- **Updated `backend/loader.py`**:
  - Implemented `load_sd15_checkpoint`, `load_sd15_unet`, and `load_sd15_clip`.
  - Implemented "Interleaved Load-and-Discard" strategy for both SD 1.5 and SDXL to minimize RAM spikes.
  - **Dtype Enforcement**:
    - UNet & CLIP: Force `float16` (even on CPU if needed for RAM saving, though usually `load_device` governs).
    - VAE: Force `float32` for precision.
  - **Split Loading**: Added ability to load SDXL components (UNet, CLIP, VAE) from separate files, verifying support for GGUF UNets + bundled CLIPs.

## 3. Findings & Verification
### Verification Script
- Created `tests/test_w01_loading.py` to exercise all loading paths.
- Script automatically detects and uses models from `Fooocus_Nex/models`.

### Test Results
1. **SD 1.5 Checkpoint (`SD_photon_v1.safetensors`)**:
   - ✅ Successfully loaded UNet, CLIP, and VAE.
   - ✅ Verified correct wrapper classes (`ModelPatcher`, `CLIP`, `VAE`).

2. **SDXL Checkpoint (`XL_juggernaut_v8.safetensors`)**:
   - ✅ Successfully loaded all components using the sequential extraction method.
   - ✅ Memory peak managed by deleting raw state dict chunks immediately after extraction.

3. **SDXL Split Loading**:
   - **UNet**: `IL_dutch_v30_Q4_K_M.gguf` (GGUF format) loaded successfully using `GGUFModelPatcher`.
   - **CLIP**: `IL_dutch_v30_clips.safetensors` loaded successfully.
   - **VAE**: `sdxl_vae.safetensors` loaded successfully.

4. **Hardware Adaptation**:
   - Verified that `resources.py` logic combined with explicit dtype casting in `loader.py` allows these large models to load without OOM on constricted systems, provided offload devices are managed (CPU offload verified).

## 4. Next Steps
- Pass control to W02 for `app.py` construction.
- `loader.py` is ready to be imported and used by the new inference runner.
