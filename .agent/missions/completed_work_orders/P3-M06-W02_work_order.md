# Work Order: Fix VAE OOM with GGUF Models

**ID:** P3-M06-W02
**Mission:** P3-M06
**Status:** Completed

## Problem Description
Validation fails with OOM when using GGUF UNet and standard/GGUF VAE on GPU (4GB target). The `backend/decode.py` module catches the initial full-decode OOM but fails during the fallback tiled-decode, likely because it does not ensure the VAE is loaded or fails to clear sufficient VRAM (e.g., evicting the large GGUF UNet).

## Objective
Modify `backend/decode.py` to explicitly manage VRAM usage during tiled decoding, ensuring the VAE is loaded and sufficient memory is cleared.

## Implementation Steps

### 1. Modify `backend/decode.py`
- [x] Calculate memory required for a single tile operation.
- [x] Call `resources.load_models_gpu` before tiled decode loop.

## Verification
User to verify with GGUF workflow.
