# Work Report: P3-M06-W02 - Fix VAE OOM with GGUF Models

**ID:** P3-M06-W02
**Status:** Completed
**Owner:** Role 2 (Implementor)

## Changes Implemented
Modified `backend/decode.py`:
- In `_decode_tiled`, added calculation of memory usage for a single tile.
- Added call to `resources.load_models_gpu` before the tiled decode loop to ensure the VAE is loaded and sufficient VRAM is cleared.

## Verification Instructions
1. Run the validation script with GGUF models:
   ```bash
   python scripts/validate_p3.py --unet "H:\webui_forge_cu121_torch21\webui\models\Stable-diffusion\unet\beretMixReal_v80_Q4_K_M.gguf" --clip "H:\webui_forge_cu121_torch21\webui\models\text_encoder\bundled_clips\beretMixReal_v80_clips.safetensors" --vae "D:\AI\Fooocus_revision\Fooocus_Nex\models\vae\sdxl_vae.safetensors" --steps 4 --device cuda
   ```
2. Verify `completed_image.png` is generated successfully.

**Verification Result (2026-02-16):**
- **512x512:** Script executed successfully. VAE decode completed without OOM.
- **1024x1024:** Script successfully triggered OOM projection and fell back to tiled decoding.
    - **Outcome:** The system stabilized using deep fallback (16x16 tiles) on 3GB VRAM.
    - **Note:** Performance is slow due to extreme memory constraints and swapping, but the process does not crash. Dtype mismatch error was also resolved.



