# Local Environment Guidelines

## Virtual Environment
- **Path**: `Fooocus_Nex/venv`
- Use this environment for all dependency resolution and execution.

## Directories
- **Checkpoints**: `Fooocus_Nex/models/checkpoints/`
  - SD 1.5 models are prefixed with `SD_`
  - SDXL models are prefixed with `XL_`
- **GGUF Models (SDXL)**: 
  - `Fooocus_Nex/models/unet/`
  - `Fooocus_Nex/models/clip/`
  - `Fooocus_Nex/models/vae/`
- **LoRA Models**: 
  - SD 1.5 LoRAs: `Fooocus_Nex/models/loras/SD15/`
  - SDXL LoRAs: `Fooocus_Nex/models/loras/SDXL/`

## Hardware Constraints
- **Local SDXL Execution**: The local hardware cannot run full SDXL checkpoints due to VRAM/RAM constraints. 
- **Workaround**: To test SDXL logic locally, ALWAYS use quantized GGUF components loaded as separate UNet, CLIP, and VAE files from their respective `models/` subdirectories. Do not attempt to load full `XL_*.safetensors` checkpoints.

## Testing and Configurations
- **Test Scripts**: All test scripts must be placed in either the `tests/` folder in the root directory, or the `Fooocus_Nex/tests/` subfolder. Do not clutter the root directory with standalone test `.py` scripts.
- **Config Files**: JSON configuration files for inference (e.g., `test_sd15_quality_config.json`, `test_sdxl_quality_config.json`) should inhabit the root directory. However, obsolete configs must be cleaned up regularly, and we should consolidate wherever possible.

## Version Control (Git)
- **NO AUTOMATED PUSHES**: The AI agent is strictly forbidden from running `git push`. The user may manually edit files alongside the agent. All `git push` operations will be executed manually by the USER. The agent may still `git add` or `git commit` if explicitly requested or appropriate.
