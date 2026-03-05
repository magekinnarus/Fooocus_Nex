# Mission P3-M02: Component-First Loader (`nex_loader.py`)
**ID:** P3-M02
**Phase:** 3 (Core Pipeline Decomposition)
**Date Issued:** 2026-02-13
**Status:** Ready

## Required Reading
Read these files FIRST before starting any work:
- `.agent/reference/P3-M01_reference_trace.md` ??Full SDXL pipeline trace (especially Stage 1: Loading)
- `.agent/summaries/02_Architecture_and_Strategy.md` ??Step-module architecture and component-first philosophy
- `.agent/rules/01_Global_Context_Rules.md` ??Consensus protocol (no cowboy coding)
- `modules/nex_loader.py` ??Current loader implementation (Phase 1.5 adapter)

## Objective
Evolve `nex_loader.py` from a thin adapter into the **definitive component-first SDXL loader** ??a single module that can load UNet, CLIP-L, CLIP-G, and VAE from any source format and return them as independent, device-managed components.

## Scope

### In Scope
1. **Checkpoint extraction** ??Accept a bundled SDXL checkpoint (.safetensors), split into UNet/CLIP/VAE state dicts by key prefix
2. **Component loading** ??Load individual UNet, CLIP-L, CLIP-G, VAE files directly
3. **GGUF support** ??Preserve existing GGUF loading path for UNet (Q4/Q5 quantized)
4. **Unified interface** ??Same downstream API regardless of source format (checkpoint vs. separate files)
5. **ModelPatcher wrapping** ??Return components wrapped for device management

### Out of Scope
- Non-SDXL architectures (SD1.5, SD3, Flux, etc.)
- LoRA loading/patching (that's `nex_patching.py`, P3-M06)
- Memory management choreography (that's `nex_memory.py`, P3-M04)
- Inference/generation pipeline

## Technical Specification

### Key Prefix Map (SDXL Checkpoint)
```python
SDXL_PREFIXES = {
    "unet": "model.diffusion_model.",
    "clip_l": "conditioner.embedders.0.transformer.text_model",
    "clip_g": "conditioner.embedders.1.model.",
    "vae": "first_stage_model.",
}
```

### CLIP State Dict Remapping (from `supported_models.py:220`)
When extracting from checkpoint, CLIP keys must be remapped:
```python
"conditioner.embedders.0.transformer.text_model" ??"clip_l.transformer.text_model"
"conditioner.embedders.1.model."                 ??"clip_g."
```
Also apply `clip_text_transformers_convert` for CLIP-G format normalization.

### Required Functions
```python
def extract_components(checkpoint_path: str) -> dict:
    """Split a bundled SDXL checkpoint into component state dicts.
    Returns: {"unet": sd, "clip_l": sd, "clip_g": sd, "vae": sd}
    """

def load_unet(source, model_options: dict = {}) -> ModelPatcher:
    """Load SDXL UNet from file path or state dict.
    Handles: .safetensors, .gguf, or pre-extracted state dict.
    Returns: ModelPatcher wrapping the UNet model.
    """

def load_clip(source, embedding_directory: str = None) -> CLIP:
    """Load SDXL CLIP (bundled L+G, or individual).
    Handles: bundled CLIP file, individual CLIP-L/CLIP-G files,
             or pre-extracted state dict.
    Returns: CLIP object wrapping SDXLClipModel.
    """

def load_vae(source) -> VAE:
    """Load SDXL VAE from file path or state dict.
    Returns: VAE object.
    """
```

### SDXL Model Config (Hardcoded ??No Detection Needed)
```python
SDXL_UNET_CONFIG = {
    "model_channels": 320,
    "use_linear_in_transformer": True,
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "context_dim": 2048,
    "adm_in_channels": 2816,
    "use_temporal_attention": False,
}
```

## Reference Code Paths
| What | ComfyUI Reference | Lines |
|------|-------------------|-------|
| Checkpoint splitting | `sd.py` ??`load_state_dict_guess_config` | 1032??110 |
| UNet loading | `sd.py` ??`load_diffusion_model_state_dict` | 1113??193 |
| CLIP key remapping | `supported_models.py` ??`SDXL.process_clip_state_dict` | 220??30 |
| CLIP model creation | `sdxl_clip.py` ??`SDXLClipModel` | 41??8 |
| VAE creation | `sd.py` ??`VAE.__init__` | 270??99 |
| SDXL config | `supported_models.py` ??`SDXL` class | 181??51 |
| Model type detection | `supported_models.py` ??`SDXL.model_type` | 195??12 |

## Constraints
1. **SDXL only** ??No multi-model dispatch. No `detect_unet_config`. Config is hardcoded.
2. **No ComfyUI imports** ??Extract the logic, don't import from ComfyUI reference.
3. **Component-first** ??The loader always returns individual components, never a bundled "model".
4. **Preserve GGUF path** ??The existing `modules/gguf/` integration must continue to work.
5. **Testable from terminal** ??Each function should be callable from a Python script.
6. **Consensus protocol** ??Get Director approval on implementation plan before writing code.

## Deliverables
1. Updated `modules/nex_loader.py` with the functions specified above
2. Test script demonstrating: load from checkpoint, load from separate files, load GGUF UNet
3. Progress report documenting any discoveries or issues encountered

## Success Criteria
- [ ] `extract_components("sdxl.safetensors")` returns 4 state dicts
- [ ] `load_unet(path)` returns a `ModelPatcher` for both .safetensors and .gguf
- [ ] `load_clip(path)` returns a `CLIP` object with working tokenizer + encoder
- [ ] `load_vae(path)` returns a working `VAE` object
- [ ] All loads work from a standalone Python script (no Fooocus startup)
