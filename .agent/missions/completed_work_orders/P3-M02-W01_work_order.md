# Work Order: P3-M02-W01

**Mission:** P3-M02 (Component-First Loader)
**Phase:** 3
**Work Order ID:** P3-M02-W01
**Status:** Ready
**Depends On:** None
**Date Issued:** 2026-02-14
**Philosophy:** Separation of Definition (Data) and Loader (Process)

## Objective
Establish the `Fooocus_Nex/backend` structure with a dedicated `defs` module for model specifications.

## Scope
1.  **Create Directory**: `Fooocus_Nex/backend/defs/`
2.  **Create File**: `Fooocus_Nex/backend/defs/sdxl.py`
    *   Contains `SDXL_PREFIXES` (The extraction map).
    *   Contains `SDXL_UNET_CONFIG` (The architecture spec).
    *   This file defines *WHAT* SDXL is.
3.  **Create File**: `Fooocus_Nex/backend/loader.py`
    *   Imports definitions from `defs/sdxl.py`.
    *   Defines the loading functions (`load_sdxl_unet`, etc.).
    *   This file defines *HOW* to load it.

## Technical Specifications

### `backend/defs/sdxl.py`
```python
PREFIXES = {
    "unet": "model.diffusion_model.",
    "clip_l": "conditioner.embedders.0.transformer.text_model",
    "clip_g": "conditioner.embedders.1.model.",
    "vae": "first_stage_model.",
}

UNET_CONFIG = {
    "model_channels": 320,
    "use_linear_in_transformer": True,
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "context_dim": 2048,
    "adm_in_channels": 2816,
    "use_temporal_attention": False,
}
```

### `backend/loader.py` Interface
```python
from .defs import sdxl

def extract_sdxl_components(ckpt_path: str) -> dict:
    # Uses sdxl.PREFIXES
    pass

def load_sdxl_unet(source, load_device, offload_device, dtype=None):
    # Uses sdxl.UNET_CONFIG
    pass
```

## Validation
*   `python -c "from Fooocus_Nex.backend.defs import sdxl; print(sdxl.UNET_CONFIG)"` works.
