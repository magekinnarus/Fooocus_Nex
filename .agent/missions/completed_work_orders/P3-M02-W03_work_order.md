# Work Order: P3-M02-W03

**Mission:** P3-M02 (Component-First Loader)
**Phase:** 3
**Work Order ID:** P3-M02-W03
**Status:** Ready
**Depends On:** P3-M02-W02
**Date Issued:** 2026-02-14
**Philosophy:** Instantiation using Definition

## Objective
Implement `load_sdxl_unet` (and friends) in `Fooocus_Nex/backend/loader.py`.

## Imports
1.  **Definitions**: `from .defs import sdxl as sdxl_def`
2.  **Runtime Classes**: `from ldm_patched.modules import model_base, model_patcher` (The Engine)

## Logic
```python
def load_sdxl_unet(source, load_device, offload_device, dtype=None):
    sd = resolve_source(source)
    
    # Instantiate using the separated Config
    model = model_base.SDXL(
        **sdxl_def.UNET_CONFIG  # Unpack the config from the def file
    )
    
    # Load weights (Generic logic or helper in loader.py)
    model.load_state_dict(sd, strict=False)
    
    # Wrap
    patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    return patcher
```

## Validation
*   Loader works using externalized config.
*   Demonstrates how easily `load_flux_unet` could be added by importing `defs.flux` and instantiating `model_base.Flux` (hypothetically).
