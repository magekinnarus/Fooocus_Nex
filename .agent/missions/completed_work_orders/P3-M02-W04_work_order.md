# Work Order: P3-M02-W04

**Mission:** P3-M02 (Component-First Loader)
**Phase:** 3
**Work Order ID:** P3-M02-W04
**Status:** Ready
**Depends On:** P3-M02-W03
**Date Issued:** 2026-02-14
**Philosophy:** Integration without Pollution

## Objective
Enable GGUF loading within `load_sdxl_unet` without polluting the clean backend with legacy code.

## Scope
1.  **Modify** `Fooocus_Nex/backend/loader.py`.
2.  **Imports**:
    *   We need `gguf_sd_loader` and `GGUFModelPatcher`.
    *   *Issue*: `modules.gguf` currently exists in legacy.
    *   *Decision*: Import from `modules.gguf`. It is a library.
    *   *Constraint*: Do NOT use `modules.gguf`'s auto-config features if they depend on Fooocus globals. Use the low-level loader.

## Logic
In `load_sdxl_unet`:
```python
if isinstance(source, str) and source.endswith(".gguf"):
    from modules.gguf.loader import gguf_sd_loader
    from modules.gguf.nodes import GGMLOps
    
    sd = gguf_sd_loader(source)
    # ... logic to create model with custom_operations=GGMLOps ...
    # ... logic to wrap in GGUFModelPatcher ...
    # Explicitly set patcher parameters (devices).
```

## Validation
*   Ensure that loading a GGUF file returns a `GGUFModelPatcher` that respects the passed `load_device` (even if GGUF manages some of its own memory, the container must be correct).
