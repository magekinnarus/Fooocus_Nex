# Work Order: P3-M02-W05

**Mission:** P3-M02 (Component-First Loader)
**Phase:** 3
**Work Order ID:** P3-M02-W05
**Status:** Ready
**Depends On:** P3-M02-W04
**Date Issued:** 2026-02-14

## Objective
Verify the new backend in isolation using `tests/p3_m02_clean_loader_test.py`.

## Scope
1.  **Create** `tests/p3_m02_clean_loader_test.py`.
2.  **Imports**: definition of isolation.
    *   `import torch`
    *   `import Fooocus_Nex.backend.loader as backend_loader`
    *   **NO** `modules.config`, `modules.default_pipeline`, etc.
3.  **Test Logic**:
    *   `ckpt_path = args.path`
    *   `device = torch.device("cpu")`
    *   `print("Extracting...")`
    *   `components = backend_loader.extract_sdxl_components(ckpt_path)`
    *   `print("Loading UNet...")`
    *   `unet = backend_loader.load_sdxl_unet(components["unet"], device, device)`
    *   `print(f"Success: {type(unet)}")`
    
## Outcome
*   Passes on CLI.
*   Proves we can build an SDXL pipeline without the "Fooocus Legacy" baggage.
