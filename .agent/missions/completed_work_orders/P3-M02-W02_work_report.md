# Work Order Report: P3-M02-W02 (Checkpoint Splitting)

**ID:** P3-M02-W02
**Phase:** 3
**Date Completed:** 2026-02-14
**Status:** Complete
**Depends On:** P3-M02-W01

## 1. Summary
Implemented the core logic for extracting SDXL components from a full checkpoint. The logic uses externalized definitions from `defs/sdxl.py` to identify component keys, strips prefixes for normalization, and manages memory using atomic pops from the source state dictionary.

## 2. Scope Outcome
- [x] Load checkpoints using `ldm_patched.modules.utils.load_torch_file`
- [x] Split keys into 4 buckets (Unet, Clip_L, Clip_G, VAE)
- [x] Atomic CLIP Splitting (L and G as separate dictionaries)
- [x] Strip source prefixes from resulting keys
- [x] Memory safety via `.pop()`

## 3. Files Modified
| File | Change Type | Description |
|------|-------------|-------------|
| `Fooocus_Nex/backend/loader.py` | Modified | Implemented `extract_sdxl_components` logic |

## 4. Verification Results
### Logic Verification
A test script `verify_w02.py` was used to validate the extraction logic with a mock state dictionary.
- **Prefix Stripping**: Confirmed (e.g., `model.diffusion_model.input_blocks...` -> `input_blocks...`).
- **Atomic Splitting**: Confirmed (CLIP-L and CLIP-G are distinct in results).
- **Memory Safety**: Confirmed (Source state dict items are removed once extracted).
- **Key Normalization**: Confirmed (Residual dots handled).

```bash
Loading checkpoint from: dummy.ckpt
Extracted unet: 1 keys
Extracted clip_l: 1 keys
Extracted clip_g: 1 keys
Extracted vae: 1 keys
Remaining keys in checkpoint state_dict: 1
Verification OK: Prefix stripping and atomic splitting working.
```

## 5. Decision Records / Findings
- **Data-Driven Process**: The extractor is fully dependent on `sdxl_def.PREFIXES`, fulfilling the objective of keeping the loading process decoupled from the model architecture data.

## 6. Recommendations
- Proceed to **P3-M02-W03** (Implement Atomic Component Loaders).
