# Mission Report: 001 — ldm_patched Foundation
**Phase:** 1
**Date Completed:** 2026-02-11
**Brief Reference:** `missions/active/001_ldm_foundation.md`

## Summary
Successfully modernized the `ldm_patched` core infrastructure by selective porting of ComfyUI 0.3.47 components. This established a robust foundation for modern weight handling (LoRA/adapters), resolved critical Colab performance regressions, and prepared the codebase for upcoming GGUF and quantized model support (Phase 2).

## Findings
### Key Discoveries
- **VRAM State Regression**: Colab environments with >=13GB VRAM were incorrectly triggering `NORMAL_VRAM` state, causing aggressive and slow weight offloading. Forcing `HIGH_VRAM` resolved the 4.68s/it lag.
- **Precision Mismatch**: Weights were loading in FP32 (10.2GB) because `ModelPatcher` skipped casting for unpatched layers. Fixed to ensure a consistent 5.1GB FP16 footprint.
- **Unified Precision Pass**: Standardizing UNet inputs (x, context, y, control) to `weight_dtype` at the start of the forward pass eliminates per-layer upcasting overhead and "self and mat2 must have the same dtype" errors.
- **Sharpness Filter Overhead**: The anisotropic filter caused significant early-step delays. A threshold check (alpha < 0.01) now skips this expensive operation when its visual impact is negligible.

### Files Modified
| File | Change Type | Description |
|------|-------------|-------------|
| `ldm_patched/modules/model_management.py` | Modified | Forced `HIGH_VRAM` on Colab; updated precision logic. |
| `ldm_patched/modules/model_patcher.py` | New | Ported modern `ModelPatcher` with Fooocus memory overrides. |
| `modules/patch.py` | Modified | Implemented Unified Precision Pass and Sharpness optimization. |
| `ldm_patched/modules/weight_adapter/` | New | Ported 9 adapter types for modern LoRA support. |
| `ldm_patched/modules/ops.py` | Modified | Updated for precision consistency. |
| `ldm_patched/modules/quant_ops.py` | New | Ported for Phase 2 readiness. |
| `ldm_patched/modules/lora.py` | Modified | Adjusted for `intermediate_dtype` support. |
| `ldm_patched/modules/model_base.py` | Modified | Added modern model classes (StableCascade_C, SD3, etc.) for `lora.py` compatibility. |
| `ldm_patched/modules/model_sampling.py` | Modified | Added modern sampling classes for model_base dependencies. |
| `extras/expansion.py` | Deleted | Removed Fooocus V2 GPT-2 expansion. |
| `modules/default_pipeline.py` | Modified | Bypassed expansion logic. |
| `modules/async_worker.py` | Modified | Removed expansion calls. |
| `modules/sdxl_styles.py` | Modified | Removed expansion formatting. |
| `modules/meta_parser.py` | Modified | Added metadata filtering for legacy "Fooocus V2" strings. |

## Issues Encountered
- **LoRA Application Regressions**: Initial porting caused LoRA effects to disappear due to incorrect device placement and initialization logic in `ModelPatcher`.
- **Double-Patching OOM**: Precision forcing in `stochastic_rounding` caused weights to accidentally upcast to FP32 during patching, doubling memory usage on T4.
- **AttributeError Cascade**: Missing modern model classes in `model_base.py` (e.g., `StableCascade_C`, `SD3`) caused `lora.py`'s patching logic to fail.

## Architectural Insights for Future Phases
> [!IMPORTANT]
> Phase 1 revealed systemic friction between Fooocus's monkey-patching and ComfyUI's evolving module internals. Key finding:

- **17+ monkey-patch targets** across `patch.py` (10), `patch_clip.py` (5), and `patch_precision.py` (2) create implicit coupling with ComfyUI internals.
- **`calculate_weight_patched`** (134 lines in `patch.py` overriding `lora.calculate_weight`) was the single biggest source of bugs — precision mismatch, LoRA regression, and double-patching OOM all traced back to this override.
- **Recommendation:** Before Phase 2, absorb weight calculation into `model_patcher.py` and create a `nex_loader.py` adapter for component-wise model loading. See `missions/active/002_targeted_refactoring.md`.

## Recommendations
- **Phase 1.5 Refactoring**: Address architectural debt before Phase 2. See mission brief 002.
- **Phase 2 Transition**: The foundation is stable for GGUF integration, but loader changes should use the new `nex_loader.py` adapter rather than patching `load_checkpoint_guess_config`.
- **Lint Cleanup**: Several lint errors remain in `patch.py` regarding type inference and missing attributes; cosmetic but should be reviewed.

## Verification
- **Performance**: Generation speeds on T4 improved from 4.68s/it back to < 1.1s/it.
- **Memory**: SDXL peak VRAM with 2 LoRAs stabilized at ~9.9GB on T4.
- **Correctness**: User verified that LoRA effects are active and OOM errors are resolved.
