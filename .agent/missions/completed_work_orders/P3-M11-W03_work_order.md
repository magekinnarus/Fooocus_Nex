# Work Order: P3-M11-W03 — Backend `ldm_patched` Import Reduction
**ID:** P3-M11-W03
**Mission:** P3-M11
**Status:** ✅ Completed
**Depends On:** P3-M11-W02

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `Fooocus_Nex/backend/lora.py` (lines 310–355 — model type checking)
- `Fooocus_Nex/backend/utils.py` (lines 1–10, 159 — checkpoint_pickle import)
- `Fooocus_Nex/backend/weight_ops.py` (line 75 — weight_adapter import)
- `Fooocus_Nex/backend/loader.py` (lines 8–9 — model_base, latent_formats imports)

## Objective
Reduce the backend's remaining `ldm_patched` imports to the absolute minimum. The backend should ideally depend only on `ldm_patched` for model architecture definitions (`model_base`, `latent_formats`) and low-level weight operations (`weight_adapter`) — everything else should be self-contained.

## Scope

### 1. Clean `backend/lora.py` — Remove Unsupported Model Type Checks

**Current state**: Lines 310–355 contain `hasattr(ldm_patched.modules.model_base, 'X')` checks for 6 model architectures that Fooocus_Nex does not support:
- `StableCascade_C` (line 312)
- `HunyuanDiT` (line 319)
- `GenmoMochi` (line 325)
- `HunyuanVideo` (line 331)
- `HiDream` (line 343)
- `ACEStep` (line 351)

**Action**: Remove these 6 `hasattr` blocks and their associated key-mapping logic. Keep only the SD1.5 and SDXL key-mapping paths that Fooocus_Nex actually uses.

**Side effect**: This may allow removing the `import ldm_patched.modules.model_base` at line 2 if no other references remain. Verify after cleanup.

### 2. Inline `checkpoint_pickle` in `backend/utils.py`

**Current state**: `backend/utils.py` imports `ldm_patched.modules.checkpoint_pickle` (line 5) and uses it at line 159 for safe `.ckpt` loading.

**The imported module is ~7 lines** — a simple `RestrictedUnpickler` class that blocks arbitrary code execution during `torch.load`.

**Action**: Inline the `RestrictedUnpickler` class directly into `backend/utils.py`. Remove the `ldm_patched` import.

### 3. Evaluate `backend/weight_ops.py` — `weight_adapter` Dependency

**Current state**: Line 75 has a lazy import: `import ldm_patched.modules.weight_adapter as weight_adapter`. This is used for `calculate_weight()` dispatching.

**Assessment**:
- `weight_adapter/` is a complex module (8 files) deeply integrated with `model_patcher.py`
- It handles weight format conversions (LoRA, LoHA, LoKR, etc.)
- Extracting it would be high-risk with limited immediate benefit

**Action**: **Document but defer**. Note this as a known remaining dependency for Phase 4. Do not attempt extraction in this mission.

### 4. Document Remaining `ldm_patched` Dependencies

After W03 changes, produce a summary table of all remaining `ldm_patched` imports in `backend/` with justification for why each one remains.

## Verification
1. **Fooocus launches** without import errors
2. **txt2img generation** works through the UI with LoRA
3. **`.ckpt` model loading** still works via `backend/utils.py` (test with SD1.5 full checkpoint)
4. **`app.py` headless test** passes for both SD1.5 and SDXL with LoRA
5. **No missing import errors** — verify with `python -c "import backend.lora; import backend.utils; import backend.weight_ops"`

## Success Criteria
- `backend/lora.py` has zero references to unsupported model architectures
- `backend/utils.py` has zero `ldm_patched` imports
- Remaining `backend/` `ldm_patched` dependencies are documented with justification
- Zero functional difference — all tests pass identically
