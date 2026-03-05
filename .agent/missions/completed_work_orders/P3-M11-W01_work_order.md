# Work Order: P3-M11-W01 — Dead Code Removal
**ID:** P3-M11-W01
**Mission:** P3-M11
**Status:** ✅ Complete
**Depends On:** —

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `.agent/summaries/06_Implementation_Observations_and_Insights.md`
- `Fooocus_Nex/modules/patch.py` (to understand `anisotropic` import redirect)

## Objective
Remove all confirmed dead files from `modules/` and strip 22 unused ComfyUI node files from `ldm_patched/contrib/`. This is a zero-logic-change cleanup — no behavior should differ after this work order.

## Scope

### 1. Delete Dead `modules/` Files

| File | Reason | Verification |
|:---|:---|:---|
| `modules/nex_loader.py` (6.3 KB) | Zero importers. Superseded by `backend/loader.py`. | `grep -r "nex_loader" --include="*.py"` returns only self-references |
| `modules/PATCH_MANIFEST.md` (2.9 KB) | Zero references from any code or doc. Obsolete tracking artifact. | `grep -r "PATCH_MANIFEST"` returns nothing |

### 2. Remove Duplicate `modules/anisotropic.py`
- **Current**: `modules/patch.py` line 7 imports `import modules.anisotropic as anisotropic`
- **Backend**: `backend/anisotropic.py` is a byte-identical copy
- **Action**: Delete `modules/anisotropic.py`. Update `modules/patch.py` line 7 to `import backend.anisotropic as anisotropic`
- **Pre-check**: Confirm files are identical with `fc /b modules\anisotropic.py backend\anisotropic.py` (or diff)

### 3. Strip Unused `ldm_patched/contrib/` Nodes

**KEEP these 3 actively imported files:**
- `external_align_your_steps.py` — imported by `modules/sample_hijack.py`
- `external_custom_sampler.py` — imported by `modules/sample_hijack.py`, `modules/core.py`
- `external_upscale_model.py` — imported by `modules/upscaler.py`

**DELETE these 22 files:**
| File | Size |
|:---|:---|
| `external.py` | 77 KB |
| `external_canny.py` | 12 KB |
| `external_clip_sdxl.py` | 3 KB |
| `external_compositing.py` | 8 KB |
| `external_freelunch.py` | 5 KB |
| `external_hypernetwork.py` | 5 KB |
| `external_hypertile.py` | 3 KB |
| `external_images.py` | 7 KB |
| `external_latent.py` | 5 KB |
| `external_mask.py` | 13 KB |
| `external_model_advanced.py` | 8 KB |
| `external_model_downscale.py` | 3 KB |
| `external_model_merging.py` | 11 KB |
| `external_perpneg.py` | 2 KB |
| `external_photomaker.py` | 8 KB |
| `external_post_processing.py` | 10 KB |
| `external_rebatch.py` | 5 KB |
| `external_sag.py` | 6 KB |
| `external_sdupscale.py` | 2 KB |
| `external_stable3d.py` | 5 KB |
| `external_tomesd.py` | 7 KB |
| `external_video_model.py` | 5 KB |

**Pre-check**: For each file, confirm zero external imports with `grep -r "filename_without_ext" --include="*.py"` excluding `ldm_patched/contrib/` self-references.

### 4. Evaluate `ldm_patched/t2ia/` for Deletion
- **CM Role 1 Finding**: Do NOT delete `ldm_patched/t2ia/`. It is actively imported and used by `ldm_patched/modules/controlnet.py` and `ldm_patched/modules/sd.py`. Removal would break these core model components. Leave intact.

## Verification
1. **Fooocus launches** without import errors (`python launch.py --preset default`)
2. **txt2img generation** produces a valid image through the UI (SD1.5 or SDXL GGUF)
3. **`app.py` headless test** still passes (`python app.py --config test_sdxl_quality_config.json`)
4. **No missing module errors** in any `import` statement

## Success Criteria
- All listed dead files are deleted
- `modules/anisotropic.py` no longer exists; `patch.py` imports from `backend/`
- `ldm_patched/contrib/` contains only 3 files (+ `__pycache__/`)
- Zero functional difference — all tests pass identically to post-M10 baseline
