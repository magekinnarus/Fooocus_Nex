# Work Order: P3-M11-W04 — UI Cleanup & Backend Wiring
**ID:** P3-M11-W04
**Mission:** P3-M11
**Status:** COMPLETED
**Depends On:** P3-M11-W01

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/summaries/07_Backend_Dependency_Map.md`
- `Fooocus_Nex/webui.py` (1016 lines — full Gradio UI definition)
- `Fooocus_Nex/launch.py` (137 lines — startup and model download)
- `Fooocus_Nex/modules/flags.py` (191 lines — sampler/scheduler lists, UI constants)
- `Fooocus_Nex/backend/sampling.py` (lines 82–94 — SAMPLER_NAMES, KSAMPLER_NAMES)
- `Fooocus_Nex/backend/schedulers.py` (scheduler function registry)

## Objective
Clean the Fooocus UI of dead/disabled features, fix the double-load console output, and wire the sampler/scheduler dropdowns to the backend's actual registries. This ensures M12 refactoring starts from a clean, accurate UI.

## Scope

### 1. Fix Double Console Load Display

**Problem**: Launching Fooocus prints startup info twice:
```
[System ARGV] ['launch.py']
Python 3.11.9 ...
Fooocus version: 2.5.5
...
[System ARGV] ['launch.py']   ← duplicate
Python 3.11.9 ...             ← duplicate
```

**Root cause**: `launch.py` executes `prepare_environment()` and all model download logic at **module scope** (lines 5, 48, 129–134). When `webui.py` does `import launch` (line 18), and Gradio's `launch()` triggers a module reload, this code re-executes.

**Action**:
- Add a module-level guard flag (e.g., `_INITIALIZED = False`) to prevent re-execution
- OR wrap all module-scope side-effects in `if __name__ == '__main__'` or equivalent
- Ensure `prepare_environment()`, `download_models()`, `config.update_files()`, and `init_cache()` run exactly once

**Risk**: Low — this is a logging/startup fix, no inference impact.

### 2. Remove Dead UI Features from `webui.py`

**Problem**: Several tabs and controls reference features that have been purged or disabled but still render in the UI, creating confusion.

**Features to remove:**

| Feature | Location in webui.py | Reason |
|---|---|---|
| Enhance tab | L308-313 (tab + input image) | Backend enhance pipeline removed |
| Enhance panel | L335-491 (~155 lines, entire enhance_input_panel) | SAM/GroundingDINO logic purged |
| SAM/GroundingDINO mask options | L253-303 (inpaint mask generation col) | SAM/DINO models not available |
| Describe tab | L305-306 | Already marked "Disabled" |
| FreeU tab | L792-798 | User decision: remove (not useful) |
| GroundingDINO debug checkbox | L747-748 | References purged DINO |

**Action**:
- Remove the Enhance tab definition and the `enhance_input_panel` block
- Remove all `enhance_ctrls`, `enhance_inpaint_mode_ctrls`, `enhance_inpaint_engine_ctrls`, `enhance_inpaint_update_ctrls` variables and their references in `ctrls` (L940-943)
- Remove SAM model dropdown, SAM options accordion, DINO prompt text, DINO debug checkbox from the inpaint mask generation section — keep only `u2net`-family models
- Remove the Describe tab
- Remove the FreeU tab and `freeu_ctrls` references (including from `ctrls` at L930)
- Remove `save_final_enhanced_image_only` checkbox (L709-710) — only relevant to enhance
- Remove `enhance_checkbox` and its panel toggle (L507-508)
- Update `load_data_outputs` (L826-833) to exclude removed controls
- Update `input_image_tab_ids` in `flags.py` (L70) to remove `'enhance_tab'` and `'describe_tab'`

**Also remove from `flags.py`:**
- `enhancement_uov_before`, `enhancement_uov_after`, `enhancement_uov_processing_order` (L13-15)
- `enhancement_uov_prompt_type_original`, `enhancement_uov_prompt_type_last_filled`, `enhancement_uov_prompt_types` (L17-19)
- `describe_type_photo`, `describe_type_anime`, `describe_types` (L96-98)
- `inpaint_mask_sam_model` list (L88) — keep `inpaint_mask_models` but remove `'sam'` entry

**Also remove from `args_manager.py`:**
- `--disable-enhance-output-sorting` argument (L33-34)
- `--enable-auto-describe-image` argument (L36-37)

**Risk**: Medium — many Gradio component cross-references. Must verify `ctrls` list length matches `async_worker.py`'s `AsyncTask` argument parsing. This is the highest-risk item in this work order.

> [!WARNING]
> The `ctrls` list order in `webui.py` must exactly match the argument unpacking in `async_worker.py`'s `AsyncTask.__init__()`. Removing controls from `ctrls` requires a corresponding update to `async_worker.py`.

### 3. Wire Sampler/Scheduler Dropdowns to Backend

**Problem**: `flags.py` defines sampler and scheduler lists as hardcoded dictionaries (L24-62). The backend at `backend/sampling.py` has a larger, authoritative set of samplers. The UI shows the old list, not the backend's.

**Action**:
- In `flags.py`, replace the hardcoded `KSAMPLER`, `SAMPLER_EXTRA`, `SAMPLERS`, `KSAMPLER_NAMES`, `SCHEDULER_NAMES`, `SAMPLER_NAMES` with imports from `backend.sampling`:
  ```python
  from backend.sampling import SAMPLER_NAMES, KSAMPLER_NAMES
  from backend.schedulers import SCHEDULER_NAMES
  ```
- Keep the `CIVITAI_NO_KARRAS` list and `SAMPLERS` CivitAI mapping dict for metadata compatibility, but derive from the backend where possible
- Verify the dropdown in `webui.py` (L659-662) still renders correctly with the expanded list

**Risk**: Low — the backend's sampler names are a superset of the current list. Existing presets/metadata that reference old sampler names will still match.

## Verification
1. **Single console output** — `launch.py` prints startup info exactly once
2. **Fooocus launches** without Gradio errors — no missing component references
3. **UI has no Enhance, Describe, FreeU, or SAM tabs/options**
4. **Sampler dropdown** shows all backend samplers (e.g., `euler_cfg_pp`, `sa_solver`)
5. **Scheduler dropdown** shows all backend schedulers (e.g., `turbo`, `align_your_steps`)
6. **txt2img generation** works with a newly-visible sampler (e.g., `euler_cfg_pp`)
7. **LoRA application** still works through the UI
8. **Preset loading** still works (test with `initial` preset)
9. **`app.py` headless test** passes

## Success Criteria
- Console output shows startup info exactly once
- Enhance tab, Describe tab, FreeU tab, SAM options completely removed from UI
- Sampler/scheduler dropdowns populated from `backend/sampling.py` and `backend/schedulers.py`
- `flags.py` no longer duplicates backend sampler/scheduler lists
- Zero functional regression — all generation tests pass
