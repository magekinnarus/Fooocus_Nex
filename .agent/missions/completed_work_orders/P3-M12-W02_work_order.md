# Work Order: P3-M12-W02 — `webui.py` Component Modularization (Revised)
**ID:** P3-M12-W02
**Mission:** P3-M12
**Status:** Completed
**Depends On:** P3-M12-W01
**Revision:** v2.0 — Phased approach after two failed monolithic attempts

## Lessons Learned from Previous Attempts
1. Gradio components have invisible parent-child scoping — moving them out of `with gr.Blocks()` context breaks the render tree
2. The `ctrls` list (L661-686) is positionally parsed by `AsyncTask.__init__` — any reordering causes silent data corruption
3. Event handler closures capture component variables by reference — extracting them into separate modules breaks those captures
4. **Never** do layout extraction + event extraction together

## Strategy: 4 Atomic Phases

Each phase is a self-contained unit that **MUST** be:
- Completed fully before the next phase starts
- Verified by launching the app + generating an image
- Committed to git before proceeding

```
Phase 1 ──► commit + test ──► Phase 2 ──► commit + test ──► Phase 3 ──► commit + test ──► Phase 4
```

---

## Phase 1: Metadata Tab Extraction (Already Done)

`modules/ui_components/metadata_ui.py` already exists from W01. This serves as the proven pattern for all subsequent extractions.

**Pattern established:**
- Component function creates Gradio components inside the caller's context
- Returns a dict of components needed by the caller
- Event handlers stay in `webui.py` for now

**Status:** ✅ Complete

---

## Phase 2: Settings Tab Extraction (Safest)

### Why first: The Settings tab (L305-400) is the most isolated. It has:
- No cross-references to image input components
- Simple event handlers (preset change, seed random, style sort)
- Self-contained `load_data_outputs` list that can be constructed from returned components

### Action

#### [NEW] `modules/ui_components/settings_panel.py`

Create a function that builds the Settings tab contents:

```python
def build_settings_tab():
    """Builds Settings tab: preset, performance, aspect ratios, image number, output format, 
    negative prompt, seed, and history link.
    
    Returns:
        dict of Gradio components keyed by name
    """
```

**Extract these components** (L305-367):
- `preset_selection` (conditional on `args_manager`)
- `performance_selection`
- `aspect_ratios_selection` + accordion
- `image_number`
- `output_format`
- `negative_prompt`
- `seed_random`, `image_seed`
- `history_link`

**DO NOT extract:** The `random_checked`, `refresh_seed`, `update_history_link` handlers — these stay inline in `webui.py` for now.

#### [MODIFY] `webui.py`

Replace the Settings tab layout block (L305-367) with:
```python
from modules.ui_components.settings_panel import build_settings_tab
# ... inside gr.Blocks() ...
with gr.Tab(label='Settings'):
    settings = build_settings_tab()
    preset_selection = settings['preset_selection']
    performance_selection = settings['performance_selection']
    # ... destructure all needed components ...
```

**CRITICAL:** After destructuring, all existing event bindings and `ctrls`/`load_data_outputs` list references continue to use the same variable names. **Nothing else changes.**

### Verification Gate
- [x] App launches without errors
- [x] Settings tab renders correctly
- [x] Preset switching works
- [x] Performance mode switching works
- [x] Generate image works (full pipeline still intact)
- [x] **Commit**

---

## Phase 3: Styles + Models Tabs Extraction

### Why second: These tabs (L368-425) are moderately isolated. Models tab has the `refresh_files` button flow and LoRA dynamic controls.

#### [NEW] `modules/ui_components/styles_panel.py`

```python
def build_styles_tab():
    """Builds Styles tab: search bar, style checkboxes, receiver.
    Returns: dict of Gradio components
    """
```

**Extract** (L368-399):
- `style_search_bar`, `style_selections`, `gradio_receiver_style_selections`

#### [NEW] `modules/ui_components/models_panel.py`

```python
def build_models_tab():
    """Builds Models tab: base model, VAE, CLIP, LoRA rows, refresh button.
    Returns: dict of Gradio components (including lora_ctrls list)
    """
```

**Extract** (L401-425):
- `base_model`, `vae_model`, `clip_model`
- LoRA row loop (must preserve exact `lora_ctrls` list ordering)
- `refresh_files` button

#### [MODIFY] `webui.py`

Same pattern as Phase 2: call builder functions, destructure, keep all event bindings.

### Verification Gate
- [x] All 3 tabs render correctly (Settings, Styles, Models)
- [x] Style search works
- [x] Model switching works
- [x] LoRA enable/disable and weight adjustment works
- [x] Refresh Files button works
- [x] Generate image works
- [x] **Commit**

---

## Phase 4: Advanced + Control + Inpaint Tabs Extraction

### Why last: These tabs (L426-563) contain the most cross-references and are directly wired into the `ctrls` list and `inpaint_ctrls` sublist.

#### [NEW] `modules/ui_components/advanced_panel.py`

```python
def build_advanced_tab():
    """Builds Advanced tab: debug tools, control, inpaint sub-tabs.
    Returns: dict of Gradio components (including inpaint_ctrls list)
    """
```

**Extract** (L426-563):
- Debug Tools: guidance_scale, sharpness, ADM scalers, CLIP skip, sampler, scheduler, overwrites, preview/seed/metadata toggles
- Control sub-tab: CN preprocessor settings, canny thresholds, mixing checkboxes
- Inpaint sub-tab: inpaint engine, strength, respective field, mask settings

**CRITICAL: `inpaint_ctrls` list** (L551-553) must be returned exactly as constructed. The `ctrls` list at L679 appends `inpaint_ctrls` — this ordering must be preserved.

#### [MODIFY] `webui.py`

Same pattern. Destructure. Keep `ctrls` list assembly and all event bindings in `webui.py`.

### Verification Gate
- [x] All tabs render correctly
- [x] All debug sliders functional
- [x] Inpaint tab functional (settings only — actual inpainting is W03)
- [x] ControlNet settings functional
- [x] Generate image with modified advanced settings works
- [x] `ctrls` ordering still matches `AsyncTask.__init__` parsing
- [x] **Commit**

---

## What This Work Order Does NOT Include

The original W02 included `ui_controller.py` (centralized event binding). **This is deferred** to a later work order because:
1. Event handlers reference many components by closure — extracting them requires a different technique (component registry or explicit passing)
2. Layout extraction alone achieves the primary goal: making `webui.py` readable and each tab independently editable
3. Event controller extraction is lower-value, higher-risk

## Success Criteria
- `modules/ui_components/` has 4+ component modules (metadata already done + 3 new)
- `webui.py` is reduced to ~350-400 lines (layout assembly + event binding)
- All UI features work identically to pre-W02 baseline
- Each phase was committed and tested independently
- `ctrls` list and `AsyncTask.__init__` positional parsing remain in sync
