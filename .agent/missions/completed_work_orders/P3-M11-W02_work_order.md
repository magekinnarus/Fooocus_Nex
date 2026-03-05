# Work Order: P3-M11-W02 — Bridge File Consolidation
**ID:** P3-M11-W02
**Mission:** P3-M11
**Status:** Completed
**Depends On:** P3-M11-W01

> [!NOTE]
> **W01 Overlap:** During W01 execution, `modules/sample_hijack.py` was already deleted and its schedulers (`turbo`, `align_your_steps`) were ported to `backend/schedulers.py`. The `clip_separate` import in `default_pipeline.py` was also removed. **Scope item #2 below is already complete.** Verify during execution and skip if confirmed.

## Mandatory Reading
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `Fooocus_Nex/modules/sample_hijack.py` (160 lines — understand monkey-patches and `clip_separate`)
- `Fooocus_Nex/modules/lora.py` (153 lines — understand `match_lora`)
- `Fooocus_Nex/modules/gguf/` (4 files — understand GGUF loading chain)
- `Fooocus_Nex/backend/loader.py` (imports from `modules.gguf`)
- `Fooocus_Nex/modules/core.py` (imports from `modules.lora` and `sample_hijack`)
- `Fooocus_Nex/modules/default_pipeline.py` (imports `clip_separate` from `sample_hijack`)

## Objective
Consolidate bridge/adapter files that exist in `modules/` but have functional equivalents in `backend/`. Move shared dependencies to their proper layer and eliminate cross-layer imports that run against the architecture.

## Scope

### 1. Move `modules/gguf/` → `backend/gguf/`

**Problem**: `backend/loader.py` directly imports from `modules.gguf`. This is a cross-layer dependency — the backend should not depend on the modules layer.

**Action**:
- [x] Move `modules/gguf/__init__.py`, `dequant.py`, `loader.py`, `ops.py`, `patcher.py` to `backend/gguf/`
- [x] Update all imports:
  - [x] `backend/loader.py` lines 138–140: `from modules.gguf.*` → `from backend.gguf.*`
  - [x] `modules/nex_loader.py` is already deleted in W01 — no update needed
- [x] Update internal imports within gguf files:
  - [x] `gguf/patcher.py` uses `ldm_patched.modules.utils` and `ldm_patched.modules.model_management` — redirect to `backend.utils` and `backend.resources` where possible
  - [x] `gguf/ops.py` uses `ldm_patched.modules.ops` and `ldm_patched.modules.model_management` — evaluate if these can be redirected; keep if deep dependency

### 2. Consolidate `modules/sample_hijack.py`

**Current state**: This file does two things:
1. Defines `clip_separate()` / `clip_separate_inner()` / `clip_separate_after_preparation()` — used by `default_pipeline.py` and `core.py`
2. Monkey-patches `ldm_patched.modules.samplers.sample` and `calculate_sigmas_scheduler` at module scope (lines 158–159)

**Action**:
- [x] **`clip_separate()` family (lines 19–82)**: Already handled in W01.
- [x] **`sample_hacked()` (lines 87–130)**: Already handled/deleted in W01.
- [x] **`calculate_sigmas_scheduler_hacked()` (lines 133–155)**: Already handled/deleted in W01.

**Risk**: The monkey-patches at lines 158–159 run at import time. Removing them may break paths that still go through `ldm_patched.modules.samplers.sample()`. Test carefully.

### 3. Consolidate `modules/lora.py`

**Current state**: Contains a single function `match_lora()` (153 lines) that matches LoRA weight keys to model keys. Imported only by `modules/core.py`.

**Action**:
- [x] Move `match_lora()` into `backend/lora.py`
- [x] Update `modules/core.py` import: `from modules.lora import match_lora` → `from backend.lora import match_lora`
- [x] Delete `modules/lora.py`

### 4. Evaluate `modules/ops.py`

**Current state**: Small file (502 bytes) providing `use_patched_ops` context manager. Imported by `modules/patch_clip.py` and `extras/ip_adapter.py`.

**Action**:
- [x] Move `use_patched_ops` to `backend/ops.py`
- [x] Update imports in `modules/patch_clip.py` and `extras/ip_adapter.py`
- [x] Delete `modules/ops.py`

## Verification
- [x] 1. **Fooocus launches** without import errors
- [x] 2. **txt2img generation** works through the UI (SD1.5 or SDXL GGUF)
- [x] 3. **LoRA application** works through the UI
- [x] 4. **GGUF model loading** works (UNet + separate CLIP/VAE) — critical test for gguf move
- [x] 5. **`app.py` headless test** passes for both SD1.5 and SDXL
- [x] 6. **No broken imports** — `grep -r "modules.gguf" --include="*.py"` returns zero hits outside comments
- [x] 7. **No broken imports** — `grep -r "modules.sample_hijack" --include="*.py"` returns zero hits

## Success Criteria
- `modules/gguf/` directory no longer exists; `backend/gguf/` has all 4 files
- `modules/lora.py` no longer exists; `match_lora()` lives in `backend/lora.py`
- `modules/sample_hijack.py` either deleted or reduced to only essential monkey-patches
- Zero functional difference — all tests pass identically
