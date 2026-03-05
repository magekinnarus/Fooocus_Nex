# Work Order: P3-M12-W01 — `async_worker.py` Decomposition & Metadata Refactor
**ID:** P3-M12-W01
**Mission:** P3-M12
**Status:** Completed
**Depends On:** —

## Mandatory Reading
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/06_Implementation_Observations_and_Insights.md`
- `.agent/temp/refactoring_assessment.md` (Section 1: Thread Orchestration)
- `Fooocus_Nex/modules/async_worker.py` (1128 lines — full file)

## Objective
Decompose `async_worker.py` from a 1128-line God Function into a clean orchestrator pattern with explicit state management and modular pipeline stages. **Expansion**: Refactor `meta_parser.py` into UI and core logic, ensure consistent prompt/negative prompt metadata ordering (original at index 0), and ensure stability for GGUF models with separate CLIP files.

## Scope

### 1. Create `modules/task_state.py` — Formal State Object

**Problem**: `AsyncTask.__init__` (lines 14–138) pops 80+ args from a list positionally. Pipeline functions then mutate `async_task` attributes freely with no contract. Additionally, 64 `gr.State(False)` placeholder slots exist in `webui.py`'s `ctrls` list for removed enhance/describe features, with corresponding `_ = args.pop()` calls in `AsyncTask.__init__` — pure tech debt from M11.

**Action**: Define a `TaskState` dataclass (or class) that:
- Has typed, named fields for all generation parameters (prompt, negative_prompt, width, height, steps, cfg, sampler, scheduler, seed, etc.)
- Has typed fields for runtime state (loras, goals, initial_latent, denoising_strength, positive_cond, negative_cond, etc.)
- Has explicit methods for state transitions (e.g., `apply_lora_settings()`, `set_resolution()`)
- `AsyncTask` can still exist for Gradio compatibility but should delegate to `TaskState` for parameter storage
- **Remove all 64 `gr.State(False)` placeholders** from `webui.py` `ctrls` list and corresponding `_ = args.pop()` calls from `AsyncTask.__init__`

### 2. Create `modules/pipeline/` — Pipeline Stage Modules

Extract the closure functions from `worker()` into standalone functions with explicit parameters:

#### `modules/pipeline/preprocessing.py`
- `process_prompt()` (currently lines 615–716) — prompt enhancement, style mixing, CLIP encoding
- `apply_overrides()` (lines 606–613)
- `apply_freeu()` (lines 718–726)
- `patch_samplers()` / `patch_discrete()` / `patch_edm()` (lines 728–747)
- `set_hyper_sd_defaults()` / `set_lightning_defaults()` / `set_lcm_defaults()` (lines 749–799)

#### `modules/pipeline/image_input.py`
- `apply_image_input()` (lines 801–879) — ControlNet, IP-Adapter, inpainting setup
- `apply_vary()` (lines 430–458)
- `apply_inpaint()` (lines 460–522)
- `apply_outpaint()` (lines 524–550)
- `apply_upscale()` (lines 552–604)
- `prepare_upscale()` (lines 881–898)

#### `modules/pipeline/inference.py`
- `process_task()` (lines 265–323) — core sampling invocation
- `callback()` (lines 1064–1071) — sampling progress callback

#### `modules/pipeline/output.py`
- `yield_result()` (lines 201–219) — image display to UI
- `build_image_wall()` (lines 221–263) — multi-image composition
- `save_and_log()` (lines 326–377) — image saving and metadata logging

#### `modules/pipeline/__init__.py`
- Re-export stage functions for clean imports

### 3. Metadata Modularization & Prompt Consistency (Expansion)

#### `modules/meta_parser.py`
- Extract Gradio UI logic into `modules/ui_components/metadata_ui.py`.
- Remove `A1111MetadataParser` and its dependencies.
- Fix `FooocusMetadataParser.to_json` to correctly restore `clip_model` filenames from stems during import.

#### `modules/pipeline/preprocessing.py`
- Ensure the original user prompt is consistently placed at index 0 in both `full_prompt` and `full_negative_prompt` metadata.
- Prevent redundant LoRA tags from being appended to the `prompt` metadata field.

#### `modules/core.py`
- Add null-safety in `Model.refresh_loras` to prevent `AttributeError` when loading GGUF models without an internal CLIP.

### 4. Simplify `async_worker.py` — Dumb Orchestrator

After extraction, `async_worker.py` should contain only:
- `AsyncTask` class (or import from `task_state.py`)
- `async_tasks` queue
- `worker()` function — minimal loop that:
  1. Dequeues tasks
  2. Calls `handler()` 
  3. Catches exceptions and yields error state
- `handler(task)` — orchestrator that:
  1. Constructs `TaskState` from `AsyncTask`
  2. Calls pipeline stages in sequence with explicit state
  3. Yields progress updates to Gradio
  4. Handles Stop/Skip interrupts
- `progressbar()` utility

**Target**: < 300 lines total.

### 4. Extraction Rules (Non-Negotiable)

Every extracted function MUST:
1. **Have explicit parameters** — no `nonlocal` captures of `async_task` or any other closure variable
2. **Return values explicitly** — no mutation of external state via side effects
3. **Be independently testable** — could be called with mock data without starting the worker thread
4. **Import cleanly** — no circular dependencies with `async_worker.py`

Exception: `progressbar()` and `yield_result()` may take a `task` parameter to update Gradio state, as this is their explicit purpose.

## Verification
1. **Fooocus launches** without import errors
2. **txt2img generation** works through the UI (SD1.5 and SDXL GGUF)
3. **LoRA application** works through the UI
4. **Skip/Stop buttons** halt generation correctly
5. **Consecutive generations** work without worker thread crashes
6. **Multi-image generation** (image_number > 1) works correctly
7. **Live previews** display during sampling
8. **No duplicate images** in gallery
9. **`async_worker.py` line count** is under 300 lines
10. **Prompt Metadata Order**: `full_prompt` and `full_negative_prompt` both start with the original prompt at index 0.
11. **GGUF LoRA**: LoRA loading on GGUF models with separate CLIP works without crashes.

## Success Criteria
- `modules/pipeline/` directory exists with at least 4 stage modules
- `modules/task_state.py` exists with typed `TaskState` class
- `async_worker.py` is under 300 lines
- Zero closure-captured state mutations — all pipeline functions have explicit parameters
- All UI features work identically to post-M11 baseline
