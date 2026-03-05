# Work Order: P4-M01-W03 — Backend Structure Migration
**ID:** P4-M01-W03
**Mission:** P4-M01
**Status:** Ready
**Depends On:** P4-M01-W02 (completed)

## Mandatory Reading
- `.agent/missions/active/P4-M01_mission_brief.md` (updated — architecture pivot section)
- `.agent/missions/active/P4-M01-W02_work_report.md` (W02 pivot rationale)
- `backend_server/api_server.py` (current working server)
- `headless_app/app.py` (reference headless script)

## Objective
Reorganize `backend_server/` from a renamed Fooocus clone into a clean API server structure.
After this work order, `backend_server/` contains only the engine (`engine/`), pipeline
orchestrators (`pipelines/`), `ldm_patched/`, and API server code. All legacy Fooocus UI code
is removed. `python api_server.py` still generates txt2img images.

## Context

`backend_server/` is currently the old `Fooocus_Nex` folder renamed. It contains:
- **~15% useful code**: `backend/`, `api_server.py`, `ldm_patched/`
- **~85% dead weight**: `modules/` (Gradio-coupled), `webui.py`, `css/`, Docker files, etc.

W02 proved the API builds directly on `backend/` functions. The `modules/` layer is not needed
for the API server. However, some `modules/` files contain algorithms (InpaintPipeline, quality
patches) that will be referenced when building future pipeline orchestrators in P4-M03.

## Scope

### 1. Rename `backend/` → `engine/`

Move `backend_server/backend/` to `backend_server/engine/`:

```
backend_server/engine/
├── __init__.py
├── loader.py
├── sampling.py
├── conditioning.py
├── decode.py
├── resources.py
├── lora.py
├── patching.py
├── k_diffusion.py
├── schedulers.py
├── precision.py
├── utils.py
├── cond_utils.py
├── float_ops.py
├── anisotropic.py
├── weight_ops.py
├── ops.py
├── clip.py
├── encode.py
├── gguf/
│   ├── loader.py
│   ├── ops.py
│   └── patcher.py
└── defs/
    ├── sdxl.py
    └── sd15.py
```

**Update all imports** in `api_server.py` and internal `engine/` cross-references from
`backend.X` → `engine.X`.

### 2. Create `pipelines/` Layer

Extract the orchestration logic from `api_server.py`'s `run_pipeline_api()` into a dedicated
pipeline module:

```python
# backend_server/pipelines/__init__.py
# backend_server/pipelines/txt2img.py

def run_txt2img(config: dict, progress_callback=None) -> dict:
    """Execute txt2img pipeline. Returns dict with image paths and seed."""
    # Logic extracted from api_server.py run_pipeline_api()
```

This separation means:
- `api_server.py` handles HTTP/WebSocket routing and request validation
- `pipelines/txt2img.py` handles the generation orchestration
- `engine/` handles the low-level inference

### 3. Create API-Specific Config

Create `backend_server/config.py` with:
- Model directory paths (checkpoint, LoRA, VAE, CLIP)
- Default model settings
- Output directory configuration
- Server defaults (port, host, CORS origins)

This replaces the dependency on the old `modules/config.py` (32KB, heavily Gradio-coupled).

### 4. Migrate Styles

Create `backend_server/styles.py`:
- Copy style definitions from `modules/sdxl_styles.py`
- Or reference the shared `sorted_styles.json` at project root

### 5. Remove Legacy Code

Delete everything from `backend_server/` that is not:
- `api_server.py`
- `engine/` (renamed from `backend/`)
- `pipelines/`
- `config.py`
- `styles.py`
- `ldm_patched/`

Specifically remove:
- `modules/` (entire directory)
- `webui.py`, `launch.py`, `shared.py`
- `css/`, `javascript/`, `presets/`
- `Dockerfile`, `docker-compose.yml`, `docker.md`, `.dockerignore`
- `extras/`, `language/`, `wildcards/`
- `build_launcher.py`, `config.txt`, `config_modification_tutorial.txt`
- `experiments_*.py`, `temp_*.py`, `fooocus_colab.ipynb`
- All other Fooocus-specific files

> [!IMPORTANT]
> The old code is **not lost** — it remains at project root level in the original Fooocus_Nex
> structure. It serves as a reference for future migration work (InpaintPipeline, quality
> patches, etc.).

### 6. Update ldm_patched Imports

`ldm_patched/` stays in `backend_server/` but internal references within `engine/` that
currently use `ldm_patched.modules.X` or `ldm_patched.ldm.X` paths should continue to work.
Verify that the `sys.path` setup in `api_server.py` correctly resolves these imports after
the restructure.

## Implementation Steps

### Step 1: Rename backend → engine
- [ ] Move `backend_server/backend/` → `backend_server/engine/`
- [ ] Update all `from backend import X` → `from engine import X` in `api_server.py`
- [ ] Update internal cross-references within `engine/` modules
- [ ] **Verification**: `python api_server.py` starts without import errors

### Step 2: Extract pipelines/txt2img.py
- [ ] Create `backend_server/pipelines/__init__.py`
- [ ] Create `backend_server/pipelines/txt2img.py` by extracting from `run_pipeline_api()`
- [ ] Update `api_server.py` to call `pipelines.txt2img.run_txt2img()`
- [ ] **Verification**: `POST /generate` still produces images

### Step 3: Create config.py and styles.py
- [ ] Create `backend_server/config.py` with model paths and defaults
- [ ] Create `backend_server/styles.py` with style loading
- [ ] Update `api_server.py` to use new config/styles modules
- [ ] **Verification**: `GET /models` and `GET /styles` still return correct data

### Step 4: Remove legacy code
- [ ] Delete all non-essential files and directories (see list in Scope §5)
- [ ] Verify no remaining imports reference deleted files
- [ ] **Verification**: `python api_server.py` starts cleanly with no import warnings
- [ ] **Verification**: `POST /generate` end-to-end test passes

### Step 5: Final validation
- [ ] Run existing test: `tests/test_api_sdxl.py`
- [ ] Verify folder structure matches target layout
- [ ] Document any deviations in work report

## Success Criteria
1. `backend_server/` contains only: `api_server.py`, `engine/`, `pipelines/`, `config.py`, `styles.py`, `ldm_patched/`
2. No `modules/` folder in `backend_server/`
3. `python api_server.py` starts and loads model
4. `POST /generate` produces images identical to pre-migration output
5. `tests/test_api_sdxl.py` passes
6. All imports resolve cleanly — no warnings about missing modules
7. Folder is deployment-ready: could be copied to Colab without shipping Gradio code
