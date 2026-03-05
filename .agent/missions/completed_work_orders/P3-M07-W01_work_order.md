# Work Order: P3-M07-W01 — Process-Flow Extraction

**Mission:** P3-M07
**Work Order ID:** P3-M07-W01
**Status:** Ready
**Assignee:** Role 2

## Objective
Trace the execution flow for SD 1.5 and SDXL inference, identifying exactly which parts of `model_management.py`, `sd.py`, and `model_patcher.py` are used. Extract these into lean backend modules.
Create `backend/defs/sd15.py` and update `backend/loader.py` to support SD 1.5 loading.

## Context
We need to remove the dependency on the heavy `ldm_patched` modules by extracting only the necessary logic. SD 1.5 allows for a full local test loop (load -> sample -> decode) on limited hardware, while SDXL will rely on GGUF or Colab.

## Tasks

### 1. Analysis & Tracing
- [ ] Trace `load_checkpoint` flow for SD 1.5.
- [ ] Trace `load_checkpoint` flow for SDXL.
- [ ] Identify the minimum code needed for `model_management` (device switching, memory calc).
- [ ] Identify the minimum code needed for `model_patcher` (calculate weight, apply patch - if simple).

### 2. Implementation — Backend Modules
- [ ] **Create `backend/defs/sd15.py`**:
    - Define SD 1.5 UNet configuration.
    - Define SD 1.5 CLIP configuration.
    - Define prefix constants (e.g., `cond_stage_model.transformer.text_model`).
- [ ] **Update `backend/loader.py`**:
    - Add `load_sd15_checkpoint(path)`:
        - Load safetensors.
        - Convert keys using `sd15.py` mappings.
        - Create `ModelPatcher` (or equivalent lean wrapper).
        - Load CLIP and VAE.
- [ ] **Create `backend/memory.py`** (if needed) or update `resources.py`:
    - Implement the "smart" memory management logic from `model_management.py` but simplified.
    - Ensure it handles the 6.7GB duplication issue (avoid loading full state dict twice if possible).

### 3. Verification
- [ ] Create a test script `tests/test_w01_loading.py`:
    - Load an SD 1.5 checkpoint using the new `loader.py`.
    - Print the keys/shapes of the loaded model.
    - Assert that no `ldm_patched` modules are imported in `sys.modules`.

## Deliverables
- `backend/defs/sd15.py`
- Updated `backend/loader.py`
- `tests/test_w01_loading.py`
- `backend/memory.py` (optional, can be integrated into `resources.py` if small)
