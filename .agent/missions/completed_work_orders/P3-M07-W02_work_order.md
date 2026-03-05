# Work Order: P3-M07-W02 — Build app.py & config.json

**Mission:** P3-M07
**Work Order ID:** P3-M07-W02
**Status:** Ready
**Assignee:** Role 2
**Prerequisites:** P3-M07-W01 (Backend modules)

## Objective
Develop the `app.py` inference runner and its configuration file `config.json`.
Implement the main logic to read config, load models (using W01 modules), run inference, and save the result.
Ensure support for both SD 1.5 (local safetensors) and SDXL (local GGUF).

## Context
This is the core deliverable of the mission. `app.py` must be a standalone script that doesn't rely on Granido or complex UI frameworks. It proves that our backend extractions works end-to-end.

## Tasks

### 1. Configuration (`config.json`)
- [ ] Create `config.json` with the schema defined in the mission brief.
- [ ] Include default settings for SDXL GGUF (e.g., specific GGUF path, dual CLIP).
- [ ] Document the schema fields in `README.md` (or a specific `docs/config.md`).

### 2. Inference Runner (`app.py`)
- [ ] **Scaffolding**:
    - Setup `argparse` (optional, for config path) or just read `config.json`.
    - Setup logging.
- [ ] **Model Loading**:
    - Implement logic to switch between `load_sd15_checkpoint` and `load_sdxl_gguf` based on config.
    - Use `backend/loader.py` and `backend/resources.py`.
    - Ensure VAE loading logic (embedded vs external) works.
- [ ] **Pipeline Execution**:
    - **Step 1: Tokenize/Encode**: Call `backend/conditioning.py` (needs `encode_text_sd15` from W01).
    - **Step 2: Sample**: Call `backend/sampling.py` (common logic).
    - **Step 3: Decode**: Call `backend/decode.py` (common logic).
    - **Step 4: Save**: Save the resulting image to `output/` with metadata (if possible).

### 3. Verification
- [ ] **Local Test (SD 1.5)**:
    - Update `config.json` to point to a local SD 1.5 safetensors file.
    - Run `python app.py`.
    - Verify output image.
- [ ] **Local Test (SDXL GGUF)**:
    - Update `config.json` to point to a local SDXL GGUF file.
    - Run `python app.py`.
    - Verify output image.
- [ ] **Resource Check**:
    - Monitor RAM usage during execution to ensure we aren't loading double weights.

## Deliverables
- `config.json`
- `app.py`
- `output/nex-00001.png` (proof of run)
