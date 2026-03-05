# Work Order: P3-M07-W03 — Colab Validation

**Mission:** P3-M07
**Work Order ID:** P3-M07-W03
**Status:** Pending
**Assignee:** Role 2
**Prerequisites:** P3-M07-W02 (Building app.py)

## Objective
Validate `app.py` in a cloud environment (Google Colab) using full-precision SDXL checkpoints, which cannot run locally.

## Context
Local tests cover SD 1.5 safetensors and SDXL GGUF. We need to ensure the standard SDXL pipeline works on high-VRAM GPUs (T4: 16GB, L4: 24GB). This confirms that our lean `app.py` correctly handles full model loading without OOM issues.

## Tasks

### 1. Colab Notebook Setup
- [ ] Create `P3-M07_Validation.ipynb` or updating existing notebook.
- [ ] **Dependencies**: Install minimal requirements (skip `xformers` if not critical, verify `torch` version).
- [ ] **Clone Repo**: Pull the latest branch with `app.py`.
- [ ] **Model Download**: Script to download SDXL 1.0 Base (6.94GB) or Juggernaut XL.

### 2. Execution
- [ ] **Run 1**: T4 (16GB VRAM) test.
    - Set `config.json` to use full SDXL checkpoint.
    - Monitor VRAM usage (should peak around 12-14GB during load, <8GB inference with sequential offload).
- [ ] **Run 2**: L4 (24GB VRAM) test (optional if T4 succeeds comfortably).
- [ ] **Output Verification**: Check that generated images are coherent and match expected quality.
- [ ] **Performance Check**: Record s/it on T4. Compare to known baselines (~2-3 s/it on T4 standard).

## Deliverables
- `P3-M07_Validation.ipynb` (or link to it)
- Screenshot/Log of successful run on T4
- Generated images from Colab session
