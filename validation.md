# Runtime Validation

This file defines the canonical automated validation contract for the post-W11
runtime model.

Historical mission work reports still describe what was true when each work
order closed. Use the commands in this file for current validation and closure
evidence.

## Environment

Run validation through the project virtual environment:

```powershell
.\venv\Scripts\python.exe tools\check_validation_env.py
```

Validated local W12 baseline:

- `transformers==4.44.2`
- `huggingface-hub==0.36.2`
- `tokenizers==0.19.1`
- `accelerate==1.13.0`

Known mismatch:

- Plain system Python on this machine currently carries an incompatible
  `huggingface-hub==1.4.1`, which breaks `transformers` import and therefore
  must not be used for mission validation.

## Optional Launch Overrides

For constrained-hardware reproduction on a roomier Colab Pro session, the UI
launch path now supports:

- `--memory-environment-profile`
- `--hardware-total-ram-mb`
- `--hardware-total-vram-mb`
- `--sdxl-resident-clean-source-device`

Example constrained Colab streaming simulation:

- keep `--colab`
- keep the roomy Colab Pro RAM budget
- add `--hardware-total-vram-mb 12288`
- if you want the simulation fully explicit, also add `--hardware-total-ram-mb 57344`
- leave `--memory-environment-profile` on `auto` or pin it to `colab_pro`
- do not use `colab_free` for fp8 Flux Fill streaming validation; that profile exists because free-tier RAM is too small for the target streaming test

Example forcing the resident SDXL clean UNet snapshot onto CPU on a roomy
Colab Pro session:

- keep `--memory-environment-profile colab_pro` or `auto`
- add `--sdxl-resident-clean-source-device cpu`
- this preserves the Colab Pro session while forcing the CPU-shadow resident lane for SDXL LoRA lifecycle validation

## Search And Compile

Ownership/runtime audit search:

```powershell
rg -n "sdxl_runtime_owner|process_diffusion|runtime_family|execution_mode|gguf_sdxl|flux_fill" backend modules tools tracked_tests tests
```

Compile sanity on the authoritative runtime surfaces:

```powershell
.\venv\Scripts\python.exe -m py_compile backend\memory_governor.py backend\resources.py backend\sdxl_runtime_policy.py backend\sdxl_streaming_runtime.py backend\sdxl_unified_runtime.py backend\staging_manager.py modules\async_worker.py modules\objr_engine.py modules\pipeline\inference.py modules\pipeline\routes.py modules\pipeline\tiled_refinement.py modules\runtime_surface_state.py modules\runtime_surface_api.py modules\ui_logic.py webui.py tools\check_validation_env.py tracked_tests\test_memory_residency.py tracked_tests\test_pipeline_routes.py tracked_tests\test_pipeline_stage_runtime.py tests\test_runtime_surface_api.py
```

## Regression Matrix

### 1. Unified SDXL Runtime And Image-Input Handoff

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_sdxl_unified_runtime.py tests\test_unified_runtime_handoff.py tests\test_gguf_runtime_handoff.py tests\test_async_worker_process_transition.py tests\test_default_pipeline_process_reset.py tests\test_super_upscale_residency.py -q
```

Covers:

- standard unified SDXL route
- unified SDXL image-input route
- authoritative runtime handoff
- GGUF compatibility/quarantine expectations
- process-transition and cleanup behavior
- tiled-refinement runtime dispatch and interrupt semantics

### 2. Authoritative Pipeline Consolidation And GGUF Seam

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_gguf_dispatch_seam.py -q
```

Covers:

- retained GGUF dispatch seam classification
- explicit compatibility-lane expectations

### 3. Runtime-Centered Memory / Hardware / Flux Fill / Runtime-Surface Sanity

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_memory_governor.py tests\test_w11_policy_simplification.py tests\test_async_worker_process_transition.py tests\test_flux_fill_integration.py tests\test_runtime_surface_api.py -q
```

Covers:

- runtime-native memory policy
- hardware-tier mapping
- Flux Fill route/session sanity
- runtime-surface preview and completed-image API ownership
- transition isolation behavior

### 4. Tracked Route / Stage Smoke

```powershell
.\venv\Scripts\python.exe -m pytest tracked_tests\test_pipeline_routes.py tracked_tests\test_pipeline_stage_runtime.py tracked_tests\test_memory_residency.py -q
```

Covers:

- route-family selection
- stage runner execution contract
- memory residency dispatch smoke

### 5. Full Suite

```powershell
.\venv\Scripts\python.exe -m pytest tests\ --ignore=tests\test_bgr.py --ignore=tests\test_objr.py -q
```

Notes:

- `tests/test_bgr.py` and `tests/test_objr.py` remain outside the closure bundle
  because of the pre-existing `args_manager` argparse incompatibility.
- Treat this command as the broad regression sweep after the targeted matrix is
  already green.

## Optional Benchmarks

These are evidence tools, not closure gates:

```powershell
.\venv\Scripts\python.exe tools\bench_sdxl_pinned_residency_matrix.py
.\venv\Scripts\python.exe tools\bench_sdxl_resident_lora_lifecycle.py --placement both
.\venv\Scripts\python.exe tools\bench_headless_gguf_txt2img.py
.\venv\Scripts\python.exe tools\bench_flux_fill_fp8_streaming.py
```
