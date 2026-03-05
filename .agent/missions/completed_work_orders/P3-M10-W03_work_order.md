# Work Order: P3-M10-W03 — Cleanup and Integration Testing
**ID:** P3-M10-W03
**Mission:** P3-M10
**Status:** Completed
**Depends On:** P3-M10-W02

## Mandatory Reading
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `Fooocus_Nex/modules/async_worker.py` (lines 1027–1422, handler flow)

## Objective
Strip removed features from `async_worker.py`, clean up residual `ldm_patched` imports across the modules layer, and conduct comprehensive integration testing through the Fooocus UI.

## Scope

### Strip Dead Features from `async_worker.py`

#### Remove Image Enhancement (lines ~1259–1403)
Delete or gut the following:
- `process_enhance()` function (lines 921–985)
- `enhance_upscale()` function (lines 987–1023)
- Enhancement loop in `handler()` (lines 1259–1403)
- `should_enhance` logic (line 1118, 1226–1227)
- Enhancement-related `AsyncTask` fields: `enhance_checkbox`, `enhance_uov_method`, `enhance_uov_processing_order`, `enhance_uov_prompt_type`, `enhance_ctrls`, `should_enhance`, `images_to_enhance_count`, `enhance_stats`, `enhance_input_image`, `save_final_enhanced_image_only` (lines 113–156)
- Enhancement step calculations (lines 1179–1193)

#### Remove Wildcards
- Remove `apply_wildcards()` calls in `process_prompt()` (lines 657–665)
- Remove `read_wildcards_in_order` from `AsyncTask.__init__` (line 42)
- Simplify `process_prompt()` by removing wildcard processing

#### Simplify `handler()` Flow
After removing enhancement, the `handler()` function (lines 1027–1403) should end at image generation/saving — approximately halving its length. The flow becomes:
1. Initialize parameters
2. Process image inputs (vary/upscale/inpaint if applicable)
3. Load models + encode prompts
4. Sample + decode
5. Save + return

### Clean Up Residual Imports

Grep `modules/` for remaining unnecessary `ldm_patched` imports and evaluate:
- `async_worker.py` line 182: `import ldm_patched.modules.model_management` — used for `InterruptProcessingException` (lines 1010, 1246, 1364). Keep this or move exception class.
- `core.py`: remaining `ldm_patched` imports after W01 changes
- `default_pipeline.py` line 7: `import ldm_patched.modules.model_management` — should be replaced by W01

### Bug Report Document
Create `.agent/missions/active/P3-M10_integration_bugs.md` documenting:
- Any issues discovered during integration testing
- Edge cases that need attention in future missions
- Performance discrepancies between backend and `ldm_patched` paths
- Missing functionality needed for ControlNet/Inpainting (future missions)

## Verification

### Functional Tests (Local: GTX 1050, SD1.5 full + SDXL GGUF)
1. **txt2img SD1.5**: Basic prompt, default settings → valid image
2. **txt2img SDXL GGUF**: Basic prompt, default settings → valid image
3. **txt2img with styles**: Apply 2–3 style presets → visible style effects
4. **txt2img with LoRA**: Apply a LoRA → visible style change
5. **Resolution sweep**: 512×512 (SD1.5), 1024×1024 (SDXL) → both produce valid images
6. **Sampler/scheduler variations**: Test at least `euler_ancestral`+`karras` and `dpmpp_2m_sde_gpu`+`karras`
7. **Quality parameters**: Sharpness 0 vs 2, adaptive CFG default vs modified → different outputs

### Regression Checks
8. **VRAM**: No OOM on GTX 1050 with either model type
9. **Inference time**: Within ±10% of pre-M10 baseline
10. **Clean startup**: Fooocus launches without errors or warnings

### Code Quality
10. Document all `ldm_patched` imports remaining in `modules/` with justification (model architecture imports are expected to stay)

## Success Criteria
- Enhancement feature code removed from `async_worker.py`
- Wildcard processing removed
- `handler()` function reduced to ~350 lines (from current ~400 lines of active code)
- All functional tests pass
- Bug report created with integration findings
- Clean Fooocus startup with no new warnings
