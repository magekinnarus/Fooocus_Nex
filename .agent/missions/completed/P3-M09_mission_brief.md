# Mission Brief: P3-M09 — Patching, Quality Features & Backend Refinement
**ID:** P3-M09
**Phase:** 3
**Date Issued:** 2026-02-22
**Status:** Ready
**Depends On:** P3-M07 (app.py runner), P3-M08 (CLIP extraction)
**Work List:** `.agent/missions/active/P3-M09_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/missions/completed/P3-M07_mission_report.md`
- `.agent/missions/completed/P3-M08_mission_report.md`

## Objective

Three goals in one mission, ordered to minimize risk:

1. **Decompose `sampling.py`** — split the 626-line file into focused modules before adding features
2. **Absorb Fooocus quality features** — adopt anisotropic sharpness, adaptive CFG, ADM scaling, and timed ADM natively into the backend
3. **Extract LoRA patching** — build `backend/patching.py` with `NexModelPatcher`, add LoRA support to `app.py`

**Why these belong together:** All three touch the sampling/conditioning pipeline. Decomposing first prevents bloat. Quality features improve image output immediately. LoRA patching enables M10 (Fooocus integration). Doing them in this order means each work order builds on a clean foundation.

## Director Priority
> The backend will be the backbone of our ultimate plan to build our own UI.
> Getting the backend to be as efficient and functional is the main priority.

## Scope

### Part A: sampling.py Decomposition (W01)

Current `sampling.py` (626 lines) mixes three concerns:

| Concern | Lines | Functions | Target File |
|---|---|---|---|
| **Condition Processing** | ~300 | `get_area_and_mult`, `cond_equal_size`, `can_concat_cond`, `cond_cat`, `calc_cond_batch`, `resolve_areas_and_cond_masks_multidim`, `calculate_start_end_timesteps`, `encode_model_conds`, `process_conds` | **`backend/cond_utils.py` [NEW]** |
| **Sampler Infrastructure** | ~120 | `Sampler`, `KSAMPLER`, `KSampler`, `KSamplerX0Inpaint`, `ksampler()`, `sample_sdxl()`, registries | stays in **`backend/sampling.py`** |
| **CFG / Guidance** | ~200 | `cfg_function`, `sampling_function`, `CFGGuider` | stays in **`backend/sampling.py`** |

After split: `sampling.py` → ~320 lines, `cond_utils.py` → ~300 lines.

### Part B: Fooocus Quality Features (W02)

Four features from Fooocus's `modules/patch.py` worth adopting natively:

| Feature | What It Does | Lines | Target |
|---|---|---|---|
| **Anisotropic Sharpness** | Bilateral blur on positive epsilon guided by prediction. Preserves edges, smooths noise. Controlled by `sharpness` param. | ~200 | **`backend/anisotropic.py` [NEW]** (copy from `modules/anisotropic.py`, already standalone) |
| **Adaptive CFG** | Blends real CFG with lower "mimicked" CFG by diffusion progress. Reduces oversaturation at high CFG. | ~12 | **`backend/sampling.py`** (add to `cfg_function`) |
| **ADM Scaling** | Scales resolution embeddings asymmetrically: positive × 1.5, negative × 0.8. Encourages detail in positive, softens negative. | ~20 | **`backend/conditioning.py`** (add params to `get_adm_embeddings_sdxl`) |
| **Timed ADM** | Swaps from "emphasized" to "consistent" ADM embeddings partway through diffusion. Prevents resolution distortion. | ~10 | **`backend/conditioning.py`** (add as processing step) |

All four are pure PyTorch — no external dependencies, no Fooocus-specific coupling.

New configurable parameters for `app.py` / JSON config:
- `sharpness` (float, default 2.0)
- `adaptive_cfg` (float, default 7.0)
- `adm_scale_positive` (float, default 1.5)
- `adm_scale_negative` (float, default 0.8)
- `adm_scaler_end` (float, default 0.3)

### Part C: LoRA Patching (W03–W04)

Extract LoRA application from `ldm_patched/modules/model_patcher.py` (1398 lines):

| Component | Extract? | Target |
|---|---|---|
| `calculate_weight` (~165 lines) | **Yes** | **`backend/patching.py` [NEW]** |
| `NexModelPatcher` (device mgmt + patch/unpatch cycle) | **Yes** | **`backend/patching.py`** |
| Clone management | **Yes** | **`backend/patching.py`** |
| Model options (`set_model_sampler_cfg_function`, etc.) | **Yes** | **`backend/patching.py`** |
| Hook/callback system | **Defer** | M10 (ControlNet) |
| `weight_adapter/` types | **Reuse** | Import from existing extracted modules |

### Out of Scope
- Hook/callback system (M10)
- ControlNet / Inpainting patches (M10)
- `modules/patch.py` replacement (M10)
- Textual inversion embeddings (future)

## Constraints
- `backend/patching.py` must have **zero** imports from `ldm_patched/modules/model_patcher.py`
- May import from `ldm_patched/modules/weight_adapter/` (already clean)
- `sampling.py` decomposition must not change any function signatures — only file locations
- Fooocus quality features must be configurable and default-off (to preserve A/B comparison capability)
- Must work with both standard PyTorch models and GGUFModelPatcher

## Deliverables
- [ ] **`backend/cond_utils.py`** — condition processing extracted from `sampling.py`
- [ ] **`backend/anisotropic.py`** — adaptive bilateral blur filter
- [ ] **`backend/patching.py`** — LoRA patching + NexModelPatcher
- [ ] **Updated `backend/sampling.py`** — slimmer, with sharpness + adaptive CFG integrated
- [ ] **Updated `backend/conditioning.py`** — ADM scaling + timed ADM
- [ ] **Updated `backend/loader.py`** — uses `NexModelPatcher`
- [ ] **Updated `app.py`** — LoRA support + quality feature params in JSON config
- [ ] **Verification** — images with/without quality features, with/without LoRA

## Success Criteria
1. `sampling.py` is under 400 lines after decomposition
2. Fooocus quality features produce visible improvement vs vanilla (A/B comparison)
3. LoRA application produces visible style change
4. Patch/unpatch cycle works — model returns to base state after generation
5. `backend/patching.py` has zero `ldm_patched/modules/model_patcher.py` imports
6. No regression in existing `app.py` SD1.5 + SDXL inference
7. All new features are configurable via JSON config

## Work Orders
Registered in `P3-M09_work_list.md`:
- `P3-M09-W01` — Decompose `sampling.py` → extract `cond_utils.py`
- `P3-M09-W02` — Absorb Fooocus quality features (anisotropic, adaptive CFG, ADM scaling)
- `P3-M09-W03` — Extract `NexModelPatcher` + `calculate_weight` into `backend/patching.py`
- `P3-M09-W04` — Add LoRA support to `app.py`, verify with real LoRA files

## Notes
- W01 is a pure refactor — no behavior change, just file reorganization. Should be quick and verifiable.
- W02 features are all from `modules/patch.py` and `modules/anisotropic.py`. They're Fooocus's "secret sauce" for image quality and are pure PyTorch with no external dependencies.
- `anisotropic.py` already has zero Fooocus-specific imports — it can be copied to `backend/` with no changes.
- The `sharpness` system requires knowing `global_diffusion_progress` (0.0→1.0 through the diffusion). Our sampling pipeline needs to track this — currently it doesn't. W02 will add this as a parameter to the sampling callback system.
- GGUF models use `GGUFModelPatcher` which extends `ModelPatcher`. `NexModelPatcher` must either support GGUF directly or provide an extension point.
- The Director observed the backend's VRAM/RAM memory signature differs from ComfyUI. If this manifests during LoRA testing, document it.
