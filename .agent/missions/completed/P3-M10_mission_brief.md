# Mission Brief: P3-M10 — Fooocus Backend Integration
**ID:** P3-M10
**Phase:** 3
**Date Issued:** 2026-02-24
**Status:** Ready
**Depends On:** P3-M09 (completed), P3-M9.5 (completed)
**Work List:** `.agent/missions/active/P3-M10_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `.agent/summaries/05_Local_Environment_Guidelines.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `app.py` (backend calling-sequence reference)

## Objective

Connect the Fooocus `modules/` frontend to the `backend/` engine, replacing `modules/patch.py`'s monkey-patches with direct `backend/` calls and routing `async_worker.py`'s generation pipeline through the native backend. This validates the backend against real-world complexity (multiple models, LoRAs, styles, resolutions) and produces a working txt2img Fooocus UI backed by the native engine.

### Strategic Approach

**Key architectural finding:**
- `async_worker.py` (1426 lines) has **zero** direct `ldm_patched` imports for core inference. It works entirely through `modules/patch.py`, `modules/core.py`, and `modules/default_pipeline.py`.
- `modules/patch.py` (397 lines) is the **sole bridge** containing 7 monkey-patches that override `ldm_patched` internals.
- `modules/default_pipeline.py` (326 lines) wraps model loading, CLIP encoding, and diffusion — all of which have native `backend/` equivalents.
- `modules/core.py` (352 lines) wraps `ldm_patched` model loading and sampling — also has backend equivalents.

**Strategy:** Replace the intermediary layers (`patch.py`, `core.py`, `default_pipeline.py`) to route through `backend/` modules. `async_worker.py` requires minimal changes because it already abstracts through these layers.

### The 7 Monkey-Patches in `modules/patch.py`

| # | Patch | What It Overrides | Backend Equivalent | Work Order |
|---|-------|-------------------|--------------------|------------|
| 1 | `patched_sampling_function` | `ldm_patched.modules.samplers.sampling_function` | `backend/sampling.py` (anisotropic sharpness + adaptive CFG already absorbed) | **W01** |
| 2 | `sdxl_encode_adm_patched` | `ldm_patched.modules.model_base.SDXL.encode_adm` | `backend/conditioning.py` → `get_adm_embeddings_sdxl()` | **W01** |
| 3 | `patched_load_models_gpu` | `ldm_patched.modules.model_management.load_models_gpu` | `backend/resources.py` → `load_models_gpu()` | **W01** |
| 4 | `BrownianTreeNoiseSamplerPatched` | `ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler` | `backend/k_diffusion.py` (already has this) | **W01** |
| 5 | `patched_unet_forward` | `UNetModel.forward` (precision casting + diffusion progress tracking) | **4/5 behaviors already in backend** — only precision casting needs migration | **W02** |
| 6 | `patched_cldm_forward` | `ControlNet.forward` (softness + timed ADM) | Future — ControlNet not in scope | Deferred |
| 7 | `patched_KSamplerX0Inpaint_forward` | Inpainting sampler forward | Future — Inpainting not in scope | Deferred |

### Confirmed Feature Removals (Director Decision)

These features are **removed from scope permanently** and their code should be stripped during refactoring:

| Feature | Code Location | Reason |
|---------|--------------|--------|
| **Image Enhancement** | `async_worker.py` lines 1259–1403, `process_enhance()`, `enhance_upscale()`, all `enhance_*` params | Auto-detection masking unreliable; replaced by intentional inpainting design |
| **Wildcards** | `apply_wildcards()`, `read_wildcards_in_order` | Director decision — not needed |
| **SAM/GroundingDINO** | Already removed from codebase | Not used |

## Scope

### In Scope
- **`modules/patch.py` [MODIFY]** — Replace `ldm_patched` monkey-patches #1–5 with `backend/` calls
- **`modules/default_pipeline.py` [MODIFY]** — Route model loading, CLIP encoding, and diffusion through `backend/`
- **`modules/core.py` [MODIFY]** — Replace `ldm_patched` model loading and sampling
- **`async_worker.py` [MODIFY]** — Strip enhancement/wildcard code, simplify `handler()` flow
- **txt2img validation** — Full txt2img workflow through Fooocus UI with new backend
- **LoRA validation** — LoRA stacking through Fooocus UI
- **`app.py` [MINOR]** — Fix remaining `ldm_patched.modules.utils` import (line 76)

### Out of Scope (Future Missions)
- ControlNet integration (`patched_cldm_forward`)
- Inpainting integration (`patched_KSamplerX0Inpaint_forward`)
- IP-Adapter / CLIP Vision
- img2img workflows
- Phase 3.5 (inpainting feature build)

## Reference Files
- `app.py` — backend calling-sequence reference (load → encode → sample → decode)
- `Fooocus_Nex/modules/patch.py` — current monkey-patch bridge (397 lines)
- `Fooocus_Nex/modules/core.py` — current model wrapper (352 lines)
- `Fooocus_Nex/modules/default_pipeline.py` — current pipeline orchestrator (326 lines)
- `Fooocus_Nex/modules/async_worker.py` — UI worker (1426 lines)
- `Fooocus_Nex/backend/sampling.py` — native sampling engine
- `Fooocus_Nex/backend/conditioning.py` — native CLIP + ADM encoding
- `Fooocus_Nex/backend/resources.py` — native memory/device management
- `Fooocus_Nex/backend/loader.py` — native model loader

## Constraints
- **Incremental approach** — each work order must leave Fooocus in a runnable state
- **Test through UI** — validation is done by generating images through the Fooocus Gradio interface
- **ControlNet/Inpainting patches (#6, #7) remain as-is** — they still monkey-patch `ldm_patched` for now; only the txt2img path is migrated
- All testing on local GTX 1050 with **SD1.5 full checkpoint** and **SDXL GGUF** — both available locally

## Deliverables
- [ ] Modified `modules/patch.py` — monkey-patches #1–5 replaced with `backend/` calls
- [ ] Modified `modules/default_pipeline.py` — routes through `backend/` for loading, encoding, diffusion
- [ ] Modified `modules/core.py` — sampling and model loading via `backend/`
- [ ] Simplified `async_worker.py` — enhancement/wildcard code removed, handler flow cleaned
- [ ] txt2img proof — image generated through Fooocus UI using backend engine
- [ ] LoRA proof — image generated with LoRA through Fooocus UI
- [ ] Fixed `app.py` — remaining `ldm_patched.modules.utils` import removed
- [ ] Bug report — documented issues discovered during integration

## Success Criteria
1. Fooocus generates a txt2img image using the `backend/` engine (not `ldm_patched` for core inference)
2. LoRA application works through the Fooocus UI
3. Multiple resolutions work correctly (at least 512×512 and 1024×1024)
4. Style presets produce expected visual results
5. No VRAM OOM on local GTX 1050 with SD1.5 full checkpoint
6. Backend's quality features (anisotropic sharpness, adaptive CFG, ADM scaling, timed ADM) function correctly through UI
7. Bug list from integration testing is documented for future missions

## Work Orders
Registered in `P3-M10_work_list.md`:
- `P3-M10-W01` — Replace `default_pipeline.py` and `core.py` internals to route through `backend/`
- `P3-M10-W02` — Replace `patch.py` monkey-patches #1–5, integrate UNet forward precision casting
- `P3-M10-W03` — Strip dead features from `async_worker.py`, clean up residual imports, integration testing
- `P3-M10-W04` — Fix integration bugs (consecutive generation hang, memory leak, skip/stop buttons)

## Notes
- `app.py` serves as the **known-good reference** for backend calling sequences. All backend module call signatures have been validated through M07–M09 testing.
- `core.py`'s `StableDiffusionModel.refresh_loras()` still uses `ldm_patched.modules.utils.load_torch_file` — this needs to switch to `backend.utils.load_torch_file`.
- `default_pipeline.py`'s `process_diffusion()` calls `modules.patch.BrownianTreeNoiseSamplerPatched.global_init()` — this should call the backend's `BrownianTreeNoiseSampler` directly.
- The `PatchSettings` per-PID dictionary pattern in `patch.py` needs a migration path. Quality settings (sharpness, adaptive_cfg, etc.) should flow through `model_options` to the backend's sampling function, matching how `app.py` already does it.
- `build_loaded()` in `patch.py` wraps `safetensors.torch.load_file` and `torch.load` with corruption detection. This utility should be preserved or replaced with equivalent error handling.
