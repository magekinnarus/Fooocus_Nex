# Project Roadmap

**Last Updated:** 2026-03-04 (Phase 4 Architecture Pivot)

## Overview

```
Long-Term --> 3D Viewport Scene Compositor with decoupled compute
Phase 6   --> Fundable Demo: depth-compositing with basic 3D viewport
Phase 5   --> Multi-pass engine, Assembler algorithm, depth compositing
Current   --> Phase 4: API Layer + Custom React Frontend (Gradio elimination)
Done      --> Phase 3: Engine modules + Inpainting architecture pivot (complete)
```

---

## Completed Work (Fooocus_Nex Phases 1??)

These phases built the knowledge base and proved the decomposition pattern.

| Phase | Achievement |
|-------|-------------|
| **1: ldm_patched Foundation** | Upgraded ComfyUI engine modules, fixed VRAM management, ported modern `model_patcher.py`, `weight_adapter/`, `ops.py`. Removed GPT-2 bloat. |
| **1.5: Targeted Refactoring** | Absorbed `calculate_weight` into `model_patcher.py`, created `nex_loader.py` component loader, formalized patch registry, startup optimization. |
| **2: GGUF & UI Refactor** | Decomposed ComfyUI-GGUF into `modules/gguf/`, built component-wise loading (UNet + CLIP + VAE), rebuilt model selection UI. |

**Key deliverable:** GGUF integration proved the decomposition pattern ??study ComfyUI ??extract algorithm ??implement cleanly. The completion process (not prototyping) revealed the architectural insights that drove the pivot.

**Reference:** Mission reports in `.agent/missions/completed/`

---

## Phase 3: Nex Engine — Core Pipeline Decomposition

> Target: Clean SDXL step modules that can generate an image end-to-end via script

### Completed — Module Extraction (structure-first approach)

| # | Task | Status | Details |
|---|------|--------|---------|
| 3.0 | SDXL pipeline trace (M01) | **Complete** | Mapped ComfyUI's SDXL lifecycle. Produced `P3-M01_reference_trace.md`. Identified ~90% dispatch overhead. |
| 3.2 | Loader refactor (M02) | **Complete** | **Delivered:** `backend/loader.py`, `backend/defs/sdxl.py`, Native GGUF. <br>**Report:** `.agent/missions/completed/P3-M02_mission_report.md`. |
| 3.3 | Backend Refinement (M03) | **Complete** | `backend/resources.py` (Memory & Devices), `backend/conditioning.py` (Text & ADM). Clean Extraction + Standalone Verification. |
| 3.4 | Sampling Engine (M04) | **Complete** | `backend/sampling.py`, `backend/schedulers.py`, `backend/k_diffusion.py`. 36 samplers (incl. CFG++), 9 schedulers (incl. `beta`). Zero `ldm_patched` imports. |
| 3.5 | VAE Decode (M05) | **Complete** | `backend/decode.py` — latent→image decode with tiled fallback. |
| 3.6 | End-to-end validation (M06) | **Failed** | Exposed critical loader memory flaw (extract-then-load duplicates 6.7 GB checkpoint) and ~67% performance gap vs ComfyUI (20 vs 12 sec/it). Prompted methodology pivot. |

### Remaining — Process-Flow Inference Runner (revised approach)

> **Methodology pivot (M06 → consolidated M07):** M06 revealed that extracting code *structure* without understanding runtime *behavior* leads to hidden memory and performance issues. The previous M07 (analysis-only), M08 (benchmark patches), and M09 (loader rebuild) have been consolidated into a single, iterative mission: **build `app.py`** — a JSON-configured inference runner that forces the same understanding while producing testable output.
>
> **Architecture expansion:** SD 1.5 added alongside SDXL for local full-checkpoint testing without Colab/GGUF dependency.

| # | Task | Status | Details |
|---|------|--------|---------|
| 3.7 | Process-flow inference runner (M07) | **Complete** | `app.py` + JSON configs. SD 1.5 + SDXL. Local (GTX 1050) + Colab (T4/L4). Performance 10.4% gap vs ComfyUI (within 20% target). <br>**Report:** `.agent/missions/completed/P3-M07_mission_report.md`. |
| 3.8 | CLIP Pipeline Extraction (M08) | **Complete** | `backend/clip.py` — Self-contained CLIP tokenizer + encoder (SD1.5 + SDXL). Zero `ldm_patched` imports. Bit-perfect parity verified. Emerged as critical dependency during M07. <br>**Report:** `.agent/missions/completed/P3-M08_mission_report.md`. |
| 3.9 | Patching, Quality Features & Backend Refinement (M09) | **Complete** | W01 decomposed `sampling.py` → `cond_utils.py`. W02 absorbed Fooocus quality features (anisotropic sharpness, adaptive CFG, ADM scaling, timed ADM). W03 extracted `NexModelPatcher` → `backend/patching.py` + `weight_ops.py`. W04 added LoRA support in `app.py` with zero VRAM regression. Post-M09 code audit cleaned 7 modules: removed dead code/stubs, debug prints, duplicate definitions, fixed duplicate except block, unused imports, migrated print→logging. <br>**Report:** `.agent/missions/completed/P3-M09_mission_report.md`. |
| 3.9.5 | Infrastructure Module Extraction (M9.5) | **Complete** | Extracted `ldm_patched.modules.float` → `backend/float_ops.py`, `utils` → `backend/utils.py` + `lora.py`, `model_management` → `backend/resources.py`. Zero legacy infrastructure imports in backend. All regression tests passed (SD1.5 + SDXL + LoRA). <br>**Report:** `.agent/missions/completed/P3-M9.5_mission_report.md`. |
| 3.10 | Fooocus Backend Integration (M10) | **Complete** | Connected Fooocus `modules/` frontend to `backend/` engine. Replaced 5 monkey-patches with direct backend calls. Created `backend/precision.py`. Stripped dead features (enhancement, wildcards). Resolved 3GB VRAM performance regressions (~90% RAM reduction). Full txt2img + LoRA through UI validated. <br>**Report:** `.agent/missions/completed/P3-M10_mission_report.md`. |
| 3.11 | Codebase Cleanup & Dead Code Removal (M11) | **Complete** | W01 ✅ (dead code + 25 contrib nodes removed). W02 ✅ (bridge consolidation). W03 ✅ (import reduction). W04 ✅ (UI cleanup, backend wiring, VAE encode fix with offload). <br>**Report:** `.agent/missions/active/P3-M11-W04_work_report.md`. |
| 3.12 | Modules Structural Refactoring (M12) | **Partial** | W01 ✅ (`async_worker.py` decomposed to 327 lines + pipeline stages). W02 ✅ (`webui.py` modularized into `ui_components/`). W03 superseded by M12-1. W04 (ControlNet extraction) absorbed into P4-M04. <br>**Brief:** `.agent/missions/active/P3-M12_mission_brief.md`. |
| 3.12.1 | Inpainting Architecture Pivot (M12-1) | **Complete** | Replaced legacy `InpaintHead` with `denoise_mask`-based pipeline. Native aspect ratios, 8×8 pixelation primer, morphological blend stitching. W01 ✅ (plumbing + core). W02 ✅ (outpaint + UI). W03 ✅ (integration + cleanup). W005 deferred (VRAM lifecycle). <br>**Brief:** `.agent/missions/active/P3-M12-1_mission_brief.md`. |
| ~~3.13~~ | ~~Extensibility & Registry System (M13)~~ | **Absorbed** | ControlNet registry scope absorbed into P4-M04 (feature porting to new UI). |
| ~~3.14~~ | ~~Advanced Processing Features (M14)~~ | **Absorbed** | Upscaling and processing pipelines absorbed into P4-M04. |

**Development environment:** SD 1.5 full checkpoint on local GTX 1050 for full loading path verification. GGUF Q4/Q5 SDXL on local for quantized component loading. Full-precision SDXL validation on Colab T4/L4.

**Success criteria:** `python app.py` works for SD 1.5 + SDXL with LoRA support. Fooocus UI generates txt2img images using the `backend/` engine instead of `ldm_patched`. ✅ **Achieved in M10.**

---

## ~~Phase 3.5: Fooocus Inpainting Implementation~~ (Superseded)

> **Status: SUPERSEDED by Phase 4.** Learning objectives achieved through P3-M12-1.

Phase 3.5 planned to build inpainting features through Gradio to learn the UI interaction layer.
P3-M12-1 completed this learning: `denoise_mask` mechanics, BB handling, pixelation primers,
mask compositing, and Gradio's limitations are now fully understood.

**Key insight gained:** Gradio's mask handling is structurally incompatible with production-grade
compositing workflows. Every masking feature fought Gradio's assumptions (transparent layers
misinterpreted as black, context bleed, dictionary unpacking conflicts). This drove the Phase 4 pivot.

Remaining items (ControlNet integration, canvas plugins, multi-layer compositing) will be
implemented directly in the custom React frontend (Phase 4).

---

## Phase 4: API Layer & Custom Frontend (Gradio Elimination)

> Target: Decouple the engine from Gradio via a REST/WebSocket API + custom React UI

**Pivot rationale:** See `.agent/summaries/09_Phase4_Architecture_Pivot.md`

**Development strategy: Local-First**
```
Development:  React UI --> localhost:8000 (FastAPI + GGUF on GTX 1050)
Production:   React UI --> zrok tunnel   (FastAPI + full SDXL on Colab L4)
```

**Architecture pivot (2026-03-04):** W02 proved the API builds directly on `backend/` functions,
bypassing `modules/` entirely. M01 scope narrowed. M02 (frontend) pulled forward to validate
the API contract visually. Old M01 W03/W04 (inpaint, model management) deferred to new M03.

| # | Mission | Scope | Status | Depends On |
|---|---------|-------|--------|------------|
| 4.1 | Backend API Server (P4-M01) | FastAPI server, txt2img endpoint, progress streaming, **backend structure migration** (clean engine/ + pipelines/ layout). Scope narrowed: 3 work orders. | In Progress (W01✅ W02✅ W03 ready) | P3-M12-1 |
| 4.2 | Core Frontend Shell (P4-M02) | React + Vite scaffold. Connection manager. txt2img visual flow. Model/LoRA/style selectors. **Pulled forward** to validate API contract. | Ready | P4-M01-W02 |
| 4.3 | Extended API Endpoints (P4-M03) | Inpaint/outpaint/vary/upscale endpoints. Model switching, interruption. Built on validated API contract + clean backend structure. | Not Started | P4-M01, P4-M02 |
| 4.4 | Canvas & Masking System (P4-M04) | Custom canvas component replacing Gradio sketch. Mask painting, compositing. Inpaint/outpaint frontend integration. | Not Started | P4-M03 |
| 4.5 | Feature Porting (P4-M05) | ControlNet pipeline extraction + UI. Upscaling pipeline + UI. Background/foreground removal. | Not Started | P4-M04 |
| 4.6 | Colab Deployment (P4-M06) | Colab notebook with FastAPI + tunnel. Connection resilience, auto-reconnect. | Not Started | P4-M01, P4-M02 |

**Absorbs from Phase 3:** M12-W04 (ControlNet extraction) -> P4-M05-W01. M13 (registry) -> P4-M05. M14 (upscaling) -> P4-M05.

**Briefs:** `.agent/missions/active/P4-M01_mission_brief.md`, `.agent/missions/active/P4-M02_mission_brief.md`

---

## Phase 5: Multi-Pass Engine & Assembler (Renumbered from Phase 4)

> Target: Engine that can construct images from depth-separated scene elements

| # | Task | Status | Depends On |
|---|------|--------|------------|
| 5.1 | Multi-pass generation | Not Started | Phase 4 API |
| 5.2 | ControlNet depth module | Not Started | P4-M04 |
| 5.3 | Assembler algorithm | Not Started | 5.1, 5.2 |
| 5.4 | Background removal module | Not Started | P4-M04 |

**Assembler algorithm:** Contextual padding, native rescaling, self-attention consistency, feathered composite. This is the core IP -- spatial relationships drive generation.

---

## Phase 6: Fundable Demo (Renumbered from Phase 5)

> Target: Proof-of-concept for investors showing depth-based scene composition

| # | Task | Status | Depends On |
|---|------|--------|------------|
| 6.1 | Basic 3D viewport UI (React-Three-Fiber) | Not Started | Phase 4 React UI |
| 6.2 | Scene-to-API translation | Not Started | 6.1 |
| 6.3 | Live composite preview | Not Started | 6.1, 6.2 |
| 6.4 | Production Colab deployment | Not Started | P4-M05 |

**What the demo proves to funders:**
- Images constructed as depth-composited spatial scenes, not text-to-pixel
- Decoupled architecture works (local UI -> remote compute)
- Spatially coherent results that text-to-image alone cannot produce

---

## Long-Term: 3D Scene Compositor

> Target: Full 3D viewport-based image construction application

| Component | Vision |
|-----------|--------|
| **3D Viewport** | Full scene composition -- camera, lights, 3D elements with position/rotation/scale |
| **Scene-to-Image Engine** | Each element generated with spatial awareness -- lighting, shadows, occlusion, scale |
| **Decoupled Compute** | Plug into any GPU provider -- Colab, RunPod, dedicated servers |
| **Open Model Depot** | Community model hosting, not vendor-locked |
| **Production Pipeline** | Multi-pass generation, upscaling, post-processing as composable modules |

This requires funding and a team. The demo exists to prove the concept and raise capital.

---

## Known Completed Tech from Fooocus_Nex (Available for API Wrapping)

| Feature | Status | Location |
|---------|--------|----------|
| GGUF loading (all K-quant types) | Working | `modules/gguf/` |
| Inpainting (denoise_mask pipeline) | Working | `modules/pipeline/inpaint.py` |
| Outpainting (2-step, 8×8 primer) | Working | `modules/pipeline/inpaint.py` |
| Self-attention consistency patch | Working | `modules/patch.py` |
| Ratio snapping (SDXL 26 buckets) | Working | `modules/util.py` + `modules/flags.py` |
| Morphological blend stitching | Working | `modules/pipeline/inpaint.py` |
| Background removal (InSPyReNet) | Identified | `Image_Gen_Project` discovery report |

---

## Naming Convention

Mission IDs follow the format **`P{phase}-M{number}`** with optional work order suffix:

| Format | Example | Meaning |
|--------|---------|---------|
| `P3-M01` | `P3-M01_mission_brief.md` | Phase 3, Mission 01 |
| `P3-M01-W01` | `P3-M01-W01_work_order.md` | Phase 3, Mission 01, Work Order 01 |
| `P3.5-M01` | `P3.5-M01_js_canvas_plugin.md` | Phase 3.5, Mission 01 |

Pre-pivot missions (001??03) retain their original numbering in `missions/completed/`.

---

## Reference Materials

| Resource | Location |
|----------|----------|
| ComfyUI reference code | `ComfyUI_reference/comfy/` (v0.3.47, commit 2f74e17) |
| ComfyUI-GGUF reference | `ComfyUI_reference/` (commit 3d673c5) |
| Fooocus_Nex codebase | `Fooocus_Nex/` |
| Completed mission reports | `.agent/missions/completed/` |
| Inference Architectural Guideline | `.agent/summaries/04_Inference_Architectural_Guideline.md` |
| Backend Dependency Map | `.agent/summaries/07_Backend_Dependency_Map.md` |
| Image_Gen_Project discovery | `D:\AI\Image_Gen_Project\.agent\tasks\ra_mission_report1.md` |
