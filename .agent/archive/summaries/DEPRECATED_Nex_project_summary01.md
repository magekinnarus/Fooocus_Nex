# Project Brief: Fooocus_Nex (Refactor & Optimization)
**Last Updated:** 2026-02-12

## 1. Project Objective
Transition from a standard Fooocus (Colab-locked) environment to a custom, high-performance fork called **Fooocus_Nex**. Goals: remove bloat (focus on Inpaint/Outpaint compositing for 3D art workflow), enable modern hardware support (L4 GPU / Torch 2.9), implement GGUF support for local debugging, and unify features currently requiring ComfyUI into a single UI.

**Context:** The Director is a 3D artist who needs this customized Fooocus for active artwork production while separately developing the `assemble-core` project (local ComfyUI-based UI). Fooocus_Nex is the production tool; assemble-core is the long-term replacement.

## 2. Infrastructure & Environment
- **Hardware:** NVIDIA L4 (24GB VRAM) on Colab Pro / NVIDIA T4 (16GB VRAM) and 13GB Available RAM in Colab Free-tier / Local PC (GTX 1050, 32GB RAM) for debugging
- **Storage Strategy:** Models in `/content` (local ephemeral) for speed; Git Repo and persistent settings (.env) in Google Drive
- **Software Stack:** Torch 2.9 baseline, NumPy 2.x, python-dotenv for secret management
- **UI Framework:** Gradio 3.41.2 ??**confirmed staying on 3.x** (no upgrade). Custom JS canvas components will be built for advanced features.
- **Reference Codebase:** `ComfyUI_reference/` at project root ??assemble-core (Comfy Version 0.3.47, Commit 2f74e17), ComfyUI-GGUF (commit 3d673c5, April 2025). Version 0.3.47 was chosen as the latest ComfyUI version before the significant frontend migration that started in 0.3.48 (`comfy_api`, `ComfyExtension` imports appeared).

## 3. Key Refactorings Completed
- **`launch.py` De-shackling:** Bypassed `entry_with_update.py` and `update.py` to stop auto-reverting code changes. Removed automated dependency management (`run_pip`) to prevent downgrades. Integrated `.env` loading.
- **Dependency Clean-up:** Removed legacy flags and forced installs for `xformers` and `GroundingDINO` incompatible with modern Torch.
- **Phase 1 ??ldm_patched Foundation (Completed 2026-02-11):**
  - Ported modern `model_patcher.py`, `weight_adapter/` (9 types), `ops.py`, `quant_ops.py` from ComfyUI 0.3.47
  - Resolved Colab VRAM state detection (forced `HIGH_VRAM` for >=13GB VRAM)
  - Fixed precision mismatch (FP32?’FP16), LoRA regressions, double-patching OOM
  - Implemented Unified Precision Pass in UNet forward and Sharpness Optimization
  - Removed Fooocus V2 GPT-2 expansion bloat

## 4. Architectural Insights & Strategic Direction
- **Monkey-Patching Debt:** Fooocus overrides 17+ functions/methods across `patch.py`, `patch_clip.py`, and `patch_precision.py`. The weight calculation patch (`calculate_weight_patched`) was the #1 source of Phase 1 bugs.
- **ComfyUI Evolution Mismatch:** ComfyUI modules grew organically from single-checkpoint assumptions. Bolted-on quantization/adapter support creates implicit coupling between `model_patcher`, `lora`, and `model_management`. Every Phase 1?? debugging struggle traces back to this.
- **Strategic Pivot (Phase 3):** Instead of upgrading `ldm_patched` in place, we will **decompose** ComfyUI's SDXL pipeline into clean `nex_*` step modules. ComfyUI's algorithms are sound ??we reorganize them, not replace them. SDXL-only scope eliminates 60-70% of multi-model dispatch complexity.
- **Step-Module Architecture:** Clean modules for loading, memory/device management, conditioning, sampling, decoding, and patching. Follows the pattern proven by GGUF integration in Phase 2.
- **Convergence:** Clean modules built for Fooocus_Nex directly serve Image_Gen_Project's headless engine.

## 5. Current Phase: Phase 3 ??Core Pipeline Decomposition
See [02_Phase_Roadmap.md](file:///d:/AI/Fooocus_revision/.agent/archive/summaries/DEPRECATED_02_Phase_Roadmap.md) for full details.

**Focus areas:**
- Trace ComfyUI's SDXL lifecycle and map every touchpoint
- Profile and fix the 2x GGUF inference performance gap
- Extract and modularize: memory/device, conditioning, sampling, patching, loading

## 6. Workflow & Pain Points
- **3D Artist workflow:** Compositing and inpainting are primary use. Inpaint mask cannot align with ControlNet tab ??a critical pain point.
- **ComfyUI dependency:** Background removal, object removal, upscaling currently require switching to ComfyUI. Goal is to extract underlying logic and integrate into Nex.
- **Colab usage:** See `Fooocus_Comprehensive.ipynb` for current Colab workflow (model download via aria2, config.txt setup, launch customization).

## 7. Confirmed Architectural Decisions
- **Gradio 3.x stays:** Advanced UI features (BBox display, context mask visualization, ControlNet semi-transparent overlay with move/rotate/scale) will be built as custom JS canvas components injected via `ui_gradio_extensions.py`. Gradio 5 does not natively support these features either.
- **Modular Processing Pipeline (not node graphs):** Composable processing chains where each step is a self-contained `PipelineModule` class (denoise ??GAN upscale ??Ultimate SD upscale). UI dynamically generates parameter panels per module. This avoids recreating ComfyUI's node complexity inside Fooocus.
- **Custom image component:** `gradio_hijack.py` (484 lines) monkeypatches Gradio's Image component. Uses Gradio 3.x-specific APIs (`IOComponent`, `ImgSerializable`, `TokenInterpretable`). 9 `grh.Image` instances in `webui.py`.
- **ComfyUI as reference, not dependency:** Proven algorithms are extracted and reimplemented as clean SDXL-focused modules. Decompose, don't replace.
- **SDXL-only target:** Wide adoption, consumer hardware viable, clean extraction path.

## 8. Agent Organization
- **Consensus Protocol:** User = Director (Non-Coder), Agent = Implementer. No file writes without approved plan.
- **PM Workflow:** `/Nex_pm` ??strategic planning and mission orchestration (the only persistent workflow)
- **Mission System:** `.agent/missions/` for task briefs and reports. Mission briefs serve as both system prompts and task definitions ??separate persona workflows are unnecessary in IDE agent context.
- **Summaries:** `.agent/summaries/` for persistent project state

## 9. Git Standards
- One logical change per commit using Conventional Commits (e.g., `feat:`, `fix:`).
