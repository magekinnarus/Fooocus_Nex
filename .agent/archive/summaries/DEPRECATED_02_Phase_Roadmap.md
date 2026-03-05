# Fooocus_Nex: Phase Roadmap
**Last Updated:** 2026-02-12 (Mission 002 Complete)

## Phase 1: ldm_patched Foundation ✅ COMPLETED
> Fix the house before adding rooms

| # | Task | Status | Details |
|---|------|--------|---------|
| 1.1 | Memory management optimization | ✅ Completed | Fully resolved Colab offloading. Stable VRAM and zero-pause generations. |
| 1.2 | Core module upgrade | ✅ Completed | `model_management.py`, `sd.py`, and `model_detection.py` aligned with modern reference. |
| 1.3 | Weight adapter infrastructure | ✅ Completed | Ported `weight_adapter/` directory (9 adapter types) — foundation for GGUF. |
| 1.4 | `model_patcher.py` introduction | ✅ Completed | Integrated modern `ModelPatcher` with custom Fooocus memory overrides. |
| 1.5 | `ops.py` / `quant_ops.py` | ✅ Completed | Modern operations layer ready for Phase 2 quantization. |
| 1.6 | Remove Fooocus V2 (GPT-2) Expansion | ✅ Completed | GPT-2 fully excised. Metadata filtering added for legacy support. |
| 1.7 | Precision & Performance Audit | ✅ Completed | Resolved 4.68s/it lag and 10.2GB loading issues. Stabilized foundations. |

**Reference:** `ComfyUI_reference/comfy/` (Comfy Version 0.3.47, Commit 2f74e17)
**Mission Report:** `missions/completed/001_ldm_foundation_mission_report.md`

---

## Phase 1.5: Targeted Refactoring ✅ COMPLETED
> Reduce coupling before adding complexity

| # | Task | Status | Details |
|---|------|--------|---------|
| 1.5.A | Patch Registry | ✅ Completed | Formalize 17+ monkey-patches into a documented manifest with version tracking |
| 1.5.B | Absorb `calculate_weight` | ✅ Completed | Move weight calculation logic from `patch.py` override into `model_patcher.py` proper — eliminates #1 Phase 1 bug source |
| 1.5.C | `nex_loader.py` adapter | ✅ Completed | Thin adapter layer for component-wise model loading (GGUF UNet + separate CLIP + separate VAE) |
| 1.5.D | Startup Optimization | ✅ Completed | Lazy-loaded `transformers`, purged SAM/GroundingDINO/BLIP bloat |

**Mission Brief:** `missions/completed/002_targeted_refactoring.md`
**Mission Report:** `missions/completed/002_targeted_refactoring_report.md`

---

## Phase 2: GGUF Integration & UI Refactor (Local Debug) ✅ COMPLETED
| # | Task | Status | Details |
|---|------|--------|---------|
| 2.1 | GGUF loader logic | ✅ Completed | Extracted and ported from ComfyUI-GGUF. Supports all K-quant types. |
| 2.2 | Virtual Loader architecture | ✅ Completed | UNET from `.gguf`, flexible VAE/CLIP sidecar support. |
| 2.3 | `path_utils.py` update | ✅ Completed | `.gguf` whitelisted in `config.py` extension list. |
| 2.4 | Model Tab UI Refactor | ✅ Completed | Repurposed Refiner for CLIP slot, moved VAE to main Models tab. |
| 2.5 | Performance "Kills" | ✅ Completed | Lazy-loading implemented. Inpainting OpenCV migration complete. |

**Mission Brief:** `missions/active/003_performance_and_gguf.md`

> [!NOTE]
> Phase 1.5.C (`nex_loader.py`) directly serves as the foundation for Phase 2's Virtual Loader.

---

## Phase 3: Core Pipeline Decomposition (SDXL-Focused)
> Decompose ComfyUI's proven SDXL logic into clean step modules

**Strategic Direction:** Instead of upgrading `ldm_patched` in place, trace ComfyUI's SDXL pipeline path, extract the core algorithms, and implement them as clean `nex_*` modules. This follows the same pattern proven by GGUF integration in Phase 2. Scope is intentionally SDXL-only — eliminates 60-70% of ComfyUI's multi-model dispatch complexity.

| # | Task | Status | Details |
|---|------|--------|---------|
| 3.0 | SDXL pipeline trace | Not Started | Map ComfyUI's complete SDXL lifecycle: load → device → condition → sample → decode → cleanup. Document every touchpoint. |
| 3.1 | Performance profiling | Not Started | Profile Fooocus_Nex vs ComfyUI GGUF inference gap. Fix trivial causes. Identify structural ones. |
| 3.2 | Memory/device module | Not Started | Extract SDXL device choreography into `nex_memory.py` — load/offload scheduling, memory budget. |
| 3.3 | Conditioning module | Not Started | Extract dual CLIP-L/CLIP-G encoding into `nex_conditioning.py` — prompt weighting, device management. |
| 3.4 | Sampling module | Not Started | Extract sampler/scheduler logic into `nex_sampling.py` — strip multi-model dispatch. |
| 3.5 | Patching module | Not Started | Formalize LoRA/adapter application into `nex_patching.py` — clean interface, no monkey-patches. |
| 3.6 | Loader refactor | Not Started | Evolve `nex_loader.py` into definitive component-first loader with clean dtype/device contracts. |

> [!NOTE]
> Each task follows the GGUF pattern: study ComfyUI → understand the algorithm → extract → implement cleanly.

---

## Phase 4: Feature Additions
| # | Task | Status | Details |
|---|------|--------|---------|
| 4.1 | Inpaint–ControlNet integration | Not Started | Custom JS canvas: BBox display with padding, context masking (CropAndStitch-style), semi-transparent ControlNet overlay with move/rotate/scale. Built on Gradio 3.x via `ui_gradio_extensions.py` |
| 4.2 | Background/object removal | Not Started | Extract logic from ComfyUI custom nodes (identify underlying library) |
| 4.3 | Modular Processing Pipeline | Not Started | `PipelineModule` base class + runner. Dynamic UI panel generation for composable processing chains |
| 4.3.1 | Denoise module | Not Started | GAN-based denoising (e.g., `1x_NoiseToner-Poisson-Detailed_108000_G.pth`) |
| 4.3.2 | GAN Upscale module | Not Started | Scale override (e.g., 2x from native 4x), model selection (e.g., `4xNomos8kSCHAT-L.pth`) |
| 4.3.3 | Ultimate SD Upscale module | Not Started | Optional GAN skip (use previous latent), tiled diffusion processing with ControlNet Tile |
| 4.4 | Unified workflow | Not Started | Eliminate need to switch between Fooocus and ComfyUI |

---

## Confirmed Decisions (Not Phase-Dependent)
- **Gradio:** Stay on 3.41.2. No upgrade. Custom JS for advanced UI features.
- **No node editor:** Modular pipeline presets with dynamic parameter panels instead.
- **No separate persona workflows:** Mission briefs replace `/Nex_research`, `/Nex_refactor`, etc.
- **ComfyUI as reference, not dependency:** Proven algorithms are extracted and reimplemented as clean modules. SDXL-focused scope.
- **SDXL-only target:** Wide adoption, consumer hardware viable, clean extraction path. New architectures added via step-module interfaces later if needed.
- **Decompose, don't replace:** ComfyUI's logic is sound. We reorganize it, not rewrite it.
