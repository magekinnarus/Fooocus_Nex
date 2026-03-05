# Mission P3-M01: SDXL Pipeline Trace & Performance Profiling
**Phase:** 3 (Core Pipeline Decomposition)
**Date Issued:** 2026-02-13
**Status:** Approved

## Required Reading
Read these files FIRST before starting any work:
- `.agent/archive/summaries/DEPRECATED_Nex_project_summary01.md` ??Project state and strategic direction
- `.agent/archive/summaries/DEPRECATED_02_Phase_Roadmap.md` ??Phase 3 decomposition roadmap
- `.agent/rules/01_Global_Context_Rules.md` ??Consensus protocol and core philosophies
- `C:\Users\ACER\.gemini\antigravity\brain\fc459324-f4f4-42f8-bb51-bbab89c339f1\strategic_assessment.md` ??Full strategic assessment

## Objective
Map the complete SDXL lifecycle through ComfyUI's codebase ??from checkpoint/GGUF loading through device placement, text encoding, sampling, VAE decoding, to memory cleanup. Produce a **pipeline trace document** that identifies every function and module involved, separating SDXL-essential logic from multi-model dispatch overhead. Simultaneously, profile the 2x inference performance gap between Fooocus_Nex GGUF and ComfyUI GGUF to identify root causes.

This mission is **research and documentation only** ??it produces the map that all subsequent Phase 3 extraction work will follow.

## Scope

### In Scope
**Part A: SDXL Pipeline Trace (ComfyUI Reference)**
- Trace the SDXL path through ComfyUI's loading pipeline (`sd.py`, `model_detection.py`, `model_management.py`)
- Map device placement choreography: what loads where, when things move to GPU, when they offload
- Trace dual CLIP (CLIP-L + CLIP-G) encoding path
- Trace UNet sampling loop ??sampler setup, scheduler, denoising steps
- Trace VAE decode path including tiled fallback
- Trace LoRA/adapter application flow
- For each touchpoint: flag what is SDXL-essential vs. multi-model dispatch overhead

**Part B: Performance Gap Profiling**
- Instrument Fooocus_Nex GGUF inference (SDXL GGUF UNet + CLIP + VAE)
- Compare against ComfyUI GGUF inference with equivalent settings
- Identify: dtype casting operations, unnecessary copies, missing optimizations
- Classify findings as trivial-fix vs. structural-cause

### Out of Scope
- Any code changes (this is research only)
- Non-SDXL model architectures (SD1.x, SD3, Flux, Cascade, etc.)
- UI changes
- ControlNet / IP-Adapter paths (future missions)

## Reference Files

### ComfyUI Reference (Primary Source for Trace)
- [sd.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/sd.py) ??Model loading entry point
- [model_management.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/model_management.py) ??Device/memory management
- [model_detection.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/model_detection.py) ??Architecture detection
- [model_patcher.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/model_patcher.py) ??Patching infrastructure
- [model_base.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/model_base.py) ??SDXL model class
- [supported_models.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/supported_models.py) ??SDXL config
- [samplers.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/samplers.py) ??Sampler/scheduler logic
- [sdxl_clip.py](file:///d:/AI/Fooocus_revision/ComfyUI_reference/comfy/sdxl_clip.py) ??Dual CLIP encoding
- ComfyUI-GGUF reference files in `ComfyUI_reference/`

### Fooocus_Nex (Comparison Target)
- [default_pipeline.py](file:///d:/AI/Fooocus_revision/Fooocus_Nex/modules/default_pipeline.py) ??Current pipeline orchestration
- [nex_loader.py](file:///d:/AI/Fooocus_revision/Fooocus_Nex/modules/nex_loader.py) ??Component loader
- [patch.py](file:///d:/AI/Fooocus_revision/Fooocus_Nex/modules/patch.py) ??Monkey-patch layer
- [core.py](file:///d:/AI/Fooocus_revision/Fooocus_Nex/modules/core.py) ??Generation core

## Constraints
- **Research only.** No file writes except the deliverable documents.
- **SDXL only.** Ignore all code paths for other architectures ??note them but don't trace them.
- **Profiling environment:** Local PC (GTX 1050, 32GB RAM) for GGUF comparisons. Colab L4 for full-speed benchmarks if needed.

## Deliverables
- [ ] **SDXL Pipeline Trace Document** ??Complete map of ComfyUI's SDXL path with function-level detail. Organized by pipeline stage (Load ??Device ??Condition ??Sample ??Decode ??Cleanup). Each stage identifies: essential logic, dispatch overhead, and Fooocus_Nex equivalent.
- [ ] **Performance Profile Report** ??Quantitative breakdown of where inference time is spent. Classification of causes as trivial-fix vs structural.
- [ ] **Extraction Priority Recommendation** ??Based on findings, rank which step modules to extract first for maximum impact.

## Success Criteria
- The trace document is detailed enough that a developer could implement `nex_*` modules from it without re-reading ComfyUI source.
- The performance gap is explained with specific, measurable causes (not speculation).
- Priority recommendations are justified by both the trace findings and the performance data.
