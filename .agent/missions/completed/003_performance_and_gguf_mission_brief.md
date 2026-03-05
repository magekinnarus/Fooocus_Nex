# Mission 003: Performance and GGUF Infrastructure

## Status: COMPLETED

## Mission Overview
This mission targets the technical debt accumulated in Phase 1.5, specifically a ~5-minute startup regression, and kicks off Phase 2 with GGUF model support and a UI refactor of the Model selection tab.

## Primary Objectives

### 1. Startup Performance (The "Import Storm" Fix)
- **Problem**: Eager top-level imports in `patch.py` and `patch_clip.py` cascade into massive library loading at startup.
- **Goal**: Reduce startup time to < 30 seconds.
- **Action**: Implement lazy-loading for `ldm_patched` dependencies. Move the `patch_all()` trigger to a just-in-time execution point.

### 2. Inpainting Optimization
- **Goal**: Reduce inter-generation delays during inpainting.
- **Action**: Optimize iterative loops in `fooocus_fill` (`modules/inpaint_worker.py`). Implement caching for LoRA patching if weights are unchanged.

### 3. Phase 2: GGUF Integration & Virtual Loader
- **Goal**: Support GGUF UNet loading with external CLIP/VAE components.
- **Files**: Update `nex_loader.py` and `path_utils.py`.
- **Architecture**: Implement a "Virtual Loader" that can piece together a session from a `.gguf` UNet and a combined CLIP `.safetensors` file.

### 4. Models Tab UI Refactor
- **Goal**: Align the UI with the modular loading strategy.
- **Action**: 
    - Repurpose the obsolete "Refiner" slot as **"CLIP Model"**.
    - Move **"VAE"** selection from Debug to the Models tab.
    - Hide the redundant "Refiner Switch" slider.

## Technical Environment Notes

### Colab Environment
- **Torch**: 2.9.0+cu126 (Working).
- **GPU**: L4.

### Local "Potato" Notebook
- **GPU**: GTX 1050 (3GB VRAM).
- **RAM**: 32GB (Strong CPU-side, weak GPU-side).
- **Software Recommendation**: 
    - **Torch**: 2.4.1+cu121.
    - **Xformers**: **Required**. Since Pascal architecture (1050) does not support Flash Attention 2, `xformers` is essential for memory efficiency and performance on a 3GB card.

## Implementation Decisions
- **CLIP Packaging**: Combined `.safetensors` for CLIP-L and CLIP-G.
- **Naming**: Original layer naming (No patching/renaming).
- **Preset Isolation**: Ensure changes don't break existing Presets logic.

## Handover to Project Manager
Gemini Pro (Project Manager) should now digest this brief and the [Implementation Plan](file:///C:/Users/ACER/.gemini/antigravity/brain/a1e37b5d-acd3-4983-bdc7-e0f7045320dd/implementation_plan.md) to generate the granular **Work Order**.
