# Mission Brief: P3-M07 — Process-Flow Inference Runner (app.py)
> **Supported architectures: SD 1.5 + SDXL**
**ID:** P3-M07  
**Phase:** 3  
**Date Issued:** 2026-02-17  
**Status:** Ready  
**Depends On:** P3-M06 (lessons learned)  
**Supersedes:** Previous P3-M07 (analysis-only), planned M08, planned M09  

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md` (esp. "Untangle-First Refinement")
- `.agent/summaries/03_Roadmap.md` (revised Phase 3 table)
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/missions/completed/P3-M06_mission_report.md` (failure analysis)

## Objective

Build `app.py` — a minimal, JSON-configured inference runner for **SD 1.5 and SDXL** that produces an image end-to-end without Gradio, ComfyUI nodes, or the full `ldm_patched` import chain. The runner must work on:
- **Local** (GTX 1050, 3 GB VRAM, 32 GB RAM)
  - SD 1.5 as full checkpoint (~2 GB safetensors) — tests full loading path locally
  - SDXL as GGUF Q4/Q5 — tests quantized component loading locally
- **Colab T4** (16 GB VRAM, 12.7 GB RAM, full-precision SDXL)
- **Colab L4** (24 GB VRAM, 56 GB RAM, full-precision SDXL)

**Why SD 1.5?** Adding SD 1.5 lets us test the full checkpoint loading and device-placement pipeline locally without Colab or GGUF. The checkpoint is small enough to fit in 3 GB VRAM, giving us a fast iteration loop for loader development. SDXL remains the primary target.

This consolidates the previously planned M07 (untangle model_management), M08 (benchmark patches), and M09 (loader rebuild) into a single, iterative, **testable** effort. Analysis and untangling happen as a byproduct of making `app.py` work — not as a separate documentation exercise.

## Core Methodology: Process-Flow Extraction

Instead of analyzing entire files (1505-line `model_management.py`, 1398-line `model_patcher.py`), we:

1. **Trace the SD 1.5 and SDXL inference flows** — follow the actual execution path from "load checkpoint" to "save PNG"
2. **Extract only the code paths that fire** — ignore multi-vendor dispatch, SD3/Flux branches, unused features
3. **Build small, flow-based modules** — each module handles one step of the process, with imports you can explain in one sentence

> **Principle:** If you can't explain what an import does in one sentence, you shouldn't be importing it.

## JSON Configuration

All inference parameters are controlled by a JSON file (`config.json`), not CLI flags. The Director can edit this file directly to change any parameter.

### Schema (draft)

```json
{
    "model": {
        "checkpoint": "path/to/model.safetensors",
        "vae": null,
        "type": "sdxl",
        "format": "safetensors"
    },
    "_model_type_options": "sd15 | sdxl",
    "_model_format_options": "safetensors | gguf",
    "inference": {
        "prompt": "a beautiful landscape, masterpiece, best quality",
        "negative_prompt": "low quality, worst quality, blurry",
        "seed": -1,
        "steps": 25,
        "cfg_scale": 7.0,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "denoise": 1.0
    },
    "output": {
        "width": 1024,
        "height": 1024,
        "directory": "./outputs",
        "filename_prefix": "nex"
    },
    "device": {
        "gpu": "auto",
        "dtype": "auto",
        "offload_to_cpu": false
    }
}
```

**Key design choices:**
- `"type": "sd15"` or `"sdxl"` — selects architecture (determines UNet config, CLIP model, default resolution)
- `"format": "safetensors"` or `"gguf"` — switches between standard and quantized loading paths
- `"seed": -1` — random seed; any positive integer for reproducible output
- `"gpu": "auto"` — auto-detects CUDA device; can be set to `"cuda:0"` or `"cpu"` explicitly
- `"dtype": "auto"` — follows the logic in `backend/resources.py`; can be overridden to `"float16"`, `"bfloat16"`, etc.
- `"vae": null` — uses checkpoint's bundled VAE; set to a path for external VAE

## Scope

### In Scope
- **`app.py`** — the inference runner (~100-200 lines: read JSON, load components, generate, save)
- **`config.json`** — default configuration file with sensible SDXL defaults
- **Process-flow extraction** — extract only the code that fires during SD 1.5 / SDXL load→encode→sample→decode into lean backend modules
- **Lean imports** — `app.py` and its backend modules should not import `model_management.py`, `sd.py`, or `patch.py` wholesale. Extract what's needed.
- **Dual architecture** — SD 1.5 (single CLIP, 512×512 native) and SDXL (dual CLIP L+G, 1024×1024 native)
- **GGUF support** — must work with both quantized (GGUF) and full-precision (safetensors) checkpoints
- **Memory management** — solve the 6.7 GB duplication problem from M06 using process-flow knowledge
- **Performance target** — within 20% of ComfyUI baseline (targeting ≤14.4 sec/it if ComfyUI does 12 sec/it)

### Out of Scope
- Gradio UI or web interface
- LoRA/adapter application (future mission)
- Inpainting or img2img (Phase 3.5)
- SD3, Flux, or other architectures beyond SD 1.5 and SDXL
- Multi-GPU or distributed inference

## Reference Files
- `Fooocus_Nex/ldm_patched/modules/model_management.py` — device/memory choreography (trace, don't import)
- `Fooocus_Nex/ldm_patched/modules/sd.py` — checkpoint loading logic (trace, don't import)
- `Fooocus_Nex/ldm_patched/modules/model_patcher.py` — patch/unpatch flow (extract minimal)
- `Fooocus_Nex/modules/patch.py` — monkey-patches (understand which affect performance)
- `ComfyUI_reference/comfy/model_management.py` — for comparison
- `Fooocus_Nex/backend/` — existing clean extractions (resources, conditioning, sampling, decode, k_diffusion, schedulers)
- `Fooocus_Nex/modules/gguf/` — GGUF loading path

## Deliverables
- [ ] **`app.py`** — JSON-configured inference runner, working on local + Colab
- [ ] **`config.json`** — default configuration with documented fields
- [ ] **Process-flow documentation** — brief notes captured during extraction, documenting which code paths from `model_management.py` / `model_patcher.py` / `sd.py` were actually needed and why
- [ ] **Updated backend modules** — any modifications to `backend/loader.py`, new modules like `backend/memory.py`, `backend/defs/sd15.py`, updated `backend/conditioning.py`, etc.
- [ ] **Proof of work** — generated images from both local (SD 1.5 + GGUF) and Colab (full-precision SDXL) runs

## Success Criteria
1. `python app.py` reads `config.json`, loads the model, generates an image, and saves it
2. Works with SD 1.5 safetensors checkpoint on local GTX 1050 (full checkpoint loading path)
3. Works with SDXL GGUF checkpoint on local GTX 1050 (quantized component loading path)
4. Works with full-precision SDXL checkpoint on Colab T4 (16 GB) and L4 (24 GB)
5. No 6.7 GB memory duplication (peak RAM stays reasonable during loading)
6. Performance within 20% of ComfyUI baseline on comparable hardware
7. `app.py` does NOT import `model_management.py`, `sd.py`, or `patch.py` directly
8. The Director can change model, prompt, resolution, and sampler by editing `config.json` alone

## Work Orders
To be registered in `P3-M07_work_list.md` by CM Role1:
- `P3-M07-W01` — Process-flow trace and lean module extraction (SD 1.5 + SDXL paths)
- `P3-M07-W02` — Build `app.py` + `config.json` (SD 1.5 local first, then SDXL GGUF local)
- `P3-M07-W03` — Colab validation (T4/L4 full-precision)
- `P3-M07-W04` — Performance profiling and optimization

## Notes
- This mission replaces the previous analysis-only M07 and the planned M08/M09. The rationale: analysis as a separate step produced documentation but no testable output. Building `app.py` forces the same understanding while producing something that runs.
- The JSON config approach is deliberately simple — no schema validation library, just `json.load()` with sensible defaults. Complexity can be added later if needed.
- The existing `backend/` modules (conditioning, sampling, decode, schedulers, k_diffusion) are available as-is. The main gaps are loading/memory management, which is where the process-flow extraction focuses.
- Performance investigation (the "8 sec/it gap") happens naturally during W04 — if `app.py` is slow, we profile and fix. No separate benchmarking mission needed.
- **SD 1.5 backend gaps**: Need `backend/defs/sd15.py` (UNet config + prefixes), `encode_text_sd15()` in `conditioning.py`, and `load_sd15_*` functions in `loader.py`. Sampling, decode, schedulers, and k_diffusion are already architecture-agnostic.
