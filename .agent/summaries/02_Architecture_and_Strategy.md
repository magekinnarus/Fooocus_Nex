# 🏗️ Architecture & Technical Strategy

## Core Principle: Decompose, Don't Replace

ComfyUI's algorithms are battle-tested and sound. We don't rewrite them — we **decompose** them. Trace ComfyUI's code for our target architecture (SDXL), extract the core logic, strip the multi-model dispatch overhead, and reorganize into clean step modules.

This pattern was proven in Phase 2 when we decomposed ComfyUI-GGUF into `modules/gguf/`.

## Target Scope: SD 1.5 + SDXL

SD 1.5 and SDXL are deliberately chosen as the supported architectures:
- **Wide adoption**: Both have active communities producing finetunes, LoRAs, ControlNets, and methods
- **Consumer hardware viable**: SD 1.5 runs on 3 GB VRAM (full precision); SDXL runs on 8 GB (full) or 3 GB (GGUF-quantized)
- **Clean extraction path**: Targeting two related architectures eliminates 60-70% of ComfyUI's multi-model dispatch complexity while giving us a full local testing path (SD 1.5) and a production target (SDXL)
- **Sufficient for the demo**: SDXL quality is production-ready for the proof-of-concept

New architectures (SD3, Flux, etc.) are not in scope. If needed later, the step-module architecture makes adding them a matter of writing new component specs, not restructuring the engine.

## The Three-Component Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    LOCAL UI       │     │   MODEL DEPOT    │     │  COMPUTE CENTER  │
│                  │     │                  │     │                  │
│  3D Viewport     │────▶│  Community       │     │  GPU Rental      │
│  Scene Composer  │     │  Model Hosting   │     │  (Colab / Cloud) │
│  Workflow Control│     │  (Open/Neutral)  │     │                  │
│                  │     │                  │     │                  │
└────────┬─────────┘     └──────────────────┘     └────────▲─────────┘
         │                                                 │
         │              ┌──────────────────┐               │
         └─────────────▶│   NEX ENGINE     │───────────────┘
                        │                  │
                        │  Pipeline Core   │
                        │  (This project)  │
                        └──────────────────┘
```

The **Nex Engine** is the bridge: it receives scene instructions from the UI and executes them on whatever compute is available.

## Step-Module Architecture

The diffusion pipeline decomposes into independent, composable steps:

```
CLIP-L + CLIP-G  ──▶  Conditioning  ──▶  UNet Sampling  ──▶  VAE Decode  ──▶  Image
      ▲                     ▲                  ▲                  ▲
      │                     │                  │                  │
  nex_loader.py      nex_conditioning.py   nex_sampling.py   nex_decode.py
                                               │
                                          nex_patching.py (LoRA, adapters)
                                               │
                                          nex_memory.py (device choreography)
```

| Module | Responsibility | Source |
|--------|---------------|--------|
| `nex_loader.py` | Component-first loading (UNet, CLIP, VAE independently) | Existing + evolve |
| `nex_memory.py` | Device placement, offloading, memory budget | Extract from ComfyUI `model_management.py` |
| `nex_conditioning.py` | Dual CLIP encoding, prompt weighting | Extract from ComfyUI `sdxl_clip.py` + `sd.py` |
| `nex_sampling.py` | Sampler/scheduler execution | Extract from ComfyUI `samplers.py` |
| `nex_decode.py` | VAE decoding, tiled fallback | Extract from ComfyUI VAE path |
| `nex_patching.py` | LoRA/adapter application | Extract from ComfyUI `model_patcher.py` + `lora.py` |

Each module is:
- **Single-responsibility**: Does one thing
- **SDXL-specific**: No multi-model branching
- **Independently testable**: Can validate with a script
- **ComfyUI-derived**: Proven algorithms, clean organization

## The Assembler Algorithm

For scene-to-image construction, the engine runs multiple passes:

1. **Background Pass**: Generate background at depth 0 (full scene context)
2. **Element Pass (per element)**: For each scene element at depth N:
   - Extract with contextual padding (128px halo for blending context)
   - Rescale to native SDXL resolution (ratio-snapped from `config`)
   - Apply self-attention patch for color/lighting consistency with background
   - Generate with element-specific prompt + ControlNet depth guidance
3. **Composite**: Blend elements back to canvas with Gaussian feathered masks, respecting depth ordering

This is the core algorithm that makes "scene-to-image" work — spatial relationships drive generation, prompts describe details.

## Development Environments

| Environment | Hardware | Purpose | Model Format |
|-------------|----------|---------|--------------|
| **Colab Pro** (L4, 24GB) | Primary production | Art generation, demo hosting | Full-precision SDXL |
| **Colab Free** (T4, 16GB) | Secondary production | When L4 unavailable | Full-precision SDXL |
| **Local PC** (GTX 1050, 32GB RAM) | Development & verification | Agent module testing, SD 1.5 full checkpoint loading, SDXL GGUF, load/unload verification, basic inference | SD 1.5: safetensors; SDXL: GGUF Q4/Q5 |

GGUF is a **first-class development tool** for SDXL, not a fallback. SD 1.5 in full precision serves as the primary local development target — small enough to load fully on the GTX 1050, enabling fast iteration on the loading, conditioning, and inference pipeline without Colab.

## Development Methodology: Learning Through Completion

A core project philosophy, proven by experience:

> **Completing a full implementation — not prototyping — reveals the architectural insights needed for the next step.**

Evidence:
- **GGUF implementation** looked simple (load GGUF instead of safetensors). Completing it properly revealed ComfyUI's package-first architecture, device placement blind spots, and performance gaps — insights that drove the entire strategic pivot.
- **Phase 3.5 (Fooocus inpainting)** applies the same principle to the UI layer. Building the full inpainting feature against real ComfyUI/Fooocus modules will expose the practical quirks of bbox handling, context masking, ControlNet alignment, and image layer management — knowledge needed to design Assemble-Core's UI.

The pattern: build it properly within the existing system → discover what matters → use those insights to build the clean version.

### Refinement: Untangle-First (Phase 3 M06 lesson)

Phase 3 M06 revealed a gap in the pattern above: extracting code *structure* without understanding runtime *behavior* led to a loader that duplicated 6.7 GB in memory and a ~67% performance gap vs ComfyUI. The root cause: ~3900 lines of tangled runtime code (`model_management.py`, `model_patcher.py`, `sd.py`, `patch.py`) with 7 monkey-patches, multi-vendor dispatch, and global mutable state.

**Revised approach for remaining Phase 3 work:** Build a testable inference runner (`app.py`) through process-flow extraction — trace the actual SD 1.5 / SDXL execution paths, extract only the code that fires, and iterate until it works. Analysis and untangling happen as a byproduct of making `app.py` work, not as separate documentation exercises.

## Consensus Protocol

- **Director** = User (3D artist, non-coder). Makes all strategic and approval decisions.
- **Agent** = Implementer. No file writes without approved plan.
- **Workflow**: Logic Discussion → Implementation Plan → Explicit "Go" from Director.
- **Git**: One logical change per commit, Conventional Commits (`feat:`, `fix:`, etc.)

