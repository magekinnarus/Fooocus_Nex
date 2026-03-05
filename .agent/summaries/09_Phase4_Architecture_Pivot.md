# Phase 4 Architecture Pivot: Gradio Elimination & Decoupled Frontend

**Date:** 2026-03-04
**Triggered by:** P3-M12-1 completion (Inpainting Architecture Pivot)

## 1. Why We're Pivoting

### The Gradio Problem

During P3-M12-1 (inpainting rewrite), every masking and compositing operation fought Gradio's
internal behavior:

| Issue | Impact | Workaround Required |
|-------|--------|---------------------|
| Transparent RGBA layers interpreted as black | Destroyed uploaded masks in Step 2 outpainting | Built `combine_image_and_mask()` defensive merger |
| Sketch component dictionary unpacking | Masks inconsistently wrapped in nested dicts | Manual `dict.get()` chains with fallback logic |
| UI state bleed between generation modes | Inpaint state contaminated txt2img params | Explicit state clearing on mode switch |
| No multi-layer compositing support | Cannot overlay context mask + inpaint mask | Blue brush feature descoped from W02 |
| Opaque canvas coordinate system | BB calculations require reverse-engineering Gradio internals | Hardcoded offset corrections |

These are not bugs — they are **structural limitations** of a framework designed for ML demos,
not production image editors. Every future feature (ControlNet guidance maps, multi-layer
compositing, context masks, 3D viewport) will encounter the same class of problem.

### The Inflection Point

P3-M12-1 proved two critical things:
1. The pipeline (`InpaintPipeline`, `process_task`, `process_diffusion`) operates independently
   of the UI. No Gradio code is called during generation.
2. The "learning through completion" pattern (`02_Architecture_and_Strategy.md`) has delivered
   its insights — we now understand `denoise_mask` mechanics, BB snapping, pixelation primers,
   morphological blending, and the full sampling chain.

The question is no longer "how does it work" but "how do we architect the system." Continuing
to build features through Gradio is testing the wrong thing.

### Alignment with Project Vision

`01_Project_Vision.md` describes three independent components:

| Component | Role | Phase 4 Mapping |
|-----------|------|-----------------|
| **Local UI** | Scene composition, workflow control | React + Vite frontend |
| **Model Depot** | Community model hosting | Future (unchanged) |
| **Compute Center** | GPU rental | localhost → Colab via tunnel |

Phase 4 is the first concrete step toward this architecture.

## 2. The Architecture

### Before (Phase 3)
```
Gradio UI ←→ async_worker.py ←→ modules/pipeline/ ←→ backend/
    ↑                                    ↑
    └── gr.State, gr.Image, yields ──────┘
              (tightly coupled)
```

### After (Phase 4)
```
React UI ──HTTP/WS──→ FastAPI Server ──→ modules/pipeline/ ──→ backend/
  (local)              (local or Colab)    (unchanged)          (unchanged)
    ↑                       ↑
    └── JSON, base64 ───────┘
        (clean API contract)
```

### Development Strategy: Local-First

```
Phase 4 Dev:   React UI ──→ localhost:8000 (FastAPI + GGUF on GTX 1050)
Phase 4 Prod:  React UI ──→ zrok tunnel   (FastAPI + full SDXL on Colab L4)
```

The API contract is identical. The only change is the URL.

## 3. Module Separation

### Backend API Server (Colab or localhost)

These modules stay server-side — they require GPU and model access:

| Module | Role |
|--------|------|
| `backend/*` | Sampling, loading, conditioning, decoding, resources |
| `modules/core.py` | Model management, VAE, ControlNet, ksampler |
| `modules/default_pipeline.py` | Model refresh, CLIP encode, process_diffusion |
| `modules/pipeline/inpaint.py` | InpaintPipeline (prepare, encode, stitch) |
| `modules/pipeline/inference.py` | process_task orchestration |
| `modules/pipeline/preprocessing.py` | Prompt processing, style application |
| `modules/pipeline/output.py` | Save, metadata, logging |
| `modules/pipeline/image_input.py` | apply_inpaint, apply_outpaint, apply_vary, apply_upscale |
| `modules/gguf/` | GGUF model support |
| `modules/config.py` | Model paths, defaults |
| `modules/flags.py` | Constants, aspect ratios |

### Frontend (always local)

These capabilities move to the React frontend:

| Capability | Replaces |
|------------|----------|
| Canvas/mask editor | `gradio_hijack.py` sketch component |
| Model selector | `ui_components/models_panel.py` |
| Prompt editor | `ui_components/prompt_panel.py` |
| Generation controls | `ui_components/settings_panel.py` |
| Style selector | `ui_components/styles_panel.py` |
| Gallery/output viewer | Gradio gallery |
| Progress display | `task_state.yields` → WebSocket |

### Hybrid Modules (require splitting)

These modules currently mix backend logic with UI coupling:

| Module | Backend Part | UI Part |
|--------|-------------|---------|
| `image_input.py` | apply_inpaint, apply_outpaint logic | Reads task_state UI params |
| `inference.py` | process_task sampling | Progress callback formatting |
| `preprocessing.py` | Prompt encoding | UI flag interpretation |
| `output.py` | Image saving | UI-specific path generation |

The API server mediates: it translates JSON requests into the format these modules expect,
and translates their outputs into JSON responses.

## 4. Technology Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| **Backend framework** | FastAPI | Async, auto-docs (Swagger), Pydantic schemas, WebSocket support |
| **Frontend framework** | React + Vite | Future React-Three-Fiber for 3D viewport. Vite = instant HMR |
| **Canvas library** | Fabric.js or Konva | Rich canvas manipulation, mask painting, layer management |
| **Progress streaming** | WebSocket | Bidirectional, supports preview images, lower overhead than SSE |
| **Image transfer** | base64 PNG in JSON | Simple, no multipart complexity. Works over tunnels |
| **Tunnel (production)** | zrok or Cloudflare | Colab → public URL. Frontend just changes the target URL |

## 5. Phase 4 Mission Overview

| Mission | Scope | Status |
|---------|-------|--------|
| **P4-M01** | Backend API Server — txt2img endpoint, progress streaming, **backend structure migration** (clean engine/ + pipelines/ layout) | In Progress (W01✅ W02✅ W03 ready) |
| **P4-M02** | Core Frontend Shell — React + Vite, txt2img visual flow, model/style selectors. **Pulled forward** to validate API contract. | Ready |
| **P4-M03** | Extended API Endpoints — Inpaint/outpaint/vary/upscale endpoints. Model switching, interruption. | Not Started |
| **P4-M04** | Canvas & Masking System — replace Gradio sketch, mask painting, compositing | Not Started |
| **P4-M05** | Feature Porting — ControlNet, upscaling, bg removal | Not Started |
| **P4-M06** | Colab Deployment — tunnel, auto-reconnect | Not Started |

### What Phase 4 Absorbs

| Original Item | Absorbed Into |
|---------------|---------------|
| Phase 3.5 (Gradio UI features) | Superseded — learning complete |
| P3-M12-W04 (ControlNet extraction) | P4-M04-W01 |
| P3-M13 (Extensibility & Registry) | P4-M04 |
| P3-M14 (Advanced Processing) | P4-M04 |
| Old Phase 4.5 (FastAPI wrapper) | P4-M01 |

### What Gets Renumbered

| Old | New | Content |
|-----|-----|---------|
| Phase 4 | **Phase 5** | Multi-pass engine, assembler algorithm |
| Phase 5 | **Phase 6** | Fundable demo (3D viewport) |

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Tunnel latency for large images | Medium | Compress transfers, thumbnails for previews, file-path returns for local mode |
| Colab runtime disconnects | High | Frontend auto-reconnect, generation state checkpointing |
| Two codebases to maintain | Medium | Clean API contract = backend changes don't require frontend changes |
| React learning curve | Low | Vite + React is well-documented; agent scaffolds the entire UI |
| Losing Gradio features not yet replicated | Low | Only txt2img, inpaint, outpaint used — all well-understood |
