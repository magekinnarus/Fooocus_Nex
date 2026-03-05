# 🎯 Project Vision: Assemble-Core

## Why This Project Exists

Images are not 2D pixel combinations — they are 2D representations of 3D spatial data. To properly construct an image, understanding the 3D scene underneath the surface is critical. Current AI image generation tools treat the process as a flat text-to-pixel pipeline, missing the spatial intelligence that makes images coherent.

This project exists to build a **scene-driven image construction engine** where the primary input is a 3D scene composition (camera, lights, elements at defined depths) and prompts serve as detail definitions for each element — not as the sole driver of generation.

## The CivitAI Trap

The current AI image ecosystem locks users into monolithic platforms that bundle the UI, the models, and the compute. CivitAI is the clearest example: your workflow, your models, and your generation capacity are all tied to one vendor. If they change pricing, terms, or shut down, you lose everything.

**Assemble-Core avoids this by separating the process into three distinct, independent components:**

| Component | Role | Independence |
|-----------|------|--------------|
| **Local UI** | 3D viewport, scene composition, workflow control | User owns it. Runs on their machine. |
| **Model Depot** | Community-driven model hosting | Open, decentralized. Not tied to any platform. |
| **Compute Center** | GPU rental for inference | Fungible. Use Colab, RunPod, or any provider. |

No single component depends on the others to function. The UI works with any compute provider. The models work with any UI. The compute serves any client.

## Long-Term Objective

Build a **3D viewport-based image construction application** where:

1. The user composes a 3D scene: camera position, lighting, 3D elements (characters, props, environment) placed in space
2. Each element has depth, position, and orientation in the scene
3. Prompts describe the visual details of each element (style, material, texture) — they supplement the spatial data, not replace it
4. The engine generates each element respecting its spatial context: lighting, shadows, occlusion, scale
5. Elements are composited into a final image that is spatially coherent because it was *constructed* from spatial data

**This is fundamentally different from text-to-image.** It's **scene-to-image** — the AI fills in visual detail guided by explicit 3D scene structure.

## The Director

The user behind this project is a **3D artist** — not a coder. Their art production workflow involves compositing and inpainting as primary tools. They understand 3D spatial construction intuitively, which is the origin of the insight that drives this project. The entire codebase is developed through AI-assisted pair programming (Director = non-coder decision maker; Agent = implementer).

## How We Got Here

The project started as **Fooocus_Nex** — a customized fork of Fooocus optimized for the Director's art production workflow on Google Colab. Through three phases of development, we:

1. **Phase 1**: Upgraded the underlying ComfyUI engine (`ldm_patched`) to modern standards
2. **Phase 1.5**: Refactored the monkey-patching architecture, created component-wise model loading
3. **Phase 2**: Added GGUF quantized model support for local debugging, rebuilt the model selection UI

During this work, a pattern emerged: every debugging struggle traced back to ComfyUI's architecture — designed for monolithic checkpoints, with component-wise support bolted on afterward. The realization: rather than continuing to patch a package-first system for component-first needs, we should **decompose ComfyUI's proven algorithms into clean, modular steps** and build the engine we actually need.

This is the pivot: Fooocus_Nex becomes the proving ground for the engine that powers Assemble-Core.
