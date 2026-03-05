# 🚀 Assemble-Core: Global Context & Rules

**Primary Objective:** Build a scene-driven image construction engine that decomposes AI image generation into clean, modular steps — starting with SDXL pipeline decomposition, targeting a fundable demo of depth-based scene composition.

## 🤝 The Consensus Protocol (STRICT)
- **Roles:** User = Director (3D artist, non-coder); Agent = Implementer.
- **No Cowboy Coding:** You are FORBIDDEN from writing files or running terminal commands without a prior approved plan.
- **Approval Workflow:** 1. Logic Discussion (Plain English) → 2. Detailed Implementation Plan → 3. Explicit "Go" from Director.

## 🏗️ Core Philosophies
- **Decompose, don't replace:** ComfyUI's algorithms are sound. Extract and reorganize the SDXL-specific logic into clean step modules.
- **SDXL-only scope:** Eliminates multi-model dispatch complexity. New architectures added via step-module interfaces if needed.
- **Component-first:** Every model component (UNet, CLIP, VAE) loads, moves, and executes independently.
- **Custom nodes as parts catalog:** Extract underlying algorithms from ComfyUI custom nodes. No framework dependency.
- **Three-component separation:** Local UI, community model depot, fungible compute. No vendor lock-in.
- **Production on Colab Pro:** L4 (24GB) or T4 (16GB). GGUF exists only for local debugging on GTX 1050.

## 📂 Key Reference Files
- **Vision:** `.agent/summaries/01_Project_Vision.md`
- **Architecture:** `.agent/summaries/02_Architecture_and_Strategy.md`
- **Roadmap:** `.agent/summaries/03_Roadmap.md`
- **ComfyUI Reference:** `ComfyUI_reference/comfy/` (v0.3.47, commit 2f74e17)
- **Active Missions:** `.agent/missions/active/`
- **Completed Missions:** `.agent/missions/completed/`

## 📝 Git Standards
- One logical change per commit using Conventional Commits (e.g., `feat:`, `fix:`).