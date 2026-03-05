# Analysis: `ldm_patched` vs `ComfyUI_reference`

**Date:** 2026-02-15
**Scope:** Comparative analysis of the legacy `ldm_patched` directory versus the clean `ComfyUI_reference` submodule.
**Purpose:** To guide future agents in removing legacy technical debt.

## 1. Executive Summary

*   **Identity:** `ldm_patched` is **structurally identical** to `ComfyUI_reference` for 95% of its file content.
*   **The Problem:** The divergence is classified into three specific types of technical debt: Missed Updates, Bad Updates, and Vestigial Baggage.

**Strategic Directive:**
For all new backend development (`Fooocus_Nex/backend/`), **DO NOT IMPORT** from `ldm_patched`. Use `ComfyUI_reference` as the sole source of truth.

---

## 2. The Three Types of Divergence

### Category A: The "Missed" Updates (Stagnation)
*Features that ComfyUI implemented natively, but Fooocus missed or kept patching over.*

1.  **Native FP8 Support:**
    *   *ComfyUI:* `model_management.py` now natively detects and supports `torch.float8_e4m3fn` (lines 49-73).
    *   *Fooocus:* Still relies on external patching, ignoring the native support.
2.  **Standardized Device Handling:**
    *   *ComfyUI:* Unified logic for `MPS`, `XPU`, and `DirectML`.
    *   *Fooocus:* Verbose, separate branches that miss recent upstream stability fixes.

### Category B: The "Bad" Updates (Self-Inflicted Regressions)
*Updates that were applied, but implemented poorly, creating conflicts that didn't exist before.*

1.  **The "High VRAM" Regression (Colab Freeze):**
    *   *The Conflict:* ComfyUI's aggressive offloading broke Colab performance.
    *   *The Misstep:* We hardcoded `google.colab` checks into `model_management.py` instead of using configuration, polluting the library.
2.  **The "Hybrid" Wrapper (Legacy Baggage):**
    *   *The Conflict:* ComfyUI moved to `ModelPatcher`; old nodes used raw models.
    *   *The Misstep:* We maintained a `LoadedModel` shim with a `legacy_mode` flag to support custom nodes that **Fooocus does not even use**.

### Category C: The "Vestigial" Updates (Dead Baggage)
*Code that was updated/imported but is practically dead code for Fooocus.*

1.  **`args_parser.py` (CLI Arguments):**
    *   *The Issue:* `ldm_patched` contains ComfyUI's full CLI parser (`--listen`, `--port`, `--preview-option`).
    *   *Reality:* Fooocus has its own fully separate configuration system (`modules/config.py`, `modules/launch.py`). This entire file is dead weight that confuses developers.
2.  **Unused Model Architectures:**
    *   *The Issue:* `model_base.py` includes definitions for `STABLE_CASCADE`, `V_PREDICTION_CONTINUOUS`, `FLOW`.
    *   *Reality:* Fooocus is strictly focused on SDXL (and potentially Flux). Importing logic for experimental architectures we don't support adds initialization overhead and potential bugs for zero benefit.
3.  **`modules/impact/` (If present):**
    *   *The Issue:* Any custom node folders inside `ldm_patched` that are not explicitly used by the core Fooocus pipeline are ghost dependencies.

---

## 3. The "Extraction" Strategy (The Fix)

To escape this trap, we must stop "fixing" `ldm_patched`.

1.  **Source:** Open `ComfyUI_reference/comfy/[module].py`.
2.  **Copy:** Copy the logic to `backend/[module].py`.
3.  **Purge:**
    *   **REMOVE** the `legacy_mode` checks (Category B).
    *   **REMOVE** the hardcoded `google.colab` checks (Category B).
    *   **DELETE** `args_parser.py` equivalents (Category C).
4.  **Re-Implement Properly:**
    *   Add a `ForceHighVRAM` **configuration option** (Fixing Category B).
    *   Use native FP8 signatures (Fixing Category A).

---
*End of Analysis*
