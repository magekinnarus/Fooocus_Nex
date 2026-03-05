# Reference: ldm_patched CLIP Extraction Map

**Mission:** P3-M08 (CLIP Pipeline Extraction: SD1.5 → SDXL)
**Date:** 2026-02-18 (Updated: 2026-02-18 — Unified loading strategy, Flux consideration)
**Source:** `Fooocus_Nex/ldm_patched/modules/` analysis + multi-session debugging findings

---

## Purpose

This document maps the exact lines of code that need to be extracted from `ldm_patched` to create a self-contained CLIP pipeline for SD1.5 inference. It serves as the implementor's guide — telling them precisely **what to copy, what to modify, and what to discard**.

---

## The Problem: 6-File Dependency Chain

The current CLIP encoding path touches 6 files with circular imports:

```
app.py → backend/loader.py → sd1_clip.py → clip_model.py → ops.py → model_management.py
                                    ↓                           ↑
                              attention.py ──────────────────────┘
```

**Total LOC in chain:** ~2,500 lines
**Lines actually needed for SD1.5 CLIP:** ~300

---

## File-by-File Extraction Guide

### 1. `clip_model.py` (189 lines) — ✅ EXTRACT MOSTLY AS-IS

| Class | Lines | Action |
|-------|-------|--------|
| `CLIPEmbeddings` | 73-80 | Copy verbatim. Pure PyTorch. |
| `CLIPAttention` | 4-21 | Copy, inline the attention function (see §4). |
| `CLIPMLP` | 27-38 | Copy verbatim. |
| `CLIPLayer` | 40-51 | Copy verbatim. |
| `CLIPEncoder` | 54-71 | Copy verbatim. |
| `CLIPTextModel_` | 83-115 | **KEY FILE.** Copy. Note: embeddings are always FP32 (line 92). |
| `CLIPTextModel` | 117-131 | Copy verbatim. Thin wrapper. |
| `CLIPVision*` | 133-189 | **SKIP** — not needed for text encoding. |

**Single impurity to remove:**
```python
# Line 2 — the ONLY import from ldm_patched
from ldm_patched.ldm.modules.attention import optimized_attention_for_device
```
→ Replace with inlined attention function (see §4 below).

### 2. `sd1_clip.py` (539 lines) — PARTIAL EXTRACT

| Component | Lines | Action |
|-----------|-------|--------|
| `SDTokenizer` | 368-493 | Extract. Only depends on `transformers.CLIPTokenizer`. |
| `SD1Tokenizer` | 496-508 | Extract. Thin wrapper around SDTokenizer. |
| `SDClipModel.__init__` | 68-101 | **Extract and simplify.** Remove `model_class` param (always CLIPTextModel). Remove `ops.manual_cast` dependency. |
| `SDClipModel.forward` | 160-192 | **Critical.** Extract the encode flow. Simplify `text_projection` handling. |
| `SDClipModel.load_sd` | 197-209 | Extract. Handles `text_projection` / `logit_scale` loading. |
| `ClipTokenWeightEncoder` | 24-59 | Extract. The per-token weighting logic. |
| `SD1ClipModel` | 511-539 | **Extract with `layer_idx` property fix** (lines 518-524). |
| `parse_parentheses`, `token_weights` | 211-257 | Extract. Pure parsing utilities. |
| `escape_important`, `unescape_important` | 259-267 | Extract. |
| `load_embed`, `safe_load_embed_zip` | 269-366 | **SKIP for now** — textual inversion embeddings, not needed for basic inference. |

**Key dependency to cut:**
```python
# Line 8 — imports model_management for intermediate_device()
from . import model_management
```
→ Replace `model_management.intermediate_device()` calls (lines 41, 58, 59) with `backend.resources.intermediate_device()`.

### 3. `ops.py` (443 lines) — TINY EXTRACT

Only ~40 lines are needed:

| Component | Lines | Action |
|-----------|-------|--------|
| `cast_bias_weight()` | 33-67 | The core casting function. Extracts weight/bias, casts to input dtype. |
| `manual_cast.Linear` | (nested in class hierarchy) | Extract the `forward_comfy_cast_weights` logic for Linear. |
| `manual_cast.LayerNorm` | (nested in class hierarchy) | Extract the `forward_comfy_cast_weights` logic for LayerNorm. |

**What ops.py actually does for CLIP:**
1. Weights are stored in FP16 (or FP8 for GGUF)
2. Embeddings output FP32 (hardcoded in `CLIPTextModel_.__init__`, line 92)
3. When a Linear layer receives FP32 input, `cast_bias_weight` casts weights to FP32 for the matmul
4. Output stays FP32 → next layer also receives FP32


### 4. `attention.py` — INLINE ONE FUNCTION

The only function needed is `optimized_attention_for_device`, which returns `optimized_attention` — a function that does:

```python
def optimized_attention(q, k, v, heads, mask=None):
    # Reshape for multi-head
    b, _, dim_head = q.shape[0], q.shape[1], q.shape[2] // heads
    q, k, v = map(lambda t: t.reshape(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    # F.scaled_dot_product_attention (PyTorch 2.0+)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out
```

> [!IMPORTANT]
> The original `attention_basic` has an `_ATTN_PRECISION = "fp32"` guard that forces Q/K to float32 before computing attention scores. This **prevents NaN overflow in FP16**. Our extraction must preserve this behavior.

---

## Precision Chain (Critical for Correctness)

This is the flow that was causing NaNs and blue noise:

```
Input Tokens (int64)
    ↓
CLIPEmbeddings.forward()          → FP32 output (hardcoded, line 92)
    ↓
CLIPEncoder layers:
    ├── LayerNorm(FP32 input)     → cast_bias_weight casts LN weights to FP32
    ├── CLIPAttention(FP32)       → cast_bias_weight casts Q/K/V weights to FP32
    │       └── attention(FP32)   → _ATTN_PRECISION="fp32" keeps scores in FP32
    ├── LayerNorm(FP32)           → same casting
    └── CLIPMLP(FP32)             → same casting
    ↓
final_layer_norm(FP32)            → FP32 output
    ↓
pooled_output → text_projection   → FP32 @ FP32 = FP32
    ↓
Output: (cond: FP32, pooled: FP32)
```

**The key insight:** The entire CLIP forward pass runs in FP32 because embeddings are FP32 and `cast_bias_weight` follows the input dtype. The `torch.autocast` in `app.py` was fighting this, causing dtype mismatches.

---

## Unified Key Normalization Strategy

> [!IMPORTANT]
> **Design principle:** One normalization function, one key format, regardless of source. The model class never needs to know where the state dict came from.

### Source Formats

| Source | CLIP-L Raw Keys | CLIP-G Raw Keys |
|--------|----------------|----------------|
| SD1.5 bundled checkpoint | `cond_stage_model.transformer.text_model.encoder...` | N/A |
| SDXL bundled checkpoint | `conditioner.embedders.0.transformer.text_model.encoder...` | `conditioner.embedders.1.model.transformer.resblocks...` |
| **Nex bundled clips** (ours) | `clip_l.text_model.encoder...` | `clip_g.text_model.encoder...` |
| Standalone HF file | `text_model.encoder.layers...` | `text_model.encoder.layers...` |

### Nex Bundled Clips Format (Proprietary — `model_extractor.py`)

Since the bundled clips file is produced by our own `tools/model_extractor.py`, we define its canonical format:

```
clip_l.text_model.encoder.layers.0.self_attn.q_proj.weight    ← HF keys with "clip_l." namespace
clip_l.text_model.encoder.layers.0.self_attn.k_proj.weight
clip_l.text_model.final_layer_norm.weight
...
clip_g.text_model.encoder.layers.0.self_attn.q_proj.weight    ← HF keys with "clip_g." namespace
clip_g.text_model.encoder.layers.0.self_attn.k_proj.weight
clip_g.text_projection                                         ← only in CLIP-G
clip_g.logit_scale                                             ← only in CLIP-G
...
```

> [!TIP]
> The extractor performs **prefix stripping AND OpenCLIP→HF key transformation at extraction time**, so the bundled file already contains model-native keys. No transformation needed at load time — just split by `clip_l.`/`clip_g.` prefix.

**Splitting a bundled file at load time:**
```python
sd = load_safetensors("model_clips.safetensors")
clip_l_sd = {k.removeprefix("clip_l."): v for k, v in sd.items() if k.startswith("clip_l.")}
clip_g_sd = {k.removeprefix("clip_g."): v for k, v in sd.items() if k.startswith("clip_g.")}
# Both dicts now in bare HF format → load_state_dict() directly
```

### Normalizer

All other source formats are handled by prefix detection and stripping:

```python
CLIP_L_PREFIXES = [
    "clip_l.",                                    # Nex bundled clips
    "conditioner.embedders.0.transformer.",        # SDXL checkpoint
    "cond_stage_model.transformer.",               # SD1.5 checkpoint
    "transformer.",                                # some standalone extractions
]

CLIP_G_PREFIXES = [
    "clip_g.",                                    # Nex bundled clips (already HF-transformed)
    "conditioner.embedders.1.model.",             # SDXL checkpoint (needs OpenCLIP→HF transform)
]

def normalize_clip_keys(sd: dict, prefixes: list) -> dict:
    """Detect and strip any known prefix → always produces model-native keys."""
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k.removeprefix(prefix): v for k, v in sd.items()}
    return sd  # already in bare format (standalone HF file)
```

**For CLIP-G from raw checkpoints:** Additional key transformations are needed beyond prefix stripping because OpenCLIP uses different layer names (`resblocks` → `encoder.layers`, `ln_1` → `layer_norm1`, `mlp.c_fc` → `mlp.fc1`, fused `attn.in_proj` → split `q/k/v`). These transformations go into the same normalizer. **Nex bundled clips skip this** because the extractor already performed the transformation.

**Result:** All sources use the **same code path:**
```python
sd = load_safetensors(path)
sd = normalize_clip_keys(sd, CLIP_L_PREFIXES)
clip_model.load_state_dict(sd)  # always matches
```

---

## CLIP Architecture Comparison (SD1.5 / SDXL / Flux)

| Aspect | SD1.5 CLIP-L | SDXL CLIP-L | SDXL CLIP-G | Flux CLIP-L |
|--------|-------------|-------------|-------------|-------------|
| Hidden states used? | ✅ Cross-attn | ✅ Cross-attn (concat with G) | ✅ Cross-attn (concat with L) | ❌ T5 handles this |
| Pooled output used? | ❌ Ignored | ❌ Ignored | ✅ ADM (`y` vector) | ✅ Vector conditioning |
| `text_projection` needed? | No | No | Yes (1280×1280) | Yes (768×768) |
| `logit_scale` needed? | No | No | Yes | Yes |
| Clip Skip | Typically -2 | Typically -1/last | Typically -1/last | N/A |
| `layer_norm_hidden_state` | **False** for clip_skip | True (default) | True (default) | True (default) |

**Design implication:** `text_projection` is an optional feature, not a universal one. Our module uses a flag:
```python
class NexClipEncoder:
    def __init__(self, config, use_projection=False):
        # ...
        if use_projection:
            self.text_projection = nn.Parameter(...)  # loaded from state dict
```

| Architecture call | `use_projection` |
|-------------------|-------------------|
| `create_sd15_clip()` | `False` |
| `create_sdxl_clip()` → CLIP-L | `False` |
| `create_sdxl_clip()` → CLIP-G | `True` |
| `create_flux_clip()` (future) | `True` |

---

## Target Module: `backend/clip.py`

The extracted module should expose:

```python
# --- Key Normalization (loading boundary) ---
def normalize_clip_l_keys(sd: dict) -> dict
def normalize_clip_g_keys(sd: dict) -> dict

# --- Tokenizer ---
class NexTokenizer:
    def __init__(self, tokenizer_path=None)
    def tokenize(self, text: str) -> list

# --- Encoder ---
class NexClipEncoder:
    def __init__(self, config: dict, dtype=torch.float16, use_projection=False)
    def load_sd(self, state_dict: dict)  # expects normalized keys
    def set_clip_skip(self, layer_idx: int)
    def encode(self, tokens: list) -> tuple[Tensor, Tensor | None]
    # Returns (hidden_states, pooled_output_or_None)

# --- Factory Helpers ---
def create_sd15_clip(sd: dict) -> tuple[NexTokenizer, NexClipEncoder]
    # normalize_clip_l_keys(sd), use_projection=False, clip_skip=-2
def create_sdxl_clip(sd_l: dict, sd_g: dict) -> tuple[NexTokenizer, NexClipEncoder]
    # normalize both, CLIP-L: use_projection=False, CLIP-G: use_projection=True
```

**Zero imports from `ldm_patched`**. Dependencies: `torch`, `transformers.CLIPTokenizer`, `backend.resources`.

---

## Verification Strategy

1. **Unit test:** Encode a prompt with both `ldm_patched` CLIP and `backend/clip.py` CLIP → compare outputs (should be identical tensor values)
2. **Integration test:** Full SD1.5 inference with extracted CLIP → produce a recognizable image
3. **Ground truth:** Use ComfyUI SD1.5 generation with same seed/prompt as reference
4. **Key normalization test:** Load same CLIP weights from (a) bundled checkpoint and (b) standalone file → verify identical model state dicts after normalization
