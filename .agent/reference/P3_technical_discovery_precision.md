# Technical Discovery: Precision Casting Chain

## Context
During Mission P3-M06 recovery, we identified a 2x performance regression caused by a manual `model.to(torch.float32)` on the CLIP model in `app.py`. We investigated how ComfyUI handles NaN-prone layers (4 & 8) without this slowdown.

## The Discovered Chain
The knowledge required to isolate this logic into a "Nex-native" module in M09 is as follows:

1. **Forward Entry**: `CLIPTextModel_.forward` receives the input tokens.
2. **Precision Switch**: It passes `dtype=torch.float32` into its sub-modules (Embeddings and Encoder).
3. **Operation Wrapper**: The layers are constructed using an `operations` class (typically `ldm_patched.modules.ops.manual_cast`).
4. **On-the-fly Casting**: Inside `ops.py`, functions like `cast_bias_weight` check the input tensor's dtype (now FP32) and cast the weights (FP16/FP8) *only during the math operation*.
   - **Key Function**: `cast_bias_weight(s, input=None)`
   - **Logic**: `weight = cast_to(s.weight, input.dtype, device)`

## 5. The Attention Layer Trap
Crucially, `ldm_patched.modules.attention` contains a hardcoded behavior for `CrossAttention`:
- **File**: `ldm_patched/ldm/modules/attention.py`
- **Variable**: `_ATTN_PRECISION = "fp32"` (Default)
- **Function**: `attention_basic` (or `optimized_attention`)
- **Behavior**: Even if inputs `q, k, v` come in as FP16, `attention_basic` explicitly casts them:
  ```python
  if _ATTN_PRECISION == "fp32":
      sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
  ```
- **Why this matters**: This single line prevents the "vicious cycle" of normalization where `x` (FP16) -> `LayerNorm` -> `Attention` (internal FP32) -> `Out` (FP16). Without this upcast, the attention scores `sim` would explode to `Inf` in FP16 for certain prompts/checkpoints.

## 6. Precision Injection Strategy (Learnings)
In M06, we attempted to use an `EmbeddingFP32Wrapper` to force the CLIP text embeddings to output FP32, hypothetically triggering the downstream FP32-compute path in `ldm_patched` ops.
- **Goal**: Avoid casting 600MB of weights to FP32; only cast activations.
- **Outcome**: Partially successful concept, but difficult to implement due to `SD1ClipModel` structure variance (Transformers vs Native).
- **Critical Requirement**: Any wrapper causing a type change must implement `__getattr__` to proxy all other attribute accesses (like `token_embedding`) to the original module, or `ldm_patched` loading logic will crash.

## Stub for M09 (Nex-native Ops)
To preserve this for the rebuild, we should implement a `backend/ops.py` that:
- Inherits from a clean `BaseOps`.
- Implements `NexLinear` and `NexLayerNorm` that automatically detect when the input is FP32 and upcast their internal weights accordingly.
- **CRITICAL**: Ensure our custom Attention mechanism respects an `force_float32_compute` flag.
- Removes the dependency on `ldm_patched.modules.model_management`.

> [!IMPORTANT]
> This document serves as the technical blueprint for the "Nex-native Ops" work in Mission P3-M09.

## 7. CLIP & Conditioning Architecture (M07 Findings)
During the audit of SD1.5 image quality (muddy or lue noise output), we uncovered significant discrepancies in how ldm_patched loads and utilizes CLIP compared to the SD1.5 standard.

### A. Missing Projection Layers
Standard SD1.5 checkpoints do *not* contain 	ext_projection or logit_scale weights. However, the CLIPTextModel architecture requires them.
- **Discovery**: ldm_patched loader loop specifically excludes these if they aren't in the checkpoint cond_stage_model prefixes.
- **Fix**: We modified sd1_clip.py to initialize these layers with **Identity** matrices (for projection) and default scaling (1.6094) if missing. This is crucial for correct embedding alignment.

### B. Clip Skip Logic Failure
The Clip Skip functionality (accessing penultimate layers) was completely broken in SD1ClipModel.
- **Bug**: The layer_idx property had no getter/setter on SD1ClipModel, so setting it in pp.py did nothing. The model always returned the *last* layer.
- **Fix**: Added a property proxy to forward layer_idx set calls to the internal clip_l model.

### C. Layer Norm Muffling`nSD1.5 Standard dictates that intermediate layer outputs (Clip Skip) should *not* be normalized by the final layer_norm.
- **Issue**: SD1ClipModel defaults to layer_norm_hidden_state=True. When Clip Skip 2 is used, this applies a normalization meant for the *final* output to the *penultimate* one, crushing the variance.
- **Result**: Disabling this norm raised the conditioning signal standard deviation from ~1.0 to ~5.0.

### D. Current State: Bluish Noise`nDespite restoring the signal strength, the output remains lue noise. This suggests:
1.  **Oversaturation**: The 5x signal boost might be *too* strong without other compensatory scaling.
2.  **VAE Mismatch**: The latent variance (~2.2) vs pixel mean (~0.2-0.6) suggests the VAE decode is shifting the distribution wildly.
3.  **Cross-Attention Indexing**: We must still verify if ttn2 layers are strictly mapped to the correct transform blocks.
