# Session Handoff: SD1.5 Logic & Signal Restoration

## Current Context
We have been debugging "muddy" and "blue noise" outputs in SD1.5 generations. The core issue was identified as a severely muffled conditioning signal (L1 Diff ~0.08 vs expected >1.0) due to multiple failures in the CLIP loading and execution chain.

## Key Accomplishments (Session M07/M08)
1.  **Restored `logit_scale` & `text_projection`**:
    - These layers are *missing* from SD1.5 checkpoints but are required by `CLIPTextModel`.
    - `ldm_patched` loader was skipping them.
    - **Fix**: Modified `sd1_clip.py` to initialize them with Identity/Default values if missing.

2.  **Fixed "Clip Skip"**:
    - `SD1ClipModel` lacked a `layer_idx` property, so setting it in `app.py` did nothing (model always used last layer).
    - **Fix**: Added property proxy to `SD1ClipModel`.

3.  **Corrected Normalization**:
    - `SD1ClipModel` was applying `layer_norm` to intermediate outputs (Clip Skip -2), which is incorrect for SD1.5.
    - **Fix**: Disabled `layer_norm_hidden_state` in `loader.py`.

## Current Status: "Bluish Noise"
- **Conditioning Strength**: Restored to healthy levels (Std ~5.0).
- **Output**: The image is now "deep fried" with blue noise, suggesting the signal might be *too* strong or the VAE is receiving garbage.
- **Hypothesis**:
    - **Oversaturation**: The identity initialization for `text_projection` might be too aggressive compared to what the UNet expects (maybe it expects a learned projection?).
    - **VAE Scaling**: The latent distribution (Std ~2.2) is very wide.
    - **UNet Attention Mismatch**: We successfully audited the `attn2` keys, but need to verify they map to the correct transformer blocks in the UNet.

## Next Session Objectives
1.  **Validate Projection Matrix**: Research if `text_projection` should be Identity or if we can extract it from a broader search (maybe it's under a different key in `cond_stage_model`).
2.  **Audit VAE Decode**: Use `verify_vae.py` to decode a known-good latent and see if it produces blue noise.
3.  **Re-enable Layer Norm?**: Test if re-enabling `layer_norm_hidden_state` AND keeping the projection fixes the "deep fried" look.

## References
- [Technical Discovery: CLIP Architecture](file:///d:/AI/Fooocus_revision/.agent/reference/P3_technical_discovery_precision.md)
- [Implementation Plan](file:///C:/Users/ACER/.gemini/antigravity/brain/dbcf793f-bc26-4105-a554-0288d551206b/implementation_plan.md)
