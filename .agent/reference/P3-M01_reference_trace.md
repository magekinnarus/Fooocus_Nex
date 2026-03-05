# P3-M01: SDXL Pipeline Trace — ComfyUI Reference

**Mission:** P3-M01 (SDXL Pipeline Trace & Performance Profiling)
**Date:** 2026-02-13
**Source:** `ComfyUI_reference/comfy/` (v0.3.47, commit 2f74e17)

---

## SDXL Lifecycle Overview

```
┌────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│   LOAD     │───▶│  CONDITION     │───▶│   SAMPLE     │───▶│  DECODE  │───▶│  IMAGE   │
│            │    │                │    │              │    │          │    │          │
│ Checkpoint │    │ CLIP-L encode  │    │ CFGGuider    │    │ VAE      │    │ Tensor → │
│ or UNet +  │    │ CLIP-G encode  │    │ K-Diffusion  │    │ Batched  │    │ PIL/save │
│ CLIP + VAE │    │ Concatenate    │    │ UNet forward │    │ + tiled  │    │          │
│            │    │ encode_adm     │    │ CFG scale    │    │ fallback │    │          │
└────────────┘    └────────────────┘    └──────────────┘    └──────────┘    └──────────┘
```

---

## Stage 1: Loading

### Entry Points
| Function | File | Purpose |
|----------|------|---------|
| `load_state_dict_guess_config()` | `sd.py:1032` | Checkpoint → UNet + CLIP + VAE |
| `load_diffusion_model_state_dict()` | `sd.py:1113` | Standalone UNet loading |
| `load_text_encoder_state_dicts()` | `sd.py:840` | Standalone CLIP loading |
| `VAE()` | `sd.py:269` | Standalone VAE loading |

### Checkpoint Loading Flow
```
load_state_dict_guess_config(sd)
  ├── unet_prefix_from_state_dict(sd)          # Find "model.diffusion_model." prefix
  │     └─ Tries known prefixes, returns match  # model_detection.py:639
  │
  ├── model_config_from_unet(sd, prefix)       # Detect architecture
  │     ├─ detect_unet_config(sd, prefix)       # Infer UNet structure from key shapes
  │     └─ model_config_from_unet_config()      # Match against supported_models list
  │           └─ Returns SDXL config (context_dim=2048, adm_in=2816, transformer_depth=[0,0,2,2,10,10])
  │
  ├── UNet: model_config.get_model(sd, prefix)
  │     └─ model_base.SDXL(config, model_type)  # Creates UNet + Timestep embedder + noise_augmentor
  │     └─ model.load_model_weights(sd, prefix) # Loads weights
  │     └─ ModelPatcher(model, load_device, offload_device)  # Wraps for device management
  │
  ├── CLIP: model_config.process_clip_state_dict(sd)
  │     └─ Remaps key prefixes:
  │         "conditioner.embedders.0.transformer.text_model" → "clip_l.transformer.text_model"
  │         "conditioner.embedders.1.model."                → "clip_g."
  │     └─ CLIP(clip_target, ...)               # SDXLClipModel with SDXLTokenizer
  │     └─ clip.load_sd(clip_sd)                # Loads both CLIP-L and CLIP-G weights
  │
  └── VAE: Extract keys with vae_key_prefix ("first_stage_model.")
        └─ VAE(sd=vae_sd)                       # AutoencoderKL for SDXL
```

### SDXL-Specific Config (supported_models.py:181)
```python
unet_config = {
    "model_channels": 320,
    "use_linear_in_transformer": True,
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "context_dim": 2048,           # CLIP-L(768) + CLIP-G(1280) = 2048
    "adm_in_channels": 2816,      # Pooled CLIP-G(1280) + 6×256(resolution embeddings) = 2816
    "use_temporal_attention": False,
}
```

### What's SDXL-Essential vs. Overhead
| Logic | SDXL Essential? | Notes |
|-------|----------------|-------|
| `unet_prefix_from_state_dict` | ✅ Needed | But we can hardcode for SDXL |
| `detect_unet_config` (572 lines!) | ❌ Overhead | Multi-model dispatch. For SDXL, config is known. |
| `model_config_from_unet_config` | ❌ Overhead | Iterates all supported_models. We know it's SDXL. |
| `process_clip_state_dict` | ✅ Needed | Prefix remapping for bundled checkpoints |
| `load_text_encoder_state_dicts` (146 lines!) | ❌ 90% overhead | Massive multi-model branching. SDXL path is ~5 lines. |
| `ModelPatcher` wrapping | ✅ Needed | Device management, LoRA patching |

---

## Stage 2: Conditioning

### CLIP Encoding (sdxl_clip.py)
```
SDXLTokenizer.tokenize_with_weights(text)
  ├── clip_l.tokenize_with_weights(text)    # SD1-style tokenizer (77 tokens)
  └── clip_g.tokenize_with_weights(text)    # OpenCLIP tokenizer (77 tokens)
        Returns: {"l": token_pairs_l, "g": token_pairs_g}

SDXLClipModel.encode_token_weights(token_weight_pairs)
  ├── clip_g.encode_token_weights(pairs_g)  # → g_out (B, T, 1280), g_pooled (B, 1280)
  ├── clip_l.encode_token_weights(pairs_l)  # → l_out (B, T, 768),  l_pooled (B, 768)
  ├── cut_to = min(l_out.shape[1], g_out.shape[1])
  └── return cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1)  # (B, T, 2048)
         + g_pooled                                              # (B, 1280)
```

### SDXL ADM Conditioning (model_base.py:444)
The UNet also receives resolution/crop metadata (SDXL's "micro-conditioning"):
```python
encode_adm(**kwargs):
    clip_pooled = sdxl_pooled(kwargs, noise_augmentor)  # (B, 1280)
    # Timestep-embed 6 scalars: height, width, crop_h, crop_w, target_h, target_w
    flat = cat([embedder(Tensor([val])) for val in [h, w, ch, cw, th, tw]])  # (1, 1536)
    return cat([clip_pooled, flat], dim=1)  # (B, 2816) → goes to UNet as c_adm / "y"
```

### Conditioning Data Flow Summary
```
Text Prompt
  ↓
Tokenize (CLIP-L + CLIP-G independently)
  ↓
Encode (CLIP-L → 768d, CLIP-G → 1280d)
  ↓
Concatenate → cross_attn (2048d)      → UNet cross-attention (c_crossattn)
CLIP-G pooled + resolution embeds     → UNet ADM conditioning (y)
```

---

## Stage 3: Sampling

### Call Chain
```
sample.py:sample()
  └── samplers.py:sample()
        └── CFGGuider(model_patcher)
              ├── .set_conds(positive, negative)
              ├── .set_cfg(cfg_scale)
              └── .sample(noise, latent_image, sampler, sigmas, ...)
                    │
                    ├── preprocess_conds_hooks(conds)
                    ├── prepare_model_patcher(model_patcher, conds)
                    │
                    └── outer_sample()
                          ├── prepare_sampling()          # Load models to GPU
                          ├── prepare_mask()              # Inpainting mask if any
                          ├── Tensors → device
                          ├── model_patcher.pre_run()     # Apply LoRA patches
                          │
                          └── inner_sample()
                                ├── process_latent_in()   # Model-specific latent transform
                                ├── process_conds()       # Resolve areas, encode model conds
                                │     ├── resolve_areas_and_cond_masks()
                                │     ├── calculate_start_end_timesteps()
                                │     ├── encode_model_conds(model.extra_conds, ...)
                                │     │     └── SDXL.extra_conds():
                                │     │           ├── c_crossattn ← prompt embeddings
                                │     │           └── y ← encode_adm (pooled + resolution)
                                │     └── pre_run_control()  # Init ControlNets
                                │
                                └── KSAMPLER.sample()
                                      └── k_diffusion sampling loop
                                            └── Per step: sampling_function()
                                                  ├── calc_cond_batch(cond, uncond)
                                                  │     └── model.apply_model(x, t, **conds)
                                                  │           └── UNet forward pass
                                                  └── cfg_function(cond_pred, uncond_pred, scale)
                                                        └── uncond + (cond - uncond) * scale
```

### Key Sampling Architecture Points
- **CFGGuider** owns the full sampling lifecycle — setup, run, cleanup
- **`process_conds`** is where model-specific conditioning happens (encode_adm for SDXL)
- **`sampling_function`** is the per-step denoising core — this is what every step calls
- **`cfg_function`** is simple linear interpolation between cond/uncond predictions
- **Hooks/wrappers** system (`patcher_extension.py`) wraps nearly everything — powerful but adds complexity

### Schedulers Available
```
simple, normal, karras, sgm_uniform, exponential, beta, linear_quadratic, kl_optimal
```
Each takes `model_sampling` and `steps`, returns sigma schedule.

---

## Stage 4: VAE Decoding (sd.py:580)

```
VAE.decode(samples_in)
  ├── Calculate memory_used for shape
  ├── load_models_gpu([patcher])            # Move VAE to GPU
  ├── Batch by available memory
  ├── Per batch:
  │     samples → vae_dtype → device
  │     first_stage_model.decode(samples)   # AutoencoderKL.decode
  │     process_output()                    # Clamp to [0,1]
  ├── On OOM: fallback to decode_tiled_()
  └── movedim(1, -1)                        # BCHW → BHWC
```

### VAE Key Points
- Memory-aware batching — calculates how many latents fit in free VRAM
- Automatic tiled fallback on OOM
- SDXL VAE latent format: 4 channels, 8× spatial compression

---

## Stage 5: Output

Latent (B, 4, H/8, W/8) → VAE decode → (B, H, W, 3) float32 [0,1] → save

---

## Extraction Priority for nex_* Modules

| Module | Essential Logic | Source | Complexity |
|--------|---------------|--------|------------|
| **nex_loader.py** | Prefix extraction, component splitting, `ModelPatcher` wrapping | `sd.py` load functions, `model_detection.py` (SDXL path only) | Medium |
| **nex_conditioning.py** | `SDXLClipModel.encode_token_weights`, `SDXL.encode_adm`, tokenizers | `sdxl_clip.py`, `sd1_clip.py`, `model_base.py:444` | Medium |
| **nex_sampling.py** | `CFGGuider`, `sampling_function`, `cfg_function`, scheduler functions | `samplers.py`, `sample.py` | High — most complex module |
| **nex_memory.py** | `load_models_gpu`, device placement, memory estimation | `model_management.py` | Medium |
| **nex_decode.py** | `VAE.decode`, tiled fallback, memory batching | `sd.py:580` (VAE class) | Low |
| **nex_patching.py** | `ModelPatcher.patch_weight`, LoRA key matching | `model_patcher.py`, `lora.py` | High — entangled with hooks system |

### Recommended Extraction Order
1. **nex_loader.py** — Foundation. Everything depends on loading correctly.
2. **nex_conditioning.py** — Second cleanest. SDXL CLIP path is well-isolated.
3. **nex_decode.py** — Simplest. VAE is nearly self-contained.
4. **nex_memory.py** — Extract device choreography from model_management.py.
5. **nex_sampling.py** — Core complexity. CFGGuider + k-diffusion loop.
6. **nex_patching.py** — Last. Deeply entangled with hooks/wrappers system.

---

## Multi-Model Dispatch Overhead (What We Eliminate)

| Code Region | Lines | SDXL-Used Lines | Overhead % |
|-------------|-------|-----------------|------------|
| `detect_unet_config` | 572 | ~30 | 95% |
| `load_text_encoder_state_dicts` | 146 | ~10 | 93% |
| `model_config_from_unet_config` | 7 (iterates 30+ classes) | 1 | 97% |
| `supported_models.py` | 1236 | ~70 (SDXL class) | 94% |
| `model_base.py` | 1500+ | ~80 (SDXL + BaseModel core) | 95% |

**Estimated total dispatch overhead eliminated: ~90% of detection/routing code.**

The remaining ~10% is the actual algorithm we need — and it's clean, proven, and well-tested.
