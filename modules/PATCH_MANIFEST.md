# Patch Registry - Fooocus Nex

This manifest lists all monkey-patches applied to external modules or internal core components during initialization in `patch_all()`.

| Target Function | Replacement Function | Source File | Purpose (Brief) |
|:---:|:---:|:---:|:---:|
| `ldm_patched.modules.model_management.load_models_gpu` | `patched_load_models_gpu` | `patch.py` | Add timing logs for model loading. |
| `ldm_patched.modules.lora.calculate_weight` | `calculate_weight_patched` | `patch.py` | Custom Fooocus LoRA/LyCORIS weight application logic. |
| `ldm_patched.controlnet.cldm.ControlNet.forward` | `patched_cldm_forward` | `patch.py` | Implementation of ControlNet softness. |
| `ldm_patched.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward` | `patched_unet_forward` | `patch.py` | Force precision casting and diffusion progress tracking. |
| `ldm_patched.modules.model_base.SDXL.encode_adm` | `sdxl_encode_adm_patched` | `patch.py` | Patch ADM scales for positive/negative prompts. |
| `ldm_patched.modules.samplers.KSamplerX0Inpaint.forward` | `patched_KSamplerX0Inpaint_forward` | `patch.py` | Latent-level inpainting energy sampling. |
| `ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler` | `BrownianTreeNoiseSamplerPatched` | `patch.py` | DirectML compatibility for noise sampling. |
| `ldm_patched.modules.samplers.sampling_function` | `patched_sampling_function` | `patch.py` | Adaptive CFG and sharpness (anisotropic filtering). |
| `safetensors.torch.load_file` | `loader` | `patch.py` | Automatic corrupted model detection and re-download. |
| `torch.load` | `loader` | `patch.py` | Automatic corrupted model detection and re-download. |
| `ldm_patched.modules.sd1_clip.ClipTokenWeightEncoder.encode_token_weights` | `patched_encode_token_weights` | `patch_clip.py` | Custom prompt weight normalization. |
| `ldm_patched.modules.sd1_clip.SDClipModel.__init__` | `patched_SDClipModel__init__` | `patch_clip.py` | Initialize CLIP with patched ops (manual cast). |
| `ldm_patched.modules.sd1_clip.SDClipModel.forward` | `patched_SDClipModel_forward` | `patch_clip.py` | Forward pass with float32 coercion for stability. |
| `ldm_patched.modules.clip_vision.ClipVisionModel.__init__` | `patched_ClipVisionModel.__init__` | `patch_clip.py` | Initialize IP-Adapter model with patched ops. |
| `ldm_patched.modules.clip_vision.ClipVisionModel.encode_image` | `patched_ClipVisionModel_encode_image` | `patch_clip.py` | Image encoding with device management. |
| `ldm_patched.ldm.modules.diffusionmodules.openaimodel.timestep_embedding` | `patched_timestep_embedding` | `patch_precision.py` | Kohya-compatible precision for time embeddings. |
| `ldm_patched.modules.model_sampling.ModelSamplingDiscrete._register_schedule` | `patched_register_schedule` | `patch_precision.py` | Kohya-compatible noise schedule registration. |
