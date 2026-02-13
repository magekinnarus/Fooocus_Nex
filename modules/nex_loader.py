import torch
import modules.config
from ldm_patched.modules.sd import load_checkpoint_guess_config, VAE
import ldm_patched.modules.utils as ldm_utils
from modules.config import path_embeddings

def load_checkpoint(ckpt_filename, vae_filename=None):
    """
    Loads a checkpoint using the standard pipeline or GGUF.
    """
    if ckpt_filename.lower().endswith(".gguf"):
        from modules.gguf.loader import gguf_sd_loader, IMG_ARCH_LIST, TXT_ARCH_LIST
        
        sd, arch = gguf_sd_loader(ckpt_filename, return_arch=True)
        
        vae = None
        if vae_filename is not None and isinstance(vae_filename, str) and vae_filename != 'None':
            print(f"[Nex] Loading VAE from {vae_filename}")
            vae_sd = ldm_utils.load_torch_file(vae_filename)
            vae = VAE(sd=vae_sd)

        if arch in IMG_ARCH_LIST:
            unet = load_unet_gguf(ckpt_filename)
            return unet, None, vae, vae_filename, None
        elif arch in TXT_ARCH_LIST:
            clip = load_clip_gguf(ckpt_filename)
            return None, clip, vae, vae_filename, None
        else:
            # Fallback/Unknown
            unet = load_unet_gguf(ckpt_filename)
            return unet, None, vae, vae_filename, None

    # Delegate to existing ComfyUI/Fooocus loader
    return load_checkpoint_guess_config(
        ckpt_filename, 
        embedding_directory=path_embeddings,
        vae_filename_param=vae_filename
    )

def load_unet_gguf(unet_path):
    from modules.gguf import gguf_sd_loader, GGUFModelPatcher, GGMLOps
    import ldm_patched.modules.sd as comfy_sd

    sd = gguf_sd_loader(unet_path)
    model = comfy_sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": GGMLOps}
    )
    if model is None:
        raise RuntimeError(f"Could not detect model type of: {unet_path}")
    
    # Upgrade ModelPatcher -> GGUFModelPatcher
    model.__class__ = GGUFModelPatcher
    model.patch_on_device = False
    return model

def load_standalone_clip(clip_path, embedding_directory=None):
    """
    Load a standalone CLIP file using the same pipeline as full checkpoint loading.
    
    This reuses ComfyUI's native process_clip_state_dict() which handles:
    - CLIP-L prefix stripping (conditioner.embedders.0.transformer.text_model -> clip_l)
    - CLIP-G OpenCLIP->HF conversion (QKV split, key renaming via transformers_convert)
    - Proper dual-CLIP routing for SDXL
    """
    import ldm_patched.modules.utils
    import ldm_patched.modules.supported_models as supported_models
    from ldm_patched.modules.sd import CLIP, load_model_weights

    sd = ldm_patched.modules.utils.load_torch_file(clip_path, safe_load=True)

    # Detect CLIP type from key patterns (same detection as model_config does)
    has_sdxl_clip_l = any(k.startswith("conditioner.embedders.0.transformer.") for k in sd)
    has_sdxl_clip_g = any(k.startswith("conditioner.embedders.1.model.") for k in sd)
    has_sd15_clip = any(k.startswith("cond_stage_model.") for k in sd)
    
    # Already in ComfyUI internal format (text_model.* without conditioner prefix)
    has_stripped_keys = any(k.startswith("text_model.") for k in sd) and not has_sdxl_clip_l

    if has_sdxl_clip_l and has_sdxl_clip_g:
        # Full SDXL dual-CLIP bundle
        model_config = supported_models.SDXL(supported_models.SDXL.unet_config)
        print("[Nex] Detected SDXL dual-CLIP bundle (CLIP-L + CLIP-G)")
    elif has_sdxl_clip_l and not has_sdxl_clip_g:
        # SDXL checkpoint but only CLIP-L portion
        model_config = supported_models.SD15(supported_models.SD15.unet_config)
        print("[Nex] Detected standalone CLIP-L (SD1.5 format)")
    elif has_sd15_clip:
        # SD1.5 style CLIP
        model_config = supported_models.SD15(supported_models.SD15.unet_config)
        print("[Nex] Detected SD1.5 CLIP")
    elif has_stripped_keys:
        # Already stripped format — use load_clip directly
        from ldm_patched.modules.sd import load_clip
        print("[Nex] Detected pre-stripped CLIP format, using load_clip()")
        return load_clip([clip_path], embedding_directory=embedding_directory)
    else:
        # Unknown format — try load_clip as fallback
        from ldm_patched.modules.sd import load_clip
        print("[Nex] Unknown CLIP format, attempting load_clip() fallback")
        return load_clip([clip_path], embedding_directory=embedding_directory)

    # === Use the exact same pipeline as load_checkpoint_guess_config ===
    
    # Step 1: Get the clip target (model class + tokenizer)
    clip_target = model_config.clip_target()
    if clip_target is None:
        raise RuntimeError(f"Model config {type(model_config).__name__} does not support CLIP loading")

    # Step 2: Create the CLIP object (this sets up the model with proper dtype/device)
    clip = CLIP(clip_target, embedding_directory=embedding_directory)

    # Step 3: Process the state dict (key conversion, QKV split, etc.)
    sd = model_config.process_clip_state_dict(sd)

    # Step 4: Load weights using the WeightsLoader pattern
    class WeightsLoader(torch.nn.Module):
        pass

    w = WeightsLoader()
    w.cond_stage_model = clip.cond_stage_model
    m, u = w.load_state_dict(sd, strict=False)
    
    # Clean up loaded keys from sd
    unexpected_keys = set(u)
    for x in list(sd.keys()):
        if x not in unexpected_keys:
            del sd[x]

    if len(m) > 0:
        # Filter out known non-essential missing keys
        essential_missing = [k for k in m if 'position_ids' not in k]
        if essential_missing:
            print(f"[Nex] CLIP missing keys: {essential_missing}")
    if len(u) > 0:
        # Filter benign unexpected keys
        notable_unexpected = [k for k in u if k not in ('logit_scale', 'text_model.embeddings.position_ids')]
        if notable_unexpected:
            print(f"[Nex] CLIP unexpected keys: {notable_unexpected}")

    return clip

def load_clip_gguf(clip_path):
    from modules.gguf import gguf_clip_loader, GGMLOps
    import ldm_patched.modules.sd as comfy_sd
    
    sd = gguf_clip_loader(clip_path)
    
    # Check for T5
    if "encoder.block.0.layer.0.SelfAttention.q.weight" in sd:
        from ldm_patched.modules import sd3_clip
        clip_target = type('Empty', (), {})()
        clip_target.clip = sd3_clip.T5RefinerModel
        clip_target.tokenizer = sd3_clip.T5Tokenizer
        clip = comfy_sd.CLIP(clip_target, embedding_directory=path_embeddings)
        clip.load_sd(sd)
        return clip
        
    # Fallback placeholder for expanded GGUF CLIP support
    return None
