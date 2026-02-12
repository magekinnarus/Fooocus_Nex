import modules.config
from ldm_patched.modules.sd import load_checkpoint_guess_config
from modules.config import path_embeddings

def load_checkpoint(ckpt_filename, vae_filename=None):
    """
    Loads a checkpoint using the standard pipeline or GGUF.
    """
    if ckpt_filename.lower().endswith(".gguf"):
        from modules.gguf.loader import gguf_sd_loader, IMG_ARCH_LIST, TXT_ARCH_LIST
        
        sd, arch = gguf_sd_loader(ckpt_filename, return_arch=True)
        
        if arch in IMG_ARCH_LIST:
            unet = load_unet_gguf(ckpt_filename)
            return unet, None, None, vae_filename, None
        elif arch in TXT_ARCH_LIST:
            clip = load_clip_gguf(ckpt_filename)
            return None, clip, None, vae_filename, None
        else:
            # Fallback/Unknown
            unet = load_unet_gguf(ckpt_filename)
            return unet, None, None, vae_filename, None

    # Delegate to existing ComfyUI/Fooocus loader
    unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(
        ckpt_filename, 
        embedding_directory=path_embeddings,
        vae_filename_param=vae_filename
    )
    return unet, clip, vae, vae_filename, clip_vision

def load_unet_gguf(unet_path):
    from modules.gguf import gguf_sd_loader, GGUFModelPatcher, GGMLOps
    import ldm_patched.modules.sd as comfy_sd

    sd = gguf_sd_loader(unet_path)
    model = comfy_sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": GGMLOps}
    )
    if model is None:
        raise RuntimeError(f"Could not detect model type of: {unet_path}")
    
    model = GGUFModelPatcher.clone(model)
    return model

def load_clip_gguf(clip_path):
    from modules.gguf import gguf_clip_loader, GGMLOps
    import ldm_patched.modules.sd as comfy_sd
    
    sd = gguf_clip_loader(clip_path)
    # This is a simplified CLIP loader for GGUF
    # In a full implementation, we'd use model_detection to find the right CLIP class
    # For GGUF models like SD3 or FLUX, they often use T5/Llama
    
    # Check for T5
    if "encoder.block.0.layer.0.SelfAttention.q.weight" in sd:
        from ldm_patched.modules import sd3_clip
        clip_target = type('Empty', (), {})()
        clip_target.clip = sd3_clip.T5RefinerModel
        clip_target.tokenizer = sd3_clip.T5Tokenizer
        clip = comfy_sd.CLIP(clip_target, embedding_directory=path_embeddings)
        clip.load_sd(sd)
        return clip
        
    # Fallback to standard guess config if logic above is insufficient
    # (Though guess config might not handle GGUF tensors correctly without ops)
    # For now, this is a placeholder for expanded GGUF CLIP support
    return None
