import modules.config
from ldm_patched.modules.sd import load_checkpoint_guess_config
from modules.config import path_embeddings

def load_checkpoint(ckpt_filename, vae_filename=None):
    """
    Loads a checkpoint using the standard pipeline or GGUF.
    """
    if ckpt_filename.lower().endswith(".gguf"):
        unet = load_unet_gguf(ckpt_filename)
        # CLIP and VAE loading for GGUF need separate handling in the caller or future expansion
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
