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
    Load a standalone CLIP file using backend.loader.
    """
    import backend.loader
    import ldm_patched.modules.utils
    
    # We need to detect if it's SDXL or SD1.5 to call the right loader.
    # We can peek at the state dict keys (lightweight).
    sd = ldm_patched.modules.utils.load_torch_file(clip_path, safe_load=True)
    
    # Heuristic detection same as before
    has_sdxl_clip_l = any(k.startswith("conditioner.embedders.0.transformer.") for k in sd)
    has_sdxl_clip_g = any(k.startswith("conditioner.embedders.1.model.") for k in sd)
    has_sd15_clip = any(k.startswith("cond_stage_model.") for k in sd)
    
    # SDXL Dual Clip
    if has_sdxl_clip_l and has_sdxl_clip_g:
        print("[Nex] Detected SDXL dual-CLIP bundle (CLIP-L + CLIP-G)")
        # load_sdxl_clip expects separate sources or same source.
        # It handles the splitting internally if keys match.
        return backend.loader.load_sdxl_clip(sd, sd)

    # SDXL CLIP-L only (rare but exists)
    elif has_sdxl_clip_l and not has_sdxl_clip_g:
        print("[Nex] Detected standalone CLIP-L (SDXL format)")
        # This is ambiguous. Is it for SDXL or SD1.5? 
        # Usually SDXL L is same architecture as SD1.5 L.
        # But keys are "conditioner...". 
        # backend.loader.load_sdxl_clip handles this if we pass as 'source_l'.
        return backend.loader.load_sdxl_clip(sd, None)

    # SD1.5 CLIP
    elif has_sd15_clip:
        print("[Nex] Detected SD1.5 CLIP")
        return backend.loader.load_sd15_clip(sd)
        
    # Standalone files (text_model.*)
    # Could be L or G.
    # Check for G specific keys?
    # OpenCLIP G usually has "text_model.encoder.layers..." OR "resnet..." if visual.
    # But we are doing text.
    # If it has "text_model.", backend.loader normalization handles it.
    # But we need to know WHICH function to call.
    
    # Generic "text_model" keys:
    # If it has 768 dim -> likely L (Standard SD1.5/SDXL-L)
    # If it has 1280 dim -> likely G (SDXL-G)
    
    # Let's check a key shape if poss.
    keys = list(sd.keys())
    if any("text_model.encoder.layers.0.layer_norm1.weight" in k for k in keys) or \
       any("text_model.embeddings.token_embedding.weight" in k for k in keys):
           
        # Check hidden size
        for k in keys:
            if "layer_norm1.weight" in k:
                w = sd[k]
                if w.shape[0] == 1280:
                    print("[Nex] Detected Standalone CLIP-G (based on dim 1280)")
                    return backend.loader.load_sdxl_clip(None, sd)
                elif w.shape[0] == 768:
                    print("[Nex] Detected Standalone CLIP-L (based on dim 768)")
                    # Could be SD1.5 or SDXL-L. 
                    # Use SD1.5 loader as default for L-only? 
                    # Or SDXL loader with only L? 
                    # If we use SD1.5 loader, we get a generic CLIP object.
                    # If we use SDXL loader, we get SDXL object.
                    # App.py / webui might expect generic CLIP for SD1.5.
                    return backend.loader.load_sd15_clip(sd)
                    
    # Fallback to legacy if we can't determine (or it's SD3/Flux which backend.loader doesn't support yet)
    # Actually, we should probably just return the backend.loader version if possible.
    
    print("[Nex] Unknown CLIP format, defaulting to SD1.5 loader")
    return backend.loader.load_sd15_clip(sd)

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
