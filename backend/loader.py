import torch
import gc
from .defs import sdxl as sdxl_def
from .defs import sd15 as sd15_def
from backend import loader, resources, sampling, conditioning, decode, clip
from ldm_patched.modules import model_base, model_patcher, latent_formats, supported_models_base
from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
import torch.nn as nn
import ldm_patched.modules.utils as utils

def heal_model_weights(model, name_prefix="Model"):
    """
    Checks for NaNs/Infs in model weights and heals them in-place.
    """
    # print(f"Checking and Healing {name_prefix} weights...")
    bad_params_count = 0
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_params_count += 1
            print(f"CRITICAL: Bad values in {name_prefix} parameter: {name}. HEALING...")
            with torch.no_grad():
                if "weight" in name and ("layer_norm" in name or "layernorm" in name):
                     param.data.nan_to_num_(nan=1.0, posinf=1.0, neginf=-1.0)
                else:
                     param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    
    if bad_params_count > 0:
        print(f"Healed {bad_params_count} parameters in {name_prefix}.")

class EmbeddingFP32Wrapper(nn.Module):
    """
    Force embedding output to FP32 to trigger safe compute paths in ldm_patched.
    """
    def __init__(self, original_embedding):
        super().__init__()
        self.original = original_embedding
    
    def forward(self, x):
        return self.original(x).float()
    
    def __getattr__(self, name):
        if name in ["original", "forward", "__init__", "__class__", "__dir__"]:
             return super().__getattr__(name) 
        return getattr(self.original, name)

def resolve_source(source):
    """
    Ensures the source is a state dict. If it's a path, loads it.
    """
    if isinstance(source, str):
        return utils.load_torch_file(source)
    return source

class ModelConfig(supported_models_base.BASE):
    """Mock config object for model instantiation, inheriting from BASE for compatibility."""
    def __init__(self, unet_config, latent_format):
        super().__init__(unet_config)
        self.latent_format = latent_format

class CLIP:
    """Isolated CLIP container to avoid modules.sd baggage."""
    def __init__(self, cond_stage_model, tokenizer, load_device, offload_device):
        self.cond_stage_model = cond_stage_model
        self.tokenizer = tokenizer
        self.patcher = model_patcher.ModelPatcher(
            self.cond_stage_model, 
            load_device=load_device, 
            offload_device=offload_device
        )
        self.layer_idx = None

    def clone(self):
        n = CLIP(self.cond_stage_model, self.tokenizer, self.patcher.load_device, self.patcher.offload_device)
        n.patcher = self.patcher.clone()
        n.layer_idx = self.layer_idx
        return n

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()
            
        load_device = self.patcher.load_device
        offload_device = self.patcher.offload_device
        
        if load_device != offload_device:
            self.patcher.model.to(load_device)
            
        try:
            cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
            with torch.no_grad():
                c_std = cond.std().item()
                c_mean = cond.mean().item()
                print(f"DEBUG CLIP Out: Mean={c_mean:.4f}, Std={c_std:.4f}, Shape={cond.shape}")
                print(f"DEBUG CLIP Values (first 5x5): \n{cond[0, :5, :5]}")
        finally:
            if load_device != offload_device:
                self.patcher.model.to(offload_device)
                
        if return_pooled:
            return cond, pooled
        return cond

class VAE:
    """Isolated VAE container to avoid modules.sd baggage."""
    def __init__(self, first_stage_model, load_device, offload_device, latent_format=None):
        self.first_stage_model = first_stage_model
        # Use SD15 as default if not specified (backward compatibility)
        self.latent_format = latent_format or latent_formats.SD15()
        self.patcher = model_patcher.ModelPatcher(
            self.first_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )

# --- SDXL Support ---

def load_sdxl_unet(source, load_device=None, offload_device=None, dtype=None):
    """
    Loads the SDXL UNet using sdxl_def.UNET_CONFIG.
    Supports .gguf integration.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    
    custom_operations = None
    patcher_class = model_patcher.ModelPatcher

    if isinstance(source, str) and source.endswith(".gguf"):
        from modules.gguf.loader import gguf_sd_loader
        from modules.gguf.ops import GGMLOps
        from modules.gguf.patcher import GGUFModelPatcher
        
        sd = gguf_sd_loader(source)
        custom_operations = GGMLOps
        patcher_class = GGUFModelPatcher
    else:
        sd = resolve_source(source)

    model = model_base.SDXL(
        model_config=ModelConfig(sdxl_def.UNET_CONFIG, latent_formats.SDXL()),
        operations=custom_operations
    )
    
    # User Requirement: SDXL UNet should be in fp16 to avoid casting to fp32 (saving RAM/VRAM)
    if dtype is None:
        dtype = torch.float16
        
    if dtype is not None:
        model.to(dtype)
        
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    return patcher_class(model, load_device=load_device, offload_device=offload_device)

def load_sdxl_clip(source_l, source_g, load_device=None, offload_device=None, dtype=None):
    """
    Loads SDXL CLIP (L and G) and returns a clean CLIP container.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device() 
    
    sd_l = resolve_source(source_l)
    sd_g = resolve_source(source_g)
    
    # Use Nex implementations
    tokenizer = clip.NexSDXLTokenizer()
    
    # User Requirement: SDXL CLIP should be in fp16
    if dtype is None:
        dtype = torch.float16 
        
    model = clip.NexSDXLClipModel(device=offload_device, dtype=dtype)
    
    with torch.no_grad():
        # NexSDXLClipModel.load_sd handles L and G appropriately
        # If they are different dicts, we load each. 
        # If they are the same (bundled), it still works.
        if sd_l is not None:
            model.load_sd(sd_l)
        if sd_g is not None and sd_g is not sd_l:
            model.load_sd(sd_g)
    
    return CLIP(model, tokenizer, load_device, offload_device)

def load_vae(source, load_device=None, offload_device=None, dtype=None, latent_format=None):
    """
    Loads VAE (SD15/SDXL compatible) and returns a clean VAE container.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.vae_offload_device()
    
    sd = resolve_source(source)
    
    # Generic VAE config (works for both SD1.5 and SDXL usually)
    ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
    model.load_state_dict(sd, strict=False)
    
    # User Requirement: VAE should be in fp32
    if dtype is None:
        dtype = torch.float32 
    
    if dtype is not None:
        model.to(dtype)
    
    return VAE(model.eval(), load_device, offload_device, latent_format=latent_format)

def load_sdxl_checkpoint(ckpt_path, load_device=None, unet_dtype=None):
    """
    Loads SDXL components sequentially and clears raw data immediately.
    """
    print(f"Loading SDXL checkpoint from: {ckpt_path}")
    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32 default)
    print("Extracting VAE...")
    vae_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["vae"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                vae_sd[new_key] = sd.pop(k).clone()
                break
    
    vae = load_vae(vae_sd, latent_format=latent_formats.SDXL())
    del vae_sd
    gc.collect()
    
    # CLIP (fp16 default)
    print("Extracting CLIP...")
    clip_l_sd = {}
    clip_g_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["clip_l"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_l_sd[new_key] = sd.pop(k).clone()
                break
        
        if k in sd: 
            for p in sdxl_def.PREFIXES["clip_g"]:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."): new_key = new_key[1:]
                    clip_g_sd[new_key] = sd.pop(k).clone()
                    break
                    
    clip = load_sdxl_clip(clip_l_sd, clip_g_sd, dtype=unet_dtype)
    del clip_l_sd
    del clip_g_sd
    gc.collect()

    # UNet (fp16 default)
    print("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k).clone()
                break
    
    if len(sd) > 0:
        print(f"Remaining keys in checkpoint: {len(sd)}")
    
    print("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    print("Loading UNet Model...")
    unet = load_sdxl_unet(unet_sd, load_device=load_device, dtype=unet_dtype)
    del unet_sd
    gc.collect()
    
    return unet, clip, vae

# --- SD 1.5 Support ---

def load_sd15_unet(source, load_device=None, offload_device=None, dtype=None):
    """
    Loads the SD 1.5 UNet using sd15_def.UNET_CONFIG.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    
    sd = resolve_source(source)

    model = model_base.BaseModel(
        model_config=ModelConfig(sd15_def.UNET_CONFIG, latent_formats.SD15()),
    )
    
    # User Requirement: SD1.5 UNet should be in fp16
    if dtype is None:
        dtype = torch.float16
        
    model.diffusion_model.to(dtype)
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    return model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

def load_sd15_clip(source, load_device=None, offload_device=None, dtype=None):
    """
    Loads SD 1.5 CLIP (L only) and returns a clean CLIP container.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    
    sd = resolve_source(source)
    
    tokenizer, encoder = clip.create_sd15_clip(sd)
    
    # Wrap in CLIP container for compatibility
    # CLIP container expects: cond_stage_model, tokenizer, patcher
    # We need to adapt NexClipEncoder to behave like cond_stage_model for the CLIP container wrapper below?
    # Actually, the CLIP container in loader.py (lines 61-108) calls:
    #   self.tokenizer.tokenize_with_weights
    #   self.cond_stage_model.clip_layer
    #   self.cond_stage_model.reset_clip_layer
    #   self.cond_stage_model.encode_token_weights
    # Our NexClipEncoder and NexTokenizer implement these.
    
    # However, create_sd15_clip returns (tokenizer, encoder).
    # We must patch encoder to be the cond_stage_model.
    
    cond_stage_model = encoder
    
    # Patcher needs a 'model' attribute on cond_stage_model usually?
    # ModelPatcher takes 'model'. NexClipEncoder IS a module, so it works.
    
    return CLIP(cond_stage_model, tokenizer, load_device, offload_device)

def load_sd15_checkpoint(ckpt_path, load_device=None, unet_dtype=None):
    """
    Loads SD 1.5 components sequentially.
    """
    print(f"Loading SD 1.5 checkpoint from: {ckpt_path}")
    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32)
    print("Extracting VAE...")
    vae_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["vae"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                vae_sd[new_key] = sd.pop(k).clone()
                break
    
    vae = load_vae(vae_sd, latent_format=latent_formats.SD15())
    del vae_sd
    gc.collect()
    
    # CLIP (fp16)
    print("Extracting CLIP...")
    clip_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["clip"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_sd[new_key] = sd.pop(k).clone()
                break
                     
    clip = load_sd15_clip(clip_sd, dtype=unet_dtype)
    heal_model_weights(clip.patcher.model, "CLIP")

    # Precision Injection (SD1.5 specific fix for NaN overflows)
    try:
        sd1_clip_model = clip.cond_stage_model
        
        # Helper to find the transformer (support NexClipEncoder vs SD1ClipModel)
        transformer = None
        
        if hasattr(sd1_clip_model, 'transformer'): 
             # NexClipEncoder or direct CLIPTextModel
             transformer = sd1_clip_model.transformer
        elif hasattr(sd1_clip_model, 'clip_l'): 
             # SD1ClipModel wrapper
             transformer = sd1_clip_model.clip_l.transformer
        elif hasattr(sd1_clip_model, 'clip'):
             # Alternatve wrapper name
             transformer = sd1_clip_model.clip.transformer
             
        # Further unwrap if needed (SD1ClipModel sometimes wraps CLIPTextModel inside)
        if transformer is not None and hasattr(transformer, 'text_model'): # Transformers library or some wrappers
             transformer = transformer.text_model

        if transformer is not None and hasattr(transformer, 'embeddings'):
           embeddings = transformer.embeddings
           # Check if already wrapped to avoid double wrapping
           if not isinstance(embeddings, EmbeddingFP32Wrapper):
               transformer.embeddings = EmbeddingFP32Wrapper(embeddings)
               # print(f"Applied Precision Injection to CLIP Embeddings.")
    except Exception as e:
        print(f"FAILED to apply precision injection to CLIP: {e}")
    except Exception as e:
        print(f"FAILED to apply precision injection to CLIP: {e}")

    del clip_sd
    gc.collect()

    # UNet (fp16)
    print("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k).clone()
                break
    
    if len(sd) > 0:
        print(f"Remaining keys in checkpoint: {len(sd)}")
    
    print("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    print("Loading UNet Model...")
    unet = load_sd15_unet(unet_sd, load_device=load_device, dtype=unet_dtype)
    heal_model_weights(unet.model, "UNet")
    del unet_sd
    gc.collect()
    
    return unet, clip, vae
