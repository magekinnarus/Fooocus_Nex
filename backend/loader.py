import torch
from .defs import sdxl as sdxl_def
from ldm_patched.modules import model_base, model_patcher, sdxl_clip, sd1_clip, latent_formats, supported_models_base
from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
import ldm_patched.modules.utils as utils

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
        
        # In a real scenario, we'd ensure the model is on GPU here
        # but for this "Clean Slate" we keep it simple.
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

class VAE:
    """Isolated VAE container to avoid modules.sd baggage."""
    def __init__(self, first_stage_model, load_device, offload_device):
        self.first_stage_model = first_stage_model
        self.patcher = model_patcher.ModelPatcher(
            self.first_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )

def extract_sdxl_components(ckpt_path: str) -> dict:
    """
    Identifies and extracts components from the checkpoint state dict using a list of known prefixes.
    Supports both standard SDXL checkpoints and bundled clip files.
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    sd = utils.load_torch_file(ckpt_path)
    
    results = {k: {} for k in sdxl_def.PREFIXES}
    
    keys = list(sd.keys())
    for k in keys:
        for component, prefixes in sdxl_def.PREFIXES.items():
            matched = False
            for p in prefixes:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."):
                        new_key = new_key[1:]
                    results[component][new_key] = sd.pop(k)
                    matched = True
                    break
            if matched:
                break
    
    for component_key, component_sd in results.items():
        print(f"Extracted {component_key}: {len(component_sd)} keys")

    if len(sd) > 0:
        print(f"Remaining keys in checkpoint state_dict: {len(sd)}")
        
    return results

def load_sdxl_unet(source, load_device, offload_device, dtype=None):
    """
    Loads the SDXL UNet using sdxl_def.UNET_CONFIG.
    Supports .gguf integration.
    """
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
    
    if dtype is not None:
        model.to(dtype)
        
    model.load_state_dict(sd, strict=False)
    
    return patcher_class(model, load_device=load_device, offload_device=offload_device)

def load_sdxl_clip(source_l, source_g, load_device, offload_device, dtype=None):
    """
    Loads SDXL CLIP (L and G) and returns a clean CLIP container.
    """
    sd_l = resolve_source(source_l)
    sd_g = resolve_source(source_g)
    
    tokenizer = sdxl_clip.SDXLTokenizer()
    model = sdxl_clip.SDXLClipModel(device=offload_device, dtype=dtype)
    
    # Map back to what SDXLClipModel expects if necessary, 
    # but here we assume prefixes were already stripped during extraction
    with torch.no_grad():
        model.clip_l.load_sd(sd_l)
        model.clip_g.load_sd(sd_g)
    
    return CLIP(model, tokenizer, load_device, offload_device)

def load_sdxl_vae(source, load_device, offload_device, dtype=None):
    """
    Loads SDXL VAE and returns a clean VAE container.
    """
    sd = resolve_source(source)
    
    # Minimal SDXL VAE config
    ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
    model.load_state_dict(sd, strict=False)
    
    if dtype is not None:
        model.to(dtype)
    
    return VAE(model.eval(), load_device, offload_device)
