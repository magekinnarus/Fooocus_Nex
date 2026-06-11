import torch
import logging
from typing import Any, Dict
import gc
from safetensors import safe_open
import torch
from .defs import sdxl as sdxl_def
from .defs import sd15 as sd15_def
from backend import resources, clip, patching, conditioning
from ldm_patched.modules import model_base, latent_formats, supported_models_base
from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
import torch.nn as nn
from . import utils

def heal_model_weights(model, name_prefix="Model"):
    """
    Checks for NaNs/Infs in model weights and heals them in-place.
    """
    # print(f"Checking and Healing {name_prefix} weights...")
    bad_params_count = 0
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_params_count += 1
            logging.warning(f"CRITICAL: Bad values in {name_prefix} parameter: {name}. HEALING...")
            with torch.no_grad():
                if "weight" in name and ("layer_norm" in name or "layernorm" in name):
                     param.data.nan_to_num_(nan=1.0, posinf=1.0, neginf=-1.0)
                else:
                     param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    
    if bad_params_count > 0:
        logging.info(f"Healed {bad_params_count} parameters in {name_prefix}.")

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

def resolve_source(source, device=None):
    """
    Ensures the source is a state dict. If it's a path, loads it.
    """
    if isinstance(source, str):
        return utils.load_torch_file(source, device=device)
    return source


def _safe_open_device_arg(device):
    if device is None:
        return "cpu"
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return 0 if device.index is None else int(device.index)
    return "cpu"


def _strip_checkpoint_prefix(key, prefix):
    new_key = key[len(prefix):]
    if new_key.startswith("."):
        new_key = new_key[1:]
    return new_key


def _extract_prefixed_safetensors_state_dict(ckpt_path, prefixes, *, device=None):
    extracted = {}
    with safe_open(ckpt_path, framework="pt", device=_safe_open_device_arg(device)) as handle:
        for key in handle.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    extracted[_strip_checkpoint_prefix(key, prefix)] = handle.get_tensor(key)
                    break
    return extracted


def _load_prefixed_safetensors_into_module(
    ckpt_path,
    prefixes,
    module,
    *,
    device=None,
    dtype=None,
):
    target_device = torch.device(device) if device is not None else None
    state_entries = module.state_dict()
    loaded_keys = set()
    unexpected_keys = []

    with safe_open(ckpt_path, framework="pt", device=_safe_open_device_arg(target_device)) as handle:
        for key in handle.keys():
            matched_prefix = None
            for prefix in prefixes:
                if key.startswith(prefix):
                    matched_prefix = prefix
                    break

            if matched_prefix is None:
                continue

            target_key = _strip_checkpoint_prefix(key, matched_prefix)
            target_tensor = state_entries.get(target_key)
            if target_tensor is None:
                unexpected_keys.append(target_key)
                continue

            source_tensor = handle.get_tensor(key)
            target_dtype = target_tensor.dtype
            if dtype is not None and torch.is_floating_point(target_tensor):
                target_dtype = dtype

            copy_tensor = source_tensor.to(
                device=target_tensor.device,
                dtype=target_dtype,
            )
            target_tensor.copy_(copy_tensor)
            loaded_keys.add(target_key)

    missing_keys = [key for key in state_entries.keys() if key not in loaded_keys]
    return missing_keys, unexpected_keys


def _extract_prefixed_state_dict(source, prefixes, *, device=None):
    if isinstance(source, str) and source.lower().endswith(".safetensors"):
        return _extract_prefixed_safetensors_state_dict(source, prefixes, device=device)

    sd = resolve_source(source)
    extracted = {}
    for key, value in sd.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[_strip_checkpoint_prefix(key, prefix)] = value.to(device=device) if device is not None and hasattr(value, "to") else value
                break
    return extracted


def _module_is_meta(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        device = getattr(tensor, "device", None)
        if device is not None:
            return device.type == "meta"
    return False


def _reload_unet_weights(target_model, source, *, device, dtype=None, prefixes=None):
    if prefixes is not None:
        sd = _extract_prefixed_state_dict(source, prefixes, device=device)
    else:
        sd = resolve_source(source, device=device)

    diffusion_model = target_model.diffusion_model
    if _module_is_meta(diffusion_model) and hasattr(diffusion_model, "to_empty"):
        diffusion_model.to_empty(device=device)
    if dtype is not None:
        diffusion_model.to(device=device, dtype=dtype)
    else:
        diffusion_model.to(device=device)
    diffusion_model.load_state_dict(sd, strict=False)

    del sd
    gc.collect()


def _build_unet_runtime_reload(source, *, dtype=None, prefixes=None):
    if not isinstance(source, str):
        return None

    def _reload(target_model, target_device):
        _reload_unet_weights(
            target_model,
            source,
            device=target_device,
            dtype=dtype,
            prefixes=prefixes,
        )

    return _reload


def _reload_sdxl_clip_weights(
    target_model,
    source_l,
    source_g,
    *,
    device,
    dtype=None,
    prefixes_l=None,
    prefixes_g=None,
):
    if hasattr(target_model, "to"):
        if dtype is None:
            target_model.to(device=device)
        else:
            target_model.to(device=device, dtype=dtype)

    def _resolve_clip_source(source, prefixes):
        if isinstance(source, str) and prefixes is not None:
            return _extract_prefixed_safetensors_state_dict(
                source,
                prefixes,
                device=torch.device("cpu"),
            )
        return resolve_source(source)

    share_source = (
        source_l == source_g
        and prefixes_l == prefixes_g
    )
    sd_l = _resolve_clip_source(source_l, prefixes_l)
    sd_g = sd_l if share_source else _resolve_clip_source(source_g, prefixes_g)

    try:
        if sd_l is not None:
            if isinstance(sd_l, dict) and not any(
                key.startswith("clip") or key.startswith("cond") or "embedders." in key
                for key in sd_l.keys()
            ):
                target_model.load_sd(sd_l, force_type="l")
            else:
                target_model.load_sd(sd_l)
        if sd_g is not None:
            if isinstance(sd_g, dict) and not any(
                key.startswith("clip") or key.startswith("cond") or "embedders." in key
                for key in sd_g.keys()
            ):
                target_model.load_sd(sd_g, force_type="g")
            else:
                target_model.load_sd(sd_g)
    finally:
        if not share_source:
            del sd_g
        del sd_l
        gc.collect()


def _build_sdxl_clip_runtime_reload(
    source_l,
    source_g,
    *,
    dtype=None,
    prefixes_l=None,
    prefixes_g=None,
):
    if source_l is None or source_g is None:
        return None

    def _reload(target_model, target_device):
        _reload_sdxl_clip_weights(
            target_model,
            source_l,
            source_g,
            device=target_device,
            dtype=dtype,
            prefixes_l=prefixes_l,
            prefixes_g=prefixes_g,
        )

    return _reload


def _stream_load_sdxl_unet_from_checkpoint(
    ckpt_path,
    *,
    load_device=None,
    offload_device=None,
    dtype=None,
    reload_source=None,
    reload_prefixes=None,
):
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16

    runtime_reload = _build_unet_runtime_reload(
        reload_source if reload_source is not None else ckpt_path,
        dtype=effective_dtype,
        prefixes=reload_prefixes,
    )

    model = model_base.SDXL(
        model_config=ModelConfig(sdxl_def.UNET_CONFIG, latent_formats.SDXL()),
    )
    model.diffusion_model.to(device=load_device, dtype=effective_dtype)

    missing, unexpected = _load_prefixed_safetensors_into_module(
        ckpt_path,
        reload_prefixes or sdxl_def.PREFIXES["unet"],
        model.diffusion_model,
        device=load_device,
        dtype=effective_dtype,
    )
    if missing:
        logging.debug("SDXL UNet: Missing keys while streaming load: %s", missing)
    if unexpected:
        logging.debug("SDXL UNet: Unexpected keys while streaming load: %s", unexpected)

    return patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        runtime_reload=runtime_reload,
        runtime_release_to_meta=runtime_reload is not None,
    )

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
        self.patcher = patching.NexModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )
        self.layer_idx = None
        self.fcs_cond_cache = {}

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def add_patches(self, patches, weight):
        return self.patcher.add_patches(patches, weight)

    def clone(self):
        n = CLIP(self.cond_stage_model, self.tokenizer, self.patcher.load_device, self.patcher.offload_device)
        n.patcher = self.patcher.clone()
        n.layer_idx = self.layer_idx
        return n

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def _apply_clip_layer_selection(self):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

    def encode_from_tokens_resident(self, tokens, return_pooled=False):
        self._apply_clip_layer_selection()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens(self, tokens, return_pooled=False):
        resources.prepare_models_for_stage(
            [self.patcher],
            stage_name="text_encode",
            target_phase=resources.MemoryPhase.PROMPT_ENCODE,
            force_full_load=True,
        )
        return self.encode_from_tokens_resident(tokens, return_pooled=return_pooled)

class VAE:
    """Isolated VAE container to avoid modules.sd baggage."""
    def __init__(self, first_stage_model, load_device, offload_device, latent_format=None):
        self.first_stage_model = first_stage_model
        # Use SD15 as default if not specified (backward compatibility)
        if latent_format is None:
            logging.debug("VAE: No latent_format provided, defaulting to SD15.")
            latent_format = latent_formats.SD15()
        self.latent_format = latent_format
        self.patcher = patching.NexModelPatcher(
            self.first_stage_model,
            load_device=load_device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(
            self.first_stage_model,
            self.patcher.load_device,
            self.patcher.offload_device,
            latent_format=self.latent_format,
        )
        n.patcher = self.patcher.clone()
        return n

    def decode(self, samples, tiled=False, tile_size=64):
        from . import decode
        return decode.decode_latent(self, samples, tiled=tiled, tile_size=tile_size)

    def encode(self, pixels):
        from . import encode
        return encode.encode_pixels(self, pixels)

# --- SDXL Support ---

def load_sdxl_unet(
    source,
    load_device=None,
    offload_device=None,
    dtype=None,
    reload_source=None,
    reload_prefixes=None,
    *,
    execution_class=None,
):
    """
    Loads the SDXL UNet using sdxl_def.UNET_CONFIG.
    Supports .gguf integration.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16
    
    custom_operations = None
    patcher_class = patching.NexModelPatcher
    runtime_reload = None

    if isinstance(source, str) and source.endswith(".gguf"):
        from backend.gguf.loader import gguf_sd_loader, is_streaming_execution_class
        from backend.gguf.ops import GGMLOps
        from backend.gguf.patcher import GGUFModelPatcher

        streaming = is_streaming_execution_class(execution_class)
        if streaming:
            load_device = torch.device("cpu") if load_device is None else torch.device(load_device)
            offload_device = torch.device("cpu") if offload_device is None else torch.device(offload_device)
            if load_device.type != "cpu" or offload_device.type != "cpu":
                raise RuntimeError("Streaming-class SDXL GGUF loads must stage weights on CPU pinned host memory.")
        sd = gguf_sd_loader(source, pin_memory=streaming, execution_class=execution_class, require_pinned_host=streaming)
        custom_operations = GGMLOps
        patcher_class = GGUFModelPatcher
    else:
        sd = resolve_source(source, device=load_device)
        runtime_reload = _build_unet_runtime_reload(
            reload_source if reload_source is not None else source,
            dtype=effective_dtype,
            prefixes=reload_prefixes,
        )

    model = model_base.SDXL(
        model_config=ModelConfig(sdxl_def.UNET_CONFIG, latent_formats.SDXL()),
        operations=custom_operations
    )
    
    # User Requirement: SDXL UNet should be in fp16 to avoid casting to fp32 (saving RAM/VRAM)
    dtype = effective_dtype
        
    if dtype is not None:
        model.diffusion_model.to(device=load_device, dtype=dtype)
    else:
        model.diffusion_model.to(device=load_device)
        
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    patcher_kwargs = {
        "load_device": load_device,
        "offload_device": offload_device,
        "runtime_reload": runtime_reload,
        "runtime_release_to_meta": runtime_reload is not None,
    }
    if isinstance(source, str) and source.endswith(".gguf"):
        patcher_kwargs["preserve_source_artifact"] = is_streaming_execution_class(execution_class)
    return patcher_class(model, **patcher_kwargs)

def patch_unet_for_quality(unet_patcher: Any, quality: Dict[str, Any]):
    """
    Monkey-patches the UNet's forward pass to support Timed ADM.
    """
    if not quality:
        return
        
    unet = unet_patcher.model.diffusion_model
    if hasattr(unet, "_nex_quality_patched"):
        return
    unet._nex_quality_patched = True

    adm_scaler_end = quality.get("adm_scaler_end", 0.3)
    
    original_forward = unet.forward
    
    def nex_patched_forward(x, timesteps, context=None, y=None, control=None, transformer_options={}, **kwargs):
        # Prevent per-layer upcasting slowness (~3-4x penalty on Windows/NVIDIA)
        # model_base.apply_model() does NOT cast everything correctly on all paths.
        from backend import precision
        x, timesteps, context, y, control = precision.cast_unet_inputs(
            x, timesteps, context=context, y=y, control=control, weight_dtype=unet.dtype
        )

        if y is not None:
             # timed_adm(y, timestep, model, adm_scaler_end)
             y = conditioning.timed_adm(y, timesteps, unet_patcher.model, adm_scaler_end=adm_scaler_end)
             
        return original_forward(x, timesteps, context=context, y=y, control=control, transformer_options=transformer_options, **kwargs)
        
    unet.forward = nex_patched_forward
    logging.info(f"[Nex] Quality: UNet patched for Timed ADM (scaler_end={adm_scaler_end})")

def patch_controlnet_for_quality(controlnet: Any, quality: Dict[str, Any]):
    """
    Monkey-patches ControlNet's forward pass to support Timed ADM and Softness.
    Accepts either a raw ControlNet module or a backend wrapper object.
    """
    if not quality:
        return

    target = getattr(controlnet, "control_model", controlnet)
    if target is None or not hasattr(target, "forward"):
        setattr(controlnet, "_nex_pending_quality", dict(quality))
        return

    if hasattr(target, "_nex_quality_patched"):
        return
    target._nex_quality_patched = True

    controlnet_softness = quality.get("controlnet_softness", 0.0)
    original_forward = target.forward

    def nex_patched_forward(x, hint, timesteps, context, y=None, **kwargs):
        if y is not None:
            y = conditioning.timed_adm(y, timesteps, target, adm_scaler_end=quality.get("adm_scaler_end", 0.3))

        outs = original_forward(x, hint, timesteps, context, y=y, **kwargs)

        if controlnet_softness > 0 and isinstance(outs, list):
            for i in range(len(outs)):
                k = 1.0 - float(i) / (len(outs) - 1) if len(outs) > 1 else 1.0
                outs[i] = outs[i] * (1.0 - controlnet_softness * k)
        return outs

    target.forward = nex_patched_forward
    logging.info(f"[Nex] Quality: ControlNet patched (softness={controlnet_softness})")

def load_sdxl_clip(
    source_l,
    source_g,
    load_device=None,
    offload_device=None,
    dtype=None,
    *,
    reload_source_l=None,
    reload_source_g=None,
    reload_prefixes_l=None,
    reload_prefixes_g=None,
):
    """
    Loads SDXL CLIP (L and G) and returns a clean CLIP container.
    """
    load_device = load_device or resources.text_encoder_load_device()
    offload_device = offload_device or resources.text_encoder_offload_device()

    same_source = source_g is source_l
    if not same_source and isinstance(source_l, str) and isinstance(source_g, str):
        # The app/runtime path passes the same bundled SDXL CLIP file twice.
        # Keep this cheap fast path so we do not resolve and load it twice.
        same_source = source_l == source_g

    sd_l = resolve_source(source_l)
    sd_g = sd_l if same_source else resolve_source(source_g)
    
    # Use Nex implementations
    tokenizer = clip.NexSDXLTokenizer()
    
    # SDXL CLIP should stay resident in fp32 so CPU/GPU prompt encode does not
    # repeatedly upcast fp16 weights to match fp32 activations at runtime.
    if dtype is None:
        dtype = torch.float32 
        
    model = clip.NexSDXLClipModel(device=offload_device, dtype=dtype)
    
    with torch.no_grad():
        # NexSDXLClipModel.load_sd handles L and G appropriately
        # If they are different dicts, we load each. 
        # If they are the same (bundled), it still works.
        if sd_l is not None:
            if isinstance(sd_l, dict) and not any(k.startswith("clip") or k.startswith("cond") or "embedders." in k for k in sd_l.keys()):
                model.load_sd(sd_l, force_type="l")
            else:
                model.load_sd(sd_l)
        if sd_g is not None and sd_g is not sd_l:
            if isinstance(sd_g, dict) and not any(k.startswith("clip") or k.startswith("cond") or "embedders." in k for k in sd_g.keys()):
                model.load_sd(sd_g, force_type="g")
            else:
                model.load_sd(sd_g)
    
    clip_container = CLIP(model, tokenizer, load_device, offload_device)
    effective_reload_source_l = (
        reload_source_l
        if reload_source_l is not None
        else (source_l if isinstance(source_l, str) else None)
    )
    effective_reload_source_g = (
        reload_source_g
        if reload_source_g is not None
        else (source_g if isinstance(source_g, str) else None)
    )
    clip_container.patcher.runtime_reload = _build_sdxl_clip_runtime_reload(
        effective_reload_source_l,
        effective_reload_source_g,
        dtype=dtype,
        prefixes_l=reload_prefixes_l,
        prefixes_g=reload_prefixes_g,
    )
    clip_container.patcher.runtime_release_to_meta = False
    return clip_container

def load_vae(source, load_device=None, offload_device=None, dtype=None, latent_format=None):
    """
    Loads VAE/AE (SD15/SDXL/Flux compatible) and returns a clean VAE container.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.vae_offload_device()

    # Try to infer latent format from filename if not provided.
    if latent_format is None and isinstance(source, str):
        normalized_source = source.replace("\\", "/").lower()
        if "/flux_fill/" in normalized_source or "/flux/" in normalized_source:
            latent_format = latent_formats.Flux()
            logging.info(f"VAE: Inferred Flux latent format from {source}")

    if latent_format is None and isinstance(source, str):
        from modules import model_taxonomy
        arch = model_taxonomy.infer_architecture_from_filename(source)
        if arch == model_taxonomy.ARCHITECTURE_SDXL:
            latent_format = latent_formats.SDXL()
            logging.info(f"VAE: Inferred SDXL latent format from {source}")
        elif arch == model_taxonomy.ARCHITECTURE_SD15:
            latent_format = latent_formats.SD15()
            logging.info(f"VAE: Inferred SD15 latent format from {source}")

    sd = resolve_source(source, device=load_device)

    if latent_format is None and "decoder.conv_in.weight" in sd:
        latent_channels = sd["decoder.conv_in.weight"].shape[1]
        if latent_channels == latent_formats.Flux.latent_channels:
            latent_format = latent_formats.Flux()
            logging.info("VAE: Inferred Flux latent format from 16-channel AE state dict")

    # Generic VAE config; derive latent/embed width from the state dict for Flux AE.
    ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
    if "decoder.conv_in.weight" in sd:
        ddconfig["z_channels"] = sd["decoder.conv_in.weight"].shape[1]
        if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd:
            ddconfig['ch_mult'] = [1, 2, 4]

    if "post_quant_conv.weight" in sd:
        model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1])
    elif "decoder.conv_in.weight" in sd:
        model = AutoencodingEngine(
            regularizer_config={'target': "ldm_patched.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
            encoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
            decoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Decoder", 'params': ddconfig},
        )
    else:
        model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logging.debug("VAE: Missing keys while loading: %s", missing)
    if unexpected:
        logging.debug("VAE: Unexpected keys while loading: %s", unexpected)

    # User Requirement: VAE should be in fp32
    if dtype is None:
        dtype = torch.float32

    if dtype is not None:
        model.to(dtype)

    return VAE(model.eval(), load_device, offload_device, latent_format=latent_format)


def load_sdxl_checkpoint(
    ckpt_path,
    load_device=None,
    offload_device=None,
    unet_dtype=None,
    *,
    clip_load_device=None,
    clip_offload_device=None,
    vae_load_device=None,
    vae_offload_device=None,
    vae_source=None,
):
    """
    Loads SDXL components sequentially and clears raw data immediately.
    """
    logging.info(f"Loading SDXL checkpoint from: {ckpt_path}")
    load_device = load_device or resources.get_torch_device()
    clip_dtype = torch.float32
    external_vae_source = vae_source
    vae_load_device = vae_load_device or vae_offload_device or resources.vae_offload_device()

    if isinstance(ckpt_path, str) and ckpt_path.lower().endswith(".safetensors"):
        if external_vae_source is not None:
            logging.info("Loading external SDXL VAE override instead of checkpoint VAE...")
            vae = load_vae(
                external_vae_source,
                load_device=vae_load_device,
                offload_device=vae_offload_device,
                latent_format=latent_formats.SDXL(),
            )
            gc.collect()
        else:
            logging.info("Extracting VAE from safetensors checkpoint...")
            vae_sd = _extract_prefixed_safetensors_state_dict(
                ckpt_path,
                sdxl_def.PREFIXES["vae"],
                device=torch.device("cpu"),
            )
            if len(vae_sd) > 0:
                vae = load_vae(
                    vae_sd,
                    load_device=vae_load_device,
                    offload_device=vae_offload_device,
                    latent_format=latent_formats.SDXL(),
                )
            else:
                logging.warning("SDXL checkpoint is missing embedded VAE weights; continuing without checkpoint VAE.")
                vae = None
            del vae_sd
            gc.collect()

        logging.info("Extracting CLIP from safetensors checkpoint...")
        clip_l_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sdxl_def.PREFIXES["clip_l"],
            device=torch.device("cpu"),
        )
        clip_g_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sdxl_def.PREFIXES["clip_g"],
            device=torch.device("cpu"),
        )
        clip = load_sdxl_clip(
            clip_l_sd,
            clip_g_sd,
            load_device=clip_load_device,
            offload_device=clip_offload_device,
            dtype=clip_dtype,
            reload_source_l=ckpt_path,
            reload_source_g=ckpt_path,
            reload_prefixes_l=sdxl_def.PREFIXES["clip_l"],
            reload_prefixes_g=sdxl_def.PREFIXES["clip_g"],
        )
        del clip_l_sd
        del clip_g_sd
        gc.collect()

        logging.info("Extracting UNet from safetensors checkpoint directly to load device...")
        unet = _stream_load_sdxl_unet_from_checkpoint(
            ckpt_path,
            load_device=load_device,
            offload_device=offload_device,
            dtype=unet_dtype,
            reload_source=ckpt_path,
            reload_prefixes=sdxl_def.PREFIXES["unet"],
        )
        gc.collect()
        return unet, clip, vae

    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32 default)
    if external_vae_source is not None:
        logging.info("Loading external SDXL VAE override instead of checkpoint VAE...")
        vae = load_vae(
            external_vae_source,
            load_device=vae_load_device,
            offload_device=vae_offload_device,
            latent_format=latent_formats.SDXL(),
        )
        gc.collect()
    else:
        logging.info("Extracting VAE...")
        vae_sd = {}
        keys = list(sd.keys())
        for k in keys:
            for p in sdxl_def.PREFIXES["vae"]:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."): new_key = new_key[1:]
                    vae_sd[new_key] = sd.pop(k)
                    break
        
        if len(vae_sd) > 0:
            vae = load_vae(
                vae_sd,
                load_device=vae_load_device,
                offload_device=vae_offload_device,
                latent_format=latent_formats.SDXL(),
            )
        else:
            logging.warning("SDXL checkpoint is missing embedded VAE weights; continuing without checkpoint VAE.")
            vae = None
        del vae_sd
        gc.collect()
    
    # CLIP (fp16 default)
    logging.info("Extracting CLIP...")
    clip_l_sd = {}
    clip_g_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["clip_l"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_l_sd[new_key] = sd.pop(k)
                break
        
        if k in sd: 
            for p in sdxl_def.PREFIXES["clip_g"]:
                if k.startswith(p):
                    new_key = k[len(p):]
                    if new_key.startswith("."): new_key = new_key[1:]
                    clip_g_sd[new_key] = sd.pop(k)
                    break
                    
    clip = load_sdxl_clip(
        clip_l_sd,
        clip_g_sd,
        load_device=clip_load_device,
        offload_device=clip_offload_device,
        dtype=clip_dtype,
        reload_source_l=ckpt_path,
        reload_source_g=ckpt_path,
        reload_prefixes_l=sdxl_def.PREFIXES["clip_l"],
        reload_prefixes_g=sdxl_def.PREFIXES["clip_g"],
    )
    del clip_l_sd
    del clip_g_sd
    gc.collect()

    # UNet (fp16 default)
    logging.info("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sdxl_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k)
                break
    
    if len(sd) > 0:
        logging.info(f"Remaining keys in checkpoint: {len(sd)}")
    
    logging.debug("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    logging.info("Loading UNet Model...")
    unet = load_sdxl_unet(
        unet_sd,
        load_device=load_device,
        offload_device=offload_device,
        dtype=unet_dtype,
        reload_source=ckpt_path,
        reload_prefixes=sdxl_def.PREFIXES["unet"],
    )
    del unet_sd
    gc.collect()
    
    return unet, clip, vae

# --- SD 1.5 Support ---

def load_sd15_unet(source, load_device=None, offload_device=None, dtype=None, reload_source=None, reload_prefixes=None):
    """
    Loads the SD 1.5 UNet using sd15_def.UNET_CONFIG.
    """
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16
    
    sd = resolve_source(source, device=load_device)
    runtime_reload = _build_unet_runtime_reload(
        reload_source if reload_source is not None else source,
        dtype=effective_dtype,
        prefixes=reload_prefixes,
    )

    model = model_base.BaseModel(
        model_config=ModelConfig(sd15_def.UNET_CONFIG, latent_formats.SD15()),
    )
    
    # User Requirement: SD1.5 UNet should be in fp16
    dtype = effective_dtype
        
    model.diffusion_model.to(device=load_device, dtype=dtype)
    model.diffusion_model.load_state_dict(sd, strict=False)
    
    return patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        runtime_reload=runtime_reload,
        runtime_release_to_meta=runtime_reload is not None,
    )

def load_sd15_clip(source, load_device=None, offload_device=None, dtype=None):
    """
    Loads SD 1.5 CLIP (L only) and returns a clean CLIP container.
    """
    load_device = load_device or resources.text_encoder_load_device()
    offload_device = offload_device or resources.text_encoder_offload_device()
    
    sd = resolve_source(source)
    
    tokenizer, encoder = clip.create_sd15_clip(sd)
    
    # NexClipEncoder implements the cond_stage_model interface directly.
    return CLIP(encoder, tokenizer, load_device, offload_device)

def load_sd15_checkpoint(ckpt_path, load_device=None, unet_dtype=None):
    """
    Loads SD 1.5 components sequentially.
    """
    logging.info(f"Loading SD 1.5 checkpoint from: {ckpt_path}")
    load_device = load_device or resources.get_torch_device()

    if isinstance(ckpt_path, str) and ckpt_path.lower().endswith(".safetensors"):
        logging.info("Extracting VAE from safetensors checkpoint...")
        vae_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["vae"],
            device=torch.device("cpu"),
        )
        vae = load_vae(vae_sd, latent_format=latent_formats.SD15())
        del vae_sd
        gc.collect()

        logging.info("Extracting CLIP from safetensors checkpoint...")
        clip_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["clip"],
            device=torch.device("cpu"),
        )
        clip = load_sd15_clip(clip_sd, dtype=unet_dtype)
        heal_model_weights(clip.patcher.model, "CLIP")
        del clip_sd
        gc.collect()

        # Precision Injection (SD1.5 specific fix for NaN overflows)
        try:
            sd1_clip_model = clip.cond_stage_model
            transformer = None

            if hasattr(sd1_clip_model, 'transformer'):
                 transformer = sd1_clip_model.transformer
            elif hasattr(sd1_clip_model, 'clip_l'):
                 transformer = sd1_clip_model.clip_l.transformer
            elif hasattr(sd1_clip_model, 'clip'):
                 transformer = sd1_clip_model.clip.transformer

            if transformer is not None and hasattr(transformer, 'text_model'):
                 transformer = transformer.text_model

            if transformer is not None and hasattr(transformer, 'embeddings'):
               embeddings = transformer.embeddings
               if not isinstance(embeddings, EmbeddingFP32Wrapper):
                   transformer.embeddings = EmbeddingFP32Wrapper(embeddings)
        except Exception as e:
            logging.error(f"FAILED to apply precision injection to CLIP: {e}")

        logging.info("Extracting UNet from safetensors checkpoint directly to load device...")
        unet_sd = _extract_prefixed_safetensors_state_dict(
            ckpt_path,
            sd15_def.PREFIXES["unet"],
            device=load_device,
        )
        unet = load_sd15_unet(
            unet_sd,
            load_device=load_device,
            dtype=unet_dtype,
            reload_source=ckpt_path,
            reload_prefixes=sd15_def.PREFIXES["unet"],
        )
        heal_model_weights(unet.model, "UNet")
        del unet_sd
        gc.collect()
        return unet, clip, vae

    sd = utils.load_torch_file(ckpt_path)
    gc.collect()

    # VAE (fp32)
    logging.info("Extracting VAE...")
    vae_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["vae"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                vae_sd[new_key] = sd.pop(k)
                break
    
    vae = load_vae(vae_sd, latent_format=latent_formats.SD15())
    del vae_sd
    gc.collect()
    
    # CLIP (fp16)
    logging.info("Extracting CLIP...")
    clip_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["clip"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                clip_sd[new_key] = sd.pop(k)
                break
                     
    clip = load_sd15_clip(clip_sd, dtype=unet_dtype)
    heal_model_weights(clip.patcher.model, "CLIP")

    # Precision Injection (SD1.5 specific fix for NaN overflows)
    try:
        sd1_clip_model = clip.cond_stage_model
        transformer = None
        
        if hasattr(sd1_clip_model, 'transformer'): 
             transformer = sd1_clip_model.transformer
        elif hasattr(sd1_clip_model, 'clip_l'): 
             transformer = sd1_clip_model.clip_l.transformer
        elif hasattr(sd1_clip_model, 'clip'):
             transformer = sd1_clip_model.clip.transformer
             
        if transformer is not None and hasattr(transformer, 'text_model'):
             transformer = transformer.text_model

        if transformer is not None and hasattr(transformer, 'embeddings'):
           embeddings = transformer.embeddings
           if not isinstance(embeddings, EmbeddingFP32Wrapper):
               transformer.embeddings = EmbeddingFP32Wrapper(embeddings)
    except Exception as e:
        logging.error(f"FAILED to apply precision injection to CLIP: {e}")

    del clip_sd
    gc.collect()

    # UNet (fp16)
    logging.info("Extracting UNet...")
    unet_sd = {}
    keys = list(sd.keys())
    for k in keys:
        for p in sd15_def.PREFIXES["unet"]:
            if k.startswith(p):
                new_key = k[len(p):]
                if new_key.startswith("."): new_key = new_key[1:]
                unet_sd[new_key] = sd.pop(k)
                break
    
    if len(sd) > 0:
        logging.info(f"Remaining keys in checkpoint: {len(sd)}")
    
    logging.debug("Deleting original checkpoint storage...")
    del sd
    gc.collect() 

    logging.info("Loading UNet Model...")
    unet = load_sd15_unet(
        unet_sd,
        load_device=load_device,
        dtype=unet_dtype,
        reload_source=ckpt_path,
        reload_prefixes=sd15_def.PREFIXES["unet"],
    )
    heal_model_weights(unet.model, "UNet")
    del unet_sd
    gc.collect()
    
    return unet, clip, vae
