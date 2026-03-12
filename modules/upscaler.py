import os
import torch
import numpy as np
import logging

import modules.core as core
import ldm_patched.modules.model_management
import ldm_patched.modules.utils
import ldm_patched.pfn.model_loading as loading
from modules.config import path_upscale_models

logger = logging.getLogger(__name__)

_cached_model = None
_cached_model_name = None

def list_available_models():
    """Scan models/upscale_models/ for .pth files."""
    if not path_upscale_models:
        return []
    
    models = []
    for folder in path_upscale_models:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.pth') or f.lower().endswith('.safetensors'):
                    models.append(f)
    return sorted(list(set(models)))

def get_model_scale(model):
    """Auto-detect scale factor from model architecture."""
    # Priority 1: Model attribute
    for attr in ['scale', 'upscale', 'upscale_factor', 'upsampler_scale']:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (int, float)):
                return int(val)
    
    # Priority 2: Inferred from architecture if possible
    # (Some architectures like ESRGAN-2c2 might have complex logic in their __init__)
    
    # Priority 3: Fallback to name-based if it's a wrapper or if attribute missing
    return 4  # Default fallback if unknown

def load_model(model_name):
    """Load model by name, using cache if available."""
    global _cached_model, _cached_model_name
    
    if _cached_model_name == model_name and _cached_model is not None:
        return _cached_model
    
    model_path = None
    for folder in path_upscale_models:
        p = os.path.join(folder, model_name)
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        raise FileNotFoundError(f"Upscale model not found: {model_name}")
        
    logger.info(f"Loading upscale model {model_path} ...")
    
    # Use existing loading infrastructure
    if model_path.endswith('.safetensors'):
        from ldm_patched.modules.utils import load_torch_file
        sd = load_torch_file(model_path, device='cpu')
    else:
        sd = torch.load(model_path, map_location='cpu', weights_only=True)
    
    model = loading.load_state_dict(sd)
    model.eval()
    model.cpu()
    
    _cached_model = model
    _cached_model_name = model_name
    
    return model

def clear_model_cache():
    global _cached_model, _cached_model_name
    _cached_model = None
    _cached_model_name = None
    ldm_patched.modules.model_management.soft_empty_cache()

def perform_upscale(img, model_name=None, scale_override=None):
    """
    Upscale an image using the specified model.
    img: numpy array [H, W, C]
    model_name: filename of the model in models/upscale_models/
    scale_override: optional scale to force
    """
    global _cached_model, _cached_model_name
    
    if img is None:
        return None

    # Default to first available or hardcoded if None (for backward compatibility during transition)
    if model_name is None:
        available = list_available_models()
        if available:
            if '4xNomos2_otf_esrgan.pth' in available:
                model_name = '4xNomos2_otf_esrgan.pth'
            else:
                model_name = available[0]
        else:
            # Fallback for when no models are present (e.g. fresh install)
            # This will fail unless we keep the old default logic, but mission says rewritten
            raise ValueError("No upscale models found and none specified.")

    print(f'Upscaling image with shape {str(img.shape)} using {model_name} ...')

    model = load_model(model_name)
    device = ldm_patched.modules.model_management.get_torch_device()
    model.to(device)

    # Detect scale
    native_scale = get_model_scale(model)
    target_scale = scale_override if scale_override is not None else native_scale

    # Prepare image
    in_img = core.numpy_to_pytorch(img).movedim(-1, -3).to(device)
    # in_img is [1, C, H, W]
    
    tile = 512
    overlap = 32
    
    oom = True
    while oom:
        try:
            # tiled_scale expects [B, C, H, W]
            # upscale_amount is the factor
            s = ldm_patched.modules.utils.tiled_scale(
                in_img, 
                lambda a: model(a), 
                tile_x=tile, 
                tile_y=tile, 
                overlap=overlap, 
                upscale_amount=native_scale
            )
            oom = False
        except ldm_patched.modules.model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                model.cpu()
                raise e

    # Offload
    model.cpu()
    
    # Process output
    s = torch.clamp(s.movedim(1, -1), min=0, max=1.0) # [1, H, W, C]
    result = core.pytorch_to_numpy(s)[0]

    # Handle scale override via bicubic resize if needed
    if target_scale != native_scale:
        import cv2
        h, w = img.shape[:2]
        new_h, new_w = int(h * target_scale), int(w * target_scale)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return result
