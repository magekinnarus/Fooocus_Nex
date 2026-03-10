import torch
import logging
from . import resources
from . import utils

# Memory estimation constant (conservative)
VAE_ENCODE_MEMORY_STRICT = 1024 

def _process_input(image):
    """Normalizes pixel input from [0, 1] to [-1, 1] and CHW format."""
    return (image.movedim(-1, 1) * 2.0) - 1.0

def encode_pixels(vae, pixels):
    """
    Encodes pixel tensor into latent space.
    
    Args:
        vae: VAE container from loader.py
        pixels: Pixel tensor [B, H, W, 3], float32 [0, 1]
        
    Returns:
        Dict: {'samples': Latent tensor [B, 4, H//8, W//8]}
    """
    device = vae.patcher.load_device
    dtype = next(vae.first_stage_model.parameters()).dtype
    output_device = "cpu"

    # Normalize and Move dim
    pixels = _process_input(pixels)
    
    # Estimate memory usage
    memory_used = (VAE_ENCODE_MEMORY_STRICT * pixels.shape[2] * pixels.shape[3]) * utils.dtype_size(dtype)
    
    # Ensure VAE is on GPU
    resources.load_models_gpu([vae.patcher], memory_required=memory_used)
    
    # We use float32 for VAE operations to prevent NaN/Inf in some VAEs
    vae.first_stage_model.to(device=device, dtype=torch.float32)
    dtype = torch.float32
    
    free_memory = resources.get_free_memory(device)
    batch_number = int(free_memory / max(1, memory_used))
    batch_number = max(1, batch_number)
    
    latents = []
    for x in range(0, pixels.shape[0], batch_number):
        batch = pixels[x:x+batch_number].to(device=device, dtype=dtype)
        latent = vae.first_stage_model.encode(batch)
        if hasattr(latent, "sample"):
            latent = latent.sample()
        
        latents.append(latent.to(output_device))
        
    output = torch.cat(latents, dim=0)
    
    # Apply latent scaling (e.g., 0.18215 for SD1.5)
    output = vae.latent_format.process_in(output)
    
    # Offload VAE from VRAM — it must not compete with UNet during inference.
    # It will be reloaded automatically by resources.load_models_gpu when needed for decode.
    resources.eject_model(vae.patcher)
    
    return {'samples': output}
