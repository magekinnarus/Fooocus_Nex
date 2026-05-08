import torch

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
    device = torch.device("cpu")
    dtype = torch.float32
    output_device = "cpu"

    # Normalize and move dim.
    pixels = _process_input(pixels)

    # Keep patcher bookkeeping synchronized if VAE was previously resident on GPU.
    if vae.patcher.current_loaded_device() != device:
        vae.patcher.detach()

    # Estimate memory usage using CPU available memory because encode is CPU-bound.
    memory_used = (VAE_ENCODE_MEMORY_STRICT * pixels.shape[2] * pixels.shape[3]) * utils.dtype_size(dtype)
    free_memory = resources.get_free_memory(device)
    batch_number = int(free_memory / max(1, memory_used))
    batch_number = max(1, batch_number)

    # Keep VAE encode on CPU so GPU headroom stays available for UNet.
    vae.first_stage_model.to(device=device, dtype=dtype)

    latents = []
    for x in range(0, pixels.shape[0], batch_number):
        batch = pixels[x : x + batch_number].to(device=device, dtype=dtype)
        latent = vae.first_stage_model.encode(batch)
        if hasattr(latent, "sample"):
            latent = latent.sample()

        latents.append(latent.to(output_device))

    output = torch.cat(latents, dim=0)

    # Apply latent scaling (e.g., 0.18215 for SD1.5)
    output = vae.latent_format.process_in(output)

    return {"samples": output}
