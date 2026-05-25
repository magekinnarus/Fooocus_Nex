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
    runtime_policy = getattr(vae, "runtime_policy", None)
    gpu_preferred = bool(getattr(runtime_policy, "prefer_gpu_vae_encode", False))
    device = torch.device("cpu")
    dtype = torch.float32
    output_device = "cpu"

    # Normalize and move dim.
    pixels = _process_input(pixels)

    # Estimate memory usage using CPU available memory because encode is CPU-bound.
    memory_used = (VAE_ENCODE_MEMORY_STRICT * pixels.shape[2] * pixels.shape[3]) * utils.dtype_size(dtype)
    if gpu_preferred:
        resources.prepare_models_for_stage(
            [vae.patcher],
            stage_name="vae_encode",
            target_phase=resources.MemoryPhase.VAE_ENCODE,
            memory_required=memory_used,
        )
        device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
    else:
        # Keep the shared residency boundary authoritative. If the VAE is still
        # resident on a non-CPU device, release it through the shared helper first.
        try:
            if vae.patcher.current_loaded_device() != device:
                resources.eject_model(vae.patcher)
        except Exception:
            pass

    first_stage_model = vae.first_stage_model
    live_param = next(first_stage_model.parameters(), None)
    if isinstance(live_param, torch.Tensor):
        device = live_param.device
        dtype = live_param.dtype

    free_memory = resources.get_free_memory(device)
    batch_number = int(free_memory / max(1, memory_used))
    batch_number = max(1, batch_number)

    latents = []
    for x in range(0, pixels.shape[0], batch_number):
        batch = pixels[x : x + batch_number].to(device=device, dtype=dtype)
        latent = first_stage_model.encode(batch)
        if hasattr(latent, "sample"):
            latent = latent.sample()

        latents.append(latent.to(output_device))

    output = torch.cat(latents, dim=0)

    # Apply latent scaling (e.g., 0.18215 for SD1.5)
    output = vae.latent_format.process_in(output)

    return {"samples": output}
