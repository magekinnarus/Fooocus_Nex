import logging

import torch

from . import resources
from . import utils

# Estimation constant from ComfyUI for AutoencoderKL
VAE_DECODE_MEMORY_STRICT = 2178


def _process_output(image):
    """Normalizes VAE output to [0, 1] range."""
    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


def _decode_memory_required(latent, dtype):
    return (VAE_DECODE_MEMORY_STRICT * latent.shape[2] * latent.shape[3] * 64) * utils.dtype_size(dtype)


def _should_force_fp32_vae_decode(vae) -> bool:
    latent_format = getattr(vae, "latent_format", None)
    latent_channels = getattr(latent_format, "latent_channels", None)
    if latent_channels == 16:
        return True
    if str(getattr(latent_format, "taesd_decoder_name", "") or "").strip().lower() == "taef1_decoder":
        return False
    return True


def _decode_tiled(vae, samples, tile_x=64, tile_y=64, overlap=16, min_free_mem=0):
    """
    Decodes in tiles to save VRAM.
    Uses ComfyUI's 3-pass averaged tiling to prevent artifacts.
    """
    dtype = next(vae.first_stage_model.parameters()).dtype
    output_device = "cpu"

    upscale_ratio = 8

    logging.info(f"VAE Tiled Decoding: tile_size={tile_x}, overlap={overlap}")

    memory_used = (VAE_DECODE_MEMORY_STRICT * tile_x * tile_y * 64) * utils.dtype_size(dtype)
    force_full_load = _should_force_fp32_vae_decode(vae)
    resources.prepare_models_for_stage(
        [vae.patcher],
        stage_name="vae_decode_tiled",
        target_phase=resources.MemoryPhase.DECODE,
        memory_required=memory_used,
        minimum_memory_required=min_free_mem,
        force_full_load=force_full_load,
    )
    device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
    if _should_force_fp32_vae_decode(vae):
        vae.first_stage_model.to(device=device, dtype=torch.float32)
    dtype = next(vae.first_stage_model.parameters()).dtype

    decode_fn = lambda a: vae.first_stage_model.decode(a.to(dtype).to(device)).float()

    p3 = utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=upscale_ratio, output_device=output_device)
    p1 = utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=upscale_ratio, output_device=output_device)
    p2 = utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=upscale_ratio, output_device=output_device)

    output = _process_output((p1 + p2 + p3) / 3.0)
    return output


def _decode_cpu_fallback(vae, latent, tile_size=64):
    """
    Decode on the shared CPU boundary when GPU co-residency is not viable.
    """
    resources.eject_model(vae.patcher)
    return decode_preloaded_vae(vae, latent, tiled=False, tile_size=tile_size)


def decode_preloaded_vae(vae, latent, tiled=False, tile_size=64):
    """
    Decode using a VAE that the caller has already attached/placed.
    No implicit load/eject behavior is performed here.
    """
    latent = vae.latent_format.process_out(latent)

    if tiled:
        pixel_samples = _decode_tiled(vae, latent, tile_x=tile_size, tile_y=tile_size)
    else:
        device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
        if _should_force_fp32_vae_decode(vae):
            vae.first_stage_model.to(device=device, dtype=torch.float32)
            dtype = torch.float32
        else:
            dtype = next(vae.first_stage_model.parameters()).dtype
        output_device = "cpu"
        memory_used = _decode_memory_required(latent, dtype)



        free_memory = resources.get_free_memory(device)
        batch_number = int(free_memory / max(1, memory_used))
        batch_number = max(1, batch_number)

        pixel_samples = None
        for x in range(0, latent.shape[0], batch_number):
            batch = latent[x:x + batch_number].to(device=device, dtype=dtype)
            out = _process_output(vae.first_stage_model.decode(batch).to(output_device).float())

            if pixel_samples is None:
                pixel_samples = torch.empty((latent.shape[0],) + tuple(out.shape[1:]), device=output_device)

            pixel_samples[x:x + batch_number] = out

    return pixel_samples.movedim(1, -1)


def decode_latent(vae, latent, tiled=False, tile_size=64):
    """
    Compatibility wrapper that manages VAE residency before calling decode_preloaded_vae.
    """
    runtime_policy = getattr(vae, "runtime_policy", None)
    vae_encode_mode = getattr(runtime_policy, "vae_encode_mode", None)
    is_transient = (vae_encode_mode == "transient_gpu")
    is_resident = (vae_encode_mode in {"gpu_resident", "gpu_preferred"})

    patcher = getattr(vae, "patcher", None)
    configured_device = getattr(patcher, "load_device", torch.device("cpu")) if patcher is not None else torch.device("cpu")
    if not isinstance(configured_device, torch.device):
        configured_device = torch.device(configured_device)
    current_loaded_device = getattr(patcher, "current_loaded_device", lambda: torch.device("cpu"))() if patcher is not None else torch.device("cpu")
    if not isinstance(current_loaded_device, torch.device):
        current_loaded_device = torch.device(current_loaded_device)

    gpu_preferred = bool(
        getattr(runtime_policy, "prefer_gpu_vae_encode", False)
        or is_transient
        or is_resident
        or configured_device.type == "cuda"
        or current_loaded_device.type == "cuda"
    )

    dtype = next(vae.first_stage_model.parameters()).dtype
    memory_used = _decode_memory_required(latent, dtype)

    try:
        resources.soft_empty_cache(force=True)

        force_full_load = _should_force_fp32_vae_decode(vae)
        resources.prepare_models_for_stage(
            [vae.patcher],
            stage_name="vae_decode",
            target_phase=resources.MemoryPhase.DECODE,
            memory_required=memory_used,
            force_full_load=force_full_load,
        )
        active_device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
        
        # Ensure model parameters are on active_device
        first_stage_model = vae.first_stage_model
        move_model = getattr(first_stage_model, "to", None)
        if callable(move_model):
            if _should_force_fp32_vae_decode(vae):
                move_model(device=active_device, dtype=torch.float32)
            else:
                move_model(device=active_device)

        active_free_memory = resources.get_free_memory(active_device)
        use_tiled_decode = tiled or (active_device.type == "cuda" and active_free_memory < memory_used)
        pixel_samples = decode_preloaded_vae(vae, latent, tiled=use_tiled_decode, tile_size=tile_size)
    except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
        logging.warning("VAE decode OOM. Retrying with tiled decoding.")
        try:
            pixel_samples = _decode_tiled(
                vae,
                vae.latent_format.process_out(latent),
                tile_x=tile_size,
                tile_y=tile_size,
                min_free_mem=1024 * 1024 * 1024,
            ).movedim(1, -1)
        except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
            logging.warning("VAE tiled decode (64x64) OOM. Retrying with smaller tiles (32x32).")
            try:
                resources.soft_empty_cache()
                pixel_samples = _decode_tiled(
                    vae,
                    vae.latent_format.process_out(latent),
                    tile_x=tile_size // 2,
                    tile_y=tile_size // 2,
                    min_free_mem=512 * 1024 * 1024,
                ).movedim(1, -1)
            except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
                logging.warning("VAE tiled decode (32x32) OOM. Falling back to CPU decode.")
                resources.soft_empty_cache()
                pixel_samples = _decode_cpu_fallback(vae, latent, tile_size=max(1, tile_size // 4))
    finally:
        if gpu_preferred and not is_resident and patcher is not None:
            try:
                resources.eject_model(patcher)
            except Exception:
                pass

    return pixel_samples
