import torch
import logging
from . import resources
from . import utils

# Estimation constant from ComfyUI for AutoencoderKL
VAE_DECODE_MEMORY_STRICT = 2178

def _process_output(image):
    """Normalizes VAE output to [0, 1] range."""
    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

def _decode_tiled(vae, samples, tile_x=64, tile_y=64, overlap=16, min_free_mem=0):
    """
    Decodes in tiles to save VRAM.
    Uses ComfyUI's 3-pass averaged tiling to prevent artifacts.
    """
    device = vae.patcher.load_device
    dtype = next(vae.first_stage_model.parameters()).dtype
    output_device = "cpu" 
    
    
    # SDXL VAE upscale ratio is always 8
    upscale_ratio = 8 
    
    logging.info(f"VAE Tiled Decoding: tile_size={tile_x}, overlap={overlap}")

    # Estimate memory usage for a single tile (latent H * W * 64 pixels * multiplier) 
    # tile_x/y are latent dimensions, so outputs are 8x larger
    memory_used = (VAE_DECODE_MEMORY_STRICT * tile_x * tile_y * 64) * utils.dtype_size(dtype)
    
    # Ensure VAE is on GPU and float32 to avoid device/dtype mismatches
    # We can pass minimum_memory_required to force eviction of other models if needed
    resources.load_models_gpu([vae.patcher], memory_required=memory_used, minimum_memory_required=min_free_mem)
    
    # Update dtype after loading, as it might have changed
    dtype = next(vae.first_stage_model.parameters()).dtype

    # We use a float32 decode function as per ComfyUI
    decode_fn = lambda a: vae.first_stage_model.decode(a.to(dtype).to(device)).float()
    
    # Pass 1: standard tiles
    p3 = utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=upscale_ratio, output_device=output_device)
    # Pass 2: tall tiles
    p1 = utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=upscale_ratio, output_device=output_device)
    # Pass 3: wide tiles
    p2 = utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=upscale_ratio, output_device=output_device)
    
    output = _process_output((p1 + p2 + p3) / 3.0)
    return output

def decode_latent(vae, latent, tiled=False, tile_size=64):
    """
    Decodes a latent tensor into pixel space.
    
    Args:
        vae: VAE container from loader.py
        latent: Latent tensor [B, 4, H, W]
        tiled: Whether to force tiled decoding (default: False)
        tile_size: Latent tile size (default: 64)
        
    Returns:
        Pixel tensor [B, H*8, W*8, 3], float32 [0, 1]
    """
    device = vae.patcher.load_device
    dtype = next(vae.first_stage_model.parameters()).dtype
    output_device = "cpu"
    
    if tiled:
        pixel_samples = _decode_tiled(vae, latent, tile_x=tile_size, tile_y=tile_size)
    else:
        try:
            # Clear cache to free sampling memory
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Estimate memory usage (latent H * W * 64 pixels * multiplier)
            memory_used = (VAE_DECODE_MEMORY_STRICT * latent.shape[2] * latent.shape[3] * 64) * utils.dtype_size(dtype)
            
            # Ensure VAE is on GPU and float32 to avoid device/dtype mismatches
            resources.load_models_gpu([vae.patcher], memory_required=memory_used)
            vae.first_stage_model.to(device=device, dtype=torch.float32)
            dtype = torch.float32
            
            free_memory = resources.get_free_memory(device)
            batch_number = int(free_memory / max(1, memory_used))
            batch_number = max(1, batch_number)
            
            pixel_samples = None
            for x in range(0, latent.shape[0], batch_number):
                batch = latent[x:x+batch_number].to(device=device, dtype=dtype)
                out = _process_output(vae.first_stage_model.decode(batch).to(output_device).float())
                
                if pixel_samples is None:
                    pixel_samples = torch.empty((latent.shape[0],) + tuple(out.shape[1:]), device=output_device)
                
                pixel_samples[x:x+batch_number] = out
                
        except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
            logging.warning("VAE decode OOM. Retrying with tiled decoding.")
            try:
                # Ask for 1GB free to force eviction of UNet on low VRAM
                pixel_samples = _decode_tiled(vae, latent, tile_x=tile_size, tile_y=tile_size, min_free_mem=1024*1024*1024)
            except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
                logging.warning("VAE tiled decode (64x64) OOM. Retrying with smaller tiles (32x32).")
                try:
                    resources.soft_empty_cache()
                    # Ask for 512MB free
                    pixel_samples = _decode_tiled(vae, latent, tile_x=tile_size // 2, tile_y=tile_size // 2, min_free_mem=512*1024*1024)
                except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
                    logging.warning("VAE tiled decode (32x32) OOM. Retrying with minimal tiles (16x16).")
                    resources.soft_empty_cache()
                    # Force free everything we can
                    resources.free_memory(1e30, vae.patcher.load_device) 
                    pixel_samples = _decode_tiled(vae, latent, tile_x=tile_size // 4, tile_y=tile_size // 4, min_free_mem=0)

    # Convert CHW to HWC: [B, 3, H, W] -> [B, H, W, 3]
    pixel_samples = pixel_samples.movedim(1, -1)
    return pixel_samples
