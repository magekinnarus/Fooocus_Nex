import sys
import os
import torch
import logging
import time

# Add root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from backend import loader, sampling, decode, resources

def test_gguf_vae_tiling():
    # Paths from user request
    unet_path = r"H:\webui_forge_cu121_torch21\webui\models\Stable-diffusion\quantization_target\quantized\beretMixReal_v80_Q4_K_M.gguf"
    clip_path = r"H:\webui_forge_cu121_torch21\webui\models\Stable-diffusion\quantization_target\clips\beretMixReal_v80_clips.safetensors"
    vae_path = os.path.join(root_dir, "models", "vae", "sdxl_vae.safetensors")

    logging.info(f"Testing with:\nUNet: {unet_path}\nCLIP: {clip_path}\nVAE: {vae_path}")

    # 1. Load Models
    logging.info("--- Loading Models ---")
    if not os.path.exists(unet_path):
        logging.error(f"UNet not found: {unet_path}")
        return
    if not os.path.exists(clip_path):
        logging.error(f"CLIP not found: {clip_path}")
        return
    if not os.path.exists(vae_path):
        # Fallback to try finding it if relative path fails
        vae_path_alt = r"models\vae\sdxl_vae.safetensors"
        if os.path.exists(vae_path_alt):
            vae_path = vae_path_alt
        else:
            logging.error(f"VAE not found at {vae_path} or {vae_path_alt}")
            # Try absolute path based on cwd if running from root
            vae_path = os.path.abspath("models/vae/sdxl_vae.safetensors")
            if not os.path.exists(vae_path):
                 logging.error(f"VAE still not found at {vae_path}")
                 return

    logging.info("Loading UNet (GGUF)...")
    unet = loader.load_sdxl_unet(unet_path)
    
    logging.info("Loading CLIP (Combined)...")
    # Both sources are the same file
    clip = loader.load_sdxl_clip(clip_path, clip_path)
    
    logging.info("Loading VAE...")
    vae = loader.load_sdxl_vae(vae_path)

    # 2. Prepare Generation
    prompt = "A cinematic shot of a futuristic city with flying cars, highly detailed, 8k"
    neg_prompt = "blur, low quality"
    steps = 5 # Low steps for speed, but enough to engage sampler
    cfg = 7.0
    sampler_name = "euler"
    scheduler_name = "normal"
    seed = 12345
    
    # 3. Encode Prompt
    logging.info("--- Encoding Prompt ---")
    # clip.encode_from_tokens returns (cond, pooled)
    pos_cond_raw, pos_pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    neg_cond_raw, neg_pooled = clip.encode_from_tokens(clip.tokenize(neg_prompt), return_pooled=True)
    
    # Wrap in Fooocus/ComfyUI-internal style conditioning structure: [{"cross_attn": cond, "pooled_output": pooled}]
    # This allows process_conds to work correctly.
    pos_cond = [{"cross_attn": pos_cond_raw, "pooled_output": pos_pooled}]
    neg_cond = [{"cross_attn": neg_cond_raw, "pooled_output": neg_pooled}]
    
    # 4. Sample (to load UNet into VRAM)
    logging.info("--- Sampling (checking UNet VRAM usage) ---")
    # Initial noise for 1024x1024 (128x128 latent)
    # create noise on CPU first
    noise = torch.randn((1, 4, 128, 128), device="cpu", dtype=torch.float32)
    
    # Process conditioning (Extract 'y' for SDXL)
    # We must move noise to GPU for process_conds to work correctly with model pathcer if needed, 
    # but sample_sdxl handles device movement. process_conds expects device arg.
    device = resources.get_torch_device()
    noise = noise.to(device)
    conds = {"positive": pos_cond, "negative": neg_cond}
    
    # We need to ensure models are loaded for process_conds to access extra_conds
    # load_sdxl_unet already returns a patcher. We need to pass the inner model which has extra_conds
    sampling.process_conds(unet.model, noise, conds, device)
    
    # Sampling will force UNet load
    try:
        samples = sampling.sample_sdxl(
            model=unet,
            noise=noise,
            positive=conds["positive"],
            negative=conds["negative"], 
            steps=steps,
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler_name, 
            seed=seed,
            latent_image=torch.zeros_like(noise)
        )
        logging.info("Sampling complete.")
    except Exception as e:
        logging.error(f"Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Decode (The Real Test)
    logging.info("--- Decoding with Tiled VAE (64x64) ---")
    start_mem = resources.get_free_memory(resources.get_torch_device())
    logging.info(f"Free VRAM before decode: {start_mem / 1024**2:.2f} MB")
    
    try:
        # User requested verification of 64x64 tiling
        # decode_latent allows forcing tiled=True
        images = decode.decode_latent(vae, samples, tiled=True, tile_size=64)
        
        logging.info("Decoding successful!")
        end_mem = resources.get_free_memory(resources.get_torch_device())
        logging.info(f"Free VRAM after decode: {end_mem / 1024**2:.2f} MB")
        logging.info(f"Image shape: {images.shape}")
        
    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gguf_vae_tiling()
