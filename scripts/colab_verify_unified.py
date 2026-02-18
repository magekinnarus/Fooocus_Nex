import sys
import os
import torch
import logging
import time
import gc
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# --- Force Configuration for Colab ---
try:
    import ldm_patched.modules.args_parser
    # Patch args BEFORE importing model_management
    ldm_patched.modules.args_parser.args.disable_xformers = True
    ldm_patched.modules.args_parser.args.attention_pytorch = True
    ldm_patched.modules.args_parser.args.always_high_vram = True
    logging.info("Forced Colab Config: Xformers Disabled, SDPA Enabled, High VRAM.")
except ImportError:
    logging.warning("Could not import ldm_patched modules.")

from backend import loader, sampling, decode, resources

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB"
    return "N/A"

@torch.no_grad()
def test_unified_loader():
    logging.info("--- Starting Verification with Unified Loader ---")
    device = resources.get_torch_device()
    model_dtype = torch.float16 
    vae_dtype = torch.float32   
    
    logging.info(f"START VRAM: {get_vram_usage()}")

    ckpt_path = "/content/models/checkpoints/IL_beretMixReal_v100.safetensors"
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    if not os.path.exists(ckpt_path):
        logging.error(f"Checkpoint not found at {ckpt_path}.")
        return

    # --- NEW UNIFIED LOADER ---
    # This single call replaces the complex extract/pop/load/del sequence
    # and handles memory efficiently internally.
    logging.info("Loading checkpoint via backend.loader.load_checkpoint...")
    unet, clip, vae = loader.load_checkpoint(ckpt_path, load_device=device, unet_dtype=model_dtype)
    
    logging.info(f"Models loaded. VRAM: {get_vram_usage()}")

    # 3. Encode & Sample
    prompt = "A cinematic shot of a futuristic city with flying cars, highly detailed, 8k"
    neg_prompt = "blur, low quality"
    
    logging.info("--- Encoding ---")
    resources.load_models_gpu([clip.patcher])
    
    pos_cond_raw, pos_pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    neg_cond_raw, neg_pooled = clip.encode_from_tokens(clip.tokenize(neg_prompt), return_pooled=True)
    
    pos_cond = [{"cross_attn": pos_cond_raw.to(device=device, dtype=model_dtype), "pooled_output": pos_pooled.to(device=device, dtype=model_dtype)}]
    neg_cond = [{"cross_attn": neg_cond_raw.to(device=device, dtype=model_dtype), "pooled_output": neg_pooled.to(device=device, dtype=model_dtype)}]
    
    logging.info("--- Sampling ---")
    resources.load_models_gpu([unet])
    
    noise = torch.randn((1, 4, 128, 128), device=device, dtype=model_dtype)
    conds = {"positive": pos_cond, "negative": neg_cond}
    sampling.process_conds(unet.model, noise, conds, device)
    
    try:
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            samples = sampling.sample_sdxl(
                model=unet, noise=noise, positive=conds["positive"], negative=conds["negative"], 
                steps=20, cfg=7.0, sampler_name="euler", scheduler="normal", seed=12345,
                latent_image=torch.zeros_like(noise).to(device=device, dtype=model_dtype)
            )
    except Exception as e:
        logging.error(f"Sampling failed: {e}")
        return

    if torch.isnan(samples).any():
        samples = torch.nan_to_num(samples)

    # 4. Decode
    logging.info("--- Decoding ---")
    
    # Explicitly unload UNet and CLIP to be absolutely safe
    if hasattr(unet, 'model'):
        unet.model.to("cpu")
    if hasattr(clip, 'patcher'):
        clip.patcher.model.to("cpu")
    
    del unet
    del clip
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # Move VAE to GPU (this will likely kick UNet out of VRAM to make space)
        # We start with aggressive requirement to ensure space
        logging.info("Moving VAE to GPU...")
        resources.load_models_gpu([vae.patcher], minimum_memory_required=2*1024*1024*1024) 
        
        logging.info("Converting samples to VAE dtype...")
        samples_f32 = samples.to(dtype=vae_dtype)
        
        # Decode
        # Using tiled=True is safer on L4 for 1024px+ even with 24GB, 
        # but standard decode valid test too.
        logging.info(f"Decoding latent shape: {samples_f32.shape}")
        images = decode.decode_latent(vae, samples_f32, tiled=True, tile_size=512)
        
        # Final safety check before casting to numpy
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        images = torch.clamp(images, 0.0, 1.0)
        
        img_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save("colab_unified_output.png")
        logging.info("Success! saved as colab_unified_output.png")

    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unified_loader()
