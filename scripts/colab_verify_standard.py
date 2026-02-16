import sys
import os
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# --- Force Configuration for Colab (No Xformers, SDPA Only) ---
try:
    import ldm_patched.modules.args_parser
    logging.info("Forcing Colab/L4 configuration: Disable Xformers, Enable PyTorch SDPA, High VRAM.")
    ldm_patched.modules.args_parser.args.disable_xformers = True
    ldm_patched.modules.args_parser.args.attention_pytorch = True
    ldm_patched.modules.args_parser.args.always_high_vram = True
except ImportError:
    logging.warning("Could not import ldm_patched.modules.args_parser to force configuration. proceeding anyway.")

# Now import backend
from backend import loader, sampling, decode, resources

def test_standard_sdxl_colab():
    # Default path for Colab - user should adjust or upload to this location
    ckpt_path = "/content/models/sd_xl_base_1.0.safetensors"
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    # Local fallback for testing on Windows if file exists
    if not os.path.exists(ckpt_path):
        # Fallback to local test path if on developer machine
        local_fallback = r"models\checkpoints\sd_xl_base_1.0.safetensors"
        if os.path.exists(local_fallback):
            ckpt_path = local_fallback
        elif os.path.exists(os.path.abspath(local_fallback)):
            ckpt_path = os.path.abspath(local_fallback)
            
    logging.info(f"Target Checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        logging.error(f"Checkpoint not found at {ckpt_path}. Please upload it or provide path.")
        return

    # 1. Extract Components
    logging.info("--- Extracting SDXL Components ---")
    components = loader.extract_sdxl_components(ckpt_path)
    
    # 2. Load Models
    logging.info("--- Loading Models ---")
    
    # UNet
    if "unet" in components and components["unet"]:
        logging.info("Loading UNet...")
        unet = loader.load_sdxl_unet(components["unet"])
    else:
        logging.error("No UNet found in checkpoint!")
        return

    # CLIP
    if "clip_l" in components and "clip_g" in components:
        logging.info("Loading CLIP...")
        clip = loader.load_sdxl_clip(components["clip_l"], components["clip_g"])
    else:
        logging.error("CLIP L/G components missing!")
        return
        
    # VAE
    if "vae" in components and components["vae"]:
        logging.info("Loading VAE (from checkpoint)...")
        vae = loader.load_sdxl_vae(components["vae"])
    else:
        logging.warning("No VAE in checkpoint, trying to load default VAE...")
        vae_path = os.path.join(root_dir, "models", "vae", "sdxl_vae.safetensors")
        if os.path.exists(vae_path):
             vae = loader.load_sdxl_vae(vae_path)
        else:
             logging.error("No VAE found!")
             return

    # 3. Sampling Setup
    prompt = "A cinematic shot of a futuristic city with flying cars, highly detailed, 8k"
    neg_prompt = "blur, low quality"
    steps = 20
    cfg = 7.0
    sampler_name = "euler"
    scheduler_name = "normal"
    seed = 12345
    
    # 4. Encode Prompt
    logging.info("--- Encoding Prompt ---")
    # clip.encode_from_tokens returns (cond, pooled)
    pos_cond_raw, pos_pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    neg_cond_raw, neg_pooled = clip.encode_from_tokens(clip.tokenize(neg_prompt), return_pooled=True)
    
    # Condition format for process_conds
    pos_cond = [{"cross_attn": pos_cond_raw, "pooled_output": pos_pooled}]
    neg_cond = [{"cross_attn": neg_cond_raw, "pooled_output": neg_pooled}]
    
    # 5. Sample
    logging.info("--- Sampling (Standard SDXL) ---")
    device = resources.get_torch_device()
    logging.info(f"Using device: {device}")
    
    # Initial noise (1024x1024)
    noise = torch.randn((1, 4, 128, 128), device=device, dtype=torch.float32)
    
    # Prepare conds
    conds = {"positive": pos_cond, "negative": neg_cond}
    sampling.process_conds(unet.model, noise, conds, device)
    
    start_time = time.time()
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
        logging.info(f"Sampling complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        logging.error(f"Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Decode (Standard / Non-Tiled)
    logging.info("--- Decoding (Standard / Non-Tiled) ---")
    start_mem = resources.get_free_memory(device)
    logging.info(f"Free VRAM before decode: {start_mem / 1024**2:.2f} MB")
    
    try:
        # Force tiled=False for standard decode test
        images = decode.decode_latent(vae, samples, tiled=False)
        
        logging.info("Decoding successful!")
        end_mem = resources.get_free_memory(device)
        logging.info(f"Free VRAM after decode: {end_mem / 1024**2:.2f} MB")
        logging.info(f"Image shape: {images.shape}")
        
    except Exception as e:
        logging.error(f"Decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_standard_sdxl_colab()
