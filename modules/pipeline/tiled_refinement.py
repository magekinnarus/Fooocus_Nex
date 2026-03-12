import numpy as np
import torch
import math
import logging
from typing import List, NamedTuple
import modules.flags as flags
import modules.core as core
import modules.default_pipeline as pipeline
from modules.util import set_image_shape_ceil, resample_image

logger = logging.getLogger(__name__)

class TileInfo(NamedTuple):
    crop: tuple  # (x1, y1, x2, y2)
    tile_image: np.ndarray
    x: int
    y: int
    w: int
    h: int

def select_tile_resolution(width, height):
    """
    Select best SDXL bucket for tile aspect ratio matching global ratio.
    """
    target_ratio = width / height
    resolutions = []
    for s in flags.sdxl_aspect_ratios:
        w, h = map(int, s.split('*'))
        resolutions.append((w, h))
    
    # Minimize ratio error
    best_res = min(resolutions, key=lambda r: abs(r[0] / r[1] - target_ratio))
    print(f'[Tiled Refinement] Global AR {target_ratio:.2f} matched to SDXL bucket {best_res[0]}x{best_res[1]}')
    return best_res

def split_into_tiles(image: np.ndarray, tile_w: int, tile_h: int, overlap: int) -> List[TileInfo]:
    """
    Splits image into overlapping tiles. 
    Handles edge tiles by shifting them to ensure full coverage without creating small slivers.
    """
    H, W, C = image.shape
    tiles = []
    
    # Effective stride (tile size minus overlap)
    stride_w = tile_w - overlap
    stride_h = tile_h - overlap
    
    # Calculate grid size
    nx = math.ceil((W - overlap) / stride_w) if W > tile_w else 1
    ny = math.ceil((H - overlap) / stride_h) if H > tile_h else 1
    
    for i in range(ny):
        for j in range(nx):
            # Candidate top-left
            x1 = j * stride_w
            y1 = i * stride_h
            
            # Shift back if overflow to ensure tiles are always exactly tile_w x tile_h
            if x1 + tile_w > W:
                x1 = max(0, W - tile_w)
            if y1 + tile_h > H:
                y1 = max(0, H - tile_h)
            
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            
            tile_image = image[y1:y2, x1:x2]
            
            tiles.append(TileInfo(
                crop=(x1, y1, x2, y2),
                tile_image=tile_image.copy(),
                x=x1, y=y1, w=tile_w, h=tile_h
            ))
            
    return tiles

def generate_gaussian_weights(tile_w, tile_h, overlap):
    """
    Generate 2D Gaussian weight map for alpha blending.
    Sigma is proportional to tile size to ensure smooth roll-off.
    """
    # Create 1D Gaussian distributions
    def get_gaussian_1d(size):
        center = (size - 1) / 2.0
        sigma = size / 4.0
        x = np.arange(size)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    w_x = get_gaussian_1d(tile_w)
    w_y = get_gaussian_1d(tile_h)
    
    # 2D weight map
    weight_map = np.outer(w_y, w_x).astype(np.float32)
    return weight_map

def stitch_tiles(tiles: List[TileInfo], full_size: tuple, tile_w: int, tile_h: int, overlap: int) -> np.ndarray:
    """
    Stitches tiles using Gaussian-weighted accumulation and normalization.
    """
    H, W, C = full_size
    output = np.zeros((H, W, C), dtype=np.float32)
    weights = np.zeros((H, W, 1), dtype=np.float32)
    
    # Base weight map for a standard tile
    base_weight_map = generate_gaussian_weights(tile_w, tile_h, overlap)[:, :, None]
    
    for t in tiles:
        x1, y1, x2, y2 = t.crop
        tile_h_actual, tile_w_actual = t.tile_image.shape[:2]
        
        # Slice weight map if tile is smaller than standard (rare due to shift-back logic)
        weight_map = base_weight_map[:tile_h_actual, :tile_w_actual]
        
        output[y1:y2, x1:x2] += t.tile_image.astype(np.float32) * weight_map
        weights[y1:y2, x1:x2] += weight_map
        
    # Normalize to prevent overlap brightening
    output /= np.maximum(weights, 1e-5)
    
    return np.clip(output, 0, 255).astype(np.uint8)

def refine_tile(tile_image: np.ndarray, task_state, denoise_strength: float) -> np.ndarray:
    """
    Runs a single tile through the diffusion pipeline.
    """
    import gc
    from backend import resources
    
    h, w = tile_image.shape[:2]
    
    # 1. Encode to latent
    pixels = core.numpy_to_pytorch(tile_image)
    candidate_vae, _ = pipeline.get_candidate_vae(steps=task_state.steps, denoise=denoise_strength)
    resources.load_models_gpu([candidate_vae.patcher])
    latent_dict = core.encode_vae(vae=candidate_vae, pixels=pixels)
    
    # 2. Diffusion
    def noop_callback(*args, **kwargs):
        pass

    import modules.pipeline.preprocessing as preprocessing
    final_scheduler_name = preprocessing.patch_samplers(task_state)

    refined_images = pipeline.process_diffusion(
        positive_cond=task_state.positive_cond,
        negative_cond=task_state.negative_cond,
        steps=task_state.steps,
        width=w,
        height=h,
        image_seed=task_state.seed,
        callback=noop_callback,
        sampler_name=task_state.sampler_name,
        scheduler_name=final_scheduler_name,
        latent=latent_dict,
        denoise=denoise_strength,
        tiled=False,
        cfg_scale=task_state.cfg_scale
    )
    
    # Cleanup tile-specific tensors
    del latent_dict, pixels
    
    return refined_images[0]

def apply_tiled_diffusion_refinement(task_state, upscaled_image: np.ndarray, progressbar_callback=None):
    """
    Main orchestrator for tiled diffusion refinement.
    """
    import gc
    from backend import resources
    
    H, W, C = upscaled_image.shape
    
    # 1. Setup parameters
    tile_w, tile_h = select_tile_resolution(W, H)
    overlap = getattr(task_state, 'upscale_tile_overlap', 128)
    denoise = getattr(task_state, 'upscale_denoise', 0.3)
    
    # 2. Split
    tiles = split_into_tiles(upscaled_image, tile_w, tile_h, overlap)
    total_tiles = len(tiles)
    
    print(f'[Tiled Refinement] Processing {total_tiles} tiles of size {tile_w}x{tile_h} with overlap {overlap} ...')
    
    refined_tiles = []
    for i, t in enumerate(tiles):
        if progressbar_callback:
            progressbar_callback(task_state, task_state.current_progress, f'Refining tile {i+1}/{total_tiles} ...')
        
        refined_img = refine_tile(t.tile_image, task_state, denoise)
        refined_tiles.append(t._replace(tile_image=refined_img))
        
        # Immediate per-tile cleanup to minimize VRAM footprint
        gc.collect()
        resources.soft_empty_cache()
    
    # 3. Stitch
    if progressbar_callback:
        progressbar_callback(task_state, task_state.current_progress, 'Stitching tiles ...')
        
    final_image = stitch_tiles(refined_tiles, (H, W, C), tile_w, tile_h, overlap)
    
    return final_image
