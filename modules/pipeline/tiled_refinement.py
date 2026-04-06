import math
import numpy as np
from typing import List, NamedTuple
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import modules.blending as blending
from backend import resources

class TileInfo(NamedTuple):
    crop: tuple  # (x1, y1, x2, y2)
    tile_image: np.ndarray
    x: int
    y: int
    w: int
    h: int

def select_tile_resolution(full_w, full_h, min_overlap=128):
    """
    Smart Auto-Tiling 2.0: Iterates through all buckets and selects the one 
    that minimizes the total tile count (nx * ny).
    """
    buckets = []
    for s in flags.sdxl_aspect_ratios:
        w, h = map(int, s.split('*'))
        buckets.append((w, h))

    best_config = None
    min_total_tiles = float('inf')
    min_waste = float('inf')

    for bw, bh in buckets:
        # Calculate nx: how many tiles of width bw to cover full_w with min_overlap
        if full_w <= bw:
            nx = 1
            overlap_w = 0
        else:
            nx = math.ceil((full_w - min_overlap) / (bw - min_overlap))
            overlap_w = (nx * bw - full_w) / (nx - 1)
        
        # Calculate ny: how many tiles of height bh to cover full_h with min_overlap
        if full_h <= bh:
            ny = 1
            overlap_h = 0
        else:
            ny = math.ceil((full_h - min_overlap) / (bh - min_overlap))
            overlap_h = (ny * bh - full_h) / (ny - 1)

        total_tiles = nx * ny
        # Total waste is a combination of overlaps and aspect ratio mismatch
        waste = (overlap_w if nx > 1 else (bw - full_w)) + (overlap_h if ny > 1 else (bh - full_h))

        if total_tiles < min_total_tiles or (total_tiles == min_total_tiles and waste < min_waste):
            min_total_tiles = total_tiles
            min_waste = waste
            best_config = ((bw, bh), nx, ny, int(overlap_w), int(overlap_h))

    bucket, nx, ny, overlap_w, overlap_h = best_config
    print(f'[Smart Tiling] Optimized Layout: {nx}x{ny} grid using {bucket[0]}x{bucket[1]} bucket.')
    print(f'[Smart Tiling] Actual Overlap: {overlap_w}px (Horiz), {overlap_h}px (Vert)')
    
    return bucket, nx, ny, overlap_w, overlap_h

def split_into_tiles(image: np.ndarray, bucket_w: int, bucket_h: int, nx: int, ny: int, overlap_w: int, overlap_h: int) -> List[TileInfo]:
    H, W, C = image.shape
    tiles = []
    
    stride_w = bucket_w - overlap_w
    stride_h = bucket_h - overlap_h
    
    for i in range(ny):
        for j in range(nx):
            x1 = j * stride_w
            y1 = i * stride_h
            
            # Boundary correction
            if x1 + bucket_w > W: x1 = W - bucket_w
            if y1 + bucket_h > H: y1 = H - bucket_h
            
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = x1 + bucket_w, y1 + bucket_h
            
            tile_image = image[y1:y2, x1:x2]
            
            # Final check: Ensure we ARE at bucket size (in case of very small images)
            if tile_image.shape[0] != bucket_h or tile_image.shape[1] != bucket_w:
                import cv2
                tile_image = cv2.resize(tile_image, (bucket_w, bucket_h), interpolation=cv2.INTER_LANCZOS4)

            tiles.append(TileInfo(
                crop=(x1, y1, x2, y2),
                tile_image=tile_image.copy(),
                x=x1, y=y1, w=bucket_w, h=bucket_h
            ))
            
    return tiles


def stitch_tiles(tiles: List[TileInfo], full_size: tuple, bucket_w: int, bucket_h: int) -> np.ndarray:
    H, W, C = full_size
    output = np.zeros((H, W, C), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    base_weight_map = blending.sin_blend_2d(bucket_w, bucket_h).cpu().numpy()
    
    for t in tiles:
        x1, y1, x2, y2 = t.crop
        output[y1:y2, x1:x2] += t.tile_image.astype(np.float32) * base_weight_map[:, :, None]
        weights[y1:y2, x1:x2] += base_weight_map
        
    output /= np.maximum(weights[:, :, None], 1e-5)
    return np.clip(output, 0, 255).astype(np.uint8)

def refine_tile(tile_image: np.ndarray, task_state, denoise_strength: float) -> np.ndarray:
    import gc
    h, w = tile_image.shape[:2]
    
    pixels = core.numpy_to_pytorch(tile_image)
    with resources.memory_phase_scope(
        resources.MemoryPhase.VAE_ENCODE,
        task=task_state,
        notes={'route': 'tiled_refine', 'tile_size': [w, h], 'denoise': float(denoise_strength)},
        end_notes={'completed': True},
    ):
        candidate_vae, _ = pipeline.get_candidate_vae(steps=task_state.steps, denoise=denoise_strength)
        resources.load_models_gpu([candidate_vae.patcher])
        latent_dict = core.encode_vae(vae=candidate_vae, pixels=pixels)
    
    import modules.pipeline.preprocessing as preprocessing
    final_scheduler_name = preprocessing.patch_samplers(task_state)

    # Note: refined_images is already a list of numpy arrays from process_diffusion
    refined_images = pipeline.process_diffusion(
        positive_cond=task_state.positive_cond,
        negative_cond=task_state.negative_cond,
        steps=task_state.steps,
        width=w,
        height=h,
        image_seed=task_state.seed,
        callback=None,
        sampler_name=task_state.sampler_name,
        scheduler_name=final_scheduler_name,
        latent=latent_dict,
        denoise=denoise_strength,
        tiled=False,
        cfg_scale=task_state.cfg_scale,
        quality={'sharpness': task_state.sharpness}
    )
    
    return refined_images[0]
def apply_tiled_diffusion_refinement(task_state, upscaled_image: np.ndarray, progressbar_callback=None):
    import gc
    from backend import resources
    
    H, W, C = upscaled_image.shape

    with resources.memory_phase_scope(
        resources.MemoryPhase.TILED_REFINE,
        task=task_state,
        notes={'image_size': [W, H]},
        end_notes={'completed': True},
    ):
        # Pre-flight cleanup: Clear everything to maximize tile headroom
        resources.unload_all_models()
        gc.collect()
        resources.soft_empty_cache()
        
        min_overlap = getattr(task_state, 'upscale_refinement_tile_overlap', 128)
        bucket, nx, ny, overlap_w, overlap_h = select_tile_resolution(W, H, min_overlap)
        bucket_w, bucket_h = bucket
        
        denoise = getattr(task_state, 'upscale_refinement_denoise', 0.382)
        tiles = split_into_tiles(upscaled_image, bucket_w, bucket_h, nx, ny, overlap_w, overlap_h)
        
        print(f'[Tiled Refinement] Processing {len(tiles)} tiles...')
        
        refined_tiles = []
        for i, t in enumerate(tiles):
            if progressbar_callback:
                progressbar_callback(task_state, int(task_state.current_progress + (i/len(tiles))*10), f'Refining tile {i+1}/{len(tiles)} ...')
            
            refined_img = refine_tile(t.tile_image, task_state, denoise)
            refined_tiles.append(t._replace(tile_image=refined_img))
            
            # Post-tile cleanup
            gc.collect()
            resources.soft_empty_cache()
        
        if progressbar_callback:
            progressbar_callback(task_state, task_state.current_progress + 10, 'Stitching tiles ...')
            
        result = stitch_tiles(refined_tiles, (H, W, C), bucket_w, bucket_h)
        
        # Final sweep: Leave the GPU clean
        resources.unload_all_models()
        gc.collect()
        resources.soft_empty_cache()
        
        return result
