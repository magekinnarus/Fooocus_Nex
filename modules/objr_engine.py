import os
import torch
import numpy as np
import gc
import logging
import math
from PIL import Image
from typing import List, Tuple

from modules import model_registry
import modules.config as config
import modules.mask_processing as mask_processing
from ldm_patched.pfn.architecture.MAT import MAT
from modules.blending import sin_blend_1d
import backend.resources as resources
from modules.util import HWC3

logger = logging.getLogger(__name__)

_model_instance = None

# --- Utility Functions (Ported from reference) ---

def mask_unsqueeze(mask: torch.Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def to_torch(image: np.ndarray, mask: np.ndarray = None, device="cpu"):
    # image: HWC uint8 -> BCHW float32 [0, 1]
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_t = image_t.unsqueeze(0).to(device)
    
    if mask is not None:
        # mask: HW uint8 -> B1HW float32 [0, 1]
        mask_t = torch.from_numpy(mask).float() / 255.0
        mask_t = mask_unsqueeze(mask_t).to(device)
        return image_t, mask_t
    return image_t

def mask_floor(mask: torch.Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)

def pad_reflect_once(x: torch.Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = torch.nn.functional.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = torch.nn.functional.pad(x, tuple(additional_padding), mode="constant")
    return x

def resize_square(image: torch.Tensor, mask: torch.Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
        
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, size=size, mode="nearest-exact")
        mask = torch.nn.functional.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)

def undo_resize_square(image: torch.Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = torch.nn.functional.interpolate(image, size=prev_size, mode="bilinear", align_corners=False)
    # Remove padding: h_orig = prev_size - pad_h, w_orig = prev_size - pad_w
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]

# --- Tiling Utilities ---


def get_segments(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [(0, length, 0, 0)] # start, end, pad_l, pad_r
    
    segments = []
    # First
    segments.append((0, tile_size - overlap, 0, overlap))
    
    while segments[-1][1] < length:
        start = segments[-1][1]
        end = start + tile_size - overlap * 2
        
        if end >= length:
            end = length
            # pad back to keep tile_size
            start_in_v = end - tile_size
            actual_start = max(0, start)
            pad_l = actual_start - start_in_v
            segments.append((actual_start, end, pad_l, 0))
        else:
            segments.append((start, end, overlap, overlap))
    return segments

# --- Core Engine ---

def load_model(model_name: str = "Places_512_FullData_G.pth") -> MAT:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if model_name != "Places_512_FullData_G.pth":
        checkpoint_path = os.path.join(config.path_removals, model_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Object removal model not found: {model_name}")
    else:
        checkpoint_path = model_registry.ensure_asset('removals.object.mat.places512', progress=True)

    logger.info(f"Loading MAT Object Removal Engine from {checkpoint_path} ...")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Remap keys
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping")
        new_state[new_key] = v

    model = MAT()
    model.load_state_dict(new_state)
    model.eval()

    # Force float32 for Pascal stability
    model.to(torch.float32)

    _model_instance = model
    return _model_instance

def unload_model():
    global _model_instance
    if _model_instance is not None:
        logger.info("Unloading MAT OBJR engine ...")
        del _model_instance
        _model_instance = None
        
    gc.collect()
    if torch.cuda.is_available():
        resources.soft_empty_cache()

@torch.inference_mode()
def remove_object(image: np.ndarray, mask: np.ndarray, seed: int = 0, mask_dilate: int = 0) -> np.ndarray:
    """
    Remove objects defined by mask.
    image: HWC uint8
    mask: HW uint8 (255 = inpaint)
    """
    if mask_dilate > 0:
        import cv2
        kernel = np.ones((mask_dilate, mask_dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    h, w, _ = image.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)
    
    torch.manual_seed(seed)
    
    # Small image path
    if h <= 512 and w <= 512:
        img_t, mask_t = to_torch(image, mask, device=device)
        # resize_square pads to square and resizes to 512
        img_sq, mask_sq, orig_info = resize_square(img_t, mask_t, 512)
        
        # Binarize mask
        mask_sq = mask_floor(mask_sq, 0.99)
        
        # MAT inference
        # Generator.forward(images_in, masks_in, z, c, ...)
        # MAT.forward(image, mask) handles the normalization and Generator call
        res_sq = model(img_sq, mask_sq)
        
        # Undo resize/padding
        res_t = undo_resize_square(res_sq, orig_info)
        
        # Composite: original * (1-mask) + result * mask
        # Ensure mask is exactly what we used for composition
        comp_mask = to_torch(np.zeros_like(image), mask, device=device)[1]
        final_t = img_t * (1.0 - comp_mask) + res_t * comp_mask
        
        final_np = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return final_np

    # Large image path (Tiled)
    logger.info(f"Using tiled processing for {w}x{h} image")
    tile_size = 512
    overlap = 64
    
    img_t, mask_t = to_torch(image, mask, device=device)
    
    output = img_t.clone()
    # Weight map for blending
    weight_total = torch.zeros((1, 1, h, w), device=device)
    # Result accumulator
    accum = torch.zeros_like(img_t)
    
    h_segs = get_segments(h, tile_size, overlap)
    w_segs = get_segments(w, tile_size, overlap)
    
    for y_start, y_end, y_pad_l, y_pad_r in h_segs:
        for x_start, x_end, x_pad_l, x_pad_r in w_segs:
            # Extract tile with padding to ensure 512x512
            tile_y_start = y_start - y_pad_l
            tile_x_start = x_start - x_pad_l
            
            tile_img = img_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]
            tile_mask = mask_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]
            
            # Optimization: Skip if no mask in this tile
            if torch.sum(tile_mask) < 1e-4:
                tile_res = tile_img
            else:
                # Run MAT on tile
                tile_mask_bin = mask_floor(tile_mask, 0.99)
                tile_res = model(tile_img, tile_mask_bin)
            
            # Build 2D weight mask for this tile
            # sin_blend_1d for edges
            w_map = torch.ones((1, 1, tile_size, tile_size), device=device)
            if y_pad_l > 0:
                w_map[:, :, :y_pad_l, :] *= sin_blend_1d(y_pad_l, device).view(1, 1, -1, 1)
            if y_pad_r > 0:
                w_map[:, :, -y_pad_r:, :] *= sin_blend_1d(y_pad_r, device).flip(0).view(1, 1, -1, 1)
            if x_pad_l > 0:
                w_map[:, :, :, :x_pad_l] *= sin_blend_1d(x_pad_l, device).view(1, 1, 1, -1)
            if x_pad_r > 0:
                w_map[:, :, :, -x_pad_r:] *= sin_blend_1d(x_pad_r, device).flip(0).view(1, 1, 1, -1)
            
            accum[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += tile_res * w_map
            weight_total[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += w_map
            
    # Final normalization and composition
    tiled_result = accum / (weight_total + 1e-8)
    final_t = img_t * (1.0 - mask_t) + tiled_result * mask_t
    
    final_np = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return final_np

def remove_object_from_file(image_path: str, mask_path: str, seed: int = 0, mask_dilate: int = 0) -> str:
    """Filepath invariant wrapper."""
    with Image.open(image_path) as img:
        img_np = HWC3(np.array(img.convert('RGBA')))
    with Image.open(mask_path) as msk:
        msk_np = np.array(msk.convert('L'))
        
    res_np = remove_object(img_np, msk_np, seed=seed, mask_dilate=mask_dilate)
    
    return mask_processing.save_to_temp_png(res_np)



