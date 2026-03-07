"""
Centralized mask processing utilities for Fooocus Inpaint/Outpaint workflows.

All Gradio sketch/image component unpacking and mask color extraction goes through
this module to prevent RGBA vs RGB shape mismatch bugs.
"""
import numpy as np
import gradio as gr
from modules.util import HWC3
from PIL import Image
import os


def rgba_to_black_bg_rgb(x):
    """
    Unlike util.HWC3 which composites transparent pixels over a white background,
    masks need transparent pixels to composite over a BLACK background (0 = no mask).
    """
    if x is None:
        return None
        
    if x.dtype in (np.float32, np.float64):
        if x.max() <= 1.0:
            x = (x * 255).astype(np.uint8)
        else:
            x = x.astype(np.uint8)
            
    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        # Composite over BLACK (0.0): y = color * alpha + 0.0 * (1.0 - alpha)
        y = color * alpha
        return y.clip(0, 255).astype(np.uint8)
    return x


def ensure_numpy(x, mode='RGB'):
    """
    Ensure the input is a numpy array. Handles None, str (filepath), 
    PIL.Image, and np.ndarray.
    """
    if x is None:
        return None
    
    if isinstance(x, str):
        if os.path.exists(x):
            try:
                with Image.open(x) as img:
                    return np.array(img.convert(mode))
            except Exception as e:
                print(f"[mask_processing] Error loading image from {x}: {e}")
                return None
        else:
            print(f"[mask_processing] File not found: {x}")
            return None
            
    if isinstance(x, Image.Image):
        return np.array(x.convert(mode))
        
    if isinstance(x, np.ndarray):
        if mode == 'RGB' and x.ndim == 3 and x.shape[2] == 4:
            return x[:, :, :3]
        return x
        
    return None


def unpack_gradio_data(data):
    """
    Safely extract an RGB image from a Gradio image/sketch component.
    """
    if data is None:
        return None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return rgba_to_black_bg_rgb(ensure_numpy(data))
    
    if isinstance(data, dict):
        img = data.get('image')
        if img is not None:
            return rgba_to_black_bg_rgb(ensure_numpy(img))
        mask = data.get('mask')
        if mask is not None:
            return rgba_to_black_bg_rgb(ensure_numpy(mask))
    
    return None


def unpack_gradio_sketch(data):
    """
    Safely extract both the base image and the drawn mask from a Gradio sketch component.
    """
    if data is None:
        return None, None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return rgba_to_black_bg_rgb(ensure_numpy(data)), None
    
    if isinstance(data, dict):
        img = data.get('image')
        mask = data.get('mask')
        
        img_out = rgba_to_black_bg_rgb(ensure_numpy(img)) if img is not None else None
        mask_out = rgba_to_black_bg_rgb(ensure_numpy(mask)) if mask is not None else None
        
        return img_out, mask_out
    
    return None, None


def extract_mask_from_layers(data):
    """
    Extract a binary mask from Gradio 5 ImageEditor's layers.
    Combines alpha channels of all layers into a single [0, 255] uint8 mask.
    """
    if not isinstance(data, dict):
        return None
    
    layers = data.get('layers')
    if not layers:
        return None
        
    mask = None
    for layer in layers:
        if layer is None:
            continue
            
        layer_arr = ensure_numpy(layer, mode='RGBA')
        if layer_arr is None:
            continue

        if layer_arr.ndim != 3 or layer_arr.shape[2] != 4:
            continue
            
        # Extract alpha as the mask
        alpha = layer_arr[:, :, 3]
        if mask is None:
            mask = alpha
        else:
            mask = np.maximum(mask, alpha)
            
    if mask is None:
        return None
        
    return (mask > 0).astype(np.uint8) * 255


def extract_color_masks_from_layers(data):
    """
    Extract specifically Blue and White masks from ImageEditor layers.
    Blue (#0000FF) = Context Mask
    White (#FFFFFF) = Inpaint Mask
    """
    if not isinstance(data, dict):
        return None, None
    
    layers = data.get('layers')
    if not layers:
        return None, None
        
    white_mask = None
    blue_mask = None
    
    for layer in layers:
        if layer is None:
            continue
            
        layer_arr = ensure_numpy(layer, mode='RGBA')
        if layer_arr is None:
            continue

        if layer_arr.ndim != 3 or layer_arr.shape[2] != 4:
            continue
            
        r, g, b, a = layer_arr[:,:,0], layer_arr[:,:,1], layer_arr[:,:,2], layer_arr[:,:,3]
        alpha_mask = a > 0
        
        # Detect colors based on predominant channels
        white_strokes = (r > 127) & (g > 127) & (b > 127) & alpha_mask
        blue_strokes = (r < 127) & (g < 127) & (b > 127) & alpha_mask
        
        if white_strokes.any():
            m = np.zeros(layer_arr.shape[:2], dtype=np.uint8)
            m[white_strokes] = 255
            white_mask = np.maximum(white_mask, m) if white_mask is not None else m
            
        if blue_strokes.any():
            m = np.zeros(layer_arr.shape[:2], dtype=np.uint8)
            m[blue_strokes] = 255
            blue_mask = np.maximum(blue_mask, m) if blue_mask is not None else m
            
    return white_mask, blue_mask


def combine_image_and_mask(data):
    """
    Merge the 'image' and 'mask' layers from a Gradio sketch dict into a single RGB array
    using element-wise maximum. Handles RGBA vs RGB mismatches safely.
    
    Returns:
        np.ndarray (H, W, 3) or None
    """
    if data is None:
        return None
    
    if isinstance(data, (np.ndarray, str, Image.Image)):
        return ensure_numpy(data)
    
    if isinstance(data, dict):
        # Gradio 5 ImageEditor / EditorValue format
        if 'composite' in data:
            # For combine_image_and_mask, we either want the composite or 
            # the background + extracted mask. In Fooocus, this is often used
            # for mask expansion or previewing.
            if 'background' in data and data['background'] is not None:
                bg = rgba_to_black_bg_rgb(ensure_numpy(data['background']))
                mask = extract_mask_from_layers(data)
                if mask is not None:
                    # Overlay mask onto background for preview
                    mask_rgb = np.stack([mask]*3, axis=-1)
                    return np.maximum(bg, mask_rgb)
                return bg
            if 'composite' in data and data['composite'] is not None:
                return rgba_to_black_bg_rgb(ensure_numpy(data['composite']))
            return None

        # Legacy Gradio 3 sketch format
        img = data.get('image')
        mask = data.get('mask')
        
        if isinstance(img, np.ndarray) and isinstance(mask, np.ndarray) and img.ndim == 3:
            return np.maximum(rgba_to_black_bg_rgb(img), rgba_to_black_bg_rgb(mask))
        elif isinstance(img, np.ndarray):
            return rgba_to_black_bg_rgb(img)
        elif isinstance(mask, np.ndarray):
            return rgba_to_black_bg_rgb(mask)
    
    return None


def extract_color_masks(raw_mask_layer):
    """
    Extract white (inpaint) and blue (context) stroke masks from a Gradio sketch component's
    raw mask layer.
    
    The raw mask layer may be (H, W, 3) or (H, W, 4). If 4 channels, the alpha channel
    is used as a stroke presence indicator.
    
    Args:
        raw_mask_layer: np.ndarray (H, W, 3) or (H, W, 4) from Gradio sketch
        
    Returns:
        (white_mask_2d, blue_mask_2d) — both (H, W) np.uint8, values 0 or 255
    """
    if raw_mask_layer is None:
        return None, None
    
    has_alpha = raw_mask_layer.ndim == 3 and raw_mask_layer.shape[2] == 4
    if has_alpha:
        alpha_mask = raw_mask_layer[:, :, 3] > 0
    else:
        alpha_mask = np.ones(raw_mask_layer.shape[:2], dtype=bool)
    
    r = raw_mask_layer[:, :, 0]
    g = raw_mask_layer[:, :, 1]
    b = raw_mask_layer[:, :, 2]
    
    white_strokes = (r > 127) & (g > 127) & (b > 127) & alpha_mask
    blue_strokes = (r < 127) & (g < 127) & (b > 127) & alpha_mask
    
    white_mask = np.zeros(raw_mask_layer.shape[:2], dtype=np.uint8)
    white_mask[white_strokes] = 255
    
    blue_mask = np.zeros(raw_mask_layer.shape[:2], dtype=np.uint8)
    blue_mask[blue_strokes] = 255
    
    return white_mask, blue_mask


def to_binary_mask(mask):
    """
    Convert any mask array to a clean binary 2D mask (0 or 255, uint8).
    
    Handles:
      - Float masks (0.0-1.0 or 0.0-255.0)
      - Multi-channel masks (takes max across channels)
      - 4-channel RGBA (strips alpha)
    
    Returns:
        np.ndarray (H, W) uint8 with values 0 or 255, or None
    """
    if mask is None:
        return None
    
    if isinstance(mask, (str, Image.Image)):
        mask = ensure_numpy(mask)
    
    if mask.dtype in (np.float32, np.float64):
        if mask.max() <= 1.0:
            mask = mask * 255.0
        mask = mask.astype(np.uint8)
    
    if mask.ndim == 3:
        if mask.shape[-1] == 4:
            mask = mask[..., :3]
        mask = np.max(mask, axis=-1)
    
    return (mask > 127).astype(np.uint8) * 255


def combine_masks(*masks):
    """
    Merge multiple 2D masks into one using element-wise maximum.
    Ignores None entries.
    
    Returns:
        np.ndarray (H, W) uint8, or None if all inputs are None
    """
    result = None
    for m in masks:
        if m is None:
            continue
        m2d = to_binary_mask(m)
        if m2d is None:
            continue
        if result is None:
            result = m2d
        else:
            result = np.maximum(result, m2d)
    return result


def expand_mask_direction(mask_2d, direction, pixels=32):
    """
    Expand white pixels in a 2D mask in the OPPOSITE direction of outpaint.
    E.g., if direction is 'Right', expand leftward (into original image).
    
    Args:
        mask_2d: (H, W) uint8 array, values 0 or 255
        direction: one of 'Left', 'Right', 'Top', 'Bottom' (case-sensitive)
        pixels: number of pixels to expand
        
    Returns:
        np.ndarray (H, W) uint8
    """
    result = mask_2d.copy()
    
    for _ in range(pixels):
        shifted = np.zeros_like(result)
        if direction == 'Right':      # Expand Left
            shifted[:, :-1] = result[:, 1:]
        elif direction == 'Left':     # Expand Right
            shifted[:, 1:] = result[:, :-1]
        elif direction == 'Top':      # Expand Bottom
            shifted[1:, :] = result[:-1, :]
        elif direction == 'Bottom':   # Expand Top
            shifted[:-1, :] = result[1:, :]
        result = np.maximum(result, shifted)
    
    return result
