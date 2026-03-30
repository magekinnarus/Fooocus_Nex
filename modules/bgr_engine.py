import os
import torch
import numpy as np
import gc
import logging
from PIL import Image
from transparent_background import Remover

import modules.config as config
import modules.mask_processing as mask_processing
from modules import model_registry
import backend.resources as resources
from modules.util import HWC3

logger = logging.getLogger(__name__)

_remover_instance = None
_cached_jit = False


def load_model(jit: bool = True) -> Remover:
    """Load InSPyReNet model (Remover) on demand."""
    global _remover_instance, _cached_jit

    if _remover_instance is not None and _cached_jit == jit:
        return _remover_instance

    checkpoint_path = model_registry.ensure_asset('removals.background.inspyrenet.base', progress=True)

    logger.info(f"Initializing InSPyReNet BGR engine (JIT={jit}) from {checkpoint_path} ...")

    _remover_instance = Remover(jit=jit, ckpt=checkpoint_path)
    _cached_jit = jit

    return _remover_instance


def remove_background(image: np.ndarray, threshold: float = 0.5, jit: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background from a numpy HWC uint8 image.
    Returns: (rgba_image, binary_mask)
    """
    remover = load_model(jit=jit)

    # Input numpy to PIL
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    pil_img = Image.fromarray(image)

    # Process
    # Remover.process returns a PIL Image in 'rgba' mode when type='rgba'
    result_rgba_pil = remover.process(pil_img, type='rgba', threshold=threshold)

    result_rgba = np.array(result_rgba_pil)

    # Extract alpha channel as mask (uint8, 0 or 255)
    mask = result_rgba[:, :, 3]
    # Ensure binary mask (0 or 255)
    mask = (mask > 127).astype(np.uint8) * 255

    return result_rgba, mask


def remove_background_from_file(filepath: str, threshold: float = 0.5, jit: bool = True) -> tuple[str, str]:
    """
    Convenience function for Filepath Invariant support.
    Loads image from path, runs BGR, saves character + mask as temp PNGs.
    Returns: (character_path, mask_path)
    """
    # Load image with robust alpha handling
    with Image.open(filepath) as img:
        img_np = HWC3(np.array(img.convert('RGBA')))

    rgba, mask = remove_background(img_np, threshold=threshold, jit=jit)

    # Save to temp PNGs
    character_path = mask_processing.save_to_temp_png(rgba)
    mask_path = mask_processing.save_to_temp_png(mask)

    return character_path, mask_path


def unload_model():
    """Clear memory and release VRAM."""
    global _remover_instance
    if _remover_instance is not None:
        logger.info("Unloading InSPyReNet BGR engine ...")
        del _remover_instance
        _remover_instance = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    resources.soft_empty_cache()

    logger.info("BGR engine memory reclaimed.")
