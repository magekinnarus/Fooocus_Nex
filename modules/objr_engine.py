import os
import torch
import numpy as np
import gc
import logging
import math
from PIL import Image
from typing import Any, List, Tuple

from modules import model_registry
import modules.config as config
import modules.mask_processing as mask_processing
from ldm_patched.pfn.architecture.MAT import MAT
from modules.blending import sin_blend_1d
import backend.resources as resources
from modules.util import HWC3

logger = logging.getLogger(__name__)

_model_instance = None
OBJR_ENGINE_MAT = "MAT (Local)"
OBJR_ENGINE_FLUX_FILL = "Flux Fill (Colab)"
OBJR_ENGINE_CHOICES = (OBJR_ENGINE_MAT, OBJR_ENGINE_FLUX_FILL)

FLUX_FILL_TIER_Q8 = "q8_0"
FLUX_FILL_TIER_Q4 = "q4_k_s"
FLUX_FILL_GUIDANCE_DEFAULT = 15.0
FLUX_FILL_LOCAL_Q8_MIN_RAM_MB = 24 * 1024
FLUX_FILL_LOCAL_Q8_MIN_VRAM_MB = 16 * 1024
FLUX_FILL_AE_ASSET_ID = "inpaint.flux_fill.ae"
FLUX_FILL_UNET_ASSET_BY_TIER = {
    FLUX_FILL_TIER_Q8: "inpaint.flux_fill.unet.q8_0",
    FLUX_FILL_TIER_Q4: "inpaint.flux_fill.unet.q4_k_s",
}
FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH = os.path.join("flux", "flux_empty_conditioning.pt")


def normalize_objr_engine(engine: str | None) -> str:
    if engine is None or str(engine).strip() == "":
        return OBJR_ENGINE_MAT

    value = str(engine).strip()
    if value in OBJR_ENGINE_CHOICES:
        return value

    aliases = {
        "mat": OBJR_ENGINE_MAT,
        "mat local": OBJR_ENGINE_MAT,
        "places_512_fulldata_g.pth": OBJR_ENGINE_MAT,
        "places512": OBJR_ENGINE_MAT,
        "flux": OBJR_ENGINE_FLUX_FILL,
        "flux fill": OBJR_ENGINE_FLUX_FILL,
        "flux fill colab": OBJR_ENGINE_FLUX_FILL,
    }
    normalized = value.lower().replace("(", "").replace(")", "").strip()
    if normalized in aliases:
        return aliases[normalized]

    raise ValueError(f"Unsupported object removal engine: {engine!r}. Expected one of {OBJR_ENGINE_CHOICES}.")


def select_flux_fill_tier(profile: Any | None = None) -> str:
    if profile is None:
        try:
            from backend import memory_governor

            profile = memory_governor.environment_profile()
        except Exception:
            profile = None
    if profile is None:
        profile = getattr(config, "resolved_memory_environment_profile", None)

    profile_name = str(getattr(profile, "name", "") or "").lower()
    total_ram_mb = float(getattr(profile, "total_ram_mb", 0.0) or 0.0)
    total_vram_mb = float(getattr(profile, "total_vram_mb", 0.0) or 0.0)

    try:
        from backend import environment_profile

        if profile_name == environment_profile.PROFILE_COLAB_PRO:
            return FLUX_FILL_TIER_Q8
        if profile_name == environment_profile.PROFILE_COLAB_FREE:
            return FLUX_FILL_TIER_Q4
    except Exception:
        if profile_name == "colab_pro":
            return FLUX_FILL_TIER_Q8
        if profile_name == "colab_free":
            return FLUX_FILL_TIER_Q4

    if total_ram_mb >= FLUX_FILL_LOCAL_Q8_MIN_RAM_MB and total_vram_mb >= FLUX_FILL_LOCAL_Q8_MIN_VRAM_MB:
        return FLUX_FILL_TIER_Q8
    return FLUX_FILL_TIER_Q4


def _normalize_flux_fill_tier(tier: str | None) -> str:
    if tier is None or str(tier).strip() == "":
        return select_flux_fill_tier()
    normalized = str(tier).strip().lower().replace("-", "_")
    if normalized in {"q8", "q8_0"}:
        return FLUX_FILL_TIER_Q8
    if normalized in {"q4", "q4_k_s"}:
        return FLUX_FILL_TIER_Q4
    raise ValueError(f"Unsupported Flux Fill tier: {tier!r}. Expected q8_0 or q4_k_s.")


def get_flux_empty_conditioning_cache_path() -> str:
    clip_root = config.get_preferred_asset_root_path(
        "clip",
        file_name=os.path.basename(FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH),
        relative_path=FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH,
    )
    return os.path.join(clip_root, FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH)


def resolve_flux_fill_asset_paths(tier: str | None = None, *, progress: bool = True) -> dict[str, str]:
    selected_tier = _normalize_flux_fill_tier(tier)
    unet_asset_id = FLUX_FILL_UNET_ASSET_BY_TIER[selected_tier]
    unet_asset = model_registry.get_asset(unet_asset_id)
    if unet_asset is None:
        raise KeyError(f"Unknown Flux Fill UNet asset id: {unet_asset_id}")

    required_asset_ids = list(unet_asset.get("requires", []))
    if FLUX_FILL_AE_ASSET_ID not in required_asset_ids:
        required_asset_ids.append(FLUX_FILL_AE_ASSET_ID)

    resolved_required_paths = {}
    for asset_id in required_asset_ids:
        resolved_required_paths[asset_id] = model_registry.ensure_asset(asset_id, progress=progress)

    unet_path = model_registry.ensure_asset(unet_asset_id, progress=progress)
    ae_path = resolved_required_paths.get(FLUX_FILL_AE_ASSET_ID) or model_registry.ensure_asset(FLUX_FILL_AE_ASSET_ID, progress=progress)

    return {
        "tier": selected_tier,
        "unet_asset_id": unet_asset_id,
        "unet_path": unet_path,
        "ae_asset_id": FLUX_FILL_AE_ASSET_ID,
        "ae_path": ae_path,
        "conditioning_cache_path": get_flux_empty_conditioning_cache_path(),
    }


def _expand_flux_fill_mask(mask: np.ndarray, *, grow: int = 16, blur: int = 6) -> np.ndarray:
    import cv2

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Flux Fill mask must be HW or HWC, got shape {mask_np.shape}.")

    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    if grow > 0:
        kernel_size = max(1, int(grow) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    if blur > 0:
        kernel_size = max(3, int(blur) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    return mask_np.clip(0, 255).astype(np.uint8)
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

@torch.inference_mode()
def remove_object_flux_fill(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = 16,
    *,
    tier: str | None = None,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    progress: bool = True,
) -> np.ndarray:
    """
    Remove objects with the direct Flux Fill runtime.
    Asset resolution is intentionally delayed until this function is called.
    """
    if image.ndim != 3:
        raise ValueError(f"Flux Fill image must be HWC, got shape {image.shape}.")
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"Flux Fill mask shape {mask.shape[:2]} does not match image shape {image.shape[:2]}.")

    flux_grow = max(0, int(mask_dilate or 0))
    flux_mask = _expand_flux_fill_mask(np.asarray(mask), grow=flux_grow, blur=6)

    asset_paths = resolve_flux_fill_asset_paths(tier=tier, progress=progress)

    from backend.flux import FluxFillConfig, run_flux_fill

    flux_config = FluxFillConfig(
        unet_path=asset_paths["unet_path"],
        ae_path=asset_paths["ae_path"],
        conditioning_cache_path=asset_paths["conditioning_cache_path"],
        tier=asset_paths["tier"],
        seed=int(seed),
        guidance=float(guidance),
    )
    result = run_flux_fill(flux_config, HWC3(image), flux_mask, extend_factor=1.2, disable_pbar=True)
    return HWC3(np.asarray(result.output_image))


def remove_object_with_engine(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = 0,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
) -> np.ndarray:
    selected_engine = normalize_objr_engine(engine)
    if selected_engine == OBJR_ENGINE_FLUX_FILL:
        return remove_object_flux_fill(image, mask, seed=seed, mask_dilate=mask_dilate, tier=flux_tier)
    return remove_object(image, mask, seed=seed, mask_dilate=mask_dilate)

def remove_object_from_file(
    image_path: str,
    mask_path: str,
    seed: int = 0,
    mask_dilate: int = 0,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
) -> str:
    """Filepath invariant wrapper with explicit MAT/Flux dispatch."""
    with Image.open(image_path) as img:
        img_np = HWC3(np.array(img.convert('RGBA')))
    with Image.open(mask_path) as msk:
        msk_np = np.array(msk.convert('L'))

    res_np = remove_object_with_engine(
        img_np,
        msk_np,
        seed=seed,
        mask_dilate=mask_dilate,
        engine=engine,
        flux_tier=flux_tier,
    )

    return mask_processing.save_to_temp_png(res_np)
