import os
from dataclasses import dataclass
import torch
import numpy as np
import gc
import logging
from PIL import Image
from typing import Any, List, Tuple

from modules import model_registry
import modules.config as config
from modules.flux_fill_surface import (
    FLUX_FILL_BLEND_ALPHA,
    FLUX_FILL_BLEND_MORPHOLOGICAL,
    FLUX_FILL_INPAINT_ROUTE_FLUX,
    FLUX_FILL_INPAINT_ROUTE_SDXL,
    OBJR_ENGINE_CHOICES,
    OBJR_ENGINE_FLUX_FILL,
    OBJR_ENGINE_MAT,
    is_flux_fill_inpaint_route,
    is_flux_fill_route_family,
    normalize_flux_fill_blend_mode,
    normalize_flux_fill_inpaint_route,
    normalize_objr_engine,
)
import modules.mask_processing as mask_processing
from ldm_patched.pfn.architecture.MAT import MAT
from modules.blending import sin_blend_1d
import backend.resources as resources
from modules.util import HWC3
from backend.flux import LegacyFluxArchivedError

logger = logging.getLogger(__name__)

_model_instance = None

FLUX_FILL_TIER_FP8 = "fp8"
FLUX_FILL_TIER_Q8 = "q8_0"
FLUX_FILL_TIER_Q4 = "q4_k_s"
FLUX_FILL_GUIDANCE_DEFAULT = 15.0
FLUX_FILL_AE_ASSET_ID = "inpaint.flux_fill.ae"
FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID = "inpaint.flux_fill.empty_conditioning"
FLUX_FILL_CLIP_L_ASSET_ID = "inpaint.flux_fill.text_encoder.clip_l"
FLUX_FILL_T5XXL_FP16_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.fp16"
FLUX_FILL_T5XXL_Q8_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.q8_0"
FLUX_FILL_T5XXL_Q4_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.q4_k_m"
FLUX_FILL_T5_VARIANT_FP16 = "fp16"
FLUX_FILL_T5_VARIANT_Q8 = "q8_0"
FLUX_FILL_T5_VARIANT_Q4 = "q4_k_m"
FLUX_FILL_T5_RESIDENT_RESERVE_RAM_MB = 4 * 1024
FLUX_FILL_T5_HYBRID_RESERVE_RAM_MB = 8 * 1024
FLUX_FILL_T5_FP16_MIN_BUDGET_MB = 24 * 1024
FLUX_FILL_T5_Q8_MIN_BUDGET_MB = 12 * 1024
FLUX_FILL_CONDITIONING_EMPTY = "empty"
FLUX_FILL_CONDITIONING_PROMPT = "prompt"
FLUX_FILL_CONDITIONING_BY_KIND = {
    FLUX_FILL_CONDITIONING_EMPTY: FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID,
}
FLUX_FILL_PROMPT_CACHE_TEMP = "temp"
FLUX_FILL_PROMPT_CACHE_PERMANENT = "permanent"
FLUX_FILL_MASK_GROW = 16
FLUX_FILL_MASK_BLUR = 6
FLUX_FILL_UNET_ASSET_BY_TIER = {
    FLUX_FILL_TIER_FP8: "inpaint.flux_fill.unet.fp8",
    FLUX_FILL_TIER_Q8: "inpaint.flux_fill.unet.q8_0",
    FLUX_FILL_TIER_Q4: "inpaint.flux_fill.unet.q4_k_s",
}
FLUX_FILL_MODEL_VARIANT_BY_TIER = {
    FLUX_FILL_TIER_FP8: "flux_fill_fp8",
    FLUX_FILL_TIER_Q8: "flux_fill_q8",
    FLUX_FILL_TIER_Q4: "flux_fill_q4_k_s",
}
FLUX_FILL_TIER_BY_MODEL_VARIANT = {variant: tier for tier, variant in FLUX_FILL_MODEL_VARIANT_BY_TIER.items()}
FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT = {
    model_variant: FLUX_FILL_UNET_ASSET_BY_TIER[tier]
    for tier, model_variant in FLUX_FILL_MODEL_VARIANT_BY_TIER.items()
}
FLUX_FILL_T5_ASSET_BY_VARIANT = {
    FLUX_FILL_T5_VARIANT_FP16: FLUX_FILL_T5XXL_FP16_ASSET_ID,
    FLUX_FILL_T5_VARIANT_Q8: FLUX_FILL_T5XXL_Q8_ASSET_ID,
    FLUX_FILL_T5_VARIANT_Q4: FLUX_FILL_T5XXL_Q4_ASSET_ID,
}
FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH = os.path.join("flux", "flux_empty_conditioning.pt")

FLUX_FILL_VRAM_CLASS_RESIDENT = "16gb_plus"
FLUX_FILL_VRAM_CLASS_CONSTRAINED = "8gb_class"
FLUX_FILL_RUNTIME_POSTURE_RESIDENT = "resident"
FLUX_FILL_RUNTIME_POSTURE_HYBRID = "streaming"
FLUX_FILL_TEXT_ENCODER_ROUTE_BUDGET_MB = {
    "": 0.0,
    "flux_fill": 0.0,
    "removal": 2048.0,
    "upscale": 4096.0,
    "txt2img": 6144.0,
    "image_input": 8192.0,
    "inpaint": 8192.0,
    "outpaint": 8192.0,
    "sdxl": 8192.0,
}


@dataclass(frozen=True)
class FluxFillRouteReconciliation:
    decision: str
    reason: str
    target_signature: tuple[str, str, str] | None = None
    active_signature_before: tuple[str, str, str] | None = None
    active_signature_after: tuple[str, str, str] | None = None
    session_started: bool = False
    session_reused: bool = False
    session_replaced: bool = False
    session_torn_down: bool = False
    next_route_family: str | None = None
    text_encoder_kept: bool | None = None
    text_encoder_action: str | None = None
    text_encoder_reason: str | None = None


@dataclass(frozen=True)
class FluxFillHardwareProfile:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    vram_class: str
    runtime_posture: str
    gpu_name: str | None = None
    cuda_capability: str | None = None
    flux_acceleration_class: str | None = None
    tensor_core_accelerated: bool = False


@dataclass(frozen=True)
class _FluxFillPolicyContext:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    gpu_name: str | None
    cuda_capability: str | None
    flux_acceleration_class: str | None
    tensor_core_accelerated: bool
    placement_plan: Any
def inspect_flux_fill_hardware(profile: Any | None = None) -> Any:
    raise LegacyFluxArchivedError()

def evaluate_flux_fill_text_encoder_residency(profile: Any | None = None, *, next_route_family: Any | None = None) -> Any:
    raise LegacyFluxArchivedError()

def select_flux_fill_tier(profile: Any | None = None) -> str:
    raise LegacyFluxArchivedError()

def normalize_flux_fill_t5_variant(variant: str | None) -> str:
    raise LegacyFluxArchivedError()

def should_keep_flux_fill_text_encoder_resident(profile: Any | None = None, *, next_route_family: Any | None = None) -> bool:
    return False

def reconcile_flux_fill_text_encoder_residency(*, profile: Any | None = None, next_route_family: Any | None = None) -> Any:
    return {"text_encoder_action": "cleared"}

def select_flux_fill_t5_variant(profile: Any | None = None, *, variant: str | None = None) -> str:
    raise LegacyFluxArchivedError()

def get_flux_fill_t5_asset_id(variant: str | None = None, *, profile: Any | None = None) -> str:
    raise LegacyFluxArchivedError()

def ensure_flux_fill_t5_asset(variant: str | None = None, *, profile: Any | None = None, progress: bool = True) -> tuple[str, str, str]:
    raise LegacyFluxArchivedError()

def normalize_flux_fill_conditioning(conditioning: str | None) -> str:
    return "empty"

def normalize_flux_fill_prompt_cache(cache_mode: str | None) -> str:
    return "temp"

def generate_flux_fill_prompt_conditioning_cache(prompt: str, **kwargs) -> str:
    raise LegacyFluxArchivedError()

def prepare_flux_fill_prompt_conditioning_cache_path(prompt: str, **kwargs) -> str:
    raise LegacyFluxArchivedError()

def generate_flux_fill_prompt_conditioning(prompt: str, **kwargs) -> Any:
    raise LegacyFluxArchivedError()

def get_flux_fill_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    raise LegacyFluxArchivedError()

def get_flux_empty_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    raise LegacyFluxArchivedError()

def safe_resolve_flux_fill_asset_paths(**kwargs) -> dict[str, Any]:
    raise LegacyFluxArchivedError()

def resolve_flux_fill_asset_paths(**kwargs) -> dict[str, Any]:
    raise LegacyFluxArchivedError()

def get_active_flux_fill_session() -> Any:
    return None

def get_active_flux_fill_session_signature() -> Any:
    return None

def has_active_flux_fill_session() -> bool:
    return False

def ensure_active_flux_fill_session(**kwargs) -> Any:
    raise LegacyFluxArchivedError()

def reconcile_active_flux_fill_session(**kwargs) -> Any:
    from collections import namedtuple
    Reconciliation = namedtuple("Reconciliation", ["decision", "text_encoder_action"])
    return Reconciliation("ignored", "cleared")

def end_active_flux_fill_session(*args, **kwargs) -> Any:
    return None

def _select_flux_fill_mode(image, mode=None):
    return "context_crop"


@torch.inference_mode()
def remove_object_flux_fill(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    raise LegacyFluxArchivedError()


@torch.inference_mode()
def run_flux_fill_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    raise LegacyFluxArchivedError()


def remove_object_with_engine(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
) -> np.ndarray:
    selected_engine = normalize_objr_engine(engine)
    if selected_engine == OBJR_ENGINE_FLUX_FILL:
        return remove_object_flux_fill(
            image,
            mask,
            seed=seed,
            mask_dilate=mask_dilate,
            mask_blur=flux_mask_blur,
            tier=flux_tier,
            conditioning=flux_conditioning,
            prompt=flux_prompt,
            prompt_cache=flux_prompt_cache,
            blend_mode=flux_blend_mode,
            steps=flux_steps,
            sampler=flux_sampler,
            scheduler=flux_scheduler,
            callback=flux_callback,
            disable_pbar=flux_disable_pbar,
        )
    return remove_object(image, mask, seed=seed, mask_dilate=mask_dilate)

def remove_object_from_file(
    image_path: str,
    mask_path: str,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
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
        flux_conditioning=flux_conditioning,
        flux_prompt=flux_prompt,
        flux_prompt_cache=flux_prompt_cache,
        flux_mask_blur=flux_mask_blur,
        flux_blend_mode=flux_blend_mode,
        flux_steps=flux_steps,
        flux_sampler=flux_sampler,
        flux_scheduler=flux_scheduler,
        flux_callback=flux_callback,
        flux_disable_pbar=flux_disable_pbar,
    )

    return mask_processing.save_to_temp_png(res_np)


def prepare_flux_fill_mask(mask: np.ndarray, *, grow: int = FLUX_FILL_MASK_GROW, blur: int = FLUX_FILL_MASK_BLUR) -> np.ndarray:
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


_expand_flux_fill_mask = prepare_flux_fill_mask


# --- MAT Helper Utilities ---

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


def get_segments(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [(0, length, 0, 0)] # start, end, pad_l, pad_r

    segments = []
    # First
    segments.append((0, tile_size - overlap, 0, overlap))

    while segments[-1][1] < length:
        start = segments[-1][1]
        tile_start = start - overlap
        if tile_start + tile_size >= length:
            end = length
            final_tile_start = max(0, length - tile_size)
            pad_l = start - final_tile_start
            segments.append((start, end, pad_l, 0))
            break

        end = start + tile_size - overlap * 2
        segments.append((start, end, overlap, overlap))
    return segments


# --- MAT Core Engine ---

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
def remove_object(image: np.ndarray, mask: np.ndarray, seed: int = 0, mask_dilate: int = FLUX_FILL_MASK_GROW) -> np.ndarray:
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
        res_sq = model(img_sq, mask_sq)

        # Undo resize/padding
        res_t = undo_resize_square(res_sq, orig_info)

        # Composite: original * (1-mask) + result * mask
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


