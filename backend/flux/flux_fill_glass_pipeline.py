from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import modules.blending as blending
import modules.flags as flags

from backend import resources
from backend.flux.flux_fill_pipeline import (
    FluxEmptyConditioning,
    FluxFillDenoiseResult,
    FluxFillDecodedImage,
    FluxFillLatentSource,
    FluxFillValidationError,
    build_flux_fill_conditioning_payloads,
    create_flux_fill_noise,
    decode_flux_fill_latent,
    denoise_flux_fill_latent,
    load_flux_ae,
    load_flux_empty_conditioning_cache,
    load_flux_fill_unet,
)

FLUX_FILL_GLASS_MODES = ("baseline", "context_crop", "debug", "scaled")
FLUX_FILL_GLASS_DEFAULT_MODE = "baseline"
FLUX_FILL_GLASS_BLEND_MODES = ("alpha", "morphological")
FLUX_FILL_GLASS_DEFAULT_BLEND_MODE = "morphological"

FLUX_FILL_CANVAS_ALIGNMENT = 16
FLUX_FILL_CANVAS_MIN_PIXELS = 1_100_000
FLUX_FILL_CANVAS_MAX_PIXELS = 1_200_000
FLUX_FILL_CANVAS_TARGET_PIXELS = 1_150_000
_SDXL_BUCKETS: tuple[tuple[int, int], ...] = tuple(
    (int(value.split("*")[0]), int(value.split("*")[1])) for value in flags.sdxl_aspect_ratios
)


def _normalize_glass_mode(mode: str | None) -> str:
    value = str(mode or FLUX_FILL_GLASS_DEFAULT_MODE).strip().lower()
    if value in FLUX_FILL_GLASS_MODES:
        return value
    raise FluxFillValidationError(
        f"Unsupported Flux Fill glass mode: {mode!r}. Expected one of {list(FLUX_FILL_GLASS_MODES)}."
    )


def _normalize_blend_mode(blend_mode: str | None) -> str:
    value = str(blend_mode or FLUX_FILL_GLASS_DEFAULT_BLEND_MODE).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"morph", "fooocus"}:
        value = "morphological"
    if value in FLUX_FILL_GLASS_BLEND_MODES:
        return value
    raise FluxFillValidationError(
        f"Unsupported Flux Fill glass blend mode: {blend_mode!r}. Expected one of {list(FLUX_FILL_GLASS_BLEND_MODES)}."
    )

def _round_to_multiple(value: float, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return max(multiple, int(round(float(value) / float(multiple))) * int(multiple))


def _scale_working_dimensions(width: int, height: int, *, target_megapixels: float, multiple: int = FLUX_FILL_CANVAS_ALIGNMENT) -> tuple[int, int, float]:
    if width < 1 or height < 1:
        raise FluxFillValidationError(f"image dimensions must be positive, got {width}x{height}.")

    current_megapixels = float(width * height) / 1_000_000.0
    target_megapixels = float(np.clip(float(target_megapixels), FLUX_FILL_CANVAS_MIN_PIXELS / 1_000_000.0, FLUX_FILL_CANVAS_MAX_PIXELS / 1_000_000.0))
    needs_rescale = (
        current_megapixels < (FLUX_FILL_CANVAS_MIN_PIXELS / 1_000_000.0)
        or current_megapixels > (FLUX_FILL_CANVAS_MAX_PIXELS / 1_000_000.0)
        or (width % multiple != 0)
        or (height % multiple != 0)
    )
    if not needs_rescale:
        return width, height, 1.0

    source_area = float(width * height)
    target_area = float(target_megapixels) * 1_000_000.0
    target_area = float(np.clip(target_area, FLUX_FILL_CANVAS_MIN_PIXELS, FLUX_FILL_CANVAS_MAX_PIXELS))
    source_aspect = float(width) / float(height)
    base_scale = float(np.sqrt(target_area / max(source_area, 1e-8)))
    base_width = _round_to_multiple(float(width) * base_scale, multiple)
    base_height = _round_to_multiple(float(height) * base_scale, multiple)

    candidates: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    def _add_candidate(candidate_width: int, candidate_height: int) -> None:
        candidate_width = max(multiple, int(candidate_width))
        candidate_height = max(multiple, int(candidate_height))
        area = float(candidate_width * candidate_height)
        if candidate_width <= 0 or candidate_height <= 0:
            return
        score = (
            0.0 if FLUX_FILL_CANVAS_MIN_PIXELS <= area <= FLUX_FILL_CANVAS_MAX_PIXELS else 1.0,
            abs(area - target_area),
            abs((float(candidate_width) / float(candidate_height)) - source_aspect),
            abs(area - source_area),
        )
        existing = candidates.get((candidate_width, candidate_height))
        if existing is None or score < existing:
            candidates[(candidate_width, candidate_height)] = score

    for offset in range(-24, 25):
        candidate_width = max(multiple, base_width + offset * multiple)
        candidate_height = _round_to_multiple(float(candidate_width) / max(source_aspect, 1e-8), multiple)
        _add_candidate(candidate_width, candidate_height)

        candidate_height = max(multiple, base_height + offset * multiple)
        candidate_width = _round_to_multiple(float(candidate_height) * source_aspect, multiple)
        _add_candidate(candidate_width, candidate_height)

    if not candidates:
        return width, height, 1.0

    best_width, best_height = min(candidates.items(), key=lambda item: item[1])[0]
    best_area = float(best_width * best_height)
    return int(best_width), int(best_height), float(np.sqrt(best_area / max(source_area, 1e-8)))


def select_flux_fill_canvas_dimensions(
    width: int,
    height: int,
    *,
    target_megapixels: float = FLUX_FILL_CANVAS_TARGET_PIXELS / 1_000_000.0,
    multiple: int = FLUX_FILL_CANVAS_ALIGNMENT,
) -> tuple[int, int, float]:
    return _scale_working_dimensions(width, height, target_megapixels=target_megapixels, multiple=multiple)


def _resize_uint8_rgb(image: np.ndarray, width: int, height: int, *, resample: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((int(width), int(height)), resample=resample),
        dtype=np.uint8,
    )


def _resize_uint8_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    mask_2d = _mask_2d(np.asarray(mask))
    resized = Image.fromarray(np.asarray(mask_2d, dtype=np.uint8)).resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _sdxl_bucket_aspect(bucket: tuple[int, int]) -> float:
    return float(bucket[0]) / float(bucket[1])


def _reduce_aspect_pair(width: int, height: int) -> tuple[int, int]:
    divisor = math.gcd(int(width), int(height))
    divisor = max(divisor, 1)
    return int(width) // divisor, int(height) // divisor


_SDXL_BUCKET_ASPECT_PAIRS: frozenset[tuple[int, int]] = frozenset(
    _reduce_aspect_pair(bucket_width, bucket_height) for bucket_width, bucket_height in _SDXL_BUCKETS
)


def is_native_flux_dimensions(width: int, height: int) -> bool:
    if width < 1 or height < 1:
        return False
    area = float(width * height)
    return (
        width % FLUX_FILL_CANVAS_ALIGNMENT == 0
        and height % FLUX_FILL_CANVAS_ALIGNMENT == 0
        and FLUX_FILL_CANVAS_MIN_PIXELS <= area <= FLUX_FILL_CANVAS_MAX_PIXELS
    )


def is_native_sdxl_dimensions(width: int, height: int) -> bool:
    return (int(width), int(height)) in _SDXL_BUCKETS


def is_sdxl_bucket_aspect_ratio(width: int, height: int) -> bool:
    if width < 1 or height < 1:
        return False
    return _reduce_aspect_pair(width, height) in _SDXL_BUCKET_ASPECT_PAIRS


def select_sdxl_bucket_for_aspect(
    aspect: float,
) -> tuple[int, int]:
    if aspect <= 0:
        raise FluxFillValidationError(f"aspect must be positive, got {aspect!r}.")
    return min(_SDXL_BUCKETS, key=lambda bucket: (abs(_sdxl_bucket_aspect(bucket) - float(aspect)), bucket[0] * bucket[1]))


def _mask_bbox_from_binary(mask_binary: np.ndarray, *, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_binary > 0)
    if ys.size == 0 or xs.size == 0:
        height, width = image_shape
        return 0, int(height), 0, int(width)
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    return y1, y2, x1, x2


def _pad_to_aspect(
    image: np.ndarray,
    mask: np.ndarray,
    target_aspect: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image_np = _ensure_uint8_rgb(image)
    mask_np = _mask_2d(_ensure_mask_shape(image_np, mask))
    height, width = image_np.shape[:2]
    current_aspect = float(width) / float(height)
    if abs(current_aspect - target_aspect) <= 1e-6:
        return image_np, mask_np, (0, 0, 0, 0)

    if current_aspect > target_aspect:
        padded_height = int(math.ceil(float(width) / float(target_aspect)))
        pad_total = max(0, padded_height - height)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    else:
        padded_width = int(math.ceil(float(height) * float(target_aspect)))
        pad_total = max(0, padded_width - width)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0

    padded_image = np.pad(image_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    padded_mask = np.pad(mask_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    return padded_image.astype(np.uint8, copy=False), padded_mask.astype(np.uint8, copy=False), (pad_top, pad_bottom, pad_left, pad_right)


def _crop_box_to_slice(crop_box: tuple[int, int, int, int]) -> tuple[slice, slice]:
    y1, y2, x1, x2 = crop_box
    return slice(int(y1), int(y2)), slice(int(x1), int(x2))


def _crop_image_and_mask(image: np.ndarray, mask: np.ndarray, crop_box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    y_slice, x_slice = _crop_box_to_slice(crop_box)
    image_crop = np.asarray(image)[y_slice, x_slice]
    mask_crop = np.asarray(mask)[y_slice, x_slice]
    return _ensure_uint8_rgb(image_crop), _mask_2d(_ensure_mask_shape(image_crop, mask_crop))


@dataclass(frozen=True)
class FluxFillGlassCropPlan:
    mode: str
    crop_box: tuple[int, int, int, int]
    target_canvas: tuple[int, int]
    target_canvas_aspect: float
    target_canvas_area: int
    crop_width: int
    crop_height: int
    normalized_width: int
    normalized_height: int
    target_width: int
    target_height: int
    scale_to_canvas: float
    mask_bbox: tuple[int, int, int, int]
    mask_bbox_width_ratio: float
    mask_bbox_height_ratio: float
    mask_area_ratio: float
    full_image_crop: bool
    large_mask_full_context: bool
    context_limited_by_bounds: bool
    crop_aspect: float
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_image_width: int = 0
    source_image_height: int = 0

    @property
    def bucket(self) -> tuple[int, int]:
        return self.target_canvas

    @property
    def bucket_aspect(self) -> float:
        return self.target_canvas_aspect

    @property
    def scale_to_bucket(self) -> float:
        return self.scale_to_canvas

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "crop_box": [int(v) for v in self.crop_box],
            "target_canvas": [int(v) for v in self.target_canvas],
            "target_canvas_aspect": float(self.target_canvas_aspect),
            "target_canvas_area": int(self.target_canvas_area),
            "crop_width": int(self.crop_width),
            "crop_height": int(self.crop_height),
            "normalized_width": int(self.normalized_width),
            "normalized_height": int(self.normalized_height),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "scale_to_canvas": float(self.scale_to_canvas),
            "mask_bbox": [int(v) for v in self.mask_bbox],
            "mask_bbox_width_ratio": float(self.mask_bbox_width_ratio),
            "mask_bbox_height_ratio": float(self.mask_bbox_height_ratio),
            "mask_area_ratio": float(self.mask_area_ratio),
            "full_image_crop": bool(self.full_image_crop),
            "large_mask_full_context": bool(self.large_mask_full_context),
            "context_limited_by_bounds": bool(self.context_limited_by_bounds),
            "crop_aspect": float(self.crop_aspect),
            "pad": [int(v) for v in self.pad],
            "source_image_width": int(self.source_image_width),
            "source_image_height": int(self.source_image_height),
            "bucket": [int(v) for v in self.target_canvas],
            "bucket_aspect": float(self.target_canvas_aspect),
            "scale_to_bucket": float(self.scale_to_canvas),
        }


def _select_context_crop_plan(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "context_crop",
) -> FluxFillGlassCropPlan:
    image_np = _ensure_uint8_rgb(image)
    mask_np = _ensure_mask_shape(image_np, mask)
    mask_binary = _binary_mask(mask_np)
    height, width = image_np.shape[:2]
    image_area = max(1, int(height * width))
    mask_area = int(mask_binary.sum())
    mask_bbox = _mask_bbox_from_binary(mask_binary, image_shape=(height, width))
    bbox_y1, bbox_y2, bbox_x1, bbox_x2 = mask_bbox
    bbox_height = max(1, int(bbox_y2 - bbox_y1))
    bbox_width = max(1, int(bbox_x2 - bbox_x1))
    source_aspect = float(width) / float(height)
    target_width, target_height, scale_to_canvas = select_flux_fill_canvas_dimensions(
        width,
        height,
        target_megapixels=FLUX_FILL_CANVAS_TARGET_PIXELS / 1_000_000.0,
        multiple=FLUX_FILL_CANVAS_ALIGNMENT,
    )
    return FluxFillGlassCropPlan(
        mode=mode,
        crop_box=(0, int(height), 0, int(width)),
        target_canvas=(int(target_width), int(target_height)),
        target_canvas_aspect=float(target_width) / float(max(target_height, 1)),
        target_canvas_area=int(target_width * target_height),
        crop_width=int(width),
        crop_height=int(height),
        normalized_width=int(target_width),
        normalized_height=int(target_height),
        target_width=int(target_width),
        target_height=int(target_height),
        scale_to_canvas=float(scale_to_canvas),
        mask_bbox=mask_bbox,
        mask_bbox_width_ratio=float(bbox_width) / float(width),
        mask_bbox_height_ratio=float(bbox_height) / float(height),
        mask_area_ratio=float(mask_area) / float(image_area),
        full_image_crop=True,
        large_mask_full_context=(float(mask_area) / float(image_area)) > (1.0 / 6.0),
        context_limited_by_bounds=False,
        crop_aspect=source_aspect,
        pad=(0, 0, 0, 0),
        source_image_width=int(width),
        source_image_height=int(height),
    )


select_flux_fill_glass_context_crop_plan = _select_context_crop_plan


def prepare_flux_fill_glass_context_crop(
    image: np.ndarray,
    mask: np.ndarray,
    plan: FluxFillGlassCropPlan,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_np = _ensure_uint8_rgb(image)
    mask_np = _ensure_mask_shape(image_np, mask)
    crop_image, crop_mask = _crop_image_and_mask(image_np, mask_np, plan.crop_box)
    normalized_image = _resize_uint8_rgb(crop_image, plan.target_width, plan.target_height, resample=Image.Resampling.LANCZOS)
    normalized_mask = _resize_uint8_mask(crop_mask, plan.target_width, plan.target_height)
    return normalized_image, normalized_mask, {
        "plan": plan.to_dict(),
        "crop_image": crop_image,
        "crop_mask": crop_mask,
        "normalized_image": normalized_image,
        "normalized_mask": normalized_mask,
    }


def restore_flux_fill_glass_context_crop(decoded_bucket_image: np.ndarray, plan: FluxFillGlassCropPlan) -> np.ndarray:
    bucket_image = _ensure_uint8_rgb(decoded_bucket_image)
    restored = _resize_uint8_rgb(bucket_image, plan.normalized_width, plan.normalized_height, resample=Image.Resampling.LANCZOS)
    if plan.pad != (0, 0, 0, 0):
        pad_top, pad_bottom, pad_left, pad_right = plan.pad
        y_end = restored.shape[0] - int(pad_bottom)
        x_end = restored.shape[1] - int(pad_right)
        restored = restored[int(pad_top):y_end, int(pad_left):x_end]
    if restored.shape[:2] != (plan.crop_height, plan.crop_width):
        restored = _resize_uint8_rgb(restored, plan.crop_width, plan.crop_height, resample=Image.Resampling.LANCZOS)
    return restored


def stitch_flux_fill_glass_context_crop(
    original_image: np.ndarray,
    mask: np.ndarray,
    plan: FluxFillGlassCropPlan,
    decoded_crop: np.ndarray,
    *,
    blend_mode: str = FLUX_FILL_GLASS_DEFAULT_BLEND_MODE,
) -> np.ndarray:
    original_rgb = _ensure_uint8_rgb(original_image)
    canvas = original_rgb.copy()
    y1, y2, x1, x2 = plan.crop_box
    crop_rgb = _ensure_uint8_rgb(decoded_crop)
    canvas[int(y1):int(y2), int(x1):int(x2)] = crop_rgb
    if blend_mode == "morphological":
        alpha = _morphological_blend_mask(mask).astype(np.float32)[..., None] / 255.0
        alpha = blending.apply_sin2_curve(alpha)
    else:
        alpha = _mask_2d(mask).astype(np.float32)[..., None] / 255.0
    composite = np.clip(
        canvas.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    return composite


@dataclass
class FluxFillGlassConfig:
    unet_path: Path | str
    ae_path: Path | str
    conditioning_cache_path: Path | str
    image_path: Path | str | None = None
    mask_path: Path | str | None = None
    output_path: Path | str | None = None
    tier: str = "q8_0"
    seed: int = 882699830973928
    steps: int = 30
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "normal"
    denoise: float = 1.0
    guidance: float = 15.0
    device: str | None = None
    debug_output_dir: Path | str | None = None
    mode: str = FLUX_FILL_GLASS_DEFAULT_MODE
    blend_mode: str = FLUX_FILL_GLASS_DEFAULT_BLEND_MODE
    target_megapixels: float = FLUX_FILL_CANVAS_TARGET_PIXELS / 1_000_000.0
    verify_c_concat: bool = True
    capture_artifacts: bool = False
    capture_tensors: bool = False
    save_composite: bool = False

    def validate_static(self, *, require_existing_assets: bool = True) -> None:
        self.mode = _normalize_glass_mode(self.mode)
        self.blend_mode = _normalize_blend_mode(self.blend_mode)
        if self.steps < 1:
            raise FluxFillValidationError(f"steps must be >= 1, got {self.steps}.")
        if self.cfg != 1.0:
            raise NotImplementedError("Prompt-conditioned/CFG Flux Fill is out of scope for W04.")
        if self.sampler != "euler":
            raise NotImplementedError(f"Unsupported Flux Fill sampler: {self.sampler!r}.")
        if self.scheduler != "normal":
            raise NotImplementedError(f"Unsupported Flux Fill scheduler: {self.scheduler!r}.")
        if self.denoise != 1.0:
            raise NotImplementedError("W04 glass baseline only supports denoise=1.0.")
        if self.guidance <= 0:
            raise FluxFillValidationError(f"guidance must be > 0, got {self.guidance}.")
        if self.target_megapixels <= 0:
            raise FluxFillValidationError(f"target_megapixels must be > 0, got {self.target_megapixels}.")
        if require_existing_assets:
            for label, value in (("UNet", self.unet_path), ("AE", self.ae_path), ("conditioning cache", self.conditioning_cache_path)):
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"{label} path does not exist: {path}")


@dataclass(frozen=True)
class FluxFillGlassResult:
    output_image: Any
    output_path: Path | None
    seed: int
    width: int
    height: int
    raw_output_image: Any | None = None
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_summary: dict[str, Any] = field(default_factory=dict)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {}
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in cpu_tensor.shape],
        "dtype": str(cpu_tensor.dtype),
        "device": str(tensor.device),
        "sha256": _sha256_bytes(cpu_tensor.numpy().tobytes()),
    }
    if cpu_tensor.numel() > 0 and cpu_tensor.dtype.is_floating_point:
        float_tensor = cpu_tensor.float()
        summary.update(
            {
                "min": float(float_tensor.min().item()),
                "max": float(float_tensor.max().item()),
                "mean": float(float_tensor.mean().item()),
                "std": float(float_tensor.std(unbiased=False).item()) if float_tensor.numel() > 1 else 0.0,
            }
        )
    return summary


def _array_summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {}
    arr = np.asarray(array)
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "sha256": _sha256_bytes(np.ascontiguousarray(arr).tobytes()),
    }
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        arr_float = arr.astype(np.float32, copy=False)
        summary.update(
            {
                "min": float(arr_float.min()),
                "max": float(arr_float.max()),
                "mean": float(arr_float.mean()),
                "std": float(arr_float.std()),
            }
        )
    return summary


def _mask_2d(mask: np.ndarray) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 2:
        return mask_np
    if mask_np.ndim == 3 and mask_np.shape[2] >= 1:
        return mask_np[:, :, 0]
    raise FluxFillValidationError(f"mask must have shape [H, W] or [H, W, C], got {mask_np.shape}.")


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (_mask_2d(mask) > 127).astype(np.uint8)


def _max_filter_opencv(x: np.ndarray, ksize: int = 3) -> np.ndarray:
    import cv2

    return cv2.dilate(x, np.ones((ksize, ksize), dtype=np.int16))


def _morphological_blend_mask(mask: np.ndarray) -> np.ndarray:
    mask_binary = _binary_mask(mask)
    x_int16 = np.zeros_like(mask_binary, dtype=np.int16)
    x_int16[mask_binary > 0] = 256
    for _ in range(32):
        maxed = _max_filter_opencv(x_int16, ksize=3) - 8
        x_int16 = np.maximum(maxed, x_int16)
    return np.clip(x_int16, 0, 255).astype(np.uint8)

def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise FluxFillValidationError(f"image must have shape [H, W, 3], got {image_np.shape}.")
    if image_np.size == 0:
        raise FluxFillValidationError("image must not be empty.")
    if not np.issubdtype(image_np.dtype, np.number):
        raise FluxFillValidationError(f"image must be numeric, got dtype {image_np.dtype}.")
    return np.clip(image_np, 0, 255).astype(np.uint8, copy=False)


def _ensure_mask_shape(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim not in (2, 3):
        raise FluxFillValidationError(f"mask must have shape [H, W] or [H, W, C], got {mask_np.shape}.")
    if mask_np.shape[:2] != image.shape[:2]:
        raise FluxFillValidationError(f"mask spatial shape {mask_np.shape[:2]} does not match image shape {image.shape[:2]}.")
    if mask_np.size == 0:
        raise FluxFillValidationError("mask must not be empty.")
    return np.clip(mask_np, 0, 255).astype(np.uint8, copy=False)


def _prepare_scaled_inputs(image: np.ndarray, mask: np.ndarray, *, target_megapixels: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_np = _ensure_uint8_rgb(image)
    mask_np = _ensure_mask_shape(image_np, mask)
    height, width = image_np.shape[:2]
    scaled_width, scaled_height, scale_factor = _scale_working_dimensions(
        width,
        height,
        target_megapixels=target_megapixels,
        multiple=FLUX_FILL_CANVAS_ALIGNMENT,
    )
    if scaled_width == width and scaled_height == height:
        return image_np, mask_np, {
            "original_width": width,
            "original_height": height,
            "working_width": width,
            "working_height": height,
            "scale_factor": 1.0,
            "scaled": False,
        }

    resized_image = _resize_uint8_rgb(image_np, scaled_width, scaled_height, resample=Image.Resampling.LANCZOS)
    resized_mask = _resize_uint8_mask(mask_np, scaled_width, scaled_height)
    return resized_image, resized_mask, {
        "original_width": width,
        "original_height": height,
        "working_width": scaled_width,
        "working_height": scaled_height,
        "scale_factor": scale_factor,
        "scaled": True,
    }


def _artifact_path(root: Path, name: str, suffix: str) -> Path:
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return root / f"{safe_name}{suffix}"


def _build_glass_concat_condition(unet_patcher: Any, *, noise: torch.Tensor, concat_latent: torch.Tensor, denoise_mask: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    model = getattr(unet_patcher, "model", unet_patcher)
    concat_cond = getattr(model, "concat_cond", None)
    if not callable(concat_cond):
        raise FluxFillValidationError("Flux model does not expose concat_cond().")
    return concat_cond(noise=noise, concat_latent_image=concat_latent, denoise_mask=denoise_mask, device=device)


class FluxFillGlassPipeline:
    route_label = "flux_fill_glass"
    stage_order = (
        "validate_contract",
        "select_working_geometry",
        "prepare_source_pixels",
        "prepare_concat_pixels",
        "encode_source_latent",
        "encode_concat_latent",
        "prepare_denoise_mask",
        "build_conditioning_payload",
        "verify_c_concat",
        "denoise",
        "decode",
        "compose_debug",
    )

    def __init__(self, config: FluxFillGlassConfig, *, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or (torch.device(config.device) if config.device else resources.get_torch_device())

    def _resolve_effective_mode(self, image: np.ndarray, mode: str) -> str:
        normalized_mode = _normalize_glass_mode(mode)
        if normalized_mode == "debug":
            width = int(image.shape[1])
            height = int(image.shape[0])
            if is_native_sdxl_dimensions(width, height):
                return "baseline"
            return "context_crop"
        return normalized_mode

    def validate_input_contract(self, image: np.ndarray, mask: np.ndarray, *, mode: str) -> dict[str, Any]:
        image_np = _ensure_uint8_rgb(image)
        mask_np = _ensure_mask_shape(image_np, mask)
        height, width = image_np.shape[:2]
        normalized_mode = _normalize_glass_mode(mode)
        effective_mode = self._resolve_effective_mode(image_np, normalized_mode)
        if effective_mode == "baseline" and not is_native_sdxl_dimensions(width, height):
            raise FluxFillValidationError(
                f"W04 glass baseline does not scale or crop; image dimensions must be native SDXL bucket resolutions. Got {width}x{height}."
            )
        return {
            "image": _array_summary(image_np),
            "mask": _array_summary(mask_np),
            "mode": normalized_mode,
            "effective_mode": effective_mode,
            "native_flux": bool(is_native_flux_dimensions(width, height)),
            "native_sdxl": bool(is_native_sdxl_dimensions(width, height)),
            "sdxl_bucket_aspect": bool(is_sdxl_bucket_aspect_ratio(width, height)),
            "no_bb": True,
            "no_scale": effective_mode == "baseline",
            "multiple_of_8": (height % 8 == 0 and width % 8 == 0),
            "multiple_of_16": (height % 16 == 0 and width % 16 == 0),
            "prefer_multiple_of_16": (height % 16 == 0 and width % 16 == 0),
            "mask_coverage": float(_binary_mask(mask_np).mean()) if mask_np.size else 0.0,
        }

    def prepare_source_pixels(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        source_pixels = _ensure_uint8_rgb(image).copy()
        return source_pixels, {
            "stage": "prepare_source_pixels",
            "pixels": _array_summary(source_pixels),
            "fill_value": None,
            "full_image": True,
        }

    def prepare_concat_pixels(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        source_pixels = _ensure_uint8_rgb(image).copy()
        mask_binary = _binary_mask(mask)
        concat_pixels = source_pixels.copy()
        concat_pixels[mask_binary > 0] = 128
        return concat_pixels, {
            "stage": "prepare_concat_pixels",
            "pixels": _array_summary(concat_pixels),
            "fill_value": 0.5,
            "mask_coverage": float(mask_binary.mean()) if mask_binary.size else 0.0,
            "full_image": True,
        }

    def _pixels_to_torch(self, pixels: np.ndarray) -> torch.Tensor:
        pixels_np = np.asarray(pixels, dtype=np.float32) / 255.0
        pixels_np = np.ascontiguousarray(pixels_np[None].copy())
        return torch.from_numpy(pixels_np).float()

    def _encode_pixels(
        self,
        pixels: np.ndarray,
        ae_path: Path | str,
        *,
        stage_name: str,
        apply_latent_format: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        vae = load_flux_ae(ae_path, load_device=self.device, offload_device=None)
        start = time.perf_counter()
        try:
            resources.load_models_gpu([vae.patcher])
            pixels_tensor = self._pixels_to_torch(pixels)
            if apply_latent_format:
                latent = vae.encode(pixels_tensor)["samples"]
            else:
                vae_device = getattr(vae.patcher, "load_device", self.device)
                first_stage_model = vae.first_stage_model
                move_model = getattr(first_stage_model, "to", None)
                if callable(move_model):
                    move_model(device=vae_device, dtype=torch.float32)

                pixels_for_vae = (pixels_tensor.movedim(-1, 1) * 2.0) - 1.0
                if pixels_for_vae.ndim == 3:
                    pixels_for_vae = pixels_for_vae.unsqueeze(0)
                pixels_for_vae = pixels_for_vae.to(device=vae_device, dtype=torch.float32)
                latent = first_stage_model.encode(pixels_for_vae)
                if hasattr(latent, "sample"):
                    latent = latent.sample()
        finally:
            try:
                vae.patcher.detach()
            finally:
                resources.soft_empty_cache()
        elapsed = time.perf_counter() - start
        summary = {
            "stage": stage_name,
            "latent": _tensor_summary(latent),
            "elapsed": elapsed,
            "latent_format": "processed" if apply_latent_format else "raw_vae",
        }
        return latent.detach().cpu(), summary

    def encode_source_latent(self, source_pixels: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._encode_pixels(
            source_pixels,
            self.config.ae_path,
            stage_name="encode_source_latent",
            apply_latent_format=True,
        )

    def encode_concat_latent(self, concat_pixels: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._encode_pixels(
            concat_pixels,
            self.config.ae_path,
            stage_name="encode_concat_latent",
            apply_latent_format=False,
        )

    def prepare_denoise_mask(self, mask: np.ndarray, latent_shape: torch.Size | tuple[int, ...]) -> tuple[torch.Tensor, dict[str, Any]]:
        mask_binary = _binary_mask(mask)
        mask_tensor = torch.from_numpy(mask_binary.astype(np.float32, copy=False))[None, None, :, :]
        latent_h = int(latent_shape[-2])
        latent_w = int(latent_shape[-1])
        if mask_tensor.shape[-2] != latent_h * 8 or mask_tensor.shape[-1] != latent_w * 8:
            raise FluxFillValidationError(
                f"mask must downsample cleanly to latent resolution; expected {(latent_h * 8, latent_w * 8)}, got {tuple(mask_tensor.shape[-2:])}."
            )
        denoise_mask = torch.nn.functional.max_pool2d(mask_tensor, kernel_size=8, stride=8)
        denoise_mask = (denoise_mask > 0.5).float()
        return denoise_mask.detach().cpu(), {
            "stage": "prepare_denoise_mask",
            "mask": _tensor_summary(denoise_mask),
            "coverage": float(denoise_mask.float().mean().item()) if denoise_mask.numel() else 0.0,
            "latent_shape": [int(dim) for dim in latent_shape],
        }

    def build_conditioning_payload(
        self,
        empty_conditioning: FluxEmptyConditioning,
        source_latent: torch.Tensor,
        concat_latent: torch.Tensor,
        denoise_mask: torch.Tensor,
    ) -> tuple[Any, dict[str, Any]]:
        payloads = build_flux_fill_conditioning_payloads(
            empty_conditioning,
            source_latent,
            denoise_mask,
            concat_latent=concat_latent,
            guidance=self.config.guidance,
            batch_size=int(source_latent.shape[0]),
            device=self.device,
            dtype=source_latent.dtype,
        )
        return payloads, {
            "stage": "build_conditioning_payload",
            "guidance": float(payloads.guidance),
            "batch_size": int(payloads.batch_size),
            "latent_image": _tensor_summary(payloads.latent_image),
            "denoise_mask": _tensor_summary(payloads.denoise_mask),
            "positive_shape": [int(dim) for dim in payloads.positive[0][0].shape],
            "negative_shape": [int(dim) for dim in payloads.negative[0][0].shape],
        }

    def verify_c_concat(self, unet_patcher: Any, *, noise: torch.Tensor, concat_latent: torch.Tensor, denoise_mask: torch.Tensor) -> tuple[torch.Tensor | None, dict[str, Any]]:
        preview = _build_glass_concat_condition(unet_patcher, noise=noise, concat_latent=concat_latent, denoise_mask=denoise_mask, device=self.device)
        preview_tensor = preview.detach().cpu() if isinstance(preview, torch.Tensor) else None
        return preview_tensor, {
            "stage": "verify_c_concat",
            "preview": _tensor_summary(preview_tensor),
            "noise": _tensor_summary(noise),
            "concat_latent": _tensor_summary(concat_latent),
            "denoise_mask": _tensor_summary(denoise_mask),
        }

    def denoise(self, latent_source: FluxFillLatentSource, *, empty_conditioning: FluxEmptyConditioning, unet_patcher: Any, disable_pbar: bool = True) -> tuple[FluxFillDenoiseResult, dict[str, Any], torch.Tensor]:
        result = denoise_flux_fill_latent(
            self.config,
            latent_source,
            empty_conditioning=empty_conditioning,
            unet_patcher=unet_patcher,
            load_device=self.device,
            offload_device=None,
            disable_pbar=disable_pbar,
            cleanup_unet=True,
        )
        return result, {
            "stage": "denoise",
            "samples": _tensor_summary(result.samples),
            "noise": _tensor_summary(result.noise),
            "sigmas": _tensor_summary(result.sigmas),
            "metadata": dict(result.metadata),
        }, result.samples.detach().cpu()

    def decode(self, samples: torch.Tensor) -> tuple[FluxFillDecodedImage, dict[str, Any]]:
        decoded = decode_flux_fill_latent(
            samples,
            self.config.ae_path,
            stitch=False,
            tiled=False,
            load_device=self.device,
            offload_device=None,
        )
        return decoded, {
            "stage": "decode",
            "raw": _array_summary(decoded.bb_image),
            "stitched": _array_summary(decoded.stitched_image),
        }

    def compose_debug(self, original_image: np.ndarray, mask: np.ndarray, decoded_image: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
        if not self.config.save_composite:
            return None, {"stage": "compose_debug", "enabled": False}
        composite = self.compose_output(original_image, mask, decoded_image)
        return composite, {"stage": "compose_debug", "enabled": True, "composite": _array_summary(composite)}

    def compose_output(self, original_image: np.ndarray, mask: np.ndarray, decoded_image: np.ndarray) -> np.ndarray:
        original_rgb = _ensure_uint8_rgb(original_image).astype(np.float32)
        decoded_rgb = _ensure_uint8_rgb(decoded_image).astype(np.float32)
        if self.config.blend_mode == "morphological":
            mask_alpha = _morphological_blend_mask(mask).astype(np.float32)[..., None] / 255.0
            mask_alpha = blending.apply_sin2_curve(mask_alpha)
        else:
            mask_alpha = _mask_2d(mask).astype(np.float32)[..., None] / 255.0
        composite = np.clip(
            decoded_rgb * mask_alpha + original_rgb * (1.0 - mask_alpha),
            0,
            255,
        ).astype(np.uint8)
        return composite

    def _debug_root(self) -> Path | None:
        if self.config.debug_output_dir is not None:
            return Path(self.config.debug_output_dir)
        if self.config.mode == "debug" or self.config.capture_artifacts or self.config.capture_tensors or self.config.save_composite:
            if self.config.output_path is not None:
                return Path(self.config.output_path).parent / f"{Path(self.config.output_path).stem}_glass_debug"
        return None

    def _write_artifact(self, root: Path, name: str, array: np.ndarray) -> str:
        from PIL import Image

        root.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(root, name, ".png")
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
        return str(path)

    def _write_tensor_artifact(self, root: Path, name: str, tensor: torch.Tensor) -> str:
        root.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(root, name, ".pt")
        torch.save(tensor.detach().cpu(), path)
        return str(path)

    def run(self, image: np.ndarray, mask: np.ndarray, *, disable_pbar: bool = True) -> FluxFillGlassResult:
        self.config.validate_static(require_existing_assets=True)
        mode = self.config.mode
        debug_summary: dict[str, Any] = {"stage_order": list(self.stage_order), "stages": {}, "artifacts": {}, "mode": mode}
        timings: dict[str, float] = {}
        root = self._debug_root()

        image_np = _ensure_uint8_rgb(image)
        mask_np = _ensure_mask_shape(image_np, mask)
        mask_binary = _binary_mask(mask_np)
        mask_bbox = _mask_bbox_from_binary(mask_binary, image_shape=image_np.shape[:2])
        bbox_y1, bbox_y2, bbox_x1, bbox_x2 = mask_bbox
        bbox_height = max(1, int(bbox_y2 - bbox_y1))
        bbox_width = max(1, int(bbox_x2 - bbox_x1))
        mask_area = int(mask_binary.sum())
        image_area = max(1, int(image_np.shape[0] * image_np.shape[1]))
        effective_mode = self._resolve_effective_mode(image_np, mode)

        def _full_geometry(target_width: int, target_height: int, scale_factor: float, scaled_flag: bool) -> dict[str, Any]:
            return {
                "mode": mode,
                "effective_mode": effective_mode,
                "crop_box": [0, int(image_np.shape[0]), 0, int(image_np.shape[1])],
                "target_canvas": [int(target_width), int(target_height)],
                "target_canvas_aspect": float(target_width) / float(max(target_height, 1)),
                "target_canvas_area": int(target_width * target_height),
                "crop_width": int(image_np.shape[1]),
                "crop_height": int(image_np.shape[0]),
                "normalized_width": int(target_width),
                "normalized_height": int(target_height),
                "target_width": int(target_width),
                "target_height": int(target_height),
                "scale_to_canvas": float(scale_factor),
                "mask_bbox": [int(v) for v in mask_bbox],
                "mask_bbox_width_ratio": float(bbox_width) / float(max(image_np.shape[1], 1)),
                "mask_bbox_height_ratio": float(bbox_height) / float(max(image_np.shape[0], 1)),
                "mask_area_ratio": float(mask_area) / float(image_area),
                "full_image_crop": True,
                "large_mask_full_context": False,
                "context_limited_by_bounds": False,
                "crop_aspect": float(image_np.shape[1]) / float(max(image_np.shape[0], 1)),
                "pad": [0, 0, 0, 0],
                "source_image_width": int(image_np.shape[1]),
                "source_image_height": int(image_np.shape[0]),
                "scaled": bool(scaled_flag),
                "bucket": [int(target_width), int(target_height)],
                "bucket_aspect": float(target_width) / float(max(target_height, 1)),
                "scale_to_bucket": float(scale_factor),
            }

        crop_plan: FluxFillGlassCropPlan | None = None
        crop_working_summary: dict[str, Any] = {}
        if mode == "scaled":
            working_image, working_mask, scale_summary = _prepare_scaled_inputs(
                image_np,
                mask_np,
                target_megapixels=self.config.target_megapixels,
            )
            geometry_summary = _full_geometry(
                working_image.shape[1],
                working_image.shape[0],
                float(scale_summary["scale_factor"]),
                True,
            )
            geometry_summary["normalized_width"] = int(image_np.shape[1])
            geometry_summary["normalized_height"] = int(image_np.shape[0])
        elif effective_mode == "context_crop":
            crop_plan = _select_context_crop_plan(image_np, mask_np, mode=mode)
            working_image, working_mask, crop_working_summary = prepare_flux_fill_glass_context_crop(image_np, mask_np, crop_plan)
            scale_summary = {
                "original_width": int(image_np.shape[1]),
                "original_height": int(image_np.shape[0]),
                "working_width": int(working_image.shape[1]),
                "working_height": int(working_image.shape[0]),
                "scale_factor": float(crop_plan.scale_to_bucket),
                "scale_to_canvas": float(crop_plan.scale_to_canvas),
                "scale_to_bucket": float(crop_plan.scale_to_bucket),
                "scaled": False,
                "crop_width": int(crop_plan.crop_width),
                "crop_height": int(crop_plan.crop_height),
                "normalized_width": int(crop_plan.normalized_width),
                "normalized_height": int(crop_plan.normalized_height),
                "target_width": int(crop_plan.target_width),
                "target_height": int(crop_plan.target_height),
            }
            geometry_summary = crop_plan.to_dict()
            geometry_summary["effective_mode"] = effective_mode
        else:
            working_image = image_np
            working_mask = mask_np
            scale_summary = {
                "original_width": int(image_np.shape[1]),
                "original_height": int(image_np.shape[0]),
                "working_width": int(image_np.shape[1]),
                "working_height": int(image_np.shape[0]),
                "scale_factor": 1.0,
                "scaled": False,
            }
            geometry_summary = _full_geometry(image_np.shape[1], image_np.shape[0], 1.0, False)
        debug_summary["scale"] = dict(scale_summary)
        debug_summary["geometry"] = dict(geometry_summary)
        debug_summary["effective_mode"] = effective_mode
        if crop_working_summary:
            debug_summary["crop_working"] = {
                "crop_image": _array_summary(crop_working_summary.get("crop_image")),
                "crop_mask": _array_summary(crop_working_summary.get("crop_mask")),
                "normalized_image": _array_summary(crop_working_summary.get("normalized_image")),
                "normalized_mask": _array_summary(crop_working_summary.get("normalized_mask")),
            }

        stage_start = time.perf_counter()
        contract = self.validate_input_contract(image_np, mask_np, mode=mode)
        timings["validate_contract"] = time.perf_counter() - stage_start
        contract["scale"] = dict(scale_summary)
        contract["geometry"] = dict(geometry_summary)
        debug_summary["contract"] = contract
        debug_summary["stages"]["validate_contract"] = contract
        debug_summary["stages"]["select_working_geometry"] = dict(geometry_summary)

        stage_start = time.perf_counter()
        source_pixels, source_pixels_summary = self.prepare_source_pixels(working_image)
        timings["prepare_source_pixels"] = time.perf_counter() - stage_start
        debug_summary["stages"]["prepare_source_pixels"] = source_pixels_summary

        stage_start = time.perf_counter()
        concat_pixels, concat_pixels_summary = self.prepare_concat_pixels(working_image, working_mask)
        timings["prepare_concat_pixels"] = time.perf_counter() - stage_start
        debug_summary["stages"]["prepare_concat_pixels"] = concat_pixels_summary

        stage_start = time.perf_counter()
        source_latent, source_latent_summary = self.encode_source_latent(source_pixels)
        timings["encode_source_latent"] = time.perf_counter() - stage_start
        debug_summary["stages"]["encode_source_latent"] = source_latent_summary

        stage_start = time.perf_counter()
        concat_latent, concat_latent_summary = self.encode_concat_latent(concat_pixels)
        timings["encode_concat_latent"] = time.perf_counter() - stage_start
        debug_summary["stages"]["encode_concat_latent"] = concat_latent_summary

        if tuple(source_latent.shape) != tuple(concat_latent.shape):
            raise FluxFillValidationError(f"source_latent shape {list(source_latent.shape)} does not match concat_latent shape {list(concat_latent.shape)}.")

        stage_start = time.perf_counter()
        denoise_mask, denoise_mask_summary = self.prepare_denoise_mask(working_mask, source_latent.shape)
        timings["prepare_denoise_mask"] = time.perf_counter() - stage_start
        debug_summary["stages"]["prepare_denoise_mask"] = denoise_mask_summary

        latent_source = FluxFillLatentSource(
            context=None,
            source_latent=source_latent,
            concat_latent=concat_latent,
            denoise_mask=denoise_mask,
            width=int(source_latent.shape[-1] * 8),
            height=int(source_latent.shape[-2] * 8),
        )

        empty_conditioning = load_flux_empty_conditioning_cache(self.config.conditioning_cache_path)
        payloads, payload_summary = self.build_conditioning_payload(empty_conditioning, source_latent, concat_latent, denoise_mask)
        debug_summary["stages"]["build_conditioning_payload"] = payload_summary

        noise = create_flux_fill_noise(source_latent, self.config.seed, device=self.device, dtype=source_latent.dtype)
        unet_patcher = load_flux_fill_unet(self.config.unet_path, load_device=self.device, offload_device=None)
        if self.config.verify_c_concat:
            c_concat_preview, c_concat_summary = self.verify_c_concat(unet_patcher, noise=noise, concat_latent=concat_latent, denoise_mask=denoise_mask)
            debug_summary["stages"]["verify_c_concat"] = c_concat_summary
            if c_concat_preview is not None:
                debug_summary["c_concat_preview"] = _tensor_summary(c_concat_preview)
        else:
            c_concat_preview = None
            debug_summary["stages"]["verify_c_concat"] = {"stage": "verify_c_concat", "enabled": False}

        stage_start = time.perf_counter()
        denoise_result, denoise_summary, samples = self.denoise(latent_source, empty_conditioning=empty_conditioning, unet_patcher=unet_patcher, disable_pbar=disable_pbar)
        timings["denoise"] = time.perf_counter() - stage_start
        debug_summary["stages"]["denoise"] = denoise_summary

        stage_start = time.perf_counter()
        decoded, decode_summary = self.decode(samples)
        timings["decode"] = time.perf_counter() - stage_start
        debug_summary["stages"]["decode"] = decode_summary

        raw_output_image = np.asarray(decoded.bb_image, dtype=np.uint8)
        restored_crop_image: np.ndarray | None = None
        if mode == "scaled":
            resized_raw = _resize_uint8_rgb(raw_output_image, image_np.shape[1], image_np.shape[0], resample=Image.Resampling.LANCZOS)
            final_output_image = self.compose_output(image_np, mask_np, resized_raw)
            composite_summary = {"stage": "compose_debug", "enabled": True, "composite": _array_summary(final_output_image), "mode": mode}
        elif effective_mode == "context_crop" and crop_plan is not None:
            restored_crop_image = restore_flux_fill_glass_context_crop(raw_output_image, crop_plan)
            final_output_image = stitch_flux_fill_glass_context_crop(
                image_np,
                mask_np,
                crop_plan,
                restored_crop_image,
                blend_mode=self.config.blend_mode,
            )
            composite_summary = {
                "stage": "compose_context_crop",
                "enabled": True,
                "plan": dict(geometry_summary),
                "restored_crop": _array_summary(restored_crop_image),
                "composite": _array_summary(final_output_image),
            }
        else:
            composite, composite_summary = self.compose_debug(image_np, mask_np, raw_output_image)
            final_output_image = composite if composite is not None else raw_output_image
        debug_summary["stages"]["compose_debug"] = composite_summary
        debug_summary["raw_output"] = _array_summary(raw_output_image)
        debug_summary["final_output"] = _array_summary(final_output_image)

        if root is not None:
            artifact_paths: dict[str, str] = {}
            if self.config.capture_artifacts or mode == "debug":
                artifact_paths["source_pixels"] = self._write_artifact(root, "source_pixels", source_pixels)
                artifact_paths["concat_pixels"] = self._write_artifact(root, "concat_pixels", concat_pixels)
                artifact_paths["mask"] = self._write_artifact(root, "mask", np.repeat(_mask_2d(working_mask)[:, :, None], 3, axis=2))
                artifact_paths["decoded_raw"] = self._write_artifact(root, "decoded_raw", raw_output_image)
                if effective_mode == "context_crop" and crop_working_summary:
                    artifact_paths["crop_source"] = self._write_artifact(root, "crop_source", crop_working_summary["crop_image"])
                    artifact_paths["crop_mask"] = self._write_artifact(root, "crop_mask", np.repeat(_mask_2d(crop_working_summary["crop_mask"])[:, :, None], 3, axis=2))
                    artifact_paths["normalized_image"] = self._write_artifact(root, "normalized_image", working_image)
                    artifact_paths["normalized_mask"] = self._write_artifact(root, "normalized_mask", np.repeat(_mask_2d(working_mask)[:, :, None], 3, axis=2))
                    if restored_crop_image is not None:
                        artifact_paths["decoded_restored_crop"] = self._write_artifact(root, "decoded_restored_crop", restored_crop_image)
                artifact_paths["decoded_final"] = self._write_artifact(root, "decoded_final", final_output_image)
            if self.config.capture_tensors or mode == "debug":
                artifact_paths["source_latent"] = self._write_tensor_artifact(root, "source_latent", source_latent)
                artifact_paths["concat_latent"] = self._write_tensor_artifact(root, "concat_latent", concat_latent)
                artifact_paths["denoise_mask"] = self._write_tensor_artifact(root, "denoise_mask", denoise_mask)
                artifact_paths["noise"] = self._write_tensor_artifact(root, "noise", noise)
                if isinstance(c_concat_preview, torch.Tensor):
                    artifact_paths["c_concat"] = self._write_tensor_artifact(root, "c_concat", c_concat_preview)
            if artifact_paths:
                debug_summary["artifacts"] = artifact_paths
                debug_summary["debug_output_dir"] = str(root)

        geometry_for_metadata = geometry_summary
        metadata = {
            "tier": self.config.tier,
            "mode": mode,
            "effective_mode": effective_mode,
            "blend_mode": self.config.blend_mode,
            "target_megapixels": float(self.config.target_megapixels),
            "unet_path": str(self.config.unet_path),
            "ae_path": str(self.config.ae_path),
            "conditioning_cache_path": str(self.config.conditioning_cache_path),
            "no_bb": True,
            "no_scale": effective_mode == "baseline",
            "verify_c_concat": bool(self.config.verify_c_concat),
            "original_width": int(image_np.shape[1]),
            "original_height": int(image_np.shape[0]),
            "working_width": int(working_image.shape[1]),
            "working_height": int(working_image.shape[0]),
            "scale_factor": float(scale_summary.get("scale_factor", 1.0)),
            "scale_to_canvas": float(geometry_for_metadata.get("scale_to_canvas", scale_summary.get("scale_factor", 1.0))),
            "scale_to_bucket": float(geometry_for_metadata.get("scale_to_bucket", scale_summary.get("scale_factor", 1.0))),
            "scaled": bool(scale_summary.get("scaled", False)),
            "crop_box": [int(v) for v in geometry_for_metadata.get("crop_box", [0, image_np.shape[0], 0, image_np.shape[1]])],
            "crop_width": int(geometry_for_metadata.get("crop_width", image_np.shape[1])),
            "crop_height": int(geometry_for_metadata.get("crop_height", image_np.shape[0])),
            "normalized_width": int(geometry_for_metadata.get("normalized_width", image_np.shape[1])),
            "normalized_height": int(geometry_for_metadata.get("normalized_height", image_np.shape[0])),
            "target_width": int(geometry_for_metadata.get("target_width", working_image.shape[1])),
            "target_height": int(geometry_for_metadata.get("target_height", working_image.shape[0])),
            "target_canvas": [int(v) for v in geometry_for_metadata.get("target_canvas", [working_image.shape[1], working_image.shape[0]])],
            "target_canvas_aspect": float(geometry_for_metadata.get("target_canvas_aspect", float(working_image.shape[1]) / float(max(working_image.shape[0], 1)))),
            "target_canvas_area": int(geometry_for_metadata.get("target_canvas_area", int(working_image.shape[1] * working_image.shape[0]))),
            "bucket": [int(v) for v in geometry_for_metadata.get("bucket", [working_image.shape[1], working_image.shape[0]])],
            "bucket_aspect": float(geometry_for_metadata.get("bucket_aspect", float(working_image.shape[1]) / float(max(working_image.shape[0], 1)))),
            "full_image_crop": bool(geometry_for_metadata.get("full_image_crop", True)),
            "large_mask_full_context": bool(geometry_for_metadata.get("large_mask_full_context", False)),
            "context_limited_by_bounds": bool(geometry_for_metadata.get("context_limited_by_bounds", False)),
            "mask_bbox": [int(v) for v in geometry_for_metadata.get("mask_bbox", [bbox_y1, bbox_y2, bbox_x1, bbox_x2])],
            "mask_bbox_width_ratio": float(geometry_for_metadata.get("mask_bbox_width_ratio", float(bbox_width) / float(max(image_np.shape[1], 1)))),
            "mask_bbox_height_ratio": float(geometry_for_metadata.get("mask_bbox_height_ratio", float(bbox_height) / float(max(image_np.shape[0], 1)))),
            "mask_area_ratio": float(geometry_for_metadata.get("mask_area_ratio", float(mask_area) / float(image_area))),
            "source_latent_shape": [int(dim) for dim in source_latent.shape],
            "concat_latent_shape": [int(dim) for dim in concat_latent.shape],
            "denoise_mask_shape": [int(dim) for dim in denoise_mask.shape],
            "conditioning_batch": int(payloads.batch_size),
            "native_flux": bool(is_native_flux_dimensions(image_np.shape[1], image_np.shape[0])),
            "native_sdxl": bool(is_native_sdxl_dimensions(image_np.shape[1], image_np.shape[0])),
            "sdxl_bucket_aspect": bool(is_sdxl_bucket_aspect_ratio(image_np.shape[1], image_np.shape[0])),
        }

        return FluxFillGlassResult(
            output_image=final_output_image,
            raw_output_image=raw_output_image,
            output_path=Path(self.config.output_path) if self.config.output_path is not None else None,
            seed=int(self.config.seed),
            width=int(final_output_image.shape[1]),
            height=int(final_output_image.shape[0]),
            timings=timings,
            metadata=metadata,
            debug_summary=debug_summary,
        )


def run_flux_fill_glass(config: FluxFillGlassConfig, image: np.ndarray, mask: np.ndarray, *, disable_pbar: bool = True) -> FluxFillGlassResult:
    return FluxFillGlassPipeline(config).run(image, mask, disable_pbar=disable_pbar)
