from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

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

FLUX_FILL_GLASS_MODES = ("baseline", "debug", "scaled")
FLUX_FILL_GLASS_DEFAULT_MODE = "baseline"


def _normalize_glass_mode(mode: str | None) -> str:
    value = str(mode or FLUX_FILL_GLASS_DEFAULT_MODE).strip().lower()
    if value in FLUX_FILL_GLASS_MODES:
        return value
    raise FluxFillValidationError(
        f"Unsupported Flux Fill glass mode: {mode!r}. Expected one of {list(FLUX_FILL_GLASS_MODES)}."
    )


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return max(multiple, int(value) // int(multiple) * int(multiple))


def _scale_working_dimensions(width: int, height: int, *, target_megapixels: float, multiple: int = 8) -> tuple[int, int, float]:
    if width < 1 or height < 1:
        raise FluxFillValidationError(f"image dimensions must be positive, got {width}x{height}.")

    current_megapixels = float(width * height) / 1_000_000.0
    needs_rescale = current_megapixels > float(target_megapixels) or (width % multiple != 0) or (height % multiple != 0)
    if not needs_rescale:
        return width, height, 1.0

    scale = 1.0 if current_megapixels <= float(target_megapixels) else float(np.sqrt(float(target_megapixels) / max(current_megapixels, 1e-8)))
    scaled_width = _round_down_to_multiple(max(1, int(round(width * scale))), multiple)
    scaled_height = _round_down_to_multiple(max(1, int(round(height * scale))), multiple)
    return scaled_width, scaled_height, min(scale, 1.0)


def _resize_uint8_rgb(image: np.ndarray, width: int, height: int, *, resample: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((int(width), int(height)), resample=resample),
        dtype=np.uint8,
    )


def _resize_uint8_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    mask_2d = _mask_2d(np.asarray(mask))
    resized = Image.fromarray(np.asarray(mask_2d, dtype=np.uint8)).resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


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
    target_megapixels: float = 1.0
    verify_c_concat: bool = True
    capture_artifacts: bool = False
    capture_tensors: bool = False
    save_composite: bool = False

    def validate_static(self, *, require_existing_assets: bool = True) -> None:
        self.mode = _normalize_glass_mode(self.mode)
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
            for label, value in (("UNet", self.unet_path), ("AE", self.ae_path), ("empty conditioning cache", self.conditioning_cache_path)):
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
        multiple=8,
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

    def validate_input_contract(self, image: np.ndarray, mask: np.ndarray, *, mode: str) -> dict[str, Any]:
        image_np = _ensure_uint8_rgb(image)
        mask_np = _ensure_mask_shape(image_np, mask)
        height, width = image_np.shape[:2]
        normalized_mode = _normalize_glass_mode(mode)
        if normalized_mode != "scaled" and (height % 8 != 0 or width % 8 != 0):
            raise FluxFillValidationError(
                f"W04 glass baseline does not scale or crop; image dimensions must be multiples of 8 (preferably 16). Got {width}x{height}."
            )
        return {
            "image": _array_summary(image_np),
            "mask": _array_summary(mask_np),
            "mode": normalized_mode,
            "no_bb": True,
            "no_scale": normalized_mode != "scaled",
            "multiple_of_8": (height % 8 == 0 and width % 8 == 0),
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

    def _encode_pixels(self, pixels: np.ndarray, ae_path: Path | str, *, stage_name: str) -> tuple[torch.Tensor, dict[str, Any]]:
        import modules.core as core

        vae = load_flux_ae(ae_path, load_device=self.device, offload_device=None)
        start = time.perf_counter()
        try:
            resources.load_models_gpu([vae.patcher])
            latent = vae.encode(core.numpy_to_pytorch(pixels))["samples"]
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
        }
        return latent.detach().cpu(), summary

    def encode_source_latent(self, source_pixels: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._encode_pixels(source_pixels, self.config.ae_path, stage_name="encode_source_latent")

    def encode_concat_latent(self, concat_pixels: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._encode_pixels(concat_pixels, self.config.ae_path, stage_name="encode_concat_latent")

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
        mask_alpha = _mask_2d(mask).astype(np.float32)[..., None] / 255.0
        composite = np.clip(
            decoded_image.astype(np.float32) * mask_alpha + _ensure_uint8_rgb(original_image).astype(np.float32) * (1.0 - mask_alpha),
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
        if mode == "scaled":
            working_image, working_mask, scale_summary = _prepare_scaled_inputs(
                image_np,
                mask_np,
                target_megapixels=self.config.target_megapixels,
            )
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
        debug_summary["scale"] = scale_summary

        stage_start = time.perf_counter()
        contract = self.validate_input_contract(image_np, mask_np, mode=mode)
        timings["validate_contract"] = time.perf_counter() - stage_start
        contract["scale"] = dict(scale_summary)
        debug_summary["contract"] = contract
        debug_summary["stages"]["validate_contract"] = contract

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
        if mode == "scaled":
            resized_raw = _resize_uint8_rgb(raw_output_image, image_np.shape[1], image_np.shape[0], resample=Image.Resampling.LANCZOS)
            final_output_image = self.compose_output(image_np, mask_np, resized_raw)
            composite_summary = {"stage": "compose_debug", "enabled": True, "composite": _array_summary(final_output_image), "mode": mode}
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

        metadata = {
            "tier": self.config.tier,
            "mode": mode,
            "target_megapixels": float(self.config.target_megapixels),
            "unet_path": str(self.config.unet_path),
            "ae_path": str(self.config.ae_path),
            "conditioning_cache_path": str(self.config.conditioning_cache_path),
            "no_bb": True,
            "no_scale": mode != "scaled",
            "verify_c_concat": bool(self.config.verify_c_concat),
            "original_width": int(image_np.shape[1]),
            "original_height": int(image_np.shape[0]),
            "working_width": int(working_image.shape[1]),
            "working_height": int(working_image.shape[0]),
            "scale_factor": float(scale_summary["scale_factor"]),
            "scaled": bool(scale_summary["scaled"]),
            "source_latent_shape": [int(dim) for dim in source_latent.shape],
            "concat_latent_shape": [int(dim) for dim in concat_latent.shape],
            "denoise_mask_shape": [int(dim) for dim in denoise_mask.shape],
            "conditioning_batch": int(payloads.batch_size),
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

