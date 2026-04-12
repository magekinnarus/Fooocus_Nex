from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys
import time

import torch

EMPTY_FLUX_CROSS_ATTN_SHAPE = (1, 256, 4096)
EMPTY_FLUX_POOLED_SHAPE = (1, 768)
EXPECTED_FLUX_FILL_CONTRACT = {
    "image_model": "flux",
    "in_channels": 96,
    "out_channels": 16,
    "vec_in_dim": 768,
    "context_in_dim": 4096,
    "depth": 19,
    "depth_single_blocks": 38,
    "guidance_embed": True,
}
IMPORTANT_FLUX_GGUF_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "vector_in.in_layer.weight",
    "guidance_in.in_layer.weight",
    "final_layer.linear.weight",
)


class FluxFillValidationError(ValueError):
    """Raised when Flux Fill runtime inputs violate the direct pipeline contract."""


class FluxFillUnsupportedModelError(NotImplementedError):
    """Raised for Flux models outside the W01 Flux Fill runtime contract."""


def _shape_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _validate_tensor_shape(name: str, tensor: Any, expected_shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise FluxFillValidationError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    actual_shape = _shape_tuple(tensor)
    if actual_shape != expected_shape:
        raise FluxFillValidationError(
            f"{name} must have shape {list(expected_shape)}, got {list(actual_shape)}."
        )
    if not torch.is_floating_point(tensor):
        raise FluxFillValidationError(f"{name} must be a floating point tensor, got dtype {tensor.dtype}.")
    if not torch.isfinite(tensor).all().item():
        raise FluxFillValidationError(f"{name} contains NaN or Inf values.")
    return tensor.detach().cpu()


def _shape_of(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "tensor_shape", None)
    if shape is None:
        shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _qtype_of(tensor: Any) -> str | None:
    tensor_type = getattr(tensor, "tensor_type", None)
    if tensor_type is None:
        return None
    return getattr(tensor_type, "name", str(tensor_type))


def _count_block_indices(state_dict: dict[str, Any], prefix: str) -> int:
    indices: set[int] = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return len(indices)


@dataclass(frozen=True)
class FluxEmptyConditioning:
    cross_attn: torch.Tensor
    pooled_output: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cross_attn = _validate_tensor_shape("cross_attn", self.cross_attn, EMPTY_FLUX_CROSS_ATTN_SHAPE)
        pooled_output = _validate_tensor_shape("pooled_output", self.pooled_output, EMPTY_FLUX_POOLED_SHAPE)
        object.__setattr__(self, "cross_attn", cross_attn)
        object.__setattr__(self, "pooled_output", pooled_output)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def repeat(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise FluxFillValidationError(f"batch_size must be >= 1, got {batch_size}.")
        cross_attn = self.cross_attn.repeat(batch_size, 1, 1)
        pooled_output = self.pooled_output.repeat(batch_size, 1)
        if dtype is not None:
            cross_attn = cross_attn.to(dtype=dtype)
            pooled_output = pooled_output.to(dtype=dtype)
        if device is not None:
            cross_attn = cross_attn.to(device=device)
            pooled_output = pooled_output.to(device=device)
        return cross_attn, pooled_output


@dataclass
class FluxFillConfig:
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

    def validate_static(self, *, require_existing_assets: bool = True) -> None:
        if self.steps < 1:
            raise FluxFillValidationError(f"steps must be >= 1, got {self.steps}.")
        if self.cfg != 1.0:
            raise NotImplementedError("Prompt-conditioned/CFG Flux Fill is out of scope for P4-M10-W01.")
        if self.sampler != "euler":
            raise NotImplementedError(f"Unsupported Flux Fill sampler: {self.sampler!r}. Expected 'euler'.")
        if self.scheduler != "normal":
            raise NotImplementedError(f"Unsupported Flux Fill scheduler: {self.scheduler!r}. Expected 'normal'.")
        if self.denoise != 1.0:
            raise NotImplementedError("Flux Fill W01 only supports denoise=1.0.")
        if self.guidance <= 0:
            raise FluxFillValidationError(f"guidance must be > 0, got {self.guidance}.")
        if require_existing_assets:
            for label, value in (
                ("UNet", self.unet_path),
                ("AE", self.ae_path),
                ("empty conditioning cache", self.conditioning_cache_path),
            ):
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"{label} path does not exist: {path}")


@dataclass
class FluxFillResult:
    output_image: Any
    output_path: Path | None
    seed: int
    width: int
    height: int
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillUNetInfo:
    path: Path
    arch: str
    detected_config: dict[str, Any]
    key_shapes: dict[str, list[int] | None]
    qtypes: dict[str, int]
    tensor_count: int


@dataclass(frozen=True)
class FluxFillLatentSource:
    context: Any
    source_latent: torch.Tensor        # VAE encode of UNMASKED original (KSampler noise start)
    concat_latent: torch.Tensor        # VAE encode of gray-masked image (c_concat condition)
    denoise_mask: torch.Tensor
    width: int
    height: int
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillConditioningPayloads:
    positive: list[list[Any]]
    negative: list[list[Any]]
    latent_image: torch.Tensor
    denoise_mask: torch.Tensor
    guidance: float
    batch_size: int


@dataclass(frozen=True)
class FluxFillDecodedImage:
    bb_image: Any
    stitched_image: Any | None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillDenoiseResult:
    samples: torch.Tensor
    noise: torch.Tensor
    sigmas: torch.Tensor
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_flux_fill_unet_config(detected_config: dict[str, Any]) -> None:
    mismatches = []
    for key, expected in EXPECTED_FLUX_FILL_CONTRACT.items():
        actual = detected_config.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise FluxFillUnsupportedModelError(
            "Unsupported Flux model for P4-M10-W01 Flux Fill runtime: " + "; ".join(mismatches)
        )


def inspect_flux_fill_gguf(
    unet_path: Path | str,
    *,
    handle_prefix: str | None = "model.diffusion_model.",
    validate_contract: bool = True,
) -> FluxFillUNetInfo:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    from backend.gguf.loader import gguf_sd_loader
    from ldm_patched.modules import model_detection

    state_dict, arch = gguf_sd_loader(str(path), handle_prefix=handle_prefix, return_arch=True)
    if arch != "flux":
        raise FluxFillValidationError(f"Expected Flux GGUF arch 'flux', got {arch!r} for {path}.")

    detected_config = model_detection.detect_unet_config(state_dict, "", dtype=None)
    if detected_config is None:
        raise FluxFillValidationError(f"Could not detect Flux model config from {path}.")
    if validate_contract:
        validate_flux_fill_unet_config(detected_config)

    qtypes: dict[str, int] = {}
    for tensor in state_dict.values():
        qtype = _qtype_of(tensor) or "torch"
        qtypes[qtype] = qtypes.get(qtype, 0) + 1

    key_shapes = {key: _shape_of(state_dict[key]) if key in state_dict else None for key in IMPORTANT_FLUX_GGUF_KEYS}
    detected_config = dict(detected_config)
    detected_config["double_blocks"] = _count_block_indices(state_dict, "double_blocks.")
    detected_config["single_blocks"] = _count_block_indices(state_dict, "single_blocks.")
    return FluxFillUNetInfo(
        path=path,
        arch=arch,
        detected_config=detected_config,
        key_shapes=key_shapes,
        qtypes=qtypes,
        tensor_count=len(state_dict),
    )


def load_flux_fill_unet(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    from backend import resources
    from backend.gguf.loader import gguf_sd_loader
    from backend.gguf.ops import GGMLOps
    from backend.gguf.patcher import GGUFModelPatcher
    from ldm_patched.modules import model_detection

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()

    state_dict, arch = gguf_sd_loader(str(path), handle_prefix=handle_prefix, return_arch=True)
    if arch != "flux":
        raise FluxFillValidationError(f"Expected Flux GGUF arch 'flux', got {arch!r} for {path}.")

    detected_config = model_detection.detect_unet_config(state_dict, "", dtype=None)
    if detected_config is None:
        raise FluxFillValidationError(f"Could not detect Flux model config from {path}.")
    validate_flux_fill_unet_config(detected_config)

    model_config = model_detection.model_config_from_unet(state_dict, "", use_base_if_no_match=False)
    if model_config is None:
        raise FluxFillValidationError(f"No supported Flux model config matched {path}.")

    model = model_config.get_model(
        state_dict,
        "",
        device=offload_device,
        model_options={"custom_operations": GGMLOps},
    )
    model.load_model_weights(state_dict, "")
    patcher = GGUFModelPatcher(model, load_device=load_device, offload_device=offload_device)
    patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": arch,
        "detected_config": dict(detected_config),
    }
    return patcher


def load_flux_ae(
    ae_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
) -> Any:
    path = Path(ae_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux AE path does not exist: {path}")

    from backend import loader, resources
    from ldm_patched.modules import latent_formats

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.vae_offload_device()
    return loader.load_vae(
        str(path),
        load_device=load_device,
        offload_device=offload_device,
        dtype=torch.float32,
        latent_format=latent_formats.Flux(),
    )


def _validate_image_mask_arrays(image: Any, mask: Any) -> None:
    image_shape = getattr(image, "shape", None)
    mask_shape = getattr(mask, "shape", None)
    if image_shape is None or len(image_shape) != 3 or int(image_shape[2]) != 3:
        raise FluxFillValidationError(f"image must have shape [H, W, 3], got {image_shape}.")
    if mask_shape is None or len(mask_shape) not in (2, 3):
        raise FluxFillValidationError(f"mask must have shape [H, W] or [H, W, C], got {mask_shape}.")
    if int(mask_shape[0]) != int(image_shape[0]) or int(mask_shape[1]) != int(image_shape[1]):
        raise FluxFillValidationError(f"mask spatial shape {mask_shape[:2]} does not match image {image_shape[:2]}.")


def _cleanup_model_patcher(model_patcher: Any) -> None:
    try:
        from backend import resources

        resources.eject_model(model_patcher)
    except Exception:
        detach = getattr(model_patcher, "detach", None)
        if callable(detach):
            detach()


import dataclasses
import numpy as np

@dataclasses.dataclass
class SimpleFluxFillContext:
    image: np.ndarray
    mask: np.ndarray

def prepare_flux_fill_latent_source(
    image: np.ndarray,
    mask: np.ndarray,
    ae_path: Path | str,
    *,
    extend_factor: float = 1.2,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    inpaint_pipeline: Any | None = None,
) -> FluxFillLatentSource:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise FluxFillValidationError("image must be an RGB numpy array [H, W, 3].")
    if not isinstance(mask, np.ndarray) or mask.ndim not in (2, 3):
        raise FluxFillValidationError("mask must be a numpy array [H, W] or [H, W, C].")

    if image.shape[:2] != mask.shape[:2]:
        raise FluxFillValidationError(f"Image shape {image.shape[:2]} and block mask shape {mask.shape[:2]} must match.")

    timings: dict[str, float] = {}
    prepare_start = time.perf_counter()
    context = SimpleFluxFillContext(image=image, mask=mask)
    timings["inpaint_prepare"] = time.perf_counter() - prepare_start

    # Base image for KSampler
    bb_image_for_source = image.copy().astype(np.float32) / 255.0

    # Gray-masked image for c_concat
    bb_image_for_concat = image.copy().astype(np.float32) / 255.0
    bb_mask_2d = mask
    if bb_mask_2d.ndim == 3:
        bb_mask_2d = bb_mask_2d[:, :, 0]
    mask_binary = (bb_mask_2d > 127).astype(np.float32)  # 1.0 = regenerate
    inv_mask = 1.0 - mask_binary  # 1.0 = keep
    for ch in range(3):
        bb_image_for_concat[:, :, ch] -= 0.5
        bb_image_for_concat[:, :, ch] *= inv_mask
        bb_image_for_concat[:, :, ch] += 0.5
    bb_image_for_concat = np.clip(bb_image_for_concat * 255.0, 0, 255).astype(np.uint8)

    vae = None
    encode_start = time.perf_counter()
    try:
        from modules.core import numpy_to_pytorch, encode_vae
        from backend import resources

        vae = load_flux_ae(ae_path, load_device=load_device, offload_device=offload_device)
        resources.load_models_gpu([vae.patcher])

        # Encode unmasked original → source_latent (KSampler noise init)
        orig_pixels = numpy_to_pytorch(image)
        source_latent = encode_vae(vae=vae, pixels=orig_pixels)["samples"]

        # Encode gray-masked → concat_latent (c_concat condition)
        # USER_REQUEST: Bypass encode.py to avoid double-normalization.
        # Flux.concat_cond (ldm_patched) will apply normalization itself.
        resources.load_models_gpu([vae.patcher])
        pixels_for_vae = (numpy_to_pytorch(bb_image_for_concat).movedim(-1, 1) * 2.0) - 1.0
        if pixels_for_vae.ndim == 3:
            pixels_for_vae = pixels_for_vae.unsqueeze(0)
        
        pixels_for_vae = pixels_for_vae.to(device=vae.patcher.load_device, dtype=torch.float32)
        
        # We manually call the base VAE model to get RAW unscaled latents.
        raw_latent = vae.first_stage_model.encode(pixels_for_vae)
        if hasattr(raw_latent, "sample"):
            raw_latent = raw_latent.sample()
            
        concat_latent = raw_latent.cpu()

        vae.patcher.detach()
        resources.soft_empty_cache()
    finally:
        if vae is not None:
            _cleanup_model_patcher(vae.patcher)
        try:
            from backend import resources as _res
            _res.soft_empty_cache()
        except Exception:
            pass
    timings["ae_encode"] = time.perf_counter() - encode_start

    # Build denoise_mask in latent space
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask_t = torch.from_numpy(mask_np).float() / 255.0
    mask_t = mask_t[None, None, :, :]  # (1, 1, H, W)
    denoise_mask = torch.nn.functional.max_pool2d(mask_t, kernel_size=8)
    denoise_mask = (denoise_mask > 0.5).float()

    if not isinstance(source_latent, torch.Tensor) or source_latent.ndim != 4 or int(source_latent.shape[1]) != 16:
        raise FluxFillValidationError(f"Flux source latent must have shape [B, 16, H, W], got {list(source_latent.shape) if isinstance(source_latent, torch.Tensor) else 'non-tensor'}.")
    if denoise_mask.ndim != 4 or int(denoise_mask.shape[1]) != 1:
        raise FluxFillValidationError(f"Flux denoise mask must have shape [B, 1, H, W], got {list(denoise_mask.shape)}.")
    if tuple(source_latent.shape[0:1] + source_latent.shape[2:4]) != tuple(denoise_mask.shape[0:1] + denoise_mask.shape[2:4]):
        raise FluxFillValidationError(
            f"Flux denoise mask shape {list(denoise_mask.shape)} does not match latent shape {list(source_latent.shape)}."
        )

    return FluxFillLatentSource(
        context=context,
        source_latent=source_latent.detach().cpu(),
        concat_latent=concat_latent.detach().cpu(),
        denoise_mask=denoise_mask.detach().cpu(),
        width=int(source_latent.shape[-1] * 8),
        height=int(source_latent.shape[-2] * 8),
        timings=timings,
    )


def decode_flux_fill_latent(
    latent: torch.Tensor,
    ae_path: Path | str,
    *,
    context: Any | None = None,
    stitch: bool = False,
    tiled: bool = False,
    tile_size: int = 64,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    inpaint_pipeline: Any | None = None,
) -> FluxFillDecodedImage:
    if not isinstance(latent, torch.Tensor):
        raise FluxFillValidationError(f"latent must be a torch.Tensor, got {type(latent).__name__}.")
    if latent.ndim != 4 or int(latent.shape[1]) != 16:
        raise FluxFillValidationError(f"Flux decoded latent must have shape [B, 16, H, W], got {list(latent.shape)}.")
    if stitch and context is None:
        raise FluxFillValidationError("context is required when stitch=True.")

    timings: dict[str, float] = {}
    vae = None
    decode_start = time.perf_counter()
    try:
        vae = load_flux_ae(ae_path, load_device=load_device, offload_device=offload_device)
        original_argv = list(sys.argv)
        try:
            sys.argv = [original_argv[0]]
            from modules import core
        finally:
            sys.argv = original_argv

        decoded = core.decode_vae(vae, {"samples": latent.detach().cpu()}, tiled=tiled)
    finally:
        if vae is not None:
            _cleanup_model_patcher(vae.patcher)
        try:
            from backend import resources

            resources.soft_empty_cache()
        except Exception:
            pass
    timings["ae_decode"] = time.perf_counter() - decode_start

    if isinstance(decoded, torch.Tensor):
        original_argv = list(sys.argv)
        try:
            sys.argv = [original_argv[0]]
            from modules.core import pytorch_to_numpy
        finally:
            sys.argv = original_argv

        decoded_images = pytorch_to_numpy(decoded)
    elif isinstance(decoded, list):
        decoded_images = decoded
    else:
        raise FluxFillValidationError(f"Unexpected Flux AE decode output type: {type(decoded).__name__}.")
    if not decoded_images:
        raise FluxFillValidationError("Flux AE decode produced no images.")

    bb_image = decoded_images[0]
    stitched_image = None
    if stitch and context is not None:
        stitch_start = time.perf_counter()
        import cv2
        import numpy as np

        canvas = context.image.copy().astype(np.float32)
        generated = bb_image.astype(np.float32)

        raw_mask = context.mask
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        
        # 1. Expand the mask slightly to ensure coverage of boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        morph_mask = cv2.dilate(raw_mask, kernel, iterations=2)
        
        # 2. Heavy blur for smooth alpha blending transition
        blur_mask = cv2.GaussianBlur(morph_mask, (63, 63), 0)
        alpha = (blur_mask / 255.0)[..., np.newaxis]
        
        merged = (generated * alpha) + (canvas * (1.0 - alpha))
        stitched_image = np.clip(merged, 0, 255).astype(np.uint8)

        timings["inpaint_stitch"] = time.perf_counter() - stitch_start

    return FluxFillDecodedImage(bb_image=bb_image, stitched_image=stitched_image, timings=timings)


def build_flux_fill_conditioning_payloads(
    empty_conditioning: FluxEmptyConditioning,
    source_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    *,
    concat_latent: torch.Tensor | None = None,
    guidance: float = 15.0,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FluxFillConditioningPayloads:
    """Build conditioning payloads matching ComfyUI InpaintModelConditioning protocol.

    Args:
        source_latent: VAE encode of UNMASKED original → KSampler latent_image.
        concat_latent: VAE encode of gray-masked image → c_concat condition.
                       If None, falls back to source_latent (legacy behavior).
        denoise_mask: Binary mask in latent space (1=regenerate, 0=keep).
    """
    if guidance <= 0:
        raise FluxFillValidationError(f"guidance must be > 0, got {guidance}.")
    if not isinstance(source_latent, torch.Tensor) or source_latent.ndim != 4 or int(source_latent.shape[1]) != 16:
        raise FluxFillValidationError("source_latent must have shape [B, 16, H, W].")
    if not isinstance(denoise_mask, torch.Tensor) or denoise_mask.ndim != 4 or int(denoise_mask.shape[1]) != 1:
        raise FluxFillValidationError("denoise_mask must have shape [B, 1, H, W].")
    if source_latent.shape[0] != denoise_mask.shape[0] or source_latent.shape[-2:] != denoise_mask.shape[-2:]:
        raise FluxFillValidationError(
            f"denoise_mask shape {list(denoise_mask.shape)} does not match source_latent {list(source_latent.shape)}."
        )

    # concat_latent for c_concat conditioning; falls back to source_latent
    cond_image = concat_latent if concat_latent is not None else source_latent

    batch = int(batch_size or source_latent.shape[0])
    cross_attn, pooled_output = empty_conditioning.repeat(batch, device=device, dtype=dtype)
    source = source_latent.detach().to(device=device or source_latent.device, dtype=dtype or source_latent.dtype)
    cond_img = cond_image.detach().to(device=device or cond_image.device, dtype=dtype or cond_image.dtype)
    mask = denoise_mask.detach().to(device=device or denoise_mask.device, dtype=dtype or denoise_mask.dtype)
    if int(source.shape[0]) != batch:
        if int(source.shape[0]) != 1:
            raise FluxFillValidationError(f"Cannot repeat source_latent batch {source.shape[0]} to {batch}.")
        source = source.repeat(batch, 1, 1, 1)
    if int(cond_img.shape[0]) != batch:
        if int(cond_img.shape[0]) != 1:
            raise FluxFillValidationError(f"Cannot repeat concat_latent batch {cond_img.shape[0]} to {batch}.")
        cond_img = cond_img.repeat(batch, 1, 1, 1)
    if int(mask.shape[0]) != batch:
        if int(mask.shape[0]) != 1:
            raise FluxFillValidationError(f"Cannot repeat denoise_mask batch {mask.shape[0]} to {batch}.")
        mask = mask.repeat(batch, 1, 1, 1)

    payload = {
        "pooled_output": pooled_output,
        "guidance": float(guidance),
        "concat_latent_image": cond_img,   # gray-masked latent for c_concat
        "denoise_mask": mask,
        "concat_mask": mask,
    }
    positive = [[cross_attn, payload.copy()]]
    negative = [[cross_attn.clone(), payload.copy()]]
    return FluxFillConditioningPayloads(
        positive=positive,
        negative=negative,
        latent_image=source,               # UNMASKED latent for KSampler
        denoise_mask=mask,
        guidance=float(guidance),
        batch_size=batch,
    )


def build_flux_concat_condition(
    flux_model_or_patcher: Any,
    *,
    noise: torch.Tensor,
    source_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    model = getattr(flux_model_or_patcher, "model", flux_model_or_patcher)
    concat_cond = getattr(model, "concat_cond", None)
    if not callable(concat_cond):
        raise FluxFillValidationError("Flux model does not expose concat_cond().")
    return concat_cond(
        noise=noise,
        concat_latent_image=source_latent,
        denoise_mask=denoise_mask,
        device=device or noise.device,
    )

def create_flux_fill_noise(
    latent: torch.Tensor,
    seed: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or int(latent.shape[1]) != 16:
        raise FluxFillValidationError("Flux Fill noise source latent must have shape [B, 16, H, W].")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(tuple(latent.shape), generator=generator, dtype=dtype or latent.dtype, device="cpu")
    if device is not None:
        noise = noise.to(device=device)
    return noise


def denoise_flux_fill_latent(
    config: FluxFillConfig,
    latent_source: FluxFillLatentSource,
    *,
    empty_conditioning: FluxEmptyConditioning | None = None,
    unet_patcher: Any | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    callback: Any | None = None,
    disable_pbar: bool = True,
    cleanup_unet: bool = True,
) -> FluxFillDenoiseResult:
    config.validate_static(require_existing_assets=False)
    if empty_conditioning is None:
        empty_conditioning = load_flux_empty_conditioning_cache(config.conditioning_cache_path)

    if not isinstance(latent_source.source_latent, torch.Tensor):
        raise FluxFillValidationError("latent_source.source_latent must be a tensor.")
    if not isinstance(latent_source.denoise_mask, torch.Tensor):
        raise FluxFillValidationError("latent_source.denoise_mask must be a tensor.")

    from backend import resources, sampling

    device = torch.device(load_device or config.device) if (load_device or config.device) else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "sampler": config.sampler,
        "scheduler": config.scheduler,
        "steps": int(config.steps),
        "cfg": float(config.cfg),
        "denoise": float(config.denoise),
        "guidance": float(config.guidance),
        "seed": int(config.seed),
        "device": str(device),
    }

    unet_load_start = time.perf_counter()
    if unet_patcher is None:
        unet_patcher = load_flux_fill_unet(
            config.unet_path,
            load_device=device,
            offload_device=offload_device,
        )
    timings["unet_load"] = time.perf_counter() - unet_load_start

    source = latent_source.source_latent.detach().to(device=device, dtype=torch.float32)
    concat = latent_source.concat_latent.detach().to(device=device, dtype=torch.float32)
    mask = latent_source.denoise_mask.detach().to(device=device, dtype=torch.float32)
    noise = create_flux_fill_noise(source, config.seed, device=device, dtype=source.dtype)

    cond_start = time.perf_counter()
    payloads = build_flux_fill_conditioning_payloads(
        empty_conditioning,
        source,
        mask,
        concat_latent=concat,
        guidance=config.guidance,
        batch_size=int(source.shape[0]),
        device=device,
        dtype=source.dtype,
    )
    timings["conditioning_prepare"] = time.perf_counter() - cond_start

    sampler_start = time.perf_counter()
    sampler = sampling.KSampler(
        unet_patcher,
        config.steps,
        device,
        config.sampler,
        config.scheduler,
        config.denoise,
        model_options=getattr(unet_patcher, "model_options", {}),
    )
    sigmas = sampler.sigmas.detach().cpu()
    timings["sigma_prepare"] = time.perf_counter() - sampler_start

    denoise_start = time.perf_counter()
    denoise_cpu_start = time.process_time()
    try:
        samples = sampler.sample(
            noise,
            payloads.positive,
            payloads.negative,
            config.cfg,
            latent_image=payloads.latent_image,
            denoise_mask=payloads.denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=config.seed,
        )
    finally:
        timings["denoise_wall"] = time.perf_counter() - denoise_start
        timings["denoise_cpu_proc"] = time.process_time() - denoise_cpu_start
        if cleanup_unet:
            _cleanup_model_patcher(unet_patcher)
            try:
                resources.soft_empty_cache()
            except Exception:
                pass

    metadata["sigma_count"] = int(sigmas.shape[-1])
    metadata["latent_shape"] = list(samples.shape)
    return FluxFillDenoiseResult(
        samples=samples.detach().cpu(),
        noise=noise.detach().cpu(),
        sigmas=sigmas,
        timings=timings,
        metadata=metadata,
    )

def run_flux_fill(
    config: FluxFillConfig,
    image: Any,
    mask: Any,
    *,
    extend_factor: float = 1.2,
    tiled_decode: bool = False,
    tile_size: int = 64,
    callback: Any | None = None,
    disable_pbar: bool = True,
) -> FluxFillResult:
    config.validate_static(require_existing_assets=True)
    cache = load_flux_empty_conditioning_cache(config.conditioning_cache_path)

    latent_source = prepare_flux_fill_latent_source(
        image,
        mask,
        config.ae_path,
        extend_factor=extend_factor,
        load_device=config.device,
        offload_device=None,
    )
    denoise_result = denoise_flux_fill_latent(
        config,
        latent_source,
        empty_conditioning=cache,
        load_device=config.device,
        offload_device=None,
        callback=callback,
        disable_pbar=disable_pbar,
    )
    decoded = decode_flux_fill_latent(
        denoise_result.samples,
        config.ae_path,
        context=latent_source.context,
        stitch=True,
        tiled=tiled_decode,
        tile_size=tile_size,
        load_device=config.device,
        offload_device=None,
    )

    output_image = decoded.stitched_image if decoded.stitched_image is not None else decoded.bb_image
    timings = {
        **latent_source.timings,
        **denoise_result.timings,
        **decoded.timings,
    }
    metadata = {
        "tier": config.tier,
        "unet_path": str(config.unet_path),
        "ae_path": str(config.ae_path),
        "conditioning_cache_path": str(config.conditioning_cache_path),
        "denoise": denoise_result.metadata,
        "source_width": int(getattr(image, "shape", [0, 0])[1]),
        "source_height": int(getattr(image, "shape", [0, 0])[0]),
        "bb_width": int(latent_source.width),
        "bb_height": int(latent_source.height),
    }
    return FluxFillResult(
        output_image=output_image,
        output_path=Path(config.output_path) if config.output_path is not None else None,
        seed=int(config.seed),
        width=int(getattr(output_image, "shape", [0, 0])[1]),
        height=int(getattr(output_image, "shape", [0, 0])[0]),
        timings=timings,
        metadata=metadata,
    )

def load_flux_empty_conditioning_cache(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> FluxEmptyConditioning:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Flux empty-conditioning cache does not exist: {cache_path}")

    try:
        payload = torch.load(cache_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=map_location)

    if not isinstance(payload, dict):
        raise FluxFillValidationError(
            f"Flux empty-conditioning cache must contain a dict, got {type(payload).__name__}."
        )

    missing = [key for key in ("cross_attn", "pooled_output") if key not in payload]
    if missing:
        raise FluxFillValidationError(
            f"Flux empty-conditioning cache is missing required key(s): {', '.join(missing)}."
        )

    return FluxEmptyConditioning(
        cross_attn=payload["cross_attn"],
        pooled_output=payload["pooled_output"],
        metadata=payload.get("metadata", {}),
    )


def save_flux_empty_conditioning_cache(
    path: Path | str,
    *,
    cross_attn: torch.Tensor,
    pooled_output: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> FluxEmptyConditioning:
    conditioning = FluxEmptyConditioning(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        metadata=metadata or {},
    )
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cross_attn": conditioning.cross_attn,
            "pooled_output": conditioning.pooled_output,
            "metadata": conditioning.metadata,
        },
        cache_path,
    )
    return conditioning









