from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import sys
import time

import numpy as np
import torch
from backend import patching as backend_patching
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
IMPORTANT_FLUX_NATIVE_DTYPE_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "vector_in.in_layer.weight",
    "guidance_in.in_layer.weight",
    "final_layer.linear.weight",
)


from backend.flux.flux_fill_loader import (
    FluxFillValidationError,
    FluxFillUnsupportedModelError,
    FluxFillUNetInfo,
    _shape_tuple,
    _validate_tensor_shape,
    _shape_of,
    _numel_of,
    _normalize_flux_checkpoint_dtype,
    _parse_torch_dtype,
    _detect_flux_fill_weight_dtype,
    _estimate_flux_fill_parameter_count,
    _qtype_of,
    _count_block_indices,
    _snapshot_first_param_runtime,
    _snapshot_module_runtime,
    validate_flux_fill_unet_config,
    inspect_flux_fill_gguf,
    inspect_flux_fill_native_unet,
    _load_flux_fill_native_probe,
    _instantiate_flux_fill_native_model,
    _load_flux_fill_native_weights_into_model,
    load_flux_fill_unet,
    load_flux_fill_native_unet,
    load_flux_ae,
)
from backend.flux.flux_streaming import (
    FluxDirectStreamModelPatcher,
    _clamp_int,
    _resolve_streaming_scheduler_policy,
    load_flux_fill_native_unet_streaming,
    _pin_module_tensors_for_streaming,
    measure_pinned_module_tensors,
    FluxAsyncLayerPrefetchScheduler,
    _sample_flux_fill_direct_streaming,
)


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
    execution_class: Any | None = None
    runtime_family: str | None = None
    runtime_posture: str | None = None
    streaming_profile: str | None = None
    resident_load_strategy: str | None = None
    fallback_model_variant: str | None = None

    def validate_static(self, *, require_existing_assets: bool = True) -> None:
        if self.steps < 1:
            raise FluxFillValidationError(f"steps must be >= 1, got {self.steps}.")
        if self.cfg != 1.0:
            raise NotImplementedError("Prompt-conditioned/CFG Flux Fill is out of scope for P4-M10-W01.")
        if not str(self.sampler or "").strip():
            raise FluxFillValidationError("sampler must be a non-empty string.")
        if not str(self.scheduler or "").strip():
            raise FluxFillValidationError("scheduler must be a non-empty string.")
        if self.denoise != 1.0:
            raise NotImplementedError("Flux Fill W01 only supports denoise=1.0.")
        if self.guidance <= 0:
            raise FluxFillValidationError(f"guidance must be > 0, got {self.guidance}.")
        if require_existing_assets:
            for label, value in (
                ("UNet", self.unet_path),
                ("AE", self.ae_path),
                ("conditioning cache", self.conditioning_cache_path),
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
class FluxFillLatentSource:
    context: Any
    source_latent: torch.Tensor        # VAE encode of UNMASKED original (KSampler noise start)
    concat_latent: torch.Tensor        # VAE encode of gray-masked image (c_concat condition)
    denoise_mask: torch.Tensor
    width: int
    height: int
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillDenoiseResult:
    samples: torch.Tensor
    noise: torch.Tensor
    sigmas: torch.Tensor
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillPrecomputedDenoiseInput:
    source_latent: torch.Tensor
    concat_latent: torch.Tensor
    denoise_mask: torch.Tensor
    empty_conditioning: FluxEmptyConditioning
    seed: int
    guidance: float
    steps: int
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "normal"
    denoise: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.source_latent, torch.Tensor) or self.source_latent.ndim != 4 or int(self.source_latent.shape[1]) != 16:
            raise FluxFillValidationError("source_latent must have shape [B, 16, H, W].")
        if not isinstance(self.concat_latent, torch.Tensor) or self.concat_latent.ndim != 4 or int(self.concat_latent.shape[1]) != 16:
            raise FluxFillValidationError("concat_latent must have shape [B, 16, H, W].")
        if not isinstance(self.denoise_mask, torch.Tensor) or self.denoise_mask.ndim != 4 or int(self.denoise_mask.shape[1]) != 1:
            raise FluxFillValidationError("denoise_mask must have shape [B, 1, H, W].")
        if self.source_latent.shape[0] != self.concat_latent.shape[0] or self.source_latent.shape[-2:] != self.concat_latent.shape[-2:]:
            raise FluxFillValidationError("source_latent and concat_latent must have the same batch and spatial dimensions.")
        if self.source_latent.shape[0] != self.denoise_mask.shape[0] or self.source_latent.shape[-2:] != self.denoise_mask.shape[-2:]:
            raise FluxFillValidationError("denoise_mask shape does not match source_latent.")
        if self.steps < 1:
            raise FluxFillValidationError(f"steps must be >= 1, got {self.steps}.")
        if self.guidance <= 0:
            raise FluxFillValidationError(f"guidance must be > 0, got {self.guidance}.")
        if self.denoise != 1.0:
            raise NotImplementedError("Flux Fill W08b precomputed denoise path only supports denoise=1.0.")


@dataclass(frozen=True)
class FluxFillPrecomputedDenoiseResult:
    samples: torch.Tensor
    noise: torch.Tensor
    sigmas: torch.Tensor
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_image_mask_arrays(image: Any, mask: Any) -> None:
    image_shape = getattr(image, "shape", None)
    mask_shape = getattr(mask, "shape", None)
    if image_shape is None or len(image_shape) != 3 or int(image_shape[2]) != 3:
        raise FluxFillValidationError(f"image must have shape [H, W, 3], got {image_shape}.")
    if mask_shape is None or len(mask_shape) not in (2, 3):
        raise FluxFillValidationError(f"mask must have shape [H, W] or [H, W, C], got {mask_shape}.")
    if int(mask_shape[0]) != int(image_shape[0]) or int(mask_shape[1]) != int(image_shape[1]):
        raise FluxFillValidationError(f"mask spatial shape {mask_shape[:2]} does not match image {image_shape[:2]}.")


def _cleanup_model_patcher(model_patcher: Any, *, cleanup: bool = True) -> None:
    if not cleanup:
        return
    try:
        flux_options = getattr(model_patcher, "model_options", {}).get("flux_fill", {})
        scheduler = flux_options.get("streaming_scheduler")
        if scheduler is not None and hasattr(scheduler, "detach"):
            scheduler.detach()
    except Exception:
        pass
    try:
        from backend import resources

        resources.eject_model(model_patcher)
    except Exception:
        detach = getattr(model_patcher, "detach", None)
        if callable(detach):
            detach()


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
    vae: Any | None = None,
    cleanup_vae: bool | None = None,
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

    owned_vae = vae is None
    encode_start = time.perf_counter()
    metadata: dict[str, Any] = {}
    try:
        from modules.core import numpy_to_pytorch, encode_vae
        from backend import resources

        if vae is None:
            vae = load_flux_ae(ae_path, load_device=load_device, offload_device=offload_device)
        resources.load_models_gpu([vae.patcher])
        vae_device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
        move_model = getattr(vae.first_stage_model, "to", None)
        if callable(move_model):
            move_model(device=vae_device, dtype=torch.float32)
        metadata["vae_runtime_before_source_encode"] = _snapshot_first_param_runtime(vae.first_stage_model)

        # Encode unmasked original → source_latent (KSampler noise init)
        orig_pixels = numpy_to_pytorch(image)
        source_latent = encode_vae(vae=vae, pixels=orig_pixels)["samples"]

        # Encode gray-masked → concat_latent (c_concat condition)
        # USER_REQUEST: Bypass encode.py to avoid double-normalization.
        # Flux.concat_cond (ldm_patched) will apply normalization itself.
        resources.load_models_gpu([vae.patcher])
        vae_device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
        move_model = getattr(vae.first_stage_model, "to", None)
        if callable(move_model):
            move_model(device=vae_device, dtype=torch.float32)
        metadata["vae_runtime_before_concat_encode"] = _snapshot_first_param_runtime(vae.first_stage_model)
        pixels_for_vae = (numpy_to_pytorch(bb_image_for_concat).movedim(-1, 1) * 2.0) - 1.0
        if pixels_for_vae.ndim == 3:
            pixels_for_vae = pixels_for_vae.unsqueeze(0)

        vae_param = next(vae.first_stage_model.parameters(), None)
        vae_input_device = vae.patcher.load_device
        vae_input_dtype = torch.float32
        if isinstance(vae_param, torch.Tensor):
            vae_input_device = vae_param.device
            vae_input_dtype = vae_param.dtype

        metadata["vae_runtime_manual_concat_input"] = {
            "device": str(vae_input_device),
            "dtype": str(vae_input_dtype),
        }
        pixels_for_vae = pixels_for_vae.to(device=vae_input_device, dtype=vae_input_dtype)

        # We manually call the base VAE model to get RAW unscaled latents.
        raw_latent = vae.first_stage_model.encode(pixels_for_vae)
        if hasattr(raw_latent, "sample"):
            raw_latent = raw_latent.sample()
            
        concat_latent = raw_latent.cpu()

        vae.patcher.detach()
        resources.soft_empty_cache()
    finally:
        should_cleanup = cleanup_vae if cleanup_vae is not None else owned_vae
        if vae is not None:
            _cleanup_model_patcher(vae.patcher, cleanup=should_cleanup)
        if should_cleanup:
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
        metadata=metadata,
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
    vae: Any | None = None,
    cleanup_vae: bool | None = None,
) -> FluxFillDecodedImage:
    if not isinstance(latent, torch.Tensor):
        raise FluxFillValidationError(f"latent must be a torch.Tensor, got {type(latent).__name__}.")
    if latent.ndim != 4 or int(latent.shape[1]) != 16:
        raise FluxFillValidationError(f"Flux decoded latent must have shape [B, 16, H, W], got {list(latent.shape)}.")
    if stitch and context is None:
        raise FluxFillValidationError("context is required when stitch=True.")

    timings: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    owned_vae = vae is None
    decode_start = time.perf_counter()
    try:
        try:
            from backend import process_transition
            posture = "Resident" if str(config.runtime_posture or "").strip().lower() == "resident" else "Streaming"
            process_transition.log_stage_telemetry("vae_decode", posture_override=posture)
        except Exception:
            pass
        if vae is None:
            vae = load_flux_ae(ae_path, load_device=load_device, offload_device=offload_device)
        original_argv = list(sys.argv)
        try:
            sys.argv = [original_argv[0]]
            from modules import core
        finally:
            sys.argv = original_argv

        metadata["vae_runtime_before_decode"] = _snapshot_first_param_runtime(vae.first_stage_model)
        decoded = core.decode_vae(vae, {"samples": latent.detach().cpu()}, tiled=tiled)
        metadata["vae_runtime_after_decode"] = _snapshot_first_param_runtime(vae.first_stage_model)
    finally:
        should_cleanup = cleanup_vae if cleanup_vae is not None else owned_vae
        if vae is not None:
            _cleanup_model_patcher(vae.patcher, cleanup=should_cleanup)
        if should_cleanup:
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

    return FluxFillDecodedImage(
        bb_image=bb_image,
        stitched_image=stitched_image,
        timings=timings,
        metadata=metadata,
    )


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
    # Flux / Flux Fill keep cfg at 1 and use a zeroed negative branch rather than
    # reusing the prompt conditioning as the negative prompt payload.
    negative_payload = payload.copy()
    negative_payload["pooled_output"] = torch.zeros_like(pooled_output)
    negative = [[torch.zeros_like(cross_attn), negative_payload]]
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
    cleanup_unet: bool | None = None,
) -> FluxFillDenoiseResult:
    config.validate_static(require_existing_assets=False)
    if empty_conditioning is None:
        empty_conditioning = load_flux_empty_conditioning_cache(config.conditioning_cache_path)

    if not isinstance(latent_source.source_latent, torch.Tensor):
        raise FluxFillValidationError("latent_source.source_latent must be a tensor.")
    if not isinstance(latent_source.denoise_mask, torch.Tensor):
        raise FluxFillValidationError("latent_source.denoise_mask must be a tensor.")

    from backend import resources, sampling
    try:
        from backend import process_transition
        posture = "Resident" if str(config.runtime_posture or "").strip().lower() == "resident" else "Streaming"
        process_transition.log_stage_telemetry("denoise", posture_override=posture)
    except Exception:
        pass

    device = torch.device(load_device or config.device) if (load_device or config.device) else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    owned_unet = unet_patcher is None
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
        "resident_unet": not owned_unet,
    }
    flux_options = getattr(unet_patcher, "model_options", {}).get("flux_fill", {}) if unet_patcher is not None else {}
    if flux_options:
        metadata["native_unet_load_diagnostics"] = {
            "detected_weight_dtype": flux_options.get("detected_config", {}).get("weight_dtype"),
            "source_weight_dtype": flux_options.get("detected_config", {}).get("source_weight_dtype"),
            "resident_weight_dtype": flux_options.get("detected_config", {}).get("resident_weight_dtype"),
            "manual_cast_dtype": flux_options.get("detected_config", {}).get("manual_cast_dtype"),
            "post_construct_runtime": flux_options.get("detected_config", {}).get("post_construct_runtime", {}),
            "post_load_runtime": flux_options.get("detected_config", {}).get("post_load_runtime", {}),
        }
    streaming_scheduler = flux_options.get("streaming_scheduler")
    if streaming_scheduler is not None and hasattr(streaming_scheduler, "reset_run"):
        streaming_scheduler.reset_run()
    unet_load_start = time.perf_counter()
    if unet_patcher is None:
        unet_patcher = load_flux_fill_unet(
            config.unet_path,
            load_device=device,
            offload_device=offload_device,
            execution_class=config.execution_class,
            runtime_family=config.runtime_family,
            runtime_posture=config.runtime_posture,
            streaming_profile=config.streaming_profile,
            resident_load_strategy=config.resident_load_strategy,
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
    flux_options = getattr(unet_patcher, "model_options", {}).get("flux_fill", {})
    metadata["native_unet_runtime_before_denoise"] = _snapshot_module_runtime(getattr(unet_patcher, "model", None))
    direct_stream_runtime = bool(flux_options.get("direct_stream_runtime", False))
    if direct_stream_runtime:
        sigmas = torch.empty(0)
    else:
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
        if direct_stream_runtime:
            samples, sigmas = _sample_flux_fill_direct_streaming(
                unet_patcher=unet_patcher,
                noise=noise,
                payloads=payloads,
                steps=config.steps,
                device=device,
                sampler_name=config.sampler,
                scheduler_name=config.scheduler,
                denoise=config.denoise,
                cfg=config.cfg,
                seed=config.seed,
                callback=callback,
                disable_pbar=disable_pbar,
            )
        else:
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
        metadata["native_unet_runtime_after_denoise"] = _snapshot_module_runtime(getattr(unet_patcher, "model", None))
        active_scheduler = getattr(unet_patcher, "model_options", {}).get("flux_fill", {}).get("streaming_scheduler")
        if active_scheduler is not None:
            if hasattr(active_scheduler, "snapshot"):
                metadata["streaming_scheduler"] = active_scheduler.snapshot()
            if hasattr(active_scheduler, "reset_run"):
                active_scheduler.reset_run(clear_prefetched=True)
        is_streaming = (str(config.runtime_posture or "").strip().lower() == "streaming")
        should_cleanup = cleanup_unet if cleanup_unet is not None else (owned_unet or is_streaming)
        metadata["cleanup_unet"] = bool(should_cleanup)
        if should_cleanup:
            _cleanup_model_patcher(unet_patcher, cleanup=should_cleanup)
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


def denoise_flux_fill_precomputed_latent(
    config: FluxFillConfig,
    denoise_input: FluxFillPrecomputedDenoiseInput,
    *,
    unet_patcher: Any | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    callback: Any | None = None,
    disable_pbar: bool = True,
    cleanup_unet: bool | None = None,
) -> FluxFillPrecomputedDenoiseResult:
    config.validate_static(require_existing_assets=False)
    denoise_input.validate()

    from backend import resources, sampling
    try:
        from backend import process_transition
        posture = "Resident" if str(config.runtime_posture or "").strip().lower() == "resident" else "Streaming"
        process_transition.log_stage_telemetry("denoise", posture_override=posture)
    except Exception:
        pass

    device = torch.device(load_device or config.device) if (load_device or config.device) else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    owned_unet = unet_patcher is None
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
        "resident_unet": not owned_unet,
        "precomputed_inputs": True,
    }
    flux_options = getattr(unet_patcher, "model_options", {}).get("flux_fill", {}) if unet_patcher is not None else {}
    if flux_options:
        metadata["native_unet_load_diagnostics"] = {
            "detected_weight_dtype": flux_options.get("detected_config", {}).get("weight_dtype"),
            "source_weight_dtype": flux_options.get("detected_config", {}).get("source_weight_dtype"),
            "resident_weight_dtype": flux_options.get("detected_config", {}).get("resident_weight_dtype"),
            "manual_cast_dtype": flux_options.get("detected_config", {}).get("manual_cast_dtype"),
            "post_construct_runtime": flux_options.get("detected_config", {}).get("post_construct_runtime", {}),
            "post_load_runtime": flux_options.get("detected_config", {}).get("post_load_runtime", {}),
        }
    streaming_scheduler = flux_options.get("streaming_scheduler")
    if streaming_scheduler is not None and hasattr(streaming_scheduler, "reset_run"):
        streaming_scheduler.reset_run()
    unet_load_start = time.perf_counter()
    if unet_patcher is None:
        unet_patcher = load_flux_fill_unet(
            config.unet_path,
            load_device=device,
            offload_device=offload_device,
            execution_class=config.execution_class,
            runtime_family=config.runtime_family,
            runtime_posture=config.runtime_posture,
            streaming_profile=config.streaming_profile,
            resident_load_strategy=config.resident_load_strategy,
        )
    timings["unet_load"] = time.perf_counter() - unet_load_start

    source = denoise_input.source_latent.detach().to(device=device, dtype=torch.float32)
    concat = denoise_input.concat_latent.detach().to(device=device, dtype=torch.float32)
    mask = denoise_input.denoise_mask.detach().to(device=device, dtype=torch.float32)
    noise = create_flux_fill_noise(source, denoise_input.seed, device=device, dtype=source.dtype)

    payloads = build_flux_fill_conditioning_payloads(
        denoise_input.empty_conditioning,
        source,
        mask,
        concat_latent=concat,
        guidance=denoise_input.guidance,
        batch_size=int(source.shape[0]),
        device=device,
        dtype=source.dtype,
    )

    sampler = sampling.KSampler(
        unet_patcher,
        denoise_input.steps,
        device,
        denoise_input.sampler,
        denoise_input.scheduler,
        denoise_input.denoise,
        model_options=getattr(unet_patcher, "model_options", {}),
    ) if not flux_options.get("direct_stream_runtime", False) else None
    sigmas = sampler.sigmas.detach().cpu() if sampler is not None else torch.empty(0)
    metadata["native_unet_runtime_before_denoise"] = _snapshot_module_runtime(getattr(unet_patcher, "model", None))

    denoise_start = time.perf_counter()
    denoise_cpu_start = time.process_time()
    try:
        if flux_options.get("direct_stream_runtime", False):
            samples, sigmas = _sample_flux_fill_direct_streaming(
                unet_patcher=unet_patcher,
                noise=noise,
                payloads=payloads,
                steps=denoise_input.steps,
                device=device,
                sampler_name=denoise_input.sampler,
                scheduler_name=denoise_input.scheduler,
                denoise=denoise_input.denoise,
                cfg=float(denoise_input.cfg),
                seed=denoise_input.seed,
                callback=callback,
                disable_pbar=disable_pbar,
            )
        else:
            samples = sampler.sample(
                noise,
                payloads.positive,
                payloads.negative,
                float(denoise_input.cfg),
                latent_image=payloads.latent_image,
                denoise_mask=payloads.denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=denoise_input.seed,
            )
    finally:
        timings["denoise_wall"] = time.perf_counter() - denoise_start
        timings["denoise_cpu_proc"] = time.process_time() - denoise_cpu_start
        metadata["native_unet_runtime_after_denoise"] = _snapshot_module_runtime(getattr(unet_patcher, "model", None))
        active_scheduler = getattr(unet_patcher, "model_options", {}).get("flux_fill", {}).get("streaming_scheduler")
        if active_scheduler is not None:
            if hasattr(active_scheduler, "snapshot"):
                metadata["streaming_scheduler"] = active_scheduler.snapshot()
            if hasattr(active_scheduler, "reset_run"):
                active_scheduler.reset_run(clear_prefetched=True)
        should_cleanup = cleanup_unet if cleanup_unet is not None else owned_unet
        metadata["cleanup_unet"] = bool(should_cleanup)
        if should_cleanup:
            _cleanup_model_patcher(unet_patcher, cleanup=should_cleanup)
            try:
                resources.soft_empty_cache()
            except Exception:
                pass

    metadata["sigma_count"] = int(sigmas.shape[-1])
    metadata["latent_shape"] = list(samples.shape)
    return FluxFillPrecomputedDenoiseResult(
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
    unet_patcher: Any | None = None,
    vae: Any | None = None,
    empty_conditioning: FluxEmptyConditioning | None = None,
    cleanup_unet: bool | None = None,
    cleanup_vae: bool | None = None,
) -> FluxFillResult:
    config.validate_static(require_existing_assets=True)
    cache = empty_conditioning if empty_conditioning is not None else load_flux_empty_conditioning_cache(config.conditioning_cache_path)

    is_streaming = (str(config.runtime_posture or "").strip().lower() == "streaming")
    effective_cleanup_vae = cleanup_vae if cleanup_vae is not None else (vae is None or is_streaming)
    effective_cleanup_unet = cleanup_unet if cleanup_unet is not None else (unet_patcher is None or is_streaming)

    latent_source = prepare_flux_fill_latent_source(
        image,
        mask,
        config.ae_path,
        extend_factor=extend_factor,
        load_device=config.device,
        offload_device=None,
        vae=vae,
        cleanup_vae=effective_cleanup_vae,
    )
    denoise_result = denoise_flux_fill_latent(
        config,
        latent_source,
        empty_conditioning=cache,
        load_device=config.device,
        offload_device=None,
        unet_patcher=unet_patcher,
        cleanup_unet=effective_cleanup_unet,
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
        vae=vae,
        cleanup_vae=effective_cleanup_vae,
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
        "latent_prep": latent_source.metadata,
        "decode": decoded.metadata,
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









