from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import sys
import time

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


def _numel_of(value: Any) -> int | None:
    if hasattr(value, "nelement") and callable(value.nelement):
        try:
            return int(value.nelement())
        except Exception:
            pass
    shape = _shape_of(value)
    if not shape:
        return None
    try:
        return int(math.prod(shape))
    except Exception:
        return None


def _normalize_flux_checkpoint_dtype(value: Any) -> torch.dtype | None:
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None and isinstance(value, torch.dtype):
        return value

    dtype_text = str(dtype if dtype is not None else value).strip()
    mapping = {
        "F16": torch.float16,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FN": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
        "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    }
    return mapping.get(dtype_text)


def _detect_flux_fill_weight_dtype(state_dict: dict[str, Any]) -> torch.dtype | None:
    for key in IMPORTANT_FLUX_NATIVE_DTYPE_KEYS:
        tensor = state_dict.get(key)
        if tensor is None:
            continue
        detected = _normalize_flux_checkpoint_dtype(tensor)
        if detected is not None:
            return detected
    return None


def _estimate_flux_fill_parameter_count(state_dict: dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        numel = _numel_of(value)
        if numel is not None:
            total += numel
    return int(total)


class FluxDirectStreamModelPatcher(backend_patching.NexModelPatcher):
    """Treat a pinned CPU-host UNet as the source artifact and skip generic reload work."""

    def model_size(self):
        return 0

    def loaded_size(self):
        return 0

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        target_device = device_to or self.load_device
        self.model.device = target_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        self.model.current_weight_patches_uuid = self.patches_uuid

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        self.load(device_to=device_to, force_patch_weights=force_patch_weights, full_load=False)
        return 0

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        target_device = device_to or self.offload_device
        self.model.device = target_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        return self.model

    def detach(self, unpatch_all=True):
        self.model.device = self.offload_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        return self.model


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
    execution_class: Any | None = None

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


def inspect_flux_fill_native_unet(
    unet_path: Path | str,
    *,
    validate_contract: bool = True,
) -> FluxFillUNetInfo:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    from backend.cpu_compiler import SafeOpenHeaderOnly
    from ldm_patched.modules import model_detection

    state_dict = _load_flux_fill_native_probe(str(path))

    detected_config = model_detection.detect_unet_config(state_dict, "", dtype=None)
    if detected_config is None:
        raise FluxFillValidationError(f"Could not detect Flux model config from {path}.")
    if validate_contract:
        validate_flux_fill_unet_config(detected_config)

    key_shapes = {key: _shape_of(state_dict[key]) if key in state_dict else None for key in IMPORTANT_FLUX_GGUF_KEYS}
    qtypes: dict[str, int] = {}
    for tensor in state_dict.values():
        qtype = _qtype_of(tensor) or str(getattr(tensor, "dtype", "torch"))
        qtypes[qtype] = qtypes.get(qtype, 0) + 1

    detected_config = dict(detected_config)
    detected_config["double_blocks"] = _count_block_indices(state_dict, "double_blocks.")
    detected_config["single_blocks"] = _count_block_indices(state_dict, "single_blocks.")
    return FluxFillUNetInfo(
        path=path,
        arch="flux",
        detected_config=detected_config,
        key_shapes=key_shapes,
        qtypes=qtypes,
        tensor_count=len(state_dict),
    )


def _load_flux_fill_native_probe(unet_path: str) -> dict[str, Any]:
    from backend.cpu_compiler import SafeOpenHeaderOnly

    if unet_path.lower().endswith(".safetensors"):
        return SafeOpenHeaderOnly(unet_path)

    from backend import loader

    state_dict = loader.resolve_source(unet_path)
    if not isinstance(state_dict, dict):
        raise FluxFillValidationError(
            f"Expected state dict while probing Flux Fill native UNet {unet_path}, got {type(state_dict).__name__}."
        )
    return state_dict


def _snapshot_first_param_runtime(module: Any) -> dict[str, Any]:
    try:
        param = next(module.parameters(), None)
    except Exception:
        param = None
    if not isinstance(param, torch.Tensor):
        return {}
    return {
        "device": str(param.device),
        "dtype": str(param.dtype),
    }


def _snapshot_module_runtime(module: Any) -> dict[str, Any]:
    root_module = getattr(module, "diffusion_model", module)
    if root_module is None:
        return {}

    def _tensor_bytes(tensor: Any) -> int:
        if not isinstance(tensor, torch.Tensor):
            return 0
        return int(tensor.nelement()) * int(tensor.element_size())

    def _accumulate(bucket: dict[str, int], key: str, value: int) -> None:
        bucket[key] = int(bucket.get(key, 0)) + int(value)

    snapshot: dict[str, Any] = {
        "first_param": _snapshot_first_param_runtime(root_module),
        "param_count": 0,
        "buffer_count": 0,
        "total_param_bytes": 0,
        "total_buffer_bytes": 0,
        "floating_param_bytes": 0,
        "param_bytes_by_dtype": {},
        "param_bytes_by_device": {},
        "buffer_bytes_by_dtype": {},
        "buffer_bytes_by_device": {},
    }

    try:
        params = list(root_module.parameters())
    except Exception:
        params = []
    for param in params:
        if not isinstance(param, torch.Tensor):
            continue
        tensor_bytes = _tensor_bytes(param)
        snapshot["param_count"] += 1
        snapshot["total_param_bytes"] += tensor_bytes
        _accumulate(snapshot["param_bytes_by_dtype"], str(param.dtype), tensor_bytes)
        _accumulate(snapshot["param_bytes_by_device"], str(param.device), tensor_bytes)
        if param.is_floating_point():
            snapshot["floating_param_bytes"] += tensor_bytes

    try:
        buffers = list(root_module.buffers())
    except Exception:
        buffers = []
    for buffer in buffers:
        if not isinstance(buffer, torch.Tensor):
            continue
        tensor_bytes = _tensor_bytes(buffer)
        snapshot["buffer_count"] += 1
        snapshot["total_buffer_bytes"] += tensor_bytes
        _accumulate(snapshot["buffer_bytes_by_dtype"], str(buffer.dtype), tensor_bytes)
        _accumulate(snapshot["buffer_bytes_by_device"], str(buffer.device), tensor_bytes)

    return snapshot


def _instantiate_flux_fill_native_model(
    path: Path,
    *,
    offload_device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    from backend import resources
    from ldm_patched.modules import model_detection
    from ldm_patched.modules import model_management
    from ldm_patched.modules import supported_models_base

    state_probe = _load_flux_fill_native_probe(str(path))
    detected_weight_dtype = _detect_flux_fill_weight_dtype(state_probe)
    detected_config = model_detection.detect_unet_config(state_probe, "", dtype=detected_weight_dtype)
    if detected_config is None:
        raise FluxFillValidationError(f"Could not detect Flux model config from {path}.")
    validate_flux_fill_unet_config(detected_config)

    model_config = model_detection.model_config_from_unet_config(detected_config, state_probe)
    if model_config is None:
        raise FluxFillValidationError(f"No supported Flux model config matched {path}.")

    inference_device = resources.get_torch_device()
    manual_cast_dtype = None
    if detected_weight_dtype is not None:
        manual_cast_dtype = model_management.unet_manual_cast(detected_weight_dtype, inference_device)
        if manual_cast_dtype is None:
            manual_cast_dtype = detected_weight_dtype
        model_config.set_manual_cast(manual_cast_dtype)
    elif isinstance(model_config, supported_models_base.BASE):
        model_config.set_manual_cast(torch.float16 if inference_device.type == "cuda" else torch.float32)

    model = model_config.get_model(
        state_probe,
        "",
        device=offload_device,
        model_options={"custom_operations": None},
    )
    detected_metadata = dict(detected_config)
    detected_metadata["weight_dtype"] = str(detected_weight_dtype) if detected_weight_dtype is not None else None
    detected_metadata["manual_cast_dtype"] = str(manual_cast_dtype) if manual_cast_dtype is not None else None
    detected_metadata["parameter_count"] = _estimate_flux_fill_parameter_count(state_probe)
    detected_metadata["post_construct_runtime"] = _snapshot_module_runtime(model)
    return model, detected_metadata


def _load_flux_fill_native_weights_into_model(
    path: Path,
    model: Any,
    *,
    target_device: torch.device,
) -> dict[str, Any]:
    from backend import loader

    flux_metadata = {
        "direct_safetensors_load": False,
        "single_host_artifact": False,
    }

    if str(path).lower().endswith(".safetensors"):
        missing, unexpected = loader._load_prefixed_safetensors_into_module(
            str(path),
            [""],
            model.diffusion_model,
            device=target_device,
            dtype=None,
        )
        if missing:
            raise FluxFillValidationError(
                f"Missing Flux native UNet keys while directly loading {path}: {missing[:8]}"
                + (" ..." if len(missing) > 8 else "")
            )
        if unexpected:
            raise FluxFillValidationError(
                f"Unexpected Flux native UNet keys while directly loading {path}: {unexpected[:8]}"
                + (" ..." if len(unexpected) > 8 else "")
            )
        flux_metadata["direct_safetensors_load"] = True
        flux_metadata["single_host_artifact"] = True
        return flux_metadata

    from backend import loader as backend_loader

    state_dict = backend_loader.resolve_source(str(path))
    if not isinstance(state_dict, dict):
        raise FluxFillValidationError(
            f"Expected state dict while loading Flux Fill native UNet {path}, got {type(state_dict).__name__}."
        )
    model.load_model_weights(state_dict, "")
    return flux_metadata


def load_flux_fill_unet(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
    execution_class: Any | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    from backend import resources
    from backend.gguf.loader import gguf_sd_loader, is_streaming_execution_class
    from backend.gguf.ops import GGMLOps
    from backend.gguf.patcher import GGUFModelPatcher
    from ldm_patched.modules import model_detection

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    streaming = is_streaming_execution_class(execution_class)
    if streaming:
        if load_device.type != "cpu" or offload_device.type != "cpu":
            raise RuntimeError("Streaming-class Flux GGUF loads must stage weights on CPU pinned host memory.")

    state_dict, arch = gguf_sd_loader(
        str(path),
        handle_prefix=handle_prefix,
        return_arch=True,
        pin_memory=streaming,
        execution_class=execution_class,
        require_pinned_host=streaming,
    )
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
    patcher = GGUFModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        preserve_source_artifact=streaming,
    )
    patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": arch,
        "detected_config": dict(detected_config),
    }
    return patcher


def load_flux_fill_native_unet(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    from backend import patching, resources

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    model, detected_config = _instantiate_flux_fill_native_model(
        path,
        offload_device=offload_device,
    )
    direct_load_metadata = _load_flux_fill_native_weights_into_model(
        path,
        model,
        target_device=offload_device,
    )
    detected_config["post_load_runtime"] = _snapshot_module_runtime(model)
    runtime_patcher = patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        preserve_source_artifact=False,
    )
    runtime_weight_bytes = int(runtime_patcher.model_size())
    runtime_weight_dtype = detected_config.get("manual_cast_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8",
        "runtime_weight_dtype": runtime_weight_dtype,
        "runtime_weight_bytes": runtime_weight_bytes,
        **direct_load_metadata,
    }
    return runtime_patcher


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(int(minimum), min(int(maximum), int(value)))


def _resolve_streaming_scheduler_policy(
    *,
    device: torch.device,
    prefetch_depth: int | None = None,
    max_prefetch_bytes: int | None = None,
    vram_guard_bytes: int | None = None,
    vram_guard_margin_bytes: int | None = None,
    prefetch_scan_ahead: int | None = None,
    bandwidth_limit_mb_s: float | None = None,
) -> dict[str, Any]:
    legacy_guard_bytes = int(2.85 * 1024 * 1024 * 1024)
    legacy_margin_bytes = 256 * 1024 * 1024
    legacy_prefetch_bytes = 256 * 1024 * 1024

    total_vram_bytes = 0
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            total_vram_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        except Exception:
            total_vram_bytes = 0

    if total_vram_bytes > 0:
        dynamic_margin_bytes = max(256 * 1024 * 1024, int(total_vram_bytes * 0.05))
        dynamic_guard_bytes = max(256 * 1024 * 1024, int(total_vram_bytes * 0.90))
        if dynamic_guard_bytes + dynamic_margin_bytes > total_vram_bytes:
            dynamic_guard_bytes = max(
                256 * 1024 * 1024,
                int(total_vram_bytes) - int(dynamic_margin_bytes),
            )
        dynamic_prefetch_bytes = _clamp_int(
            int(total_vram_bytes * 0.04),
            minimum=256 * 1024 * 1024,
            maximum=1024 * 1024 * 1024,
        )
    else:
        dynamic_margin_bytes = legacy_margin_bytes
        dynamic_guard_bytes = legacy_guard_bytes
        dynamic_prefetch_bytes = legacy_prefetch_bytes

    resolved_prefetch_depth = max(0, int(prefetch_depth if prefetch_depth is not None else 1))
    resolved_prefetch_scan_ahead = max(1, int(prefetch_scan_ahead if prefetch_scan_ahead is not None else 1))
    resolved_max_prefetch_bytes = int(
        max_prefetch_bytes if max_prefetch_bytes is not None else dynamic_prefetch_bytes
    )
    resolved_vram_guard_bytes = int(
        vram_guard_bytes if vram_guard_bytes is not None else dynamic_guard_bytes
    )
    resolved_vram_guard_margin_bytes = max(
        0,
        int(vram_guard_margin_bytes if vram_guard_margin_bytes is not None else dynamic_margin_bytes),
    )
    resolved_bandwidth_limit_mb_s = (
        float(bandwidth_limit_mb_s) if bandwidth_limit_mb_s is not None and float(bandwidth_limit_mb_s) > 0.0 else 0.0
    )
    return {
        "prefetch_depth": int(resolved_prefetch_depth),
        "max_prefetch_bytes": int(resolved_max_prefetch_bytes),
        "vram_guard_bytes": int(resolved_vram_guard_bytes),
        "vram_guard_margin_bytes": int(resolved_vram_guard_margin_bytes),
        "prefetch_scan_ahead": int(resolved_prefetch_scan_ahead),
        "total_vram_bytes": int(total_vram_bytes),
        "bandwidth_limit_mb_s": float(resolved_bandwidth_limit_mb_s),
    }


def load_flux_fill_native_unet_streaming(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
    prefetch_depth: int | None = None,
    max_prefetch_bytes: int | None = None,
    vram_guard_bytes: int | None = None,
    vram_guard_margin_bytes: int | None = None,
    prefetch_scan_ahead: int | None = None,
    bandwidth_limit_mb_s: float | None = None,
) -> Any:
    from backend import resources

    host_load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    host_offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    if host_load_device.type != "cpu" or host_offload_device.type != "cpu":
        raise RuntimeError("Native Flux Fill streaming loads must stage weights on CPU host memory.")

    compute_device = resources.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
    path = Path(unet_path)
    model, detected_config = _instantiate_flux_fill_native_model(
        path,
        offload_device=host_offload_device,
    )
    direct_load_metadata = _load_flux_fill_native_weights_into_model(
        path,
        model,
        target_device=host_offload_device,
    )
    runtime_patcher = FluxDirectStreamModelPatcher(
        model,
        load_device=compute_device,
        offload_device=host_offload_device,
        preserve_source_artifact=True,
    )
    runtime_weight_dtype = detected_config.get("manual_cast_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8_streaming",
        "runtime_weight_dtype": runtime_weight_dtype,
        **direct_load_metadata,
    }
    pinned_bytes = _pin_module_tensors_for_streaming(getattr(runtime_patcher, "model", None))
    scheduler_policy = _resolve_streaming_scheduler_policy(
        device=compute_device,
        prefetch_depth=prefetch_depth,
        max_prefetch_bytes=max_prefetch_bytes,
        vram_guard_bytes=vram_guard_bytes,
        vram_guard_margin_bytes=vram_guard_margin_bytes,
        prefetch_scan_ahead=prefetch_scan_ahead,
        bandwidth_limit_mb_s=bandwidth_limit_mb_s,
    )
    streaming_scheduler = FluxAsyncLayerPrefetchScheduler(
        prefetch_depth=scheduler_policy["prefetch_depth"],
        max_prefetch_bytes=scheduler_policy["max_prefetch_bytes"],
        vram_guard_bytes=scheduler_policy["vram_guard_bytes"],
        vram_guard_margin_bytes=scheduler_policy["vram_guard_margin_bytes"],
        prefetch_scan_ahead=scheduler_policy["prefetch_scan_ahead"],
        bandwidth_limit_mb_s=scheduler_policy["bandwidth_limit_mb_s"],
    )
    scheduled_module_count = streaming_scheduler.attach(getattr(runtime_patcher, "model", None))
    flux_options = runtime_patcher.model_options.setdefault("flux_fill", {})
    flux_options["host_pinned_bytes"] = int(max(pinned_bytes, measure_pinned_module_tensors(getattr(runtime_patcher, "model", None))))
    flux_options["non_blocking_supported"] = bool(resources.device_supports_non_blocking(torch.device("cuda"))) if torch.cuda.is_available() else False
    flux_options["single_host_artifact"] = bool(flux_options.get("direct_safetensors_load", False))
    flux_options["streaming_scheduler"] = streaming_scheduler
    flux_options["streaming_scheduler_kind"] = "flux_async_layer_prefetch_v1"
    flux_options["scheduled_module_count"] = int(scheduled_module_count)
    flux_options["direct_stream_runtime"] = True
    flux_options["compute_device"] = str(compute_device)
    flux_options["host_load_device"] = str(host_load_device)
    flux_options["host_offload_device"] = str(host_offload_device)
    flux_options["streaming_scheduler_policy"] = dict(scheduler_policy)
    return runtime_patcher


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


def _pin_module_tensors_for_streaming(module: Any) -> int:
    if module is None or not torch.cuda.is_available():
        return 0

    pinned_bytes = 0
    for _, submodule in module.named_modules():
        for _, param in submodule.named_parameters(recurse=False):
            if param is None or param.device.type != "cpu" or param.is_pinned():
                continue
            pinned = param.data.contiguous().pin_memory()
            param.data = pinned
            pinned_bytes += int(pinned.numel() * pinned.element_size())
        for name, buf in submodule.named_buffers(recurse=False):
            if buf is None or buf.device.type != "cpu" or buf.is_pinned():
                continue
            pinned = buf.contiguous().pin_memory()
            submodule._buffers[name] = pinned
            pinned_bytes += int(pinned.numel() * pinned.element_size())
    return pinned_bytes


def measure_pinned_module_tensors(module: Any) -> int:
    if module is None:
        return 0

    pinned_bytes = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.is_pinned():
            pinned_bytes += int(tensor.numel() * tensor.element_size())
    return pinned_bytes


class FluxAsyncLayerPrefetchScheduler:
    def __init__(
        self,
        *,
        prefetch_depth: int = 1,
        max_prefetch_bytes: int | None = None,
        vram_guard_bytes: int | None = None,
        vram_guard_margin_bytes: int = 0,
        prefetch_scan_ahead: int = 1,
        bandwidth_limit_mb_s: float | None = None,
    ) -> None:
        self.prefetch_depth = max(0, int(prefetch_depth))
        self.max_prefetch_bytes = int(max_prefetch_bytes) if max_prefetch_bytes is not None else None
        self.vram_guard_bytes = int(vram_guard_bytes) if vram_guard_bytes is not None else None
        self.vram_guard_margin_bytes = max(0, int(vram_guard_margin_bytes))
        self.prefetch_scan_ahead = max(1, int(prefetch_scan_ahead))
        self.bandwidth_limit_mb_s = (
            float(bandwidth_limit_mb_s)
            if bandwidth_limit_mb_s is not None and float(bandwidth_limit_mb_s) > 0.0
            else None
        )
        self._hooks: list[Any] = []
        self._ordered_modules: list[Any] = []
        self._module_indices: dict[int, int] = {}
        self._module_names: dict[int, str] = {}
        self._prefetched: dict[tuple[int, str, str, str], tuple[torch.Tensor, torch.Tensor | None, Any, dict[str, Any] | None]] = {}
        self._streams: dict[str, list[Any]] = {}
        self._stats: dict[str, Any] = {}
        self.reset_run(clear_prefetched=True)

    def attach(self, model: Any) -> int:
        self.detach()
        diffusion_model = getattr(model, "diffusion_model", model)
        ordered: list[Any] = []
        module_names: dict[int, str] = {}
        for name, module in diffusion_model.named_modules():
            if module is diffusion_model:
                continue
            if not hasattr(module, "comfy_cast_weights"):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
                continue
            ordered.append(module)
            module_names[id(module)] = str(name or module.__class__.__name__)

        self._ordered_modules = ordered
        self._module_indices = {id(module): index for index, module in enumerate(ordered)}
        self._module_names = module_names
        for index, module in enumerate(ordered):
            setattr(module, "_nex_streaming_scheduler", self)
            setattr(module, "_nex_streaming_scheduler_index", index)
            self._hooks.append(module.register_forward_pre_hook(self._build_prefetch_hook(index)))

        self._stats["module_count"] = len(ordered)
        return len(ordered)

    def detach(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
        for module in self._ordered_modules:
            try:
                if getattr(module, "_nex_streaming_scheduler", None) is self:
                    delattr(module, "_nex_streaming_scheduler")
            except Exception:
                pass
            try:
                if hasattr(module, "_nex_streaming_scheduler_index"):
                    delattr(module, "_nex_streaming_scheduler_index")
            except Exception:
                pass
        self._ordered_modules = []
        self._module_indices = {}
        self._module_names = {}
        self.reset_run(clear_prefetched=True)

    def reset_run(self, *, clear_prefetched: bool = True) -> None:
        if clear_prefetched:
            self._prefetched.clear()
        self._stats = {
            "enabled": bool(torch.cuda.is_available()),
            "prefetch_depth": int(self.prefetch_depth),
            "max_prefetch_bytes": int(self.max_prefetch_bytes or 0),
            "vram_guard_bytes": int(self.vram_guard_bytes or 0),
            "vram_guard_margin_bytes": int(self.vram_guard_margin_bytes),
            "prefetch_scan_ahead": int(self.prefetch_scan_ahead),
            "bandwidth_limit_mb_s": float(self.bandwidth_limit_mb_s or 0.0),
            "module_count": len(self._ordered_modules),
            "prefetch_enqueued": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_bytes": 0,
            "prefetch_copy_wall_s": 0.0,
            "prefetch_copy_cuda_ms": 0.0,
            "prefetch_throttle_cuda_ms": 0.0,
            "prefetch_throttle_events": 0,
            "prefetch_skipped_size": 0,
            "prefetch_skipped_vram": 0,
            "prefetch_scan_considered": 0,
            "direct_copy_bytes": 0,
            "direct_copy_cuda_ms": 0.0,
            "direct_throttle_cuda_ms": 0.0,
            "direct_throttle_events": 0,
            "direct_copy_calls": 0,
            "direct_copy_stream_uses": 0,
            "bandwidth_throttle_cuda_ms": 0.0,
            "bandwidth_throttle_events": 0,
            "stream_waits": 0,
            "sync_calls": 0,
            "module_profiles": {},
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._stats)
        snapshot["streams"] = {device: len(streams) for device, streams in self._streams.items()}
        snapshot["prefetched_entries"] = len(self._prefetched)
        profiles = list(self._stats.get("module_profiles", {}).values())
        snapshot["top_direct_modules"] = sorted(
            profiles,
            key=lambda item: float(item.get("direct_wall_s", 0.0)),
            reverse=True,
        )[:12]
        snapshot["top_prefetch_modules"] = sorted(
            profiles,
            key=lambda item: int(item.get("prefetch_hits", 0)),
            reverse=True,
        )[:12]
        snapshot["top_vram_skipped_modules"] = sorted(
            profiles,
            key=lambda item: int(item.get("prefetch_skipped_vram", 0)),
            reverse=True,
        )[:12]
        return snapshot

    def is_enabled_for(self, device: torch.device | None) -> bool:
        return bool(
            torch.cuda.is_available()
            and device is not None
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and self._ordered_modules
        )

    def _device_key(self, device: torch.device) -> str:
        return str(device)

    def _stream_for_index(self, device: torch.device, module_index: int):
        if not self.is_enabled_for(device):
            return None
        device_key = self._device_key(device)
        streams = self._streams.get(device_key)
        if streams is None:
            streams = [torch.cuda.Stream(device=device, priority=0), torch.cuda.Stream(device=device, priority=0)]
            self._streams[device_key] = streams
        return streams[int(module_index) % len(streams)]

    def stream_for_module(self, module: Any, *, device: torch.device):
        module_index = int(self._module_indices.get(id(module), 0))
        stream = self._stream_for_index(device, module_index)
        if stream is not None:
            self._stats["direct_copy_stream_uses"] += 1
        return stream

    def sync_stream(self, device: torch.device, stream: Any) -> None:
        if stream is None or not self.is_enabled_for(device):
            return
        torch.cuda.current_stream(device).wait_stream(stream)
        self._stats["sync_calls"] += 1
        self._stats["stream_waits"] += 1

    def consume_prefetched(
        self,
        module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        key = (id(module), str(device), str(dtype), str(bias_dtype))
        entry = self._prefetched.pop(key, None)
        if entry is None:
            self._stats["prefetch_misses"] += 1
            return None
        weight, bias, stream, copy_token = entry
        self.sync_stream(device, stream)
        self._settle_timed_copy(copy_token)
        self._stats["prefetch_hits"] += 1
        profile = self._profile(module)
        profile["prefetch_hits"] += 1
        return weight, bias

    def _build_prefetch_hook(self, module_index: int):
        def _hook(module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                return
            input_tensor = args[0]
            if not isinstance(input_tensor, torch.Tensor):
                return
            device = input_tensor.device
            if not self.is_enabled_for(device):
                return
            selected_indices = self._select_prefetch_indices(module_index)
            enqueued = 0
            for next_index in selected_indices:
                next_module = self._ordered_modules[next_index]
                if self._prefetch_module(
                    next_module,
                    device=device,
                    dtype=input_tensor.dtype,
                    bias_dtype=input_tensor.dtype,
                    module_index=next_index,
                ):
                    enqueued += 1
                    if enqueued >= self.prefetch_depth:
                        break

        return _hook

    def _select_prefetch_indices(self, module_index: int) -> list[int]:
        if self.prefetch_depth <= 0:
            return []
        start_index = int(module_index) + 1
        if start_index >= len(self._ordered_modules):
            return []
        end_index = min(len(self._ordered_modules), start_index + int(self.prefetch_scan_ahead))
        candidates: list[tuple[tuple[int, int, int], int]] = []
        for next_index in range(start_index, end_index):
            self._stats["prefetch_scan_considered"] += 1
            module = self._ordered_modules[next_index]
            priority = self._module_prefetch_priority(module, next_index)
            candidates.append((priority, next_index))
        candidates.sort(reverse=True)
        return [index for _priority, index in candidates]

    def _prefetch_module(
        self,
        module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
        module_index: int,
    ) -> bool:
        if not self.is_enabled_for(device):
            return False
        key = (id(module), str(device), str(dtype), str(bias_dtype))
        if key in self._prefetched:
            return False

        from backend import resources

        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
            return False

        stream = self._stream_for_index(device, module_index)
        if stream is None:
            return False

        projected_bytes = self._project_prefetch_bytes(
            weight=weight,
            bias=getattr(module, "bias", None),
            dtype=dtype,
            bias_dtype=bias_dtype,
        )
        if self.max_prefetch_bytes is not None and projected_bytes > self.max_prefetch_bytes:
            self._stats["prefetch_skipped_size"] += 1
            self._profile(module)["prefetch_skipped_size"] += 1
            return False
        if self.vram_guard_bytes is not None and self._would_exceed_vram_guard(
            device=device,
            projected_bytes=projected_bytes,
        ):
            self._stats["prefetch_skipped_vram"] += 1
            self._profile(module)["prefetch_skipped_vram"] += 1
            return False

        non_blocking = bool(resources.device_supports_non_blocking(device))
        has_weight_function = len(getattr(module, "weight_function", []) or []) > 0
        copy_start = time.perf_counter()
        copy_token = self._begin_timed_copy(
            kind="prefetch",
            device=device,
            stream=stream,
            transfer_bytes=projected_bytes,
        )
        weight_copy = resources.cast_to(weight, dtype=dtype, device=device, non_blocking=non_blocking, copy=has_weight_function, stream=stream)
        if has_weight_function:
            with torch.cuda.stream(stream):
                for function in module.weight_function:
                    weight_copy = function(weight_copy)

        bias = getattr(module, "bias", None)
        bias_copy = None
        if isinstance(bias, torch.Tensor):
            has_bias_function = len(getattr(module, "bias_function", []) or []) > 0
            bias_copy = resources.cast_to(bias, dtype=bias_dtype, device=device, non_blocking=non_blocking, copy=has_bias_function, stream=stream)
            if has_bias_function:
                with torch.cuda.stream(stream):
                    for function in module.bias_function:
                        bias_copy = function(bias_copy)

        self._end_timed_copy(copy_token, stream=stream)
        self._prefetched[key] = (weight_copy, bias_copy, stream, copy_token)
        self._stats["prefetch_enqueued"] += 1
        self._stats["prefetch_bytes"] += int(projected_bytes)
        self._stats["prefetch_copy_wall_s"] += float(time.perf_counter() - copy_start)
        profile = self._profile(module)
        profile["prefetch_enqueued"] += 1
        profile["prefetch_bytes"] += int(projected_bytes)
        return True

    def _project_prefetch_bytes(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> int:
        total = int(weight.numel()) * int(torch.tensor([], dtype=dtype).element_size())
        if isinstance(bias, torch.Tensor):
            total += int(bias.numel()) * int(torch.tensor([], dtype=bias_dtype).element_size())
        return total

    def estimate_module_transfer_bytes(
        self,
        module: Any,
        *,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> int:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return 0
        return self._project_prefetch_bytes(
            weight=weight,
            bias=getattr(module, "bias", None),
            dtype=dtype,
            bias_dtype=bias_dtype,
        )

    def _target_copy_ms(self, transfer_bytes: int) -> float:
        if self.bandwidth_limit_mb_s is None or self.bandwidth_limit_mb_s <= 0.0:
            return 0.0
        if transfer_bytes <= 0:
            return 0.0
        return (float(transfer_bytes) / (float(self.bandwidth_limit_mb_s) * 1024.0 * 1024.0)) * 1000.0

    def _apply_bandwidth_throttle(self, *, device: torch.device, delay_ms: float) -> float:
        if delay_ms <= 0.0:
            return 0.0
        if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        try:
            clock_rate_khz = int(torch.cuda.get_device_properties(device).clock_rate)
        except Exception:
            return 0.0
        if clock_rate_khz <= 0:
            return 0.0
        cycles = int(max(1, round(float(delay_ms) * float(clock_rate_khz))))
        try:
            with torch.cuda.device(device):
                torch.cuda._sleep(cycles)
            return float(delay_ms)
        except Exception:
            return 0.0

    def _module_prefetch_priority(self, module: Any, module_index: int) -> tuple[int, int, int]:
        label = self._module_label(module)
        score = 0
        if "txt_attn.qkv" in label:
            score += 9
        if "txt_mod.lin" in label:
            score += 8
        if "single_blocks." in label and ".linear2" in label:
            score += 7
        if "single_blocks." in label and ".modulation.lin" in label:
            score += 5
        if "img_attn.qkv" in label:
            score += 4
        if ".linear2" in label:
            score += 3
        if ".img_mlp.0" in label or ".txt_mlp.0" in label:
            score += 2
        if ".proj" in label:
            score += 1

        # Bias toward later modules so early blocks do not consume all available overlap budget.
        later_bias = int(module_index)
        transfer_bytes = self.estimate_module_transfer_bytes(
            module,
            dtype=torch.float16,
            bias_dtype=torch.float16,
        )
        size_bias = int(transfer_bytes // (16 * 1024 * 1024))
        return (score, later_bias, size_bias)

    def record_module_wall(
        self,
        module: Any,
        *,
        path: str,
        wall_s: float,
    ) -> None:
        profile = self._profile(module)
        if path == "prefetch":
            profile["prefetch_consumes"] += 1
            profile["prefetch_wall_s"] += float(max(0.0, wall_s))
        elif path == "direct":
            profile["direct_calls"] += 1
            profile["direct_wall_s"] += float(max(0.0, wall_s))

    def begin_direct_copy(
        self,
        module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
        stream: Any,
    ) -> dict[str, Any] | None:
        transfer_bytes = self.estimate_module_transfer_bytes(
            module,
            dtype=dtype,
            bias_dtype=bias_dtype,
        )
        if transfer_bytes <= 0:
            return None
        return self._begin_timed_copy(
            kind="direct",
            device=device,
            stream=stream,
            transfer_bytes=transfer_bytes,
        )

    def end_direct_copy(self, token: dict[str, Any] | None, *, stream: Any) -> None:
        self._end_timed_copy(token, stream=stream)

    def settle_direct_copy(self, token: dict[str, Any] | None) -> None:
        self._settle_timed_copy(token)

    def _begin_timed_copy(
        self,
        *,
        kind: str,
        device: torch.device,
        stream: Any,
        transfer_bytes: int,
    ) -> dict[str, Any] | None:
        token: dict[str, Any] = {
            "kind": str(kind),
            "bytes": int(max(0, transfer_bytes)),
            "device": device,
            "start_event": None,
            "end_event": None,
        }
        if (
            stream is not None
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            token["start_event"] = start_event
            token["end_event"] = end_event
        return token

    def _end_timed_copy(self, token: dict[str, Any] | None, *, stream: Any) -> None:
        if not token or stream is None:
            return
        end_event = token.get("end_event")
        if end_event is not None:
            end_event.record(stream)

    def _settle_timed_copy(self, token: dict[str, Any] | None) -> None:
        if not token:
            return
        kind = str(token.get("kind", ""))
        transfer_bytes = int(token.get("bytes", 0))
        device = token.get("device")
        elapsed_ms = 0.0
        start_event = token.get("start_event")
        end_event = token.get("end_event")
        if start_event is not None and end_event is not None:
            try:
                end_event.synchronize()
            except Exception:
                pass
            try:
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception:
                elapsed_ms = 0.0
        target_ms = self._target_copy_ms(transfer_bytes)
        throttle_ms = max(0.0, float(target_ms) - float(elapsed_ms))
        applied_throttle_ms = 0.0
        if isinstance(device, torch.device):
            applied_throttle_ms = self._apply_bandwidth_throttle(device=device, delay_ms=throttle_ms)
        if kind == "prefetch":
            self._stats["prefetch_copy_cuda_ms"] += elapsed_ms
            self._stats["prefetch_throttle_cuda_ms"] += applied_throttle_ms
            if applied_throttle_ms > 0.0:
                self._stats["prefetch_throttle_events"] += 1
        elif kind == "direct":
            self._stats["direct_copy_calls"] += 1
            self._stats["direct_copy_bytes"] += transfer_bytes
            self._stats["direct_copy_cuda_ms"] += elapsed_ms
            self._stats["direct_throttle_cuda_ms"] += applied_throttle_ms
            if applied_throttle_ms > 0.0:
                self._stats["direct_throttle_events"] += 1
        if applied_throttle_ms > 0.0:
            self._stats["bandwidth_throttle_cuda_ms"] += applied_throttle_ms
            self._stats["bandwidth_throttle_events"] += 1

    def _module_label(self, module: Any) -> str:
        module_id = id(module)
        name = self._module_names.get(module_id)
        index = int(self._module_indices.get(module_id, -1))
        if name:
            return f"{index}:{name}"
        return f"{index}:{module.__class__.__name__}"

    def _profile(self, module: Any) -> dict[str, Any]:
        profiles = self._stats.setdefault("module_profiles", {})
        module_id = id(module)
        profile = profiles.get(module_id)
        if profile is None:
            profile = {
                "label": self._module_label(module),
                "index": int(self._module_indices.get(module_id, -1)),
                "class_name": module.__class__.__name__,
                "prefetch_enqueued": 0,
                "prefetch_hits": 0,
                "prefetch_consumes": 0,
                "prefetch_bytes": 0,
                "prefetch_wall_s": 0.0,
                "prefetch_skipped_size": 0,
                "prefetch_skipped_vram": 0,
                "direct_calls": 0,
                "direct_wall_s": 0.0,
            }
            profiles[module_id] = profile
        return profile

    def _would_exceed_vram_guard(self, *, device: torch.device, projected_bytes: int) -> bool:
        if self.vram_guard_bytes is None:
            return False
        try:
            current_allocated = int(torch.cuda.memory_allocated(device))
            projected_usage = current_allocated + int(projected_bytes) + int(self.vram_guard_margin_bytes)
            if projected_usage > int(self.vram_guard_bytes):
                return True
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            required_free_bytes = int(projected_bytes) + int(self.vram_guard_margin_bytes)
            return int(free_bytes) < required_free_bytes
        except Exception:
            return False


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


def _sample_flux_fill_direct_streaming(
    *,
    unet_patcher: Any,
    noise: torch.Tensor,
    payloads: FluxFillConditioningPayloads,
    steps: int,
    device: torch.device,
    sampler_name: str,
    scheduler_name: str,
    denoise: float,
    cfg: float,
    seed: int,
    callback: Any | None,
    disable_pbar: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from backend import sampling

    model_options = getattr(unet_patcher, "model_options", {})
    sampler_instance = sampling.KSampler(
        unet_patcher,
        steps,
        device,
        sampler_name,
        scheduler_name,
        denoise,
        model_options=model_options,
    )
    guider = sampling.prepare_sampler_conds(
        unet_patcher,
        noise,
        payloads.positive,
        payloads.negative,
        cfg,
        sampler_name=sampler_name,
        latent_image=payloads.latent_image,
        denoise_mask=payloads.denoise_mask,
        seed=seed,
        model_options=model_options,
        quality=getattr(sampler_instance, "quality", {}),
        inner_model=getattr(unet_patcher, "model", None),
    )
    samples = sampling.sample_prepared_sdxl(
        guider,
        noise,
        sampler_instance.sigmas,
        sampler=sampling.ksampler(sampler_name),
        latent_image=payloads.latent_image,
        denoise_mask=payloads.denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        attach_model=False,
    )
    return samples, sampler_instance.sigmas.detach().cpu()

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
        if streaming_scheduler is not None and hasattr(streaming_scheduler, "snapshot"):
            metadata["streaming_scheduler"] = streaming_scheduler.snapshot()
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
        if streaming_scheduler is not None and hasattr(streaming_scheduler, "snapshot"):
            metadata["streaming_scheduler"] = streaming_scheduler.snapshot()
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

    latent_source = prepare_flux_fill_latent_source(
        image,
        mask,
        config.ae_path,
        extend_factor=extend_factor,
        load_device=config.device,
        offload_device=None,
        vae=vae,
        cleanup_vae=cleanup_vae,
    )
    denoise_result = denoise_flux_fill_latent(
        config,
        latent_source,
        empty_conditioning=cache,
        load_device=config.device,
        offload_device=None,
        unet_patcher=unet_patcher,
        cleanup_unet=cleanup_unet,
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
        cleanup_vae=cleanup_vae,
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









