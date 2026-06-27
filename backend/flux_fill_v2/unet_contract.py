from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from backend import resources


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


_FLUX_STREAMING_SOURCE_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class FluxFillUNetInfo:
    path: Path
    arch: str
    detected_config: dict[str, Any]
    key_shapes: dict[str, list[int] | None]
    qtypes: dict[str, int]
    tensor_count: int


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


def _parse_torch_dtype(value: Any) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return None
    return _normalize_flux_checkpoint_dtype(value)


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


def validate_flux_fill_unet_config(detected_config: dict[str, Any]) -> None:
    mismatches = []
    for key, expected in EXPECTED_FLUX_FILL_CONTRACT.items():
        actual = detected_config.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise FluxFillUnsupportedModelError(
            "Unsupported Flux model for greenfield Flux Fill runtime: " + "; ".join(mismatches)
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


def _instantiate_flux_fill_native_model(
    path: Path,
    *,
    offload_device: torch.device,
    construction_device: torch.device | None = None,
) -> tuple[Any, dict[str, Any]]:
    from backend import precision
    from ldm_patched.modules import model_detection
    from ldm_patched.modules import supported_models_base

    state_probe = _load_flux_fill_native_probe(str(path))
    source_weight_dtype = _detect_flux_fill_weight_dtype(state_probe)
    parameter_count = _estimate_flux_fill_parameter_count(state_probe)
    inference_device = resources.get_torch_device()
    resident_weight_dtype = source_weight_dtype
    if source_weight_dtype is not None:
        resident_weight_dtype = precision.unet_dtype(
            device=inference_device,
            model_params=parameter_count,
            weight_dtype=source_weight_dtype,
        )
    detected_config = model_detection.detect_unet_config(state_probe, "", dtype=resident_weight_dtype)
    if detected_config is None:
        raise FluxFillValidationError(f"Could not detect Flux model config from {path}.")
    validate_flux_fill_unet_config(detected_config)

    model_config = model_detection.model_config_from_unet_config(detected_config, state_probe)
    if model_config is None:
        raise FluxFillValidationError(f"No supported Flux model config matched {path}.")

    manual_cast_dtype = None
    if resident_weight_dtype is not None:
        manual_cast_dtype = precision.unet_manual_cast(resident_weight_dtype, inference_device)
        model_config.set_manual_cast(manual_cast_dtype)
    elif isinstance(model_config, supported_models_base.BASE):
        model_config.set_manual_cast(torch.float16 if inference_device.type == "cuda" else torch.float32)

    model = model_config.get_model(
        state_probe,
        "",
        device=construction_device if construction_device is not None else offload_device,
        model_options={"custom_operations": None},
    )
    detected_metadata = dict(detected_config)
    detected_metadata["weight_dtype"] = str(source_weight_dtype) if source_weight_dtype is not None else None
    detected_metadata["source_weight_dtype"] = str(source_weight_dtype) if source_weight_dtype is not None else None
    detected_metadata["resident_weight_dtype"] = str(resident_weight_dtype) if resident_weight_dtype is not None else None
    detected_metadata["manual_cast_dtype"] = str(manual_cast_dtype) if manual_cast_dtype is not None else None
    detected_metadata["parameter_count"] = parameter_count
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
        load_metrics: dict[str, Any] = {}
        missing, unexpected = loader._load_prefixed_safetensors_into_module(
            str(path),
            [""],
            model.diffusion_model,
            device=target_device,
            dtype=None,
            chunk_bytes=_FLUX_STREAMING_SOURCE_CHUNK_BYTES,
            realize_pinned_targets=bool(torch.cuda.is_available() and target_device.type == "cpu"),
            load_metrics=load_metrics,
            raw_byte_stream=True,
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
        flux_metadata["realized_pinned_bytes"] = int(load_metrics.get("realized_pinned_bytes", 0))
        flux_metadata["realized_pinned_tensor_count"] = int(load_metrics.get("realized_pinned_tensor_count", 0))
        flux_metadata["progressive_pinned_realization"] = bool(flux_metadata["realized_pinned_tensor_count"] > 0)
        flux_metadata["raw_sequential_stream"] = True
        return flux_metadata

    from backend import loader as backend_loader

    state_dict = backend_loader.resolve_source(str(path))
    if not isinstance(state_dict, dict):
        raise FluxFillValidationError(
            f"Expected state dict while loading Flux Fill native UNet {path}, got {type(state_dict).__name__}."
        )
    model.load_model_weights(state_dict, "")
    return flux_metadata
