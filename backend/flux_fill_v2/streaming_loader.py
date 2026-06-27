from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from backend import resources
from backend.flux_fill_v2.patcher import FluxDirectStreamModelPatcher
from backend.flux_fill_v2.scheduler import FluxAsyncLayerPrefetchScheduler
from backend.flux_fill_v2.unet_contract import (
    FluxFillValidationError,
    _instantiate_flux_fill_native_model,
    _load_flux_fill_native_weights_into_model,
    validate_flux_fill_unet_config,
)

logger = logging.getLogger(__name__)


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(int(minimum), min(int(maximum), int(value)))


FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1 = "open_c64_d1_s1"
FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1 = "open_c128_d1_s1"
FLUX_FILL_SANCTIONED_STREAMING_PROFILES: dict[str, dict[str, int]] = {
    FLUX_FILL_STREAMING_PROFILE_OPEN_C64_D1_S1: {
        "prefetch_depth": 1,
        "max_prefetch_bytes": 64 * 1024 * 1024,
        "prefetch_scan_ahead": 1,
    },
    FLUX_FILL_STREAMING_PROFILE_OPEN_C128_D1_S1: {
        "prefetch_depth": 1,
        "max_prefetch_bytes": 128 * 1024 * 1024,
        "prefetch_scan_ahead": 1,
    },
}


def resolve_flux_fill_sanctioned_streaming_profile(profile_name: str | None) -> tuple[str | None, dict[str, int]]:
    normalized = str(profile_name or "").strip().lower().replace("-", "_")
    if normalized == "":
        return None, {}
    try:
        return normalized, dict(FLUX_FILL_SANCTIONED_STREAMING_PROFILES[normalized])
    except KeyError as exc:
        supported = ", ".join(sorted(FLUX_FILL_SANCTIONED_STREAMING_PROFILES))
        raise FluxFillValidationError(
            f"Unsupported Flux Fill streaming profile {profile_name!r}. Expected one of: {supported}."
        ) from exc


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


def load_flux_fill_unet_streaming(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
    execution_class: Any | None = None,
    runtime_family: str | None = None,
    streaming_profile: str | None = None,
    prefetch_depth: int | None = None,
    prefetch_chunk_mb: int | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    selected_runtime_family = str(runtime_family or "").strip().lower().replace("-", "_")
    if selected_runtime_family == "":
        selected_runtime_family = "native_fp8" if path.suffix.lower() == ".safetensors" else "gguf"

    if selected_runtime_family == "native_fp8" or path.suffix.lower() == ".safetensors":
        return load_flux_fill_native_unet_streaming(
            path,
            load_device=load_device,
            offload_device=offload_device,
            execution_class=execution_class,
            streaming_profile=streaming_profile,
            prefetch_depth=prefetch_depth,
            max_prefetch_bytes=prefetch_chunk_mb * 1024 * 1024 if prefetch_chunk_mb is not None else None,
        )

    from backend.gguf.loader import gguf_sd_loader
    from backend.gguf.ops import GGMLOps
    from backend.gguf.patcher import GGUFModelPatcher
    from ldm_patched.modules import model_detection

    host_load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    host_offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    if host_load_device.type != "cpu" or host_offload_device.type != "cpu":
        raise RuntimeError("Streaming-class Flux GGUF loads must stage weights on CPU pinned host memory.")

    state_dict, arch = gguf_sd_loader(
        str(path),
        handle_prefix=handle_prefix,
        return_arch=True,
        pin_memory=True,
        execution_class=execution_class,
        require_pinned_host=True,
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
        device=host_offload_device,
        model_options={"custom_operations": GGMLOps},
    )
    model.load_model_weights(state_dict, "")
    patcher = GGUFModelPatcher(
        model,
        load_device=host_load_device,
        offload_device=host_offload_device,
        preserve_source_artifact=True,
    )
    patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": arch,
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "runtime_family": selected_runtime_family,
        "runtime_posture": "streaming",
        "streaming_profile": streaming_profile,
        "prefetch_depth": prefetch_depth,
        "prefetch_chunk_mb": prefetch_chunk_mb,
    }
    return patcher


def load_flux_fill_native_unet_streaming(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
    streaming_profile: str | None = None,
    prefetch_depth: int | None = None,
    max_prefetch_bytes: int | None = None,
    vram_guard_bytes: int | None = None,
    vram_guard_margin_bytes: int | None = None,
    prefetch_scan_ahead: int | None = None,
    bandwidth_limit_mb_s: float | None = None,
) -> Any:
    host_load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    host_offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    if host_load_device.type != "cpu" or host_offload_device.type != "cpu":
        raise RuntimeError("Native Flux Fill streaming loads must stage weights on CPU host memory.")

    compute_device = resources.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
    path = Path(unet_path)
    construction_device = host_offload_device
    if torch.cuda.is_available() and str(path).lower().endswith(".safetensors"):
        construction_device = torch.device("meta")
    model, detected_config = _instantiate_flux_fill_native_model(
        path,
        offload_device=host_offload_device,
        construction_device=construction_device,
    )
    prepared_pinned_bytes = _prepare_module_tensors_for_pinned_load(getattr(model, "diffusion_model", model))
    if prepared_pinned_bytes > 0:
        logger.debug(
            "[Flux Telemetry] Prepared pinned-host UNet targets before direct load bytes=%s path=%s",
            prepared_pinned_bytes,
            path,
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
    runtime_weight_dtype = detected_config.get("resident_weight_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8_streaming",
        "runtime_family": "native_fp8",
        "runtime_posture": "streaming",
        "runtime_weight_dtype": runtime_weight_dtype,
        "compute_weight_dtype": detected_config.get("manual_cast_dtype"),
        **direct_load_metadata,
    }
    pinned_bytes = measure_pinned_module_tensors(getattr(runtime_patcher, "model", None))
    if pinned_bytes <= 0:
        pinned_bytes = _pin_module_tensors_for_streaming(getattr(runtime_patcher, "model", None))
    selected_profile_name, sanctioned_profile = resolve_flux_fill_sanctioned_streaming_profile(streaming_profile)
    scheduler_policy = _resolve_streaming_scheduler_policy(
        device=compute_device,
        prefetch_depth=prefetch_depth if prefetch_depth is not None else sanctioned_profile.get("prefetch_depth"),
        max_prefetch_bytes=max_prefetch_bytes if max_prefetch_bytes is not None else sanctioned_profile.get("max_prefetch_bytes"),
        vram_guard_bytes=vram_guard_bytes,
        vram_guard_margin_bytes=vram_guard_margin_bytes,
        prefetch_scan_ahead=prefetch_scan_ahead if prefetch_scan_ahead is not None else sanctioned_profile.get("prefetch_scan_ahead"),
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
    scheduled_module_count = streaming_scheduler.attach(getattr(runtime_patcher, "model", None), device=compute_device)
    flux_options = runtime_patcher.model_options.setdefault("flux_fill", {})
    measured_pinned_bytes = measure_pinned_module_tensors(getattr(runtime_patcher, "model", None))
    flux_options["host_pinned_bytes"] = int(max(prepared_pinned_bytes, pinned_bytes, measured_pinned_bytes))
    flux_options["non_blocking_supported"] = bool(resources.device_supports_non_blocking(torch.device("cuda"))) if torch.cuda.is_available() else False
    flux_options["single_host_artifact"] = bool(flux_options.get("direct_safetensors_load", False))
    flux_options["streaming_scheduler"] = streaming_scheduler
    flux_options["streaming_scheduler_kind"] = "flux_async_layer_prefetch_v1"
    flux_options["streaming_profile"] = selected_profile_name
    flux_options["scheduled_module_count"] = int(scheduled_module_count)
    flux_options["direct_stream_runtime"] = True
    flux_options["compute_device"] = str(compute_device)
    flux_options["host_load_device"] = str(host_load_device)
    flux_options["host_offload_device"] = str(host_offload_device)
    flux_options["streaming_scheduler_policy"] = dict(scheduler_policy)
    return runtime_patcher


def _prepare_module_tensors_for_pinned_load(module: Any) -> int:
    if module is None or not torch.cuda.is_available():
        return 0

    try:
        state_keys = set(module.state_dict().keys())
    except Exception:
        state_keys = set()

    prepared_bytes = 0
    for module_name, submodule in module.named_modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            full_key = f"{module_name}.{param_name}" if module_name else param_name
            if state_keys and full_key not in state_keys:
                continue
            if param is None:
                continue
            param_device = getattr(param, "device", None)
            param_device_type = getattr(param_device, "type", None)
            if param_device_type not in {"cpu", "meta"}:
                continue
            if param_device_type == "cpu" and param.is_pinned():
                continue
            try:
                pinned_target = torch.empty_like(param.data, device="cpu", pin_memory=True)
            except Exception as exc:
                logger.debug(
                    "[Flux Telemetry] Failed to preallocate pinned parameter target key=%s error=%s",
                    full_key,
                    exc,
                )
                continue
            submodule._parameters[param_name] = torch.nn.Parameter(
                pinned_target,
                requires_grad=bool(getattr(param, "requires_grad", False)),
            )
            prepared_bytes += int(pinned_target.numel() * pinned_target.element_size())

        for buffer_name, buf in submodule.named_buffers(recurse=False):
            full_key = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if state_keys and full_key not in state_keys:
                continue
            if buf is None:
                continue
            buffer_device = getattr(buf, "device", None)
            buffer_device_type = getattr(buffer_device, "type", None)
            if buffer_device_type not in {"cpu", "meta"}:
                continue
            if buffer_device_type == "cpu" and buf.is_pinned():
                continue
            try:
                pinned_target = torch.empty_like(buf, device="cpu", pin_memory=True)
            except Exception as exc:
                logger.debug(
                    "[Flux Telemetry] Failed to preallocate pinned buffer target key=%s error=%s",
                    full_key,
                    exc,
                )
                continue
            submodule._buffers[buffer_name] = pinned_target
            prepared_bytes += int(pinned_target.numel() * pinned_target.element_size())

    return prepared_bytes


def _sample_flux_fill_direct_streaming(
    *,
    unet_patcher: Any,
    noise: torch.Tensor,
    positive: Any,
    negative: Any,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor,
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
        positive,
        negative,
        cfg,
        sampler_name=sampler_name,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
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
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        attach_model=False,
    )
    return samples, sampler_instance.sigmas.detach().cpu()


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
