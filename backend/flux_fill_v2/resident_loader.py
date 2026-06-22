from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backend import patching as backend_patching
from backend import resources
from backend.flux_fill_v2.unet_contract import (
    FluxFillValidationError,
    _instantiate_flux_fill_native_model,
    _load_flux_fill_native_weights_into_model,
    _parse_torch_dtype,
    _snapshot_module_runtime,
    validate_flux_fill_unet_config,
)


def load_flux_fill_unet_resident(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
    execution_class: Any | None = None,
    runtime_family: str | None = None,
    resident_load_strategy: str | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    selected_runtime_family = str(runtime_family or "").strip().lower().replace("-", "_")
    if selected_runtime_family == "":
        selected_runtime_family = "native_fp8" if path.suffix.lower() == ".safetensors" else "gguf"

    if selected_runtime_family == "native_fp8" or path.suffix.lower() == ".safetensors":
        return load_flux_fill_native_unet(
            path,
            load_device=load_device,
            offload_device=offload_device,
            execution_class=execution_class,
            resident_load_strategy=resident_load_strategy,
        )

    from backend.gguf.loader import gguf_sd_loader
    from backend.gguf.ops import GGMLOps
    from backend.gguf.patcher import GGUFModelPatcher
    from ldm_patched.modules import model_detection

    resident_load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    resident_offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    if str(resident_load_strategy or "").strip().lower().replace("-", "_") == "sticky_no_cpu_shadow":
        if resident_load_device.type != "cuda":
            resident_load_device = resources.get_torch_device()
        if resident_load_device.type != "cuda":
            raise RuntimeError("Flux Fill sticky resident fp8 path requires a CUDA load device.")
        resident_offload_device = resident_load_device

    state_dict, arch = gguf_sd_loader(
        str(path),
        handle_prefix=handle_prefix,
        return_arch=True,
        pin_memory=False,
        execution_class=execution_class,
        require_pinned_host=False,
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
        device=resident_offload_device,
        model_options={"custom_operations": GGMLOps},
    )
    model.load_model_weights(state_dict, "")
    patcher = GGUFModelPatcher(
        model,
        load_device=resident_load_device,
        offload_device=resident_offload_device,
        preserve_source_artifact=False,
    )
    patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": arch,
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "runtime_family": selected_runtime_family,
        "runtime_posture": "resident",
        "resident_load_strategy": resident_load_strategy,
    }
    return patcher


def load_flux_fill_native_unet(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
    resident_load_strategy: str | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

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
    runtime_patcher = backend_patching.NexModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        preserve_source_artifact=False,
        runtime_weight_dtype_override=_parse_torch_dtype(detected_config.get("resident_weight_dtype")),
    )
    runtime_weight_bytes = int(runtime_patcher.model_size())
    runtime_weight_dtype = detected_config.get("resident_weight_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_fill"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8",
        "runtime_family": "native_fp8",
        "runtime_posture": "resident",
        "resident_load_strategy": resident_load_strategy,
        "runtime_weight_dtype": runtime_weight_dtype,
        "runtime_weight_bytes": runtime_weight_bytes,
        "compute_weight_dtype": detected_config.get("manual_cast_dtype"),
        **direct_load_metadata,
    }
    return runtime_patcher
