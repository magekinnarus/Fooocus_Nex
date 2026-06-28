from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backend import patching as backend_patching
from backend import resources
from backend.flux_fill_v3.unet_contract import (
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
        selected_runtime_family = "native_fp8"

    return load_flux_fill_native_unet(
        path,
        load_device=load_device,
        offload_device=offload_device,
        execution_class=execution_class,
        resident_load_strategy=resident_load_strategy,
    )


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
    if str(resident_load_strategy or "").strip().lower().replace("-", "_") == "sticky_no_cpu_shadow":
        if load_device.type != "cuda":
            load_device = resources.get_torch_device()
        if load_device.type != "cuda":
            raise RuntimeError("Flux Fill sticky resident fp8 path requires a CUDA load device.")
        offload_device = load_device

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
