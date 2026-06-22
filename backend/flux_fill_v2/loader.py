from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.gguf.loader import is_streaming_execution_class
from backend.flux_fill_v2.conditioning_loader import FluxEmptyConditioning, load_flux_empty_conditioning_cache
from backend.flux_fill_v2.patcher import FluxDirectStreamModelPatcher
from backend.flux_fill_v2.resident_loader import load_flux_fill_native_unet, load_flux_fill_unet_resident
from backend.flux_fill_v2.scheduler import FluxAsyncLayerPrefetchScheduler
from backend.flux_fill_v2.streaming_loader import (
    _sample_flux_fill_direct_streaming,
    load_flux_fill_native_unet_streaming,
    load_flux_fill_unet_streaming,
    measure_pinned_module_tensors,
    resolve_flux_fill_sanctioned_streaming_profile,
)
from backend.flux_fill_v2.unet_contract import (
    EXPECTED_FLUX_FILL_CONTRACT,
    FluxFillUNetInfo,
    FluxFillUnsupportedModelError,
    FluxFillValidationError,
    IMPORTANT_FLUX_GGUF_KEYS,
    IMPORTANT_FLUX_NATIVE_DTYPE_KEYS,
    inspect_flux_fill_gguf,
    inspect_flux_fill_native_unet,
    validate_flux_fill_unet_config,
)
from backend.flux_fill_v2.vae_loader import load_flux_ae


def load_flux_fill_unet(
    unet_path: Path | str,
    *,
    load_device: Any | None = None,
    offload_device: Any | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
    execution_class: Any | None = None,
    runtime_family: str | None = None,
    runtime_posture: str | None = None,
    streaming_profile: str | None = None,
    prefetch_depth: int | None = None,
    prefetch_chunk_mb: int | None = None,
    resident_load_strategy: str | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Fill UNet path does not exist: {path}")

    selected_runtime_family = str(runtime_family or "").strip().lower().replace("-", "_")
    selected_runtime_posture = str(runtime_posture or "").strip().lower().replace("-", "_")
    if selected_runtime_posture == "":
        selected_runtime_posture = "streaming" if is_streaming_execution_class(execution_class) else "resident"
    if selected_runtime_family == "":
        selected_runtime_family = "native_fp8" if path.suffix.lower() == ".safetensors" else "gguf"

    if selected_runtime_posture == "streaming":
        return load_flux_fill_unet_streaming(
            path,
            load_device=load_device,
            offload_device=offload_device,
            handle_prefix=handle_prefix,
            execution_class=execution_class,
            runtime_family=selected_runtime_family,
            streaming_profile=streaming_profile,
            prefetch_depth=prefetch_depth,
            prefetch_chunk_mb=prefetch_chunk_mb,
        )

    return load_flux_fill_unet_resident(
        path,
        load_device=load_device,
        offload_device=offload_device,
        handle_prefix=handle_prefix,
        execution_class=execution_class,
        runtime_family=selected_runtime_family,
        resident_load_strategy=resident_load_strategy,
    )


__all__ = [
    "EXPECTED_FLUX_FILL_CONTRACT",
    "IMPORTANT_FLUX_GGUF_KEYS",
    "IMPORTANT_FLUX_NATIVE_DTYPE_KEYS",
    "FluxFillValidationError",
    "FluxFillUnsupportedModelError",
    "FluxFillUNetInfo",
    "FluxEmptyConditioning",
    "FluxDirectStreamModelPatcher",
    "FluxAsyncLayerPrefetchScheduler",
    "validate_flux_fill_unet_config",
    "inspect_flux_fill_gguf",
    "inspect_flux_fill_native_unet",
    "resolve_flux_fill_sanctioned_streaming_profile",
    "measure_pinned_module_tensors",
    "load_flux_fill_unet",
    "load_flux_fill_unet_streaming",
    "load_flux_fill_unet_resident",
    "load_flux_fill_native_unet_streaming",
    "load_flux_fill_native_unet",
    "load_flux_ae",
    "load_flux_empty_conditioning_cache",
    "_sample_flux_fill_direct_streaming",
]
