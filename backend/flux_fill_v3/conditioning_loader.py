from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import psutil
import torch

from backend import resources
from backend.flux_fill_v3.unet_contract import FluxFillValidationError, _validate_tensor_shape

logger = logging.getLogger(__name__)

EMPTY_FLUX_CROSS_ATTN_SHAPE = (1, 256, 4096)
EMPTY_FLUX_POOLED_SHAPE = (1, 768)


def _append_process_memory_summary(parts: list[str]) -> None:
    try:
        process = psutil.Process()
        process_info = process.memory_info()
        process_rss_mb = float(process_info.rss) / (1024 * 1024)
        parts.append(f"proc_rss={process_rss_mb:.1f}MB")

        full_info = process.memory_full_info()
        for label, attribute in (
            ("proc_shared", "shared"),
            ("proc_uss", "uss"),
            ("proc_pss", "pss"),
        ):
            value = getattr(full_info, attribute, None)
            if value is not None:
                value_mb = float(value) / (1024 * 1024)
                parts.append(f"{label}={value_mb:.1f}MB")
    except Exception:
        pass


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


def format_flux_conditioning_memory_summary(*, tag: str | None = None) -> str:
    parts: list[str] = []
    if tag:
        parts.append(f"tag={tag}")
    try:
        snapshot = resources.capture_memory_snapshot(notes={"tag": tag or "flux_conditioning"})
        parts.append(f"phase={snapshot.phase}")
        if snapshot.total_ram_mb is not None:
            parts.append(f"ram_total={snapshot.total_ram_mb:.1f}MB")
        if snapshot.free_ram_mb is not None:
            parts.append(f"ram_available={snapshot.free_ram_mb:.1f}MB")
        if snapshot.total_ram_mb is not None and snapshot.free_ram_mb is not None:
            parts.append(f"ram_unavailable_est={snapshot.total_ram_mb - snapshot.free_ram_mb:.1f}MB")
        if snapshot.total_vram_mb is not None:
            parts.append(f"vram_total={snapshot.total_vram_mb:.1f}MB")
        if snapshot.free_vram_mb is not None:
            parts.append(f"vram_free={snapshot.free_vram_mb:.1f}MB")
    except Exception:
        parts.append("memory_snapshot=unavailable")

    _append_process_memory_summary(parts)

    return " ".join(parts)


def load_flux_empty_conditioning_cache(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> FluxEmptyConditioning:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Flux empty-conditioning cache does not exist: {cache_path}")

    cache_size_mb = None
    try:
        cache_size_mb = float(cache_path.stat().st_size) / (1024 * 1024)
    except OSError:
        pass

    logger.debug(
        "[Flux Telemetry] Conditioning cache load begin path=%s size=%s %s",
        cache_path,
        "n/a" if cache_size_mb is None else f"{cache_size_mb:.3f}MB",
        format_flux_conditioning_memory_summary(tag="conditioning_cache_load_begin"),
    )

    try:
        payload = torch.load(cache_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=map_location)
    except Exception:
        logger.exception(
            "[Flux Telemetry] Conditioning cache load failed path=%s size=%s %s",
            cache_path,
            "n/a" if cache_size_mb is None else f"{cache_size_mb:.3f}MB",
            format_flux_conditioning_memory_summary(tag="conditioning_cache_load_failed"),
        )
        raise

    if not isinstance(payload, dict):
        raise FluxFillValidationError(
            f"Flux empty-conditioning cache must contain a dict, got {type(payload).__name__}."
        )

    missing = [key for key in ("cross_attn", "pooled_output") if key not in payload]
    if missing:
        raise FluxFillValidationError(
            f"Flux empty-conditioning cache is missing required key(s): {', '.join(missing)}."
        )

    metadata = payload.get("metadata", {})
    cross_attn = payload["cross_attn"]
    pooled_output = payload["pooled_output"]
    logger.debug(
        "[Flux Telemetry] Conditioning cache load complete path=%s posture=%s conditioning_kind=%s "
        "cross_attn_shape=%s pooled_shape=%s %s",
        cache_path,
        metadata.get("posture"),
        metadata.get("conditioning_kind"),
        tuple(getattr(cross_attn, "shape", ())),
        tuple(getattr(pooled_output, "shape", ())),
        format_flux_conditioning_memory_summary(tag="conditioning_cache_load_complete"),
    )

    return FluxEmptyConditioning(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        metadata=metadata,
    )
