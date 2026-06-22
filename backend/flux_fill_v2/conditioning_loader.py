from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from backend.flux_fill_v2.unet_contract import FluxFillValidationError, _validate_tensor_shape


EMPTY_FLUX_CROSS_ATTN_SHAPE = (1, 256, 4096)
EMPTY_FLUX_POOLED_SHAPE = (1, 768)


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
