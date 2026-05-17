"""Nex-owned production home for the unified SDXL runtime spine.

W07c1 uses this module to lock ownership. W07c2 should implement the CPU-first
artifact build entrypoints here. W07c3 should implement the stream-like
denoise and decode entrypoints here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from backend.sdxl_runtime_contract import (
    BaseModelAvailability,
    CompiledUnetArtifact,
    GpuAttachedExecutionState,
    InjectedFeatureArtifact,
    PromptConditioningArtifact,
    SDXL_RUNTIME_SURFACE_CONTRACTS,
    UnifiedSDXLRuntimeProtocol,
    UnifiedSDXLRuntimeSeams,
)


@dataclass(frozen=True)
class UnifiedSDXLRuntimeConfig:
    """Configuration shared by unified SDXL runtime execution modes."""

    model_variant: str
    execution_class: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    clip_layer: int = -2
    batch_size: int = 1


@dataclass
class UnifiedSDXLPreparedInputs:
    """Prepared runtime artifacts consumed by the denoise entrypoint."""

    base_model: BaseModelAvailability | None = None
    compiled_unet: CompiledUnetArtifact | None = None
    conditioning: PromptConditioningArtifact | None = None
    injected_features: dict[str, InjectedFeatureArtifact] = field(default_factory=dict)
    payload: Any = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class UnifiedSDXLDenoiseResult:
    """Denoise output plus the explicit GPU-attached execution state."""

    samples: torch.Tensor
    execution_state: GpuAttachedExecutionState | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class UnifiedSDXLRuntime(UnifiedSDXLRuntimeProtocol):
    """Unified Nex-owned SDXL runtime spine.

    This class is intentionally skeletal in W07c1. It exists to give W07c2 and
    W07c3 a stable production-owned target instead of building directly against
    the benchmark harness.
    """

    route_label = "sdxl_unified_runtime"

    seams = UnifiedSDXLRuntimeSeams(
        task_start_owner="modules.async_worker",
        prompt_conditioning_owner="backend.conditioning",
        compiled_unet_owner="backend.sdxl_unified_runtime",
        denoise_owner="backend.sdxl_unified_runtime",
        decode_owner="backend.sdxl_unified_runtime",
    )

    def __init__(self, config: UnifiedSDXLRuntimeConfig) -> None:
        self.config = config
        self.base_model: BaseModelAvailability | None = None
        self.compiled_unet: CompiledUnetArtifact | None = None
        self.conditioning: PromptConditioningArtifact | None = None
        self.injected_features: dict[str, InjectedFeatureArtifact] = {}
        self.execution_state: GpuAttachedExecutionState | None = None

    def load_components(self) -> float:
        raise NotImplementedError(
            "W07c2/W07c3 must implement component loading in backend.sdxl_unified_runtime."
        )

    def prepare_inputs(self) -> tuple[UnifiedSDXLPreparedInputs, dict[str, float]]:
        raise NotImplementedError(
            "W07c2 must implement CPU-first artifact preparation in backend.sdxl_unified_runtime."
        )

    def denoise_prepared_inputs(
        self,
        prepared_inputs: UnifiedSDXLPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> UnifiedSDXLDenoiseResult:
        raise NotImplementedError(
            "W07c3 must implement stream-like denoise in backend.sdxl_unified_runtime."
        )

    def decode_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        raise NotImplementedError(
            "W07c3 must implement the unified decode transition in backend.sdxl_unified_runtime."
        )

    def close(self) -> None:
        self.execution_state = None

    def patched_weights_for_block(self, block_id: str) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["patched_weights_for_block"]
        raise NotImplementedError(
            "W07c2/W07c3 must implement compiled block retrieval in backend.sdxl_unified_runtime."
        )

    def injected_features_for_block(self, block_id: str, timestep: Any, context: Any) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["injected_features_for_block"]
        raise NotImplementedError(
            "W07c2/W07c3 must implement feature injection retrieval in backend.sdxl_unified_runtime."
        )
