"""Unified SDXL runtime contract for the fp16 spine.

This module names the stable runtime entrypoints and artifact categories used
by W07c follow-through. It intentionally avoids policy, routing, and loader
behavior so the runtime spine can be implemented without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


SDXL_RUNTIME_PHASES: tuple[str, ...] = (
    "startup",
    "model_refresh",
    "prompt_encode",
    "diffusion",
    "decode",
    "finalize",
)

SDXL_RUNTIME_ENTRYPOINTS: tuple[str, ...] = (
    "load_components",
    "prepare_inputs",
    "denoise_prepared_inputs",
    "decode_latent",
    "close",
)

SDXL_RUNTIME_BLOCK_SURFACES: tuple[str, ...] = (
    "patched_weights_for_block",
    "injected_features_for_block",
)


@dataclass(frozen=True)
class RuntimeSurfaceContract:
    """Caller/owner/cache boundary for a runtime surface."""

    surface_name: str
    caller_responsibility: str
    owner_responsibility: str
    artifact_kind: str
    cache_scope: str
    invalidated_by: tuple[str, ...]


@dataclass(frozen=True)
class BaseModelAvailability:
    """Base model availability, separate from any compiled runtime artifact."""

    family: str
    variant: str
    source_path: str | None = None
    fingerprint: str | None = None
    loaded: bool = False
    reusable: bool = True


@dataclass(frozen=True)
class CompiledUnetArtifact:
    """CPU-owned fp16 UNet artifact prepared for attach or stream-like use."""

    family: str
    variant: str
    execution_class: str
    source_path: str | None = None
    source_fingerprint: str | None = None
    artifact_fingerprint: str | None = None
    pinned_cpu_mb: float = 0.0
    gpu_mb: float = 0.0
    reusable: bool = True


@dataclass(frozen=True)
class PromptConditioningArtifact:
    """Reusable prompt-conditioning payload emitted after CPU prompt encode."""

    family: str
    variant: str
    prompt_fingerprint: str
    clip_identity: str
    clip_layer_idx: int | None = None
    conditioning_fingerprint: str | None = None
    pooled_fingerprint: str | None = None
    reusable: bool = True


@dataclass(frozen=True)
class InjectedFeatureArtifact:
    """Reusable side-feature payload injected during diffusion."""

    family: str
    variant: str
    block_id: str
    timestep_key: str
    context_key: str
    feature_fingerprint: str | None = None
    reusable: bool = True


@dataclass(frozen=True)
class GpuAttachedExecutionState:
    """Explicit GPU-attached execution state for the active runtime phase."""

    execution_class: str
    device: str = "cuda"
    active_phase: str = "diffusion"
    attached_component_ids: tuple[str, ...] = field(default_factory=tuple)
    stream_budget_mb: float = 0.0
    headroom_mb: float = 0.0


SDXL_RUNTIME_SURFACE_CONTRACTS: dict[str, RuntimeSurfaceContract] = {
    "patched_weights_for_block": RuntimeSurfaceContract(
        surface_name="patched_weights_for_block",
        caller_responsibility=(
            "Call only after a compiled UNet artifact has been selected for the request and "
            "only for the active diffusion block."
        ),
        owner_responsibility=(
            "Return the already-compiled block payload for execution without asking the caller "
            "to rebuild LoRA or base-model patch state."
        ),
        artifact_kind="compiled_unet",
        cache_scope=(
            "May reuse the compiled UNet artifact across requests while the base model identity, "
            "LoRA stack fingerprint, compiler profile, and execution class remain unchanged."
        ),
        invalidated_by=(
            "base_model_change",
            "lora_stack_change",
            "compiler_profile_change",
            "execution_class_change",
        ),
    ),
    "injected_features_for_block": RuntimeSurfaceContract(
        surface_name="injected_features_for_block",
        caller_responsibility=(
            "Call only during diffusion with the current block id, timestep, and runtime context."
        ),
        owner_responsibility=(
            "Return side features prepared outside the denoise hot path and scoped to the current "
            "request or reusable feature artifact."
        ),
        artifact_kind="injected_feature",
        cache_scope=(
            "May reuse prepared feature artifacts while the prompt payload, feature-source "
            "fingerprint, execution class, and runtime family remain unchanged."
        ),
        invalidated_by=(
            "prompt_change",
            "feature_source_change",
            "execution_class_change",
            "runtime_family_change",
        ),
    ),
}


@runtime_checkable
class UnifiedSDXLRuntimeProtocol(Protocol):
    """Contract for the unified SDXL runtime spine."""

    route_label: str

    def load_components(self) -> float:
        """Load or attach the base model components required by the runtime."""

    def prepare_inputs(self) -> tuple[Any, dict[str, float]]:
        """Build reusable conditioning artifacts and sampler inputs."""

    def denoise_prepared_inputs(
        self,
        prepared_inputs: Any,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> Any:
        """Run diffusion against already prepared runtime artifacts."""

    def decode_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Transition from denoise state into VAE decode state."""

    def close(self) -> None:
        """Release runtime-owned attachments and transient state."""

    def patched_weights_for_block(self, block_id: str) -> Any:
        """Return patched block weights for stream-like execution.

        Caller responsibility and cache boundaries are defined by
        ``SDXL_RUNTIME_SURFACE_CONTRACTS["patched_weights_for_block"]``.
        """

    def injected_features_for_block(self, block_id: str, timestep: Any, context: Any) -> Any:
        """Return side features injected into the denoise path for a block.

        Caller responsibility and cache boundaries are defined by
        ``SDXL_RUNTIME_SURFACE_CONTRACTS["injected_features_for_block"]``.
        """


@dataclass(frozen=True)
class UnifiedSDXLRuntimeSeams:
    """Named ownership seams for the unified SDXL runtime."""

    task_start_owner: str
    prompt_conditioning_owner: str
    compiled_unet_owner: str
    denoise_owner: str
    decode_owner: str
    entrypoints: tuple[str, ...] = SDXL_RUNTIME_ENTRYPOINTS
    block_surfaces: tuple[str, ...] = SDXL_RUNTIME_BLOCK_SURFACES
    phases: tuple[str, ...] = SDXL_RUNTIME_PHASES
