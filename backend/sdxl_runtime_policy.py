from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import environment_profile
from backend.staging_manager import ExecutionClass, PlacementSolver, ResidencyMode


EXECUTION_FAMILY_STANDARD = "standard_sdxl"
EXECUTION_FAMILY_GGUF_STAGED = "sdxl_gguf_staged"
GPU_PREFERRED_VRAM_THRESHOLD_MB = 12 * 1024

CLIP_RESIDENCY_CPU_ONLY = "cpu_only"
CLIP_RESIDENCY_GPU_THEN_OFFLOAD = "gpu_then_offload"
CLIP_RESIDENCY_GPU_RESIDENT = "gpu_resident"

VAE_ENCODE_CPU_DEFAULT = "cpu_default"
VAE_ENCODE_GPU_PREFERRED = "gpu_preferred"
VAE_POSTURE_TRANSIENT_GPU = "transient_gpu"
VAE_POSTURE_GPU_RESIDENT = "gpu_resident"

SDXL_RESIDENCY_CLASS_FULL = "full_resident"
SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING = "unified_streaming"
SDXL_RESIDENCY_CLASS_GGUF_STAGED = "gguf_staged_residency"
SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING = "gguf_true_streaming"
SDXL_GGUF_DEPRECATED_NOTE = "sdxl_gguf_deprecated"


@dataclass(frozen=True)
class SDXLExecutionPolicy:
    enabled: bool
    architecture: str | None = None
    execution_class: Any | None = None
    execution_family: str | None = None
    residency_class: str | None = None
    clip_residency_mode: str | None = None
    vae_encode_mode: str | None = None
    keep_clip_loaded: bool = False
    prefer_clip_gpu: bool = False
    prefer_gpu_vae_encode: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    stream_budget_mb: float = 256.0

    def cache_domain(self) -> tuple[str | None, str | None, str | None]:
        return self.execution_family, self.residency_class, self.clip_residency_mode


def is_gguf_checkpoint_name(base_model_name: Any | None) -> bool:
    return str(base_model_name or "").strip().lower().endswith(".gguf")


def is_legacy_sdxl_gguf_selection(*, architecture: Any | None, base_model_name: Any | None) -> bool:
    normalized_architecture = str(architecture or "").strip().lower()
    return normalized_architecture == "sdxl" and is_gguf_checkpoint_name(base_model_name)


def is_dedicated_gguf_execution_policy(policy: Any | None) -> bool:
    if policy is None:
        return False
    execution_family = str(getattr(policy, 'execution_family', None) or '').strip().lower()
    residency_class = str(getattr(policy, 'residency_class', None) or '').strip().lower()
    return (
        execution_family == EXECUTION_FAMILY_GGUF_STAGED
        or residency_class == SDXL_RESIDENCY_CLASS_GGUF_STAGED
        or residency_class == SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING
    )


def policy_marks_legacy_sdxl_gguf(policy: Any | None) -> bool:
    if policy is None:
        return False
    notes = tuple(str(item) for item in (getattr(policy, 'notes', ()) or ()))
    architecture = str(getattr(policy, 'architecture', None) or '').strip().lower()
    return architecture == 'sdxl' and SDXL_GGUF_DEPRECATED_NOTE in notes


def normalize_residency_class(residency_class: Any | None) -> str:
    normalized = str(residency_class or "").strip().lower()
    if normalized in {
        SDXL_RESIDENCY_CLASS_FULL,
        SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING,
        SDXL_RESIDENCY_CLASS_GGUF_STAGED,
        SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING,
    }:
        return normalized
    return SDXL_RESIDENCY_CLASS_FULL


def resolve_sdxl_execution_policy(
    *,
    architecture: str | None,
    base_model_name: Any | None,
    profile: Any | None = None,
    requested_residency_class: Any | None = None,
) -> SDXLExecutionPolicy:
    normalized_architecture = str(architecture or "").strip().lower() or None

    is_gguf_model = is_gguf_checkpoint_name(base_model_name)
    is_legacy_sdxl_gguf = is_legacy_sdxl_gguf_selection(
        architecture=normalized_architecture,
        base_model_name=base_model_name,
    )
    allow_gguf_runtime_family = is_gguf_model and not is_legacy_sdxl_gguf

    task_id = "sdxl"
    if normalized_architecture == "flux":
        task_id = "flux_fill"

    if profile is not None:
        plan = PlacementSolver.solve(
            vram_total_mb=getattr(profile, "total_vram_mb", 8192.0),
            ram_total_mb=getattr(profile, "total_ram_mb", 16384.0),
            task_id=task_id,
        )
    else:
        plan = PlacementSolver.solve_from_system(task_id=task_id)

    unet_streaming = plan.execution_class == ExecutionClass.SDXL_STREAMING_T1 or plan.unet.mode in {
        ResidencyMode.CPU_RESIDENT,
        ResidencyMode.DISK_PAGED,
    }
    residency_class = SDXL_RESIDENCY_CLASS_FULL
    if allow_gguf_runtime_family:
        residency_class = (
            SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING
            if unet_streaming
            else SDXL_RESIDENCY_CLASS_GGUF_STAGED
        )
    elif normalized_architecture == "sdxl" and unet_streaming:
        residency_class = SDXL_RESIDENCY_CLASS_UNIFIED_STREAMING

    clip_residency_mode = CLIP_RESIDENCY_CPU_ONLY
    if plan.clip.mode == ResidencyMode.GPU_RESIDENT:
        clip_residency_mode = CLIP_RESIDENCY_GPU_RESIDENT

    vae_encode_mode = VAE_ENCODE_CPU_DEFAULT
    if plan.vae.residency_mode == ResidencyMode.GPU_RESIDENT.value:
        vae_encode_mode = VAE_POSTURE_GPU_RESIDENT
    elif plan.vae.residency_mode == ResidencyMode.TRANSIENT_GPU.value:
        vae_encode_mode = VAE_POSTURE_TRANSIENT_GPU
    elif plan.vae.mode == ResidencyMode.GPU_RESIDENT:
        vae_encode_mode = VAE_POSTURE_GPU_RESIDENT

    notes = [plan.tier.name, plan.execution_class.name, plan.unet.mode.name]
    if is_legacy_sdxl_gguf:
        notes.append(SDXL_GGUF_DEPRECATED_NOTE)

    stream_budget_mb = 0.0 if plan.unet.mode == ResidencyMode.GPU_RESIDENT else 256.0

    return SDXLExecutionPolicy(
        enabled=True,
        architecture=normalized_architecture,
        execution_class=plan.execution_class,
        execution_family=EXECUTION_FAMILY_GGUF_STAGED if allow_gguf_runtime_family else EXECUTION_FAMILY_STANDARD,
        residency_class=residency_class,
        clip_residency_mode=clip_residency_mode,
        vae_encode_mode=vae_encode_mode,
        keep_clip_loaded=(plan.clip.mode == ResidencyMode.GPU_RESIDENT),
        prefer_clip_gpu=(plan.clip.mode == ResidencyMode.GPU_RESIDENT),
        prefer_gpu_vae_encode=(plan.vae.mode == ResidencyMode.GPU_RESIDENT),
        notes=tuple(notes),
        stream_budget_mb=stream_budget_mb,
    )
