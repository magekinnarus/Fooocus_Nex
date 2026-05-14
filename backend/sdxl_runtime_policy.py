from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import environment_profile


EXECUTION_FAMILY_STANDARD = "standard_sdxl"
EXECUTION_FAMILY_GGUF_STAGED = "sdxl_gguf_staged"
GPU_PREFERRED_VRAM_THRESHOLD_MB = 12 * 1024

CLIP_RESIDENCY_CPU_ONLY = "cpu_only"
CLIP_RESIDENCY_GPU_THEN_OFFLOAD = "gpu_then_offload"
CLIP_RESIDENCY_GPU_RESIDENT = "gpu_resident"

VAE_ENCODE_CPU_DEFAULT = "cpu_default"
VAE_ENCODE_GPU_PREFERRED = "gpu_preferred"

SDXL_RESIDENCY_CLASS_FULL = "full_resident"
SDXL_RESIDENCY_CLASS_GGUF_STAGED = "gguf_staged_residency"
SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING = "gguf_true_streaming"


@dataclass(frozen=True)
class SDXLExecutionPolicy:
    enabled: bool
    architecture: str | None = None
    execution_family: str | None = None
    residency_class: str | None = None
    clip_residency_mode: str | None = None
    vae_encode_mode: str | None = None
    keep_clip_loaded: bool = False
    prefer_clip_gpu: bool = False
    prefer_gpu_vae_encode: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def cache_domain(self) -> tuple[str | None, str | None, str | None]:
        return self.execution_family, self.residency_class, self.clip_residency_mode


def is_gguf_checkpoint_name(base_model_name: Any | None) -> bool:
    return str(base_model_name or "").strip().lower().endswith(".gguf")


def normalize_residency_class(residency_class: Any | None) -> str:
    normalized = str(residency_class or "").strip().lower()
    if normalized in {
        SDXL_RESIDENCY_CLASS_FULL,
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
    if normalized_architecture != "sdxl":
        return SDXLExecutionPolicy(enabled=False, architecture=normalized_architecture)

    profile_name = str(getattr(profile, "name", "") or "").strip().lower()
    total_vram_mb = float(getattr(profile, "total_vram_mb", 0.0) or 0.0)
    is_gguf_model = is_gguf_checkpoint_name(base_model_name)
    normalized_residency = normalize_residency_class(requested_residency_class)

    execution_family = EXECUTION_FAMILY_GGUF_STAGED if is_gguf_model else EXECUTION_FAMILY_STANDARD
    residency_class = SDXL_RESIDENCY_CLASS_GGUF_STAGED if is_gguf_model else SDXL_RESIDENCY_CLASS_FULL
    clip_residency_mode = CLIP_RESIDENCY_CPU_ONLY
    vae_encode_mode = VAE_ENCODE_CPU_DEFAULT
    keep_clip_loaded = False
    notes: list[str] = []

    if normalized_residency == SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING:
        return SDXLExecutionPolicy(
            enabled=True,
            architecture=normalized_architecture,
            execution_family=EXECUTION_FAMILY_GGUF_STAGED if is_gguf_model else EXECUTION_FAMILY_STANDARD,
            residency_class=normalized_residency,
            clip_residency_mode=CLIP_RESIDENCY_GPU_THEN_OFFLOAD,
            vae_encode_mode=VAE_ENCODE_CPU_DEFAULT,
            keep_clip_loaded=False,
            prefer_clip_gpu=True,
            prefer_gpu_vae_encode=False,
            notes=("benchmark_only_true_streaming",),
        )

    if profile_name in {
        environment_profile.PROFILE_COLAB_FREE,
        environment_profile.PROFILE_COLAB_PRO,
    } or total_vram_mb >= GPU_PREFERRED_VRAM_THRESHOLD_MB:
        clip_residency_mode = CLIP_RESIDENCY_GPU_RESIDENT
        vae_encode_mode = VAE_ENCODE_GPU_PREFERRED
        keep_clip_loaded = True
        if profile_name == environment_profile.PROFILE_COLAB_FREE:
            notes.append("colab_free_gpu_preferred")
        elif profile_name == environment_profile.PROFILE_COLAB_PRO:
            notes.append("colab_pro_gpu_preferred")
        else:
            notes.append("high_vram_gpu_preferred")
    elif is_gguf_model and profile_name == environment_profile.PROFILE_LOCAL_LOW_VRAM:
        clip_residency_mode = CLIP_RESIDENCY_CPU_ONLY
        vae_encode_mode = VAE_ENCODE_CPU_DEFAULT
        keep_clip_loaded = False
        notes.append("local_low_vram_gguf_cpu_phase1")
    elif is_gguf_model:
        clip_residency_mode = CLIP_RESIDENCY_GPU_THEN_OFFLOAD
        vae_encode_mode = VAE_ENCODE_CPU_DEFAULT
        notes.append("gguf_staged_default")
    else:
        notes.append("standard_cpu_default")

    return SDXLExecutionPolicy(
        enabled=True,
        architecture=normalized_architecture,
        execution_family=execution_family,
        residency_class=residency_class,
        clip_residency_mode=clip_residency_mode,
        vae_encode_mode=vae_encode_mode,
        keep_clip_loaded=keep_clip_loaded,
        prefer_clip_gpu=clip_residency_mode in {CLIP_RESIDENCY_GPU_THEN_OFFLOAD, CLIP_RESIDENCY_GPU_RESIDENT},
        prefer_gpu_vae_encode=vae_encode_mode == VAE_ENCODE_GPU_PREFERRED,
        notes=tuple(notes),
    )
