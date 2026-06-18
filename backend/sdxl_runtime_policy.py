from __future__ import annotations

import logging
import os
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
    runtime_family: str | None = None
    execution_mode: str | None = None
    hardware_tier: str | None = None
    allow_cpu_shadow: bool = False
    prefer_clip_gpu: bool = False
    prefer_gpu_vae_encode: bool = False
    stream_budget_mb: float = 256.0
    resident_clean_source_device: str = "cpu"
    notes: tuple[str, ...] = field(default_factory=tuple)

    # Legacy compatibility fields
    execution_class: Any = None
    execution_family: str | None = None
    residency_class: str | None = None
    clip_residency_mode: str | None = None
    vae_encode_mode: str | None = None
    keep_clip_loaded: bool | None = None

    def __post_init__(self) -> None:
        # Since the dataclass is frozen, we use object.__setattr__ to initialize/override fields.
        runtime_family = self.runtime_family
        execution_mode = self.execution_mode
        prefer_clip_gpu = self.prefer_clip_gpu
        prefer_gpu_vae_encode = self.prefer_gpu_vae_encode

        # Infer from residency_class
        if self.residency_class is not None:
            if self.residency_class == "gguf_true_streaming":
                if runtime_family is None:
                    runtime_family = "gguf_sdxl"
                if execution_mode is None:
                    execution_mode = "streaming"
            elif self.residency_class == "gguf_staged_residency":
                if runtime_family is None:
                    runtime_family = "gguf_sdxl"
                if execution_mode is None:
                    execution_mode = "resident"
            elif self.residency_class == "unified_streaming":
                if runtime_family is None:
                    runtime_family = "unified_sdxl"
                if execution_mode is None:
                    execution_mode = "streaming"
            elif self.residency_class == "full_resident":
                if runtime_family is None:
                    runtime_family = "unified_sdxl"
                if execution_mode is None:
                    execution_mode = "resident"
                object.__setattr__(self, "allow_cpu_shadow", True)

        # Infer from execution_family
        if self.execution_family is not None and runtime_family is None:
            if self.execution_family == "sdxl_gguf_staged":
                runtime_family = "gguf_sdxl"
            else:
                runtime_family = "unified_sdxl"

        # Infer from clip_residency_mode or keep_clip_loaded
        if self.clip_residency_mode is not None:
            if self.clip_residency_mode == "gpu_resident":
                prefer_clip_gpu = True
        elif self.keep_clip_loaded is not None:
            if self.keep_clip_loaded:
                prefer_clip_gpu = True

        # Infer from vae_encode_mode
        if self.vae_encode_mode is not None:
            if self.vae_encode_mode in ("transient_gpu", "gpu_resident"):
                prefer_gpu_vae_encode = True

        # Default values if still None
        if runtime_family is None:
            runtime_family = "unified_sdxl"
        if execution_mode is None:
            execution_mode = "resident"

        object.__setattr__(self, "runtime_family", runtime_family)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "prefer_clip_gpu", prefer_clip_gpu)
        object.__setattr__(self, "prefer_gpu_vae_encode", prefer_gpu_vae_encode)

        # Compute legacy properties if they are None
        # 1. execution_family
        if self.execution_family is None:
            val = "sdxl_gguf_staged" if runtime_family == "gguf_sdxl" else "standard_sdxl"
            object.__setattr__(self, "execution_family", val)

        # 2. residency_class
        if self.residency_class is None:
            if runtime_family == "gguf_sdxl":
                val = "gguf_true_streaming" if execution_mode == "streaming" else "gguf_staged_residency"
            else:
                val = "unified_streaming" if execution_mode == "streaming" else "full_resident"
            object.__setattr__(self, "residency_class", val)

        # 3. clip_residency_mode
        if self.clip_residency_mode is None:
            val = "gpu_resident" if prefer_clip_gpu else "cpu_only"
            object.__setattr__(self, "clip_residency_mode", val)

        # 4. vae_encode_mode
        if self.vae_encode_mode is None:
            if prefer_gpu_vae_encode:
                val = "transient_gpu" if execution_mode == "streaming" else "gpu_resident"
            else:
                val = "cpu_default"
            object.__setattr__(self, "vae_encode_mode", val)

        # 5. keep_clip_loaded
        if self.keep_clip_loaded is None:
            object.__setattr__(self, "keep_clip_loaded", prefer_clip_gpu)

        # 6. execution_class
        if self.execution_class is None:
            from backend.staging_manager import ExecutionClass
            if runtime_family == "gguf_sdxl":
                val = ExecutionClass.SDXL_STREAMING_T1 if execution_mode == "streaming" else ExecutionClass.SDXL_RESIDENT_T2
            elif execution_mode == "streaming":
                val = ExecutionClass.SDXL_STREAMING_T1
            else:
                if self.hardware_tier in ("LOW_VRAM", "NORMAL_VRAM"):
                    val = ExecutionClass.SDXL_RESIDENT_T2
                else:
                    val = ExecutionClass.SDXL_GPU_GREEDY_T3PLUS
            object.__setattr__(self, "execution_class", val)

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


def _sdxl_process_class(policy) -> str:
    from backend import process_transition

    if policy is None or not bool(getattr(policy, 'enabled', False)):
        return process_transition.PROCESS_CLASS_STANDARD_SDXL
    
    runtime_family = getattr(policy, 'runtime_family', None)
    execution_mode = getattr(policy, 'execution_mode', None)

    if runtime_family == 'gguf_sdxl':
        if execution_mode == 'streaming':
            return process_transition.PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING
        return process_transition.PROCESS_CLASS_SDXL_GGUF_STAGED

    execution_family = str(getattr(policy, 'execution_family', None) or '').strip().lower()
    residency_class = str(getattr(policy, 'residency_class', None) or '').strip().lower()
    if policy_marks_legacy_sdxl_gguf(policy):
        return process_transition.PROCESS_CLASS_STANDARD_SDXL
    if execution_family == EXECUTION_FAMILY_GGUF_STAGED or residency_class == SDXL_RESIDENCY_CLASS_GGUF_STAGED:
        return process_transition.PROCESS_CLASS_SDXL_GGUF_STAGED
    if residency_class == SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING:
        return process_transition.PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING
        
    return process_transition.PROCESS_CLASS_STANDARD_SDXL


def _sdxl_route_family(policy, base_model_name=None) -> str:
    if policy_marks_legacy_sdxl_gguf(policy):
        return 'sdxl'
    if policy is not None:
        if getattr(policy, 'runtime_family', None) == 'gguf_sdxl':
            return 'gguf'
        execution_family = str(getattr(policy, 'execution_family', None) or '').strip().lower()
        residency_class = str(getattr(policy, 'residency_class', None) or '').strip().lower()
        if (
            execution_family == EXECUTION_FAMILY_GGUF_STAGED
            or residency_class == SDXL_RESIDENCY_CLASS_GGUF_STAGED
            or residency_class == SDXL_RESIDENCY_CLASS_GGUF_TRUE_STREAMING
        ):
            return 'gguf'
    if is_gguf_checkpoint_name(base_model_name):
        return 'gguf'
    return 'sdxl'


def resolve_sdxl_process_key(
    *,
    base_model_name,
    vae_name=None,
    clip_name=None,
    sdxl_policy=None,
    loras=None,
):
    from backend import process_transition

    route_family = _sdxl_route_family(sdxl_policy, base_model_name)
    if route_family == 'gguf':
        identity = [str(base_model_name or '')]
    else:
        identity = [
            str(base_model_name or ''),
            str(clip_name or ''),
        ]

    if loras:
        for lora in sorted(loras):
            identity.append(str(lora))

    return process_transition.build_process_key(
        family=process_transition.PROCESS_FAMILY_SDXL,
        process_class=_sdxl_process_class(sdxl_policy),
        authoritative_identity=tuple(identity),
        execution_family=getattr(sdxl_policy, 'execution_family', None) if sdxl_policy is not None else None,
        residency_class=getattr(sdxl_policy, 'residency_class', None) if sdxl_policy is not None else None,
        route_family=route_family,
    )



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


def _resident_clean_source_device_override() -> str | None:
    raw_value = str(os.environ.get("FOOOCUS_NEX_SDXL_RESIDENT_CLEAN_SOURCE_DEVICE", "") or "").strip().lower()
    if raw_value in {"", "auto"}:
        return None
    if raw_value in {"cpu", "cuda"}:
        return raw_value
    logging.warning(
        "Ignoring unsupported FOOOCUS_NEX_SDXL_RESIDENT_CLEAN_SOURCE_DEVICE=%r; expected auto, cpu, or cuda.",
        raw_value,
    )
    return None


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

    allow_cpu_shadow = False
    if not allow_gguf_runtime_family and not unet_streaming:
        allow_cpu_shadow = True

    from backend.staging_manager import HardwareTier
    resident_clean_source_device = "cpu"
    if plan.tier == HardwareTier.HIGH_VRAM:
        resident_clean_source_device = "cuda"

    notes = [plan.tier.name, plan.execution_class.name, plan.unet.mode.name]
    clean_source_override = _resident_clean_source_device_override()
    if clean_source_override is not None:
        resident_clean_source_device = clean_source_override
        notes.append(f"resident_clean_source_device={clean_source_override}")
    if is_legacy_sdxl_gguf:
        notes.append(SDXL_GGUF_DEPRECATED_NOTE)

    stream_budget_mb = 0.0 if plan.unet.mode == ResidencyMode.GPU_RESIDENT else 256.0

    return SDXLExecutionPolicy(
        enabled=True,
        architecture=normalized_architecture,
        runtime_family="gguf_sdxl" if allow_gguf_runtime_family else "unified_sdxl",
        execution_mode="streaming" if unet_streaming else "resident",
        hardware_tier=plan.tier.name,
        allow_cpu_shadow=allow_cpu_shadow,
        prefer_clip_gpu=(plan.clip.mode == ResidencyMode.GPU_RESIDENT),
        prefer_gpu_vae_encode=(plan.vae.mode == ResidencyMode.GPU_RESIDENT),
        stream_budget_mb=stream_budget_mb,
        resident_clean_source_device=resident_clean_source_device,
        notes=tuple(notes),
    )
