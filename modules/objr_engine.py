import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
import torch
import numpy as np
import gc
import logging
import math
from PIL import Image
from typing import Any, List, Tuple

from modules import model_registry
import modules.config as config
import modules.mask_processing as mask_processing
from ldm_patched.pfn.architecture.MAT import MAT
from modules.blending import sin_blend_1d
import backend.resources as resources
from modules.util import HWC3
from backend.flux import FluxEmptyConditioning
from backend.flux.flux_fill_session import FluxFillSession, FluxPromptConditioningCache
from backend.flux.flux_runtime import FluxFillPipelineConfig
from backend.flux.text_conditioning import encode_flux_prompt_conditioning, save_flux_prompt_conditioning_cache

logger = logging.getLogger(__name__)

_model_instance = None
OBJR_ENGINE_MAT = "MAT512 (initial removal pass)"
OBJR_ENGINE_FLUX_FILL = "Flux Fill (refinement pass)"
OBJR_ENGINE_CHOICES = (OBJR_ENGINE_MAT, OBJR_ENGINE_FLUX_FILL)

FLUX_FILL_TIER_FP8 = "fp8"
FLUX_FILL_TIER_Q8 = "q8_0"
FLUX_FILL_TIER_Q4 = "q4_k_s"
FLUX_FILL_GUIDANCE_DEFAULT = 15.0
FLUX_FILL_AE_ASSET_ID = "inpaint.flux_fill.ae"
FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID = "inpaint.flux_fill.empty_conditioning"
FLUX_FILL_CLIP_L_ASSET_ID = "inpaint.flux_fill.text_encoder.clip_l"
FLUX_FILL_T5XXL_FP16_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.fp16"
FLUX_FILL_T5XXL_Q8_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.q8_0"
FLUX_FILL_T5XXL_Q4_ASSET_ID = "inpaint.flux_fill.text_encoder.t5xxl.q4_k_m"
FLUX_FILL_T5_VARIANT_FP16 = "fp16"
FLUX_FILL_T5_VARIANT_Q8 = "q8_0"
FLUX_FILL_T5_VARIANT_Q4 = "q4_k_m"
FLUX_FILL_T5_RESIDENT_RESERVE_RAM_MB = 4 * 1024
FLUX_FILL_T5_HYBRID_RESERVE_RAM_MB = 8 * 1024
FLUX_FILL_T5_FP16_MIN_BUDGET_MB = 24 * 1024
FLUX_FILL_T5_Q8_MIN_BUDGET_MB = 12 * 1024
FLUX_FILL_CONDITIONING_EMPTY = "empty"
FLUX_FILL_CONDITIONING_PROMPT = "prompt"
FLUX_FILL_INPAINT_ROUTE_SDXL = "sdxl"
FLUX_FILL_INPAINT_ROUTE_FLUX = "flux"
FLUX_FILL_CONDITIONING_BY_KIND = {
    FLUX_FILL_CONDITIONING_EMPTY: FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID,
}
FLUX_FILL_PROMPT_CACHE_TEMP = "temp"
FLUX_FILL_PROMPT_CACHE_PERMANENT = "permanent"
FLUX_FILL_MASK_GROW = 16
FLUX_FILL_MASK_BLUR = 6
FLUX_FILL_BLEND_ALPHA = "alpha"
FLUX_FILL_BLEND_MORPHOLOGICAL = "morphological"
FLUX_FILL_UNET_ASSET_BY_TIER = {
    FLUX_FILL_TIER_FP8: "inpaint.flux_fill.unet.fp8",
    FLUX_FILL_TIER_Q8: "inpaint.flux_fill.unet.q8_0",
    FLUX_FILL_TIER_Q4: "inpaint.flux_fill.unet.q4_k_s",
}
FLUX_FILL_MODEL_VARIANT_BY_TIER = {
    FLUX_FILL_TIER_FP8: "flux_fill_fp8",
    FLUX_FILL_TIER_Q8: "flux_fill_q8",
    FLUX_FILL_TIER_Q4: "flux_fill_q4_k_s",
}
FLUX_FILL_TIER_BY_MODEL_VARIANT = {variant: tier for tier, variant in FLUX_FILL_MODEL_VARIANT_BY_TIER.items()}
FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT = {
    model_variant: FLUX_FILL_UNET_ASSET_BY_TIER[tier]
    for tier, model_variant in FLUX_FILL_MODEL_VARIANT_BY_TIER.items()
}
FLUX_FILL_T5_ASSET_BY_VARIANT = {
    FLUX_FILL_T5_VARIANT_FP16: FLUX_FILL_T5XXL_FP16_ASSET_ID,
    FLUX_FILL_T5_VARIANT_Q8: FLUX_FILL_T5XXL_Q8_ASSET_ID,
    FLUX_FILL_T5_VARIANT_Q4: FLUX_FILL_T5XXL_Q4_ASSET_ID,
}
FLUX_FILL_EMPTY_CONDITIONING_RELATIVE_PATH = os.path.join("flux", "flux_empty_conditioning.pt")

_active_flux_fill_session: FluxFillSession | None = None
_active_flux_fill_session_signature: tuple[str, str, str] | None = None
FLUX_FILL_VRAM_CLASS_RESIDENT = "16gb_plus"
FLUX_FILL_VRAM_CLASS_CONSTRAINED = "8gb_class"
FLUX_FILL_RUNTIME_POSTURE_RESIDENT = "resident"
FLUX_FILL_RUNTIME_POSTURE_HYBRID = "hybrid_offload"
FLUX_FILL_TEXT_ENCODER_ROUTE_BUDGET_MB = {
    "": 0.0,
    "flux_fill": 0.0,
    "removal": 2048.0,
    "upscale": 4096.0,
    "txt2img": 6144.0,
    "image_input": 8192.0,
    "inpaint": 8192.0,
    "outpaint": 8192.0,
    "sdxl": 8192.0,
}


@dataclass(frozen=True)
class FluxFillRouteReconciliation:
    decision: str
    reason: str
    target_signature: tuple[str, str, str] | None = None
    active_signature_before: tuple[str, str, str] | None = None
    active_signature_after: tuple[str, str, str] | None = None
    session_started: bool = False
    session_reused: bool = False
    session_replaced: bool = False
    session_torn_down: bool = False
    next_route_family: str | None = None
    text_encoder_kept: bool | None = None
    text_encoder_action: str | None = None
    text_encoder_reason: str | None = None


@dataclass(frozen=True)
class FluxFillHardwareProfile:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    vram_class: str
    runtime_posture: str
    gpu_name: str | None = None
    cuda_capability: str | None = None
    flux_acceleration_class: str | None = None
    tensor_core_accelerated: bool = False


@dataclass(frozen=True)
class _FluxFillPolicyContext:
    profile_name: str
    total_ram_mb: float
    available_ram_mb: float
    total_vram_mb: float
    available_vram_mb: float
    is_colab: bool
    gpu_name: str | None
    cuda_capability: str | None
    flux_acceleration_class: str | None
    tensor_core_accelerated: bool
    placement_plan: Any


def normalize_objr_engine(engine: str | None) -> str:
    if engine is None or str(engine).strip() == "":
        return OBJR_ENGINE_MAT

    value = str(engine).strip()
    if value in OBJR_ENGINE_CHOICES:
        return value

    aliases = {
        "mat": OBJR_ENGINE_MAT,
        "mat local": OBJR_ENGINE_MAT,
        "mat512": OBJR_ENGINE_MAT,
        "mat512 initial removal pass": OBJR_ENGINE_MAT,
        "places_512_fulldata_g.pth": OBJR_ENGINE_MAT,
        "places512": OBJR_ENGINE_MAT,
        "flux": OBJR_ENGINE_FLUX_FILL,
        "flux fill": OBJR_ENGINE_FLUX_FILL,
        "flux fill colab": OBJR_ENGINE_FLUX_FILL,
        "flux fill refinement pass": OBJR_ENGINE_FLUX_FILL,
    }
    normalized = value.lower().replace("(", "").replace(")", "").strip()
    if normalized in aliases:
        return aliases[normalized]

    raise ValueError(f"Unsupported object removal engine: {engine!r}. Expected one of {OBJR_ENGINE_CHOICES}.")


def _resolve_flux_fill_profile(profile: Any | None = None) -> tuple[Any | None, Any | None]:
    snapshot = None
    if profile is None:
        try:
            from backend import memory_governor

            profile = memory_governor.environment_profile()
            snapshot = memory_governor.capture_snapshot(notes={"purpose": "flux_fill_policy"})
        except Exception:
            profile = None
            snapshot = None
    if profile is None:
        profile = getattr(config, "resolved_memory_environment_profile", None)
    return profile, snapshot


def _resolve_flux_fill_policy_context(profile: Any | None = None) -> _FluxFillPolicyContext:
    profile, snapshot = _resolve_flux_fill_profile(profile)
    profile_name = str(getattr(profile, "name", "") or "").lower()
    profile_notes = getattr(profile, "notes", {}) or {}
    total_ram_mb = float(getattr(profile, "total_ram_mb", 0.0) or getattr(snapshot, "total_ram_mb", 0.0) or 16384.0)
    total_vram_mb = float(getattr(profile, "total_vram_mb", 0.0) or getattr(snapshot, "total_vram_mb", 0.0) or 0.0)
    available_ram_mb = float(
        getattr(profile, "free_ram_mb", 0.0)
        or getattr(profile, "available_ram_mb", 0.0)
        or getattr(snapshot, "free_ram_mb", 0.0)
        or total_ram_mb
        or 0.0
    )
    available_vram_mb = float(
        getattr(profile, "free_vram_mb", 0.0)
        or getattr(profile, "available_vram_mb", 0.0)
        or getattr(snapshot, "free_vram_mb", 0.0)
        or total_vram_mb
        or 0.0
    )
    is_colab = bool(getattr(profile, "is_colab", False))
    from backend.staging_manager import PlacementSolver, ResourceLedger

    planning_ledger = None
    used_host_ram_mb = max(0.0, float(total_ram_mb) - float(available_ram_mb))
    if used_host_ram_mb > 0.0:
        planning_ledger = ResourceLedger()
        # Reflect current host pressure into the staging solve without re-deciding policy locally.
        planning_ledger.register_load(
            "__host_ram_in_use__",
            current_device="cpu",
            residency_mode="cpu_only",
            family="system",
            variant="host_ram_in_use",
            pinned_cpu_mb=0.0,
            host_ram_mb=used_host_ram_mb,
            reusable=False,
        )
    placement_plan = PlacementSolver.solve(
        vram_total_mb=total_vram_mb,
        ram_total_mb=total_ram_mb,
        task_id="flux_fill",
        current_ledger=planning_ledger,
    )
    return _FluxFillPolicyContext(
        profile_name=profile_name,
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
        total_vram_mb=total_vram_mb,
        available_vram_mb=available_vram_mb,
        is_colab=is_colab,
        gpu_name=str(profile_notes.get("gpu_name") or "").strip() or None,
        cuda_capability=str(profile_notes.get("cuda_capability") or "").strip() or None,
        flux_acceleration_class=str(profile_notes.get("flux_acceleration_class") or "").strip() or None,
        tensor_core_accelerated=bool(profile_notes.get("tensor_core_accelerated", False)),
        placement_plan=placement_plan,
    )


def _hardware_profile_from_policy_context(policy_context: _FluxFillPolicyContext) -> FluxFillHardwareProfile:
    placement_plan = policy_context.placement_plan

    if str(getattr(placement_plan, "runtime_posture", "") or "").lower() == "resident":
        vram_class = FLUX_FILL_VRAM_CLASS_RESIDENT
        runtime_posture = FLUX_FILL_RUNTIME_POSTURE_RESIDENT
    else:
        vram_class = FLUX_FILL_VRAM_CLASS_CONSTRAINED
        runtime_posture = FLUX_FILL_RUNTIME_POSTURE_HYBRID

    return FluxFillHardwareProfile(
        profile_name=policy_context.profile_name,
        total_ram_mb=policy_context.total_ram_mb,
        available_ram_mb=policy_context.available_ram_mb,
        total_vram_mb=policy_context.total_vram_mb,
        available_vram_mb=policy_context.available_vram_mb,
        is_colab=policy_context.is_colab,
        vram_class=vram_class,
        runtime_posture=runtime_posture,
        gpu_name=policy_context.gpu_name,
        cuda_capability=policy_context.cuda_capability,
        flux_acceleration_class=policy_context.flux_acceleration_class,
        tensor_core_accelerated=policy_context.tensor_core_accelerated,
    )


def inspect_flux_fill_hardware(profile: Any | None = None) -> FluxFillHardwareProfile:
    return _hardware_profile_from_policy_context(_resolve_flux_fill_policy_context(profile))


def _resolve_flux_fill_placement_plan(profile: Any | None = None):
    return _resolve_flux_fill_policy_context(profile).placement_plan


def _flux_fill_tier_for_model_variant(model_variant: str | None) -> str:
    return FLUX_FILL_TIER_BY_MODEL_VARIANT.get(str(model_variant or "").strip().lower(), FLUX_FILL_TIER_Q4)


def _normalize_route_family(route_family: Any | None) -> str:
    value = str(route_family or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value == "flux":
        return "flux_fill"
    return value


def _flux_fill_next_route_budget_mb(route_family: Any | None) -> float:
    normalized_route_family = _normalize_route_family(route_family)
    if normalized_route_family == "":
        return 0.0
    return float(FLUX_FILL_TEXT_ENCODER_ROUTE_BUDGET_MB.get(normalized_route_family, 6144.0))


def _resolve_flux_fill_t5_mode(profile: Any | None = None) -> str | None:
    plan = _resolve_flux_fill_placement_plan(profile)
    return str(getattr(plan, "t5_mode", "") or "").strip().lower() or None


def _variant_for_flux_fill_t5_mode(t5_mode: str | None) -> str:
    normalized_t5_mode = str(t5_mode or "").strip().lower()
    if normalized_t5_mode in {"cpu_fp16_resident", "disk_paged_fp16"}:
        return FLUX_FILL_T5_VARIANT_FP16
    if normalized_t5_mode in {"cpu_q8_resident", "disk_paged_q8"}:
        return FLUX_FILL_T5_VARIANT_Q8
    return FLUX_FILL_T5_VARIANT_FP16


def evaluate_flux_fill_text_encoder_residency(
    profile: Any | None = None,
    *,
    next_route_family: Any | None = None,
) -> dict[str, Any]:
    policy_context = _resolve_flux_fill_policy_context(profile)
    hardware = _hardware_profile_from_policy_context(policy_context)
    t5_mode = str(getattr(policy_context.placement_plan, "t5_mode", "") or "").strip().lower() or None
    normalized_next_route_family = _normalize_route_family(next_route_family)
    # Cache retention is a bridge concern only. The staging plan has already selected the sanctioned T5 mode.
    resident_cost_mb = (
        FLUX_FILL_T5_RESIDENT_RESERVE_RAM_MB
        if hardware.runtime_posture == FLUX_FILL_RUNTIME_POSTURE_RESIDENT
        else FLUX_FILL_T5_HYBRID_RESERVE_RAM_MB
    )
    next_route_budget_mb = _flux_fill_next_route_budget_mb(normalized_next_route_family)
    required_t5_budget_mb = (
        FLUX_FILL_T5_FP16_MIN_BUDGET_MB
        if t5_mode in {"cpu_fp16_resident", "disk_paged_fp16"}
        else FLUX_FILL_T5_Q8_MIN_BUDGET_MB
    )
    baseline_keep = t5_mode == "cpu_fp16_resident"
    available_after_next_route_mb = max(0.0, float(policy_context.available_ram_mb) - float(next_route_budget_mb))
    keep_resident = baseline_keep and available_after_next_route_mb >= float(resident_cost_mb + required_t5_budget_mb)
    policy = "flux_fill_fp16_disk_paged_default"
    if t5_mode == "disk_paged_q8":
        policy = "flux_fill_q8_disk_paged_fallback"
    elif keep_resident:
        policy = "flux_fill_fp16_resident_roomy_cache"

    return {
        "keep_resident": keep_resident,
        "baseline_keep": baseline_keep,
        "next_route_family": normalized_next_route_family or None,
        "resident_cost_mb": float(resident_cost_mb),
        "next_route_budget_mb": float(next_route_budget_mb),
        "required_t5_budget_mb": float(required_t5_budget_mb),
        "available_ram_mb": float(policy_context.available_ram_mb),
        "available_after_next_route_mb": float(available_after_next_route_mb),
        "total_ram_mb": float(policy_context.total_ram_mb),
        "t5_mode": t5_mode,
        "hardware": hardware,
        "policy": policy,
    }


def select_flux_fill_tier(profile: Any | None = None) -> str:
    plan = _resolve_flux_fill_placement_plan(profile)
    return _flux_fill_tier_for_model_variant(getattr(plan, "model_variant", None))


def _normalize_flux_fill_tier(tier: str | None) -> str:
    if tier is None or str(tier).strip() == "":
        return select_flux_fill_tier()
    normalized = str(tier).strip().lower().replace("-", "_")
    if normalized in {"fp8", "native_fp8"}:
        return FLUX_FILL_TIER_FP8
    if normalized in {"q8", "q8_0"}:
        return FLUX_FILL_TIER_Q8
    if normalized in {"q4", "q4_k_s"}:
        return FLUX_FILL_TIER_Q4
    raise ValueError(f"Unsupported Flux Fill tier: {tier!r}. Expected fp8, q8_0, or q4_k_s.")


def normalize_flux_fill_t5_variant(variant: str | None) -> str:
    if variant is None or str(variant).strip() == "":
        return FLUX_FILL_T5_VARIANT_FP16
    normalized = str(variant).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fp16", "float16", "t5xxl_fp16"}:
        return FLUX_FILL_T5_VARIANT_FP16
    if normalized in {"q8", "q8_0", "t5xxl_q8_0"}:
        return FLUX_FILL_T5_VARIANT_Q8
    if normalized in {"q4", "q4_k_m", "t5xxl_q4_k_m"}:
        return FLUX_FILL_T5_VARIANT_Q4
    raise ValueError(f"Unsupported Flux Fill T5 variant: {variant!r}. Expected fp16, q8_0, or q4_k_m.")


def should_keep_flux_fill_text_encoder_resident(
    profile: Any | None = None,
    *,
    next_route_family: Any | None = None,
) -> bool:
    return bool(
        evaluate_flux_fill_text_encoder_residency(
            profile,
            next_route_family=next_route_family,
        ).get("keep_resident", False)
    )


def reconcile_flux_fill_text_encoder_residency(
    *,
    profile: Any | None = None,
    next_route_family: Any | None = None,
) -> dict[str, Any]:
    decision = evaluate_flux_fill_text_encoder_residency(profile, next_route_family=next_route_family)
    if decision["keep_resident"]:
        decision["text_encoder_action"] = "kept"
        return decision

    try:
        from backend.flux import text_conditioning as flux_text_conditioning

        flux_text_conditioning.clear_flux_prompt_text_encoder_cache()
        decision["text_encoder_action"] = "cleared"
    except Exception as exc:
        decision["text_encoder_action"] = "clear_failed"
        decision["text_encoder_error"] = str(exc)
    return decision


def select_flux_fill_t5_variant(profile: Any | None = None, *, variant: str | None = None) -> str:
    override = variant
    if override is None:
        override = os.environ.get("FOOOCUS_NEX_FLUX_FILL_T5_VARIANT")
    if override is not None and str(override).strip() != "":
        # Manual compatibility override. Production policy still defaults to staging-owned fp16 selection.
        return normalize_flux_fill_t5_variant(override)

    t5_mode = _resolve_flux_fill_t5_mode(profile)
    if t5_mode is not None:
        return _variant_for_flux_fill_t5_mode(t5_mode)

    return FLUX_FILL_T5_VARIANT_FP16


def get_flux_fill_t5_asset_id(variant: str | None = None, *, profile: Any | None = None) -> str:
    selected_variant = select_flux_fill_t5_variant(profile, variant=variant)
    return FLUX_FILL_T5_ASSET_BY_VARIANT[selected_variant]


def ensure_flux_fill_t5_asset(
    variant: str | None = None,
    *,
    profile: Any | None = None,
    progress: bool = True,
) -> tuple[str, str, str]:
    selected_variant = select_flux_fill_t5_variant(profile, variant=variant)
    primary_asset_id = FLUX_FILL_T5_ASSET_BY_VARIANT[selected_variant]
    try:
        return selected_variant, primary_asset_id, model_registry.ensure_asset(primary_asset_id, progress=progress)
    except Exception as exc:
        if selected_variant != FLUX_FILL_T5_VARIANT_FP16:
            raise

        # Explicit compatibility fallback boundary: only use q8 when the sanctioned fp16 asset is unavailable.
        fallback_variant = FLUX_FILL_T5_VARIANT_Q8
        fallback_asset_id = FLUX_FILL_T5_ASSET_BY_VARIANT[fallback_variant]
        logger.warning(
            "Flux Fill fp16 T5 asset is unavailable; falling back to %s. Recovery boundary: %s",
            fallback_variant,
            exc,
        )
        return fallback_variant, fallback_asset_id, model_registry.ensure_asset(fallback_asset_id, progress=progress)


def normalize_flux_fill_conditioning(conditioning: str | None) -> str:
    if conditioning is None or str(conditioning).strip() == "":
        return FLUX_FILL_CONDITIONING_EMPTY

    value = str(conditioning).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"empty", "empty_conditioning", "empty_cond"}:
        return FLUX_FILL_CONDITIONING_EMPTY

    raise ValueError(
        "Unsupported Flux Fill conditioning: "
        f"{conditioning!r}. Expected empty conditioning. Non-empty prompts generate prompt conditioning at runtime."
    )


def normalize_flux_fill_prompt_cache(cache_mode: str | None) -> str:
    value = str(cache_mode or FLUX_FILL_PROMPT_CACHE_TEMP).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"permanent", "persist", "persistent"}:
        return FLUX_FILL_PROMPT_CACHE_PERMANENT
    return FLUX_FILL_PROMPT_CACHE_TEMP


def normalize_flux_fill_blend_mode(blend_mode: str | None) -> str:
    value = str(blend_mode or FLUX_FILL_BLEND_MORPHOLOGICAL).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"morphological", "morph", "fooocus"}:
        return FLUX_FILL_BLEND_MORPHOLOGICAL
    return FLUX_FILL_BLEND_ALPHA


def normalize_flux_fill_inpaint_route(route: str | None) -> str:
    if route is None or str(route).strip() == "":
        return FLUX_FILL_INPAINT_ROUTE_SDXL

    value = str(route).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {
        FLUX_FILL_INPAINT_ROUTE_SDXL,
        "sdxl_inpaint",
        "sdxl_inpaint_route",
        "sdxl_inpaint_model",
        "sdxl_inpaint_pipeline",
    }:
        return FLUX_FILL_INPAINT_ROUTE_SDXL
    if value in {
        FLUX_FILL_INPAINT_ROUTE_FLUX,
        "flux_fill",
        "flux_fill_inpaint",
        "flux_fill_route",
        "flux_inpaint",
    }:
        return FLUX_FILL_INPAINT_ROUTE_FLUX
    raise ValueError(
        "Unsupported Flux Fill inpaint route: "
        f"{route!r}. Expected sdxl or flux."
    )


def is_flux_fill_inpaint_route(route: str | None) -> bool:
    return normalize_flux_fill_inpaint_route(route) == FLUX_FILL_INPAINT_ROUTE_FLUX


def is_flux_fill_route_family(route_family: str | None) -> bool:
    value = str(route_family or "").strip().lower()
    return value in {"removal", "flux_fill"}


def _safe_prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")
    return (slug or "prompt")[:48]


def _primary_clip_root() -> Path:
    roots = getattr(config, "paths_clips", None)
    if isinstance(roots, (list, tuple)) and len(roots) > 0:
        return Path(roots[0])
    root = getattr(config, "path_clip", None)
    if isinstance(root, (list, tuple)) and len(root) > 0:
        return Path(root[0])
    if root:
        return Path(root)
    return Path("models") / "clip"


def _flux_fill_prompt_cache_path(prompt: str, clip_l_path: str, t5_path: str, cache_mode: str | None) -> Path:
    payload = "\n".join([prompt, str(clip_l_path), str(t5_path)]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    filename = f"{_safe_prompt_slug(prompt)}_{digest}.pt"
    if normalize_flux_fill_prompt_cache(cache_mode) == FLUX_FILL_PROMPT_CACHE_PERMANENT:
        return _primary_clip_root() / "flux" / "generated_conditioning" / filename
    return Path(config.path_temp_outputs) / "flux_conditioning" / filename


def _cleanup_flux_prompt_conditioning_host_memory(reason: str) -> None:
    resources.cleanup_memory(
        reason,
        gc_collect=True,
        force_cache=True,
        trim_host=True,
        target_phase=resources.MemoryPhase.PROMPT_ENCODE,
        notes={"route_family": "flux_fill", "component": "text_conditioning"},
    )

def generate_flux_fill_prompt_conditioning_cache(
    prompt: str,
    *,
    cache_mode: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    t5_variant: str | None = None,
    progress: bool = True,
) -> str:
    prompt_text = str(prompt or "").strip()
    if prompt_text == "":
        raise ValueError("Flux Fill prompt conditioning requires a non-empty prompt.")

    clip_l_path = model_registry.ensure_asset(FLUX_FILL_CLIP_L_ASSET_ID, progress=progress)
    _resolved_t5_variant, _resolved_t5_asset_id, t5_path = ensure_flux_fill_t5_asset(
        t5_variant,
        progress=progress,
    )
    output_path = _flux_fill_prompt_cache_path(prompt_text, clip_l_path, t5_path, cache_mode)
    if output_path.exists():
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_resident = should_keep_flux_fill_text_encoder_resident()
    try:
        if not keep_resident:
            _cleanup_flux_prompt_conditioning_host_memory("flux_prompt_conditioning_cache_preflight")
        save_flux_prompt_conditioning_cache(
            prompt_text,
            clip_l_path=Path(clip_l_path),
            t5_path=Path(t5_path),
            output_path=output_path,
            keep_resident=keep_resident,
        )
        return str(output_path)
    finally:
        if not keep_resident:
            _cleanup_flux_prompt_conditioning_host_memory("flux_prompt_conditioning_cache_postflight")


def generate_flux_fill_prompt_conditioning(
    prompt: str,
    *,
    t5_variant: str | None = None,
    progress: bool = True,
) -> FluxEmptyConditioning:
    prompt_text = str(prompt or "").strip()
    if prompt_text == "":
        raise ValueError("Flux Fill prompt conditioning requires a non-empty prompt.")

    clip_l_path = model_registry.ensure_asset(FLUX_FILL_CLIP_L_ASSET_ID, progress=progress)
    _resolved_t5_variant, _resolved_t5_asset_id, t5_path = ensure_flux_fill_t5_asset(
        t5_variant,
        progress=progress,
    )
    keep_resident = should_keep_flux_fill_text_encoder_resident()
    try:
        if not keep_resident:
            _cleanup_flux_prompt_conditioning_host_memory("flux_prompt_conditioning_preflight")
        return encode_flux_prompt_conditioning(
            prompt_text,
            clip_l_path=Path(clip_l_path),
            t5_path=Path(t5_path),
            keep_resident=keep_resident,
        )
    finally:
        if not keep_resident:
            _cleanup_flux_prompt_conditioning_host_memory("flux_prompt_conditioning_postflight")

def get_flux_fill_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    selected_conditioning = normalize_flux_fill_conditioning(conditioning)
    asset_id = FLUX_FILL_CONDITIONING_BY_KIND[selected_conditioning]
    return model_registry.ensure_asset(asset_id, progress=progress)


def get_flux_empty_conditioning_cache_path(conditioning: str | None = None, *, progress: bool = True) -> str:
    return get_flux_fill_conditioning_cache_path(conditioning, progress=progress)


def _component_is_gpu_resident(component: Any | None) -> bool:
    if component is None:
        return False
    residency_mode = str(getattr(component, "residency_mode", "") or "").strip().lower()
    if residency_mode:
        return residency_mode == "gpu_resident"
    mode = getattr(component, "mode", None)
    return str(getattr(mode, "value", "") or mode or "").strip().lower() == "gpu_resident"


def resolve_flux_fill_asset_paths(
    tier: str | None = None,
    *,
    conditioning: str | None = None,
    conditioning_cache_path: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    policy_plan = _resolve_flux_fill_placement_plan()
    fallback_tier = FLUX_FILL_TIER_Q4
    # `tier=` remains an explicit compatibility override; the production path consumes the staging-selected variant.
    selected_tier = (
        _normalize_flux_fill_tier(tier)
        if tier is not None and str(tier).strip() != ""
        else _flux_fill_tier_for_model_variant(getattr(policy_plan, "model_variant", None))
    )
    selected_model_variant = FLUX_FILL_MODEL_VARIANT_BY_TIER[selected_tier]
    runtime_family = getattr(policy_plan, "runtime_family", None)
    runtime_posture = getattr(policy_plan, "runtime_posture", None)
    streaming_profile = getattr(policy_plan, "streaming_profile", None)
    resident_load_strategy = getattr(policy_plan, "resident_load_strategy", None)
    fallback_model_variant = getattr(policy_plan, "fallback_model_variant", None) or FLUX_FILL_MODEL_VARIANT_BY_TIER[fallback_tier]
    fallback_reason = None
    fallback_engaged = False

    force_q4_fallback = str(os.environ.get("FOOOCUS_NEX_FLUX_FILL_FORCE_Q4_FALLBACK", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_q4_fallback and selected_model_variant != fallback_model_variant:
        logger.warning(
            "Flux Fill policy requested %s, but FOOOCUS_NEX_FLUX_FILL_FORCE_Q4_FALLBACK is set; using %s instead.",
            selected_model_variant,
            fallback_model_variant,
        )
        selected_model_variant = fallback_model_variant
        selected_tier = fallback_tier
        fallback_reason = "env_force_q4_fallback"
        fallback_engaged = True

    selected_conditioning = "prompt" if conditioning_cache_path else normalize_flux_fill_conditioning(conditioning)

    def _resolve_unet_bundle(model_variant: str) -> tuple[str, str, str]:
        unet_asset_id = FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT[model_variant]
        unet_asset = model_registry.get_asset(unet_asset_id)
        if unet_asset is None:
            raise KeyError(f"Unknown Flux Fill UNet asset id: {unet_asset_id}")

        required_asset_ids = list(unet_asset.get("requires", []))
        if FLUX_FILL_AE_ASSET_ID not in required_asset_ids:
            required_asset_ids.append(FLUX_FILL_AE_ASSET_ID)

        resolved_required_paths = {}
        for asset_id in required_asset_ids:
            resolved_required_paths[asset_id] = model_registry.ensure_asset(asset_id, progress=progress)

        unet_path = model_registry.ensure_asset(unet_asset_id, progress=progress)
        ae_path = resolved_required_paths.get(FLUX_FILL_AE_ASSET_ID) or model_registry.ensure_asset(
            FLUX_FILL_AE_ASSET_ID,
            progress=progress,
        )
        return unet_asset_id, unet_path, ae_path

    try:
        unet_asset_id, unet_path, ae_path = _resolve_unet_bundle(selected_model_variant)
    except Exception as exc:
        if selected_model_variant != fallback_model_variant:
            # Explicit compatibility fallback boundary: preserve Q4 when the primary asset cannot be resolved.
            logger.warning(
                "Flux Fill primary runtime %s is unavailable; falling back to %s. Recovery boundary: %s",
                selected_model_variant,
                fallback_model_variant,
                exc,
            )
            selected_model_variant = fallback_model_variant
            selected_tier = fallback_tier
            fallback_reason = str(exc)
            fallback_engaged = True
            unet_asset_id, unet_path, ae_path = _resolve_unet_bundle(selected_model_variant)
        else:
            raise

    if selected_model_variant != getattr(policy_plan, "model_variant", None):
        runtime_family = "gguf"
        streaming_profile = None
        resident_load_strategy = None

    conditioning_asset_id = None
    if conditioning_cache_path:
        resolved_conditioning_cache_path = str(conditioning_cache_path)
    else:
        conditioning_asset_id = FLUX_FILL_CONDITIONING_BY_KIND[selected_conditioning]
        resolved_conditioning_cache_path = get_flux_empty_conditioning_cache_path(selected_conditioning, progress=progress)

    return {
        "tier": selected_tier,
        "model_variant": selected_model_variant,
        "unet_asset_id": unet_asset_id,
        "unet_path": unet_path,
        "ae_asset_id": FLUX_FILL_AE_ASSET_ID,
        "ae_path": ae_path,
        "conditioning_kind": selected_conditioning,
        "conditioning_asset_id": conditioning_asset_id,
        "conditioning_cache_path": resolved_conditioning_cache_path,
        "runtime_family": runtime_family,
        "runtime_posture": runtime_posture,
        "streaming_profile": streaming_profile,
        "resident_load_strategy": resident_load_strategy,
        "keep_vae_resident": _component_is_gpu_resident(getattr(policy_plan, "vae", None)),
        "fallback_model_variant": fallback_model_variant,
        "fallback_engaged": bool(fallback_engaged),
        "fallback_reason": str(fallback_reason) if fallback_reason is not None else "",
        "policy_execution_class": getattr(policy_plan, "execution_class", None),
        "policy_hardware_tier": getattr(policy_plan, "hardware_tier", None),
        "policy_t5_mode": getattr(policy_plan, "t5_mode", None),
    }


def _flux_fill_session_signature(asset_paths: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        str(asset_paths["tier"]),
        str(asset_paths["unet_path"]),
        str(asset_paths["ae_path"]),
        str(asset_paths.get("runtime_family", "")),
        str(asset_paths.get("runtime_posture", "")),
        str(asset_paths.get("streaming_profile", "")),
        str(asset_paths.get("resident_load_strategy", "")),
        str(asset_paths.get("keep_vae_resident", "")),
    )


def get_active_flux_fill_session() -> FluxFillSession | None:
    return _active_flux_fill_session


def get_active_flux_fill_session_signature() -> tuple[str, str, str] | None:
    return _active_flux_fill_session_signature


def has_active_flux_fill_session() -> bool:
    session = get_active_flux_fill_session()
    return session is not None and bool(session.started)


def _build_flux_fill_session(asset_paths: dict[str, Any]) -> FluxFillSession:
    session_config = FluxFillPipelineConfig(
        unet_path=asset_paths["unet_path"],
        ae_path=asset_paths["ae_path"],
        conditioning_cache_path=asset_paths["conditioning_cache_path"],
        tier=asset_paths["tier"],
        device=None,
        execution_class=asset_paths.get("policy_execution_class") or None,
        runtime_family=asset_paths.get("runtime_family") or None,
        runtime_posture=asset_paths.get("runtime_posture") or None,
        streaming_profile=asset_paths.get("streaming_profile") or None,
        resident_load_strategy=asset_paths.get("resident_load_strategy") or None,
        keep_vae_resident=asset_paths.get("keep_vae_resident"),
        fallback_model_variant=asset_paths.get("fallback_model_variant") or None,
    )
    conditioning_provider = FluxPromptConditioningCache(
        resolve_conditioning=generate_flux_fill_prompt_conditioning,
        resolve_cache_path=generate_flux_fill_prompt_conditioning_cache,
    )
    return FluxFillSession(session_config, conditioning_provider=conditioning_provider)


def ensure_active_flux_fill_session(
    *,
    tier: str | None = None,
    conditioning: str | None = None,
    progress: bool = True,
) -> FluxFillSession:
    global _active_flux_fill_session, _active_flux_fill_session_signature

    selected_tier = _normalize_flux_fill_tier(tier) if tier is not None and str(tier).strip() != "" else None
    asset_paths = resolve_flux_fill_asset_paths(tier=selected_tier, conditioning=conditioning, progress=progress)
    session_signature = _flux_fill_session_signature(asset_paths)

    if _active_flux_fill_session is not None and _active_flux_fill_session_signature == session_signature and _active_flux_fill_session.started:
        return _active_flux_fill_session

    end_active_flux_fill_session(reason="flux_session_replaced")
    session = _build_flux_fill_session(asset_paths)
    session.start()
    _active_flux_fill_session = session
    _active_flux_fill_session_signature = session_signature
    return session


def reconcile_active_flux_fill_session(
    *,
    route_family: str,
    selected_engine: str,
    tier: str | None = None,
    conditioning: str | None = None,
    task_state: Any | None = None,
    progress: bool = True,
) -> FluxFillRouteReconciliation:
    global _active_flux_fill_session, _active_flux_fill_session_signature

    active_signature = get_active_flux_fill_session_signature()
    normalized_route_family = _normalize_route_family(route_family)
    active_profile = resources.active_memory_environment_profile()

    if not is_flux_fill_route_family(normalized_route_family) or selected_engine != OBJR_ENGINE_FLUX_FILL:
        text_residency = reconcile_flux_fill_text_encoder_residency(
            profile=active_profile,
            next_route_family=normalized_route_family or getattr(task_state, "current_tab", None),
        )
        if get_active_flux_fill_session() is None:
            return FluxFillRouteReconciliation(
                decision="ignored",
                reason=f"non_flux_route:{route_family}",
                active_signature_before=active_signature,
                next_route_family=normalized_route_family or None,
                text_encoder_kept=bool(text_residency.get("keep_resident", False)),
                text_encoder_action=str(text_residency.get("text_encoder_action", "kept")),
                text_encoder_reason="route_transition_without_active_session",
            )

        end_active_flux_fill_session(reason=f"route_switch:{route_family}")
        return FluxFillRouteReconciliation(
            decision="torn_down",
            reason=f"route_switch:{route_family}",
            active_signature_before=active_signature,
            session_torn_down=True,
            next_route_family=normalized_route_family or None,
            text_encoder_kept=bool(text_residency.get("keep_resident", False)),
            text_encoder_action=str(text_residency.get("text_encoder_action", "kept")),
            text_encoder_reason="route_transition_reconciled",
        )

    target_asset_paths = resolve_flux_fill_asset_paths(tier=tier, conditioning=conditioning, progress=progress)
    target_signature = _flux_fill_session_signature(target_asset_paths)

    if active_signature == target_signature and has_active_flux_fill_session():
        return FluxFillRouteReconciliation(
            decision="reused",
            reason="compatible_flux_residency",
            target_signature=target_signature,
            active_signature_before=active_signature,
            active_signature_after=active_signature,
            session_reused=True,
        )

    session_was_active = get_active_flux_fill_session() is not None
    if session_was_active:
        end_active_flux_fill_session(reason="flux_session_replaced")

    session = _build_flux_fill_session(target_asset_paths)
    session.start()
    _active_flux_fill_session = session
    _active_flux_fill_session_signature = target_signature
    return FluxFillRouteReconciliation(
        decision="started" if not session_was_active else "replaced",
        reason="flux_session_started" if not session_was_active else "flux_session_replaced",
        target_signature=target_signature,
        active_signature_before=active_signature,
        active_signature_after=target_signature,
        session_started=not session_was_active,
        session_replaced=session_was_active,
    )


def end_active_flux_fill_session(*, reason: str | None = None) -> dict[str, Any] | None:
    global _active_flux_fill_session, _active_flux_fill_session_signature

    session = _active_flux_fill_session
    if session is None:
        return None

    try:
        if reason:
            logger.info("Ending active Flux Fill session: %s", reason)
        return session.end()
    finally:
        _active_flux_fill_session = None
        _active_flux_fill_session_signature = None




def prepare_flux_fill_mask(mask: np.ndarray, *, grow: int = FLUX_FILL_MASK_GROW, blur: int = FLUX_FILL_MASK_BLUR) -> np.ndarray:
    import cv2

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Flux Fill mask must be HW or HWC, got shape {mask_np.shape}.")

    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    if grow > 0:
        kernel_size = max(1, int(grow) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    if blur > 0:
        kernel_size = max(3, int(blur) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    return mask_np.clip(0, 255).astype(np.uint8)


_expand_flux_fill_mask = prepare_flux_fill_mask
# --- Utility Functions (Ported from reference) ---

def mask_unsqueeze(mask: torch.Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def to_torch(image: np.ndarray, mask: np.ndarray = None, device="cpu"):
    # image: HWC uint8 -> BCHW float32 [0, 1]
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_t = image_t.unsqueeze(0).to(device)
    
    if mask is not None:
        # mask: HW uint8 -> B1HW float32 [0, 1]
        mask_t = torch.from_numpy(mask).float() / 255.0
        mask_t = mask_unsqueeze(mask_t).to(device)
        return image_t, mask_t
    return image_t

def mask_floor(mask: torch.Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)

def pad_reflect_once(x: torch.Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = torch.nn.functional.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = torch.nn.functional.pad(x, tuple(additional_padding), mode="constant")
    return x

def resize_square(image: torch.Tensor, mask: torch.Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
        
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, size=size, mode="nearest-exact")
        mask = torch.nn.functional.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)

def undo_resize_square(image: torch.Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = torch.nn.functional.interpolate(image, size=prev_size, mode="bilinear", align_corners=False)
    # Remove padding: h_orig = prev_size - pad_h, w_orig = prev_size - pad_w
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]

# --- Tiling Utilities ---


def get_segments(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [(0, length, 0, 0)] # start, end, pad_l, pad_r
    
    segments = []
    # First
    segments.append((0, tile_size - overlap, 0, overlap))
    
    while segments[-1][1] < length:
        start = segments[-1][1]
        tile_start = start - overlap
        if tile_start + tile_size >= length:
            end = length
            final_tile_start = max(0, length - tile_size)
            pad_l = start - final_tile_start
            segments.append((start, end, pad_l, 0))
            break

        end = start + tile_size - overlap * 2
        segments.append((start, end, overlap, overlap))
    return segments

# --- Core Engine ---

def load_model(model_name: str = "Places_512_FullData_G.pth") -> MAT:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if model_name != "Places_512_FullData_G.pth":
        checkpoint_path = os.path.join(config.path_removals, model_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Object removal model not found: {model_name}")
    else:
        checkpoint_path = model_registry.ensure_asset('removals.object.mat.places512', progress=True)

    logger.info(f"Loading MAT Object Removal Engine from {checkpoint_path} ...")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Remap keys
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping")
        new_state[new_key] = v

    model = MAT()
    model.load_state_dict(new_state)
    model.eval()

    # Force float32 for Pascal stability
    model.to(torch.float32)

    _model_instance = model
    return _model_instance

def unload_model():
    global _model_instance
    if _model_instance is not None:
        logger.info("Unloading MAT OBJR engine ...")
        del _model_instance
        _model_instance = None
        
    gc.collect()
    if torch.cuda.is_available():
        resources.soft_empty_cache()

@torch.inference_mode()
def remove_object(image: np.ndarray, mask: np.ndarray, seed: int = 0, mask_dilate: int = FLUX_FILL_MASK_GROW) -> np.ndarray:
    """
    Remove objects defined by mask.
    image: HWC uint8
    mask: HW uint8 (255 = inpaint)
    """
    if mask_dilate > 0:
        import cv2
        kernel = np.ones((mask_dilate, mask_dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    h, w, _ = image.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.to(device)
    
    torch.manual_seed(seed)
    
    # Small image path
    if h <= 512 and w <= 512:
        img_t, mask_t = to_torch(image, mask, device=device)
        # resize_square pads to square and resizes to 512
        img_sq, mask_sq, orig_info = resize_square(img_t, mask_t, 512)
        
        # Binarize mask
        mask_sq = mask_floor(mask_sq, 0.99)
        
        # MAT inference
        # Generator.forward(images_in, masks_in, z, c, ...)
        # MAT.forward(image, mask) handles the normalization and Generator call
        res_sq = model(img_sq, mask_sq)
        
        # Undo resize/padding
        res_t = undo_resize_square(res_sq, orig_info)
        
        # Composite: original * (1-mask) + result * mask
        # Ensure mask is exactly what we used for composition
        comp_mask = to_torch(np.zeros_like(image), mask, device=device)[1]
        final_t = img_t * (1.0 - comp_mask) + res_t * comp_mask
        
        final_np = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return final_np

    # Large image path (Tiled)
    logger.info(f"Using tiled processing for {w}x{h} image")
    tile_size = 512
    overlap = 64
    
    img_t, mask_t = to_torch(image, mask, device=device)
    
    output = img_t.clone()
    # Weight map for blending
    weight_total = torch.zeros((1, 1, h, w), device=device)
    # Result accumulator
    accum = torch.zeros_like(img_t)
    
    h_segs = get_segments(h, tile_size, overlap)
    w_segs = get_segments(w, tile_size, overlap)
    
    for y_start, y_end, y_pad_l, y_pad_r in h_segs:
        for x_start, x_end, x_pad_l, x_pad_r in w_segs:
            # Extract tile with padding to ensure 512x512
            tile_y_start = y_start - y_pad_l
            tile_x_start = x_start - x_pad_l
            
            tile_img = img_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]
            tile_mask = mask_t[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size]
            
            # Optimization: Skip if no mask in this tile
            if torch.sum(tile_mask) < 1e-4:
                tile_res = tile_img
            else:
                # Run MAT on tile
                tile_mask_bin = mask_floor(tile_mask, 0.99)
                tile_res = model(tile_img, tile_mask_bin)
            
            # Build 2D weight mask for this tile
            # sin_blend_1d for edges
            w_map = torch.ones((1, 1, tile_size, tile_size), device=device)
            if y_pad_l > 0:
                w_map[:, :, :y_pad_l, :] *= sin_blend_1d(y_pad_l, device).view(1, 1, -1, 1)
            if y_pad_r > 0:
                w_map[:, :, -y_pad_r:, :] *= sin_blend_1d(y_pad_r, device).flip(0).view(1, 1, -1, 1)
            if x_pad_l > 0:
                w_map[:, :, :, :x_pad_l] *= sin_blend_1d(x_pad_l, device).view(1, 1, 1, -1)
            if x_pad_r > 0:
                w_map[:, :, :, -x_pad_r:] *= sin_blend_1d(x_pad_r, device).flip(0).view(1, 1, 1, -1)
            
            accum[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += tile_res * w_map
            weight_total[:, :, tile_y_start : tile_y_start + tile_size, tile_x_start : tile_x_start + tile_size] += w_map
            
    # Final normalization and composition
    tiled_result = accum / (weight_total + 1e-8)
    final_t = img_t * (1.0 - mask_t) + tiled_result * mask_t
    
    final_np = (final_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return final_np

def _select_flux_fill_mode(image: np.ndarray, requested_mode: str | None = None) -> str:
    value = str(requested_mode or "").strip().lower()
    if value:
        if value in {"baseline", "context_crop", "debug", "scaled"}:
            return value
        raise ValueError(f"Unsupported Flux Fill pipeline mode: {requested_mode!r}. Expected baseline, context_crop, debug, or scaled.")

    height, width = image.shape[:2]
    if width % 8 == 0 and height % 8 == 0:
        try:
            from backend.flux import is_native_sdxl_dimensions

            if is_native_sdxl_dimensions(width, height):
                return "baseline"
        except Exception:
            pass
    return "context_crop"


@torch.inference_mode()
def remove_object_flux_fill(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    """
    Remove objects with the Flux Pipeline runtime.
    Asset resolution is intentionally delayed until this function is called.
    """
    if image.ndim != 3:
        raise ValueError(f"Flux Fill image must be HWC, got shape {image.shape}.")
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"Flux Fill mask shape {mask.shape[:2]} does not match image shape {image.shape[:2]}.")

    flux_mask = prepare_flux_fill_mask(np.asarray(mask), grow=int(mask_dilate), blur=int(mask_blur))
    selected_mode = _select_flux_fill_mode(np.asarray(image), mode)
    selected_blend_mode = normalize_flux_fill_blend_mode(blend_mode)
    active_session = get_active_flux_fill_session()

    if active_session is not None:
        result = active_session.generate_removal(
            HWC3(image),
            flux_mask,
            prompt=prompt,
            seed=int(seed),
            steps=int(steps),
            sampler=sampler,
            scheduler=scheduler,
            guidance=float(guidance),
            mode=selected_mode,
            blend_mode=selected_blend_mode,
            callback=callback,
            disable_pbar=disable_pbar,
            progress=progress,
        )
        return HWC3(np.asarray(result.output_image))

    prompt_cache_path = None
    if str(prompt or "").strip():
        prompt_cache_path = generate_flux_fill_prompt_conditioning_cache(
            str(prompt or ""),
            cache_mode=prompt_cache,
            progress=progress,
        )

    asset_paths = resolve_flux_fill_asset_paths(
        tier=tier,
        conditioning=conditioning,
        conditioning_cache_path=prompt_cache_path,
        progress=progress,
    )

    from backend.flux import FluxFillPipelineConfig, run_flux_fill_pipeline

    flux_config = FluxFillPipelineConfig(
        unet_path=asset_paths["unet_path"],
        ae_path=asset_paths["ae_path"],
        conditioning_cache_path=asset_paths["conditioning_cache_path"],
        tier=asset_paths["tier"],
        seed=int(seed),
        steps=int(steps),
        sampler=str(sampler),
        scheduler=str(scheduler),
        guidance=float(guidance),
        mode=selected_mode,
        blend_mode=selected_blend_mode,
        execution_class=asset_paths.get("policy_execution_class"),
        runtime_family=asset_paths.get("runtime_family"),
        runtime_posture=asset_paths.get("runtime_posture"),
        streaming_profile=asset_paths.get("streaming_profile"),
        resident_load_strategy=asset_paths.get("resident_load_strategy"),
        keep_vae_resident=asset_paths.get("keep_vae_resident"),
        fallback_model_variant=asset_paths.get("fallback_model_variant"),
    )
    result = run_flux_fill_pipeline(
        flux_config,
        HWC3(image),
        flux_mask,
        callback=callback,
        disable_pbar=disable_pbar,
    )
    return HWC3(np.asarray(result.output_image))


@torch.inference_mode()
def run_flux_fill_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    mask_blur: int = FLUX_FILL_MASK_BLUR,
    tier: str | None = None,
    conditioning: str | None = None,
    prompt: str | None = None,
    prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    guidance: float = FLUX_FILL_GUIDANCE_DEFAULT,
    steps: int = 30,
    sampler: str = "euler",
    scheduler: str = "normal",
    callback: Any | None = None,
    disable_pbar: bool = True,
    progress: bool = True,
    mode: str | None = None,
) -> np.ndarray:
    """Run Flux Fill from the Inpaint tab using the same Flux session contract."""
    return remove_object_flux_fill(
        image,
        mask,
        seed=seed,
        mask_dilate=mask_dilate,
        mask_blur=mask_blur,
        tier=tier,
        conditioning=conditioning,
        prompt=prompt,
        prompt_cache=prompt_cache,
        blend_mode=blend_mode,
        guidance=guidance,
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        callback=callback,
        disable_pbar=disable_pbar,
        progress=progress,
        mode=mode,
    )


def remove_object_with_engine(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
) -> np.ndarray:
    selected_engine = normalize_objr_engine(engine)
    if selected_engine == OBJR_ENGINE_FLUX_FILL:
        return remove_object_flux_fill(
            image,
            mask,
            seed=seed,
            mask_dilate=mask_dilate,
            mask_blur=flux_mask_blur,
            tier=flux_tier,
            conditioning=flux_conditioning,
            prompt=flux_prompt,
            prompt_cache=flux_prompt_cache,
            blend_mode=flux_blend_mode,
            steps=flux_steps,
            sampler=flux_sampler,
            scheduler=flux_scheduler,
            callback=flux_callback,
            disable_pbar=flux_disable_pbar,
        )
    return remove_object(image, mask, seed=seed, mask_dilate=mask_dilate)

def remove_object_from_file(
    image_path: str,
    mask_path: str,
    seed: int = 0,
    mask_dilate: int = FLUX_FILL_MASK_GROW,
    *,
    engine: str | None = OBJR_ENGINE_MAT,
    flux_tier: str | None = None,
    flux_conditioning: str | None = None,
    flux_prompt: str | None = None,
    flux_prompt_cache: str | None = FLUX_FILL_PROMPT_CACHE_TEMP,
    flux_mask_blur: int = FLUX_FILL_MASK_BLUR,
    flux_blend_mode: str | None = FLUX_FILL_BLEND_MORPHOLOGICAL,
    flux_steps: int = 30,
    flux_sampler: str = "euler",
    flux_scheduler: str = "normal",
    flux_callback: Any | None = None,
    flux_disable_pbar: bool = True,
) -> str:
    """Filepath invariant wrapper with explicit MAT/Flux dispatch."""
    with Image.open(image_path) as img:
        img_np = HWC3(np.array(img.convert('RGBA')))
    with Image.open(mask_path) as msk:
        msk_np = np.array(msk.convert('L'))

    res_np = remove_object_with_engine(
        img_np,
        msk_np,
        seed=seed,
        mask_dilate=mask_dilate,
        engine=engine,
        flux_tier=flux_tier,
        flux_conditioning=flux_conditioning,
        flux_prompt=flux_prompt,
        flux_prompt_cache=flux_prompt_cache,
        flux_mask_blur=flux_mask_blur,
        flux_blend_mode=flux_blend_mode,
        flux_steps=flux_steps,
        flux_sampler=flux_sampler,
        flux_scheduler=flux_scheduler,
        flux_callback=flux_callback,
        flux_disable_pbar=flux_disable_pbar,
    )

    return mask_processing.save_to_temp_png(res_np)
