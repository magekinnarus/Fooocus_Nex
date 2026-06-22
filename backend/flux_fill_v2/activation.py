from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from backend.process_transition import (
    PROCESS_CLASS_FLUX_FILL,
    PROCESS_FAMILY_FLUX_FILL,
    ProcessKey,
    build_process_key,
    clear_active_runtime,
    set_active_runtime,
)
from backend.flux_fill_v2.contracts import UNetSpineKind

logger = logging.getLogger(__name__)

FLUX_FILL_TIER_FP8 = "fp8"
FLUX_FILL_TIER_Q8 = "q8_0"
FLUX_FILL_TIER_Q4 = "q4_k_s"
FLUX_FILL_AE_ASSET_ID = "inpaint.flux_fill.ae"
FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID = "inpaint.flux_fill.empty_conditioning"
FLUX_FILL_UNET_ASSET_BY_TIER = {
    FLUX_FILL_TIER_FP8: "inpaint.flux_fill.unet.fp8",
    FLUX_FILL_TIER_Q8: "inpaint.flux_fill.unet.q8_0",
    FLUX_FILL_TIER_Q4: "inpaint.flux_fill.unet.q4_k_s",
}
FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT = {
    "flux_fill_fp8": FLUX_FILL_UNET_ASSET_BY_TIER[FLUX_FILL_TIER_FP8],
    "flux_fill_q8": FLUX_FILL_UNET_ASSET_BY_TIER[FLUX_FILL_TIER_Q8],
    "flux_fill_q4_k_s": FLUX_FILL_UNET_ASSET_BY_TIER[FLUX_FILL_TIER_Q4],
}
FLUX_FILL_MODEL_VARIANT_BY_TIER = {
    FLUX_FILL_TIER_FP8: "flux_fill_fp8",
    FLUX_FILL_TIER_Q8: "flux_fill_q8",
    FLUX_FILL_TIER_Q4: "flux_fill_q4_k_s",
}


@dataclass(frozen=True)
class FluxFillActivationAssets:
    unet_path: str
    ae_path: str
    conditioning_cache_path: str
    model_variant: str
    conditioning_kind: str


def _assign_task_state_attr(task_state: Any, name: str, value: Any) -> None:
    if task_state is None:
        return
    try:
        setattr(task_state, name, value)
    except Exception:
        pass


def _normalize_flux_fill_conditioning(value: Any) -> str:
    normalized = str(value or "empty").strip().lower().replace("-", "_").replace(" ", "_")
    return "empty" if normalized in {"", "empty"} else normalized


def _normalize_flux_fill_tier(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in FLUX_FILL_MODEL_VARIANT_BY_TIER:
        return normalized
    return None


def _resolve_flux_fill_model_variant(task_state: Any) -> str:
    explicit_variant = str(getattr(task_state, "flux_fill_model_variant", "") or "").strip()
    if explicit_variant in FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT:
        return explicit_variant

    explicit_tier = _normalize_flux_fill_tier(
        getattr(task_state, "flux_fill_tier", None) or getattr(task_state, "flux_tier", None)
    )
    if explicit_tier is not None:
        return FLUX_FILL_MODEL_VARIANT_BY_TIER[explicit_tier]

    return "flux_fill_fp8"


def resolve_flux_fill_assets(task_state: Any) -> FluxFillActivationAssets | None:
    from modules import model_registry

    conditioning_kind = _normalize_flux_fill_conditioning(getattr(task_state, "flux_fill_conditioning", None))

    direct_unet_path = str(
        getattr(task_state, "flux_fill_unet_path", None) or getattr(task_state, "unet_path", None) or ""
    ).strip()
    direct_ae_path = str(
        getattr(task_state, "flux_fill_ae_path", None) or getattr(task_state, "ae_path", None) or ""
    ).strip()
    direct_conditioning_path = str(
        getattr(task_state, "flux_fill_conditioning_cache_path", None)
        or getattr(task_state, "conditioning_cache_path", None)
        or ""
    ).strip()

    model_variant = _resolve_flux_fill_model_variant(task_state)
    unet_asset_id = FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT.get(
        model_variant,
        FLUX_FILL_UNET_ASSET_BY_MODEL_VARIANT["flux_fill_fp8"],
    )
    unet_path = direct_unet_path or model_registry.resolve_asset_path(unet_asset_id)
    ae_path = direct_ae_path or model_registry.resolve_asset_path(FLUX_FILL_AE_ASSET_ID)

    if direct_conditioning_path:
        conditioning_cache_path = direct_conditioning_path
    elif conditioning_kind == "empty":
        conditioning_cache_path = model_registry.resolve_asset_path(FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID)
    else:
        # Prompt-conditioned routing remains deferred to W03R until a real T5 posture lands.
        return None

    assets = FluxFillActivationAssets(
        unet_path=str(unet_path),
        ae_path=str(ae_path),
        conditioning_cache_path=str(conditioning_cache_path),
        model_variant=model_variant,
        conditioning_kind=conditioning_kind,
    )
    _assign_task_state_attr(task_state, "flux_fill_model_variant", assets.model_variant)
    _assign_task_state_attr(task_state, "flux_fill_unet_path", assets.unet_path)
    _assign_task_state_attr(task_state, "flux_fill_ae_path", assets.ae_path)
    _assign_task_state_attr(task_state, "flux_fill_conditioning_cache_path", assets.conditioning_cache_path)
    return assets


def resolve_flux_fill_process_key(
    task_state: Any,
    *,
    route_family: str | None = None,
    selected_engine: str | None = None,
) -> ProcessKey | None:
    """ Greenfield Process Key Resolver.
    
    This function resolves the process key for the greenfield runtime, returning
    None if the request does not target the greenfield Flux Fill runtime.
    """
    from modules.flux_fill_surface import OBJR_ENGINE_FLUX_FILL, normalize_objr_engine

    # Resolve selected engine
    if selected_engine is None and task_state is not None:
        selected_engine = normalize_objr_engine(getattr(task_state, "objr_engine", None))

    # Identify if route family or task state options target greenfield Flux Fill
    is_flux_fill = False
    if str(route_family or "").strip().lower() == "flux_fill":
        # Check if inpaint_route is set to flux or objr_engine targets Flux Fill
        inpaint_route = getattr(task_state, "inpaint_route", None)
        if inpaint_route == "flux" or selected_engine == OBJR_ENGINE_FLUX_FILL:
            is_flux_fill = True
    elif selected_engine == OBJR_ENGINE_FLUX_FILL:
        is_flux_fill = True

    if not is_flux_fill:
        return None

    spine_kind = resolve_flux_fill_spine_kind(task_state)
    assets = resolve_flux_fill_assets(task_state)
    if assets is None:
        return None

    identity = tuple(
        sorted(
            (
                ("ae_path", assets.ae_path),
                ("conditioning_cache_path", assets.conditioning_cache_path),
                ("model_variant", assets.model_variant),
                ("unet_path", assets.unet_path),
                ("unet_spine", spine_kind.value),
            )
        )
    )

    return build_process_key(
        family=PROCESS_FAMILY_FLUX_FILL,
        process_class=PROCESS_CLASS_FLUX_FILL,
        authoritative_identity=identity,
        residency_class="resident" if spine_kind == UNetSpineKind.RESIDENT else "streaming",
        route_family="flux_fill",
    )


def resolve_flux_fill_spine_kind(task_state: Any) -> UNetSpineKind:
    """ Resolves the greenfield UNetSpineKind from task state parameters. """
    if task_state is None:
        return UNetSpineKind.STREAMING

    # Greenfield runtime posture option
    posture = str(
        getattr(task_state, "flux_fill_runtime_posture", None)
        or getattr(task_state, "flux_fill_unet_spine", None)
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if posture == "resident":
        return UNetSpineKind.RESIDENT

    return UNetSpineKind.STREAMING


def sync_flux_fill_process_activation(
    route: Any,
    task_state: Any,
    requested_process_key: ProcessKey | None,
) -> Any:
    """ Greenfield route process activation synchronizer. """
    if (
        requested_process_key is not None
        and requested_process_key.family == PROCESS_FAMILY_FLUX_FILL
    ):
        spine_kind = resolve_flux_fill_spine_kind(task_state)
        safe_to_retain = (spine_kind == UNetSpineKind.RESIDENT)

        set_active_runtime(
            family=PROCESS_FAMILY_FLUX_FILL,
            key=requested_process_key,
            route_owner=route.route_id,
            safe_to_retain=safe_to_retain,
        )
    else:
        clear_active_runtime()
    return None
