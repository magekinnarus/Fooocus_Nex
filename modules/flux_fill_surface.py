"""Lightweight Flux Fill / removal surface constants and normalizers."""

OBJR_ENGINE_MAT = "MAT512 (initial removal pass)"
OBJR_ENGINE_FLUX_FILL = "Flux Fill (refinement pass)"
OBJR_ENGINE_CHOICES = (OBJR_ENGINE_MAT, OBJR_ENGINE_FLUX_FILL)
OBJR_ENGINE_DROPDOWN_CHOICES = (
    ("MAT512 initial removal pass", OBJR_ENGINE_MAT),
    ("Flux Fill refinement pass", OBJR_ENGINE_FLUX_FILL),
)

FLUX_FILL_INPAINT_ROUTE_SDXL = "sdxl"
FLUX_FILL_INPAINT_ROUTE_FLUX = "flux"
FLUX_FILL_INPAINT_ROUTE_CHOICES = (
    ("SDXL Inpaint", FLUX_FILL_INPAINT_ROUTE_SDXL),
    ("Flux Fill", FLUX_FILL_INPAINT_ROUTE_FLUX),
)

FLUX_FILL_BLEND_ALPHA = "alpha"
FLUX_FILL_BLEND_MORPHOLOGICAL = "morphological"


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

    raise ValueError(
        f"Unsupported object removal engine: {engine!r}. Expected one of {OBJR_ENGINE_CHOICES}."
    )


def normalize_flux_fill_blend_mode(blend_mode: str | None) -> str:
    value = str(blend_mode or FLUX_FILL_BLEND_MORPHOLOGICAL).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"morphological", "morph", "fooocus"}:
        return FLUX_FILL_BLEND_MORPHOLOGICAL
    return FLUX_FILL_BLEND_ALPHA


def normalize_flux_fill_inpaint_route(route: str | None) -> str:
    value = str(route or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {
        "",
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
    return FLUX_FILL_INPAINT_ROUTE_SDXL


def is_flux_fill_inpaint_route(route: str | None) -> bool:
    return normalize_flux_fill_inpaint_route(route) == FLUX_FILL_INPAINT_ROUTE_FLUX


def is_flux_fill_route_family(route_family: str | None) -> bool:
    value = str(route_family or "").strip().lower()
    return value in {"removal", "flux_fill"}
