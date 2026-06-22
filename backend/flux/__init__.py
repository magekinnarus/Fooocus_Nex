"""Flux runtime helper compatibility stubs for Fooocus_Nex (Legacy Archived)."""

from typing import Any

class LegacyFluxArchivedError(NotImplementedError):
    """Raised when any part of the archived legacy Flux Fill pipeline is invoked."""
    def __init__(self, message="Legacy Flux Fill has been archived. Greenfield runtime pivot in progress."):
        super().__init__(message)


# Dummy constants/structures
EMPTY_FLUX_CROSS_ATTN_SHAPE = (1, 512, 4096)
EMPTY_FLUX_POOLED_SHAPE = (1, 768)
EXPECTED_FLUX_FILL_CONTRACT = {}

class FluxEmptyConditioning:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillConditioningPayloads:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillDecodedImage:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillDenoiseResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillPipeline:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillPipelineConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillPipelineCropPlan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillPipelineResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillLatentSource:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillResult:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillUNetInfo:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

class FluxFillUnsupportedModelError(Exception):
    pass

class FluxFillValidationError(ValueError):
    pass

def is_native_flux_dimensions(width: int, height: int) -> bool:
    return False

def is_native_sdxl_dimensions(width: int, height: int) -> bool:
    return False

def prepare_flux_fill_pipeline_context_crop(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def restore_flux_fill_pipeline_context_crop(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def select_flux_fill_pipeline_context_crop_plan(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def select_flux_fill_canvas_dimensions(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def select_sdxl_bucket_for_aspect(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def stitch_flux_fill_pipeline_context_crop(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def run_flux_fill_pipeline(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def build_flux_concat_condition(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def build_flux_fill_conditioning_payloads(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def create_flux_fill_noise(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def decode_flux_fill_latent(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def denoise_flux_fill_latent(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def inspect_flux_fill_gguf(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def load_flux_ae(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def load_flux_empty_conditioning_cache(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def load_flux_fill_unet(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def prepare_flux_fill_latent_source(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def run_flux_fill(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def save_flux_empty_conditioning_cache(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def validate_flux_fill_unet_config(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()
