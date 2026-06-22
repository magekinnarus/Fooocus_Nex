from typing import Any
from backend.flux import LegacyFluxArchivedError

class FluxFillPipelineConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise LegacyFluxArchivedError()

def run_flux_fill_pipeline(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()
