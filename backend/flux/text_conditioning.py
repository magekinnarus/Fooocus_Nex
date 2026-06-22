from typing import Any
from backend.flux import LegacyFluxArchivedError

def encode_flux_prompt_conditioning(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def save_flux_prompt_conditioning_cache(*args: Any, **kwargs: Any) -> Any:
    raise LegacyFluxArchivedError()

def clear_flux_prompt_text_encoder_cache(*args: Any, **kwargs: Any) -> Any:
    # This can be a no-op or raise the exception. Let's make it a no-op so that cleanup/reconciliation doesn't crash.
    pass
