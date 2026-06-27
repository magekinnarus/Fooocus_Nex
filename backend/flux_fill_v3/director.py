from __future__ import annotations

from backend.flux_fill_v3.contracts import FluxFillRequest
from backend.flux_fill_v3.assembly import FluxAssembly
from backend.flux_fill_v3.runtime_state import acquire_active_flux_streaming_spine
from backend.flux_fill_v3.t5_worker import DiskPagedTextWorker
from backend.flux_fill_v3.vae_worker import TransientVaeWorker


class FluxAssemblyDirector:
    """Authoritative director for assembly selection and instantiation."""

    @staticmethod
    def select_assembly(request: FluxFillRequest) -> FluxAssembly:
        # In P4-M17-W07R, the director only supports and instantiates the streaming assembly:
        # streaming_unet + disk_paged_text + transient_vae
        spine, _reused = acquire_active_flux_streaming_spine(request)
        text_worker = DiskPagedTextWorker(request)
        vae_worker = TransientVaeWorker(request)
        return FluxAssembly(
            spine,
            text_worker,
            vae_worker,
            release_spine_after_execute=False,
        )
