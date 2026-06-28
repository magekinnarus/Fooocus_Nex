from __future__ import annotations

from typing import Any
import torch
from backend import patching as backend_patching

def _detach_flux_streaming_scheduler(model_options: Any) -> None:
    if not isinstance(model_options, dict):
        return

    detached_scheduler_ids: set[int] = set()
    for options_key in ("flux_fill", "flux_dev", "flux"):
        options = model_options.get(options_key, {})
        if not isinstance(options, dict):
            continue
        scheduler = options.get("streaming_scheduler", None)
        if scheduler is None:
            continue
        scheduler_id = id(scheduler)
        if scheduler_id in detached_scheduler_ids:
            continue
        detached_scheduler_ids.add(scheduler_id)
        detach = getattr(scheduler, "detach", None)
        if callable(detach):
            try:
                detach()
            except Exception:
                pass

class FluxDirectStreamModelPatcher(backend_patching.NexModelPatcher):
    """Treat a pinned CPU-host UNet as the source artifact and skip generic reload work."""

    def model_size(self):
        return 0

    def loaded_size(self):
        return 0

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        target_device = device_to or self.load_device
        self.model.device = target_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        self.model.current_weight_patches_uuid = self.patches_uuid

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        self.load(device_to=device_to, force_patch_weights=force_patch_weights, full_load=False)
        return 0

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        target_device = device_to or self.offload_device
        self.model.device = target_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0
        return self.model

    def detach(self, unpatch_all=True):
        self.model.device = self.offload_device
        self.model.model_lowvram = False
        self.model.model_loaded_weight_memory = 0
        self.model.lowvram_patch_counter = 0

        # Flux-owned scheduler cleanup stays local to the streaming runtime seam.
        _detach_flux_streaming_scheduler(getattr(self, "model_options", None))
        return self.model
