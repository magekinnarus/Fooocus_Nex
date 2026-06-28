from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from backend import resources

def load_flux_ae(
    ae_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
) -> Any:
    path = Path(ae_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux AE path does not exist: {path}")

    from backend import loader as backend_loader
    from ldm_patched.modules import latent_formats

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.vae_offload_device()
    return backend_loader.load_vae(
        str(path),
        load_device=load_device,
        offload_device=offload_device,
        dtype=torch.float32,
        latent_format=latent_formats.Flux(),
    )
