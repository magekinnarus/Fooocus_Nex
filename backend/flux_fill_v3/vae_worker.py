from __future__ import annotations

import gc
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend import resources
from backend.flux_fill_v3.contracts import FluxFillRequest, FluxLatentArtifactBundle
from backend.flux_fill_v3.runtime_state import (
    get_cached_latent_artifact_bundle,
    set_cached_latent_artifact_bundle,
)

logger = logging.getLogger(__name__)


def hash_ndarray(arr: np.ndarray | None) -> str:
    if arr is None:
        return "none"
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode('utf-8'))
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    digest.update(arr.tobytes())
    return digest.hexdigest()


def compute_artifact_fingerprint(request: FluxFillRequest) -> str:
    image_hash = hash_ndarray(request.image)
    mask_hash = hash_ndarray(request.mask)
    raw_key = (
        f"ae:{request.ae_path}|"
        f"image:{image_hash}|"
        f"mask:{mask_hash}|"
        f"category:{request.category}|"
        f"blend:{request.blend_mode}|"
        f"mp:{request.target_megapixels}"
    )
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()


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


def _encode_vae_latents(
    vae: Any, image: np.ndarray, mask: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from backend import encode as backend_vae_encode
    from modules.core import numpy_to_pytorch

    _attach_vae(vae, device)

    # 1. Encode unmasked source image
    orig_pixels = numpy_to_pytorch(image)
    source_latent = backend_vae_encode.encode_preloaded_pixels(vae, orig_pixels)["samples"]

    # 2. Gray-masked image for concat encoding
    bb_image_for_concat = image.copy().astype(np.float32) / 255.0
    mask_binary = (mask > 127).astype(np.float32)
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[:, :, 0]
    inv_mask = 1.0 - mask_binary
    for ch in range(3):
        bb_image_for_concat[:, :, ch] -= 0.5
        bb_image_for_concat[:, :, ch] *= inv_mask
        bb_image_for_concat[:, :, ch] += 0.5
    bb_image_for_concat = np.clip(bb_image_for_concat * 255.0, 0, 255).astype(np.uint8)
    concat_pixels = numpy_to_pytorch(bb_image_for_concat)
    concat_latent = backend_vae_encode.encode_preloaded_pixels(vae, concat_pixels)["samples"]

    # 3. Build denoise_mask
    mask_t = torch.from_numpy(mask).float() / 255.0
    if mask_t.ndim == 3:
        mask_t = mask_t[:, :, 0]
    mask_t = mask_t[None, None, :, :]
    denoise_mask = torch.nn.functional.max_pool2d(mask_t, kernel_size=8)
    denoise_mask = (denoise_mask > 0.5).float()

    return source_latent, concat_latent, denoise_mask


def _attach_vae(vae: Any, device: torch.device) -> None:
    patcher = getattr(vae, "patcher", None)
    if patcher is None:
        return

    patch_model = getattr(patcher, "patch_model", None)
    if callable(patch_model):
        patch_model(device_to=device, lowvram_model_memory=0)

    active_device = getattr(patcher, "current_loaded_device", lambda: patcher.load_device)()
    move_model = getattr(vae.first_stage_model, "to", None)
    if callable(move_model):
        move_model(device=active_device, dtype=torch.float32)


def _decode_vae_latents(vae: Any, latent: torch.Tensor, device: torch.device | None = None) -> np.ndarray:
    from backend import decode as backend_vae_decode

    gc.collect()
    try:
        resources.soft_empty_cache(force=True)
    except Exception:
        pass

    if device is not None:
        _attach_vae(vae, device)
    decoded = backend_vae_decode.decode_preloaded_vae(vae, latent.detach().cpu(), tiled=False)

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        from modules.core import pytorch_to_numpy
    except Exception:
        # Fallback for mock environments
        pytorch_to_numpy = lambda x: (x.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    finally:
        sys.argv = original_argv

    decoded_images = pytorch_to_numpy(decoded)
    return decoded_images[0]


class TransientVaeWorker:
    """Authoritative Greenfield Transient VAE Worker Contract.

    Loads VAE dynamically, performs operation, and immediately ejects.
    Checks and populates in-memory cache to support fast re-execution.
    """
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request

    def compute_fingerprint(self) -> str:
        return compute_artifact_fingerprint(self.request)

    def prepare_latents(self, device: torch.device) -> FluxLatentArtifactBundle:
        """Loads VAE transiently, runs encode steps, builds artifacts, and ejects VAE.
        
        Checks the in-memory cache first to avoid re-encoding.
        """
        fingerprint = self.compute_fingerprint()
        logger.debug(f"[Flux Telemetry] VAE latent artifacts check for fingerprint={fingerprint}")
        
        cached_bundle = get_cached_latent_artifact_bundle(fingerprint)
        if cached_bundle is not None:
            logger.debug(f"[Flux Telemetry] VAE latent artifacts cache hit for fingerprint={fingerprint}")
            return FluxLatentArtifactBundle(
                source_latent=cached_bundle.source_latent,
                concat_latent=cached_bundle.concat_latent,
                denoise_mask=cached_bundle.denoise_mask,
                fingerprint=cached_bundle.fingerprint,
                vae_load_time=0.0,
                vae_encode_time=0.0,
            )

        logger.debug(f"[Flux Telemetry] VAE latent artifacts cache miss for fingerprint={fingerprint}. Running VAE encode...")
        logger.debug(f"[Flux Telemetry] Loading VAE transiently from: {self.request.ae_path}")
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(self.request.ae_path, load_device="cpu", offload_device="cpu")
        vae_load_time = time.perf_counter() - vae_load_start
        logger.debug(f"[Flux Telemetry] VAE loaded in {vae_load_time:.3f}s. Encoding latents...")

        try:
            encode_start = time.perf_counter()
            source_latent, concat_latent, denoise_mask = _encode_vae_latents(
                vae, self.request.image, self.request.mask, device
            )
            vae_encode_time = time.perf_counter() - encode_start
            logger.debug(f"[Flux Telemetry] VAE encoding finished in {vae_encode_time:.3f}s.")
            
            bundle = FluxLatentArtifactBundle(
                source_latent=source_latent,
                concat_latent=concat_latent,
                denoise_mask=denoise_mask,
                fingerprint=fingerprint,
                vae_load_time=vae_load_time,
                vae_encode_time=vae_encode_time,
            )
            set_cached_latent_artifact_bundle(bundle)
            logger.debug(f"[Flux Telemetry] VAE latent artifacts generated and cached. Load time={vae_load_time:.3f}s, Encode time={vae_encode_time:.3f}s")
            return bundle
        finally:
            self._eject_vae(vae)

    def decode(self, samples: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float, float]:
        """Loads VAE transiently, decodes samples, and ejects VAE."""
        logger.debug(f"[Flux Telemetry] Loading VAE transiently to decode from: {self.request.ae_path}")
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(self.request.ae_path, load_device="cpu", offload_device="cpu")
        vae_load_time = time.perf_counter() - vae_load_start
        logger.debug(f"[Flux Telemetry] VAE loaded in {vae_load_time:.3f}s. Decoding latents...")

        try:
            decode_start = time.perf_counter()
            output_image = _decode_vae_latents(vae, samples, device)
            vae_decode_time = time.perf_counter() - decode_start
            logger.debug(f"[Flux Telemetry] VAE decoding finished in {vae_decode_time:.3f}s.")
            return output_image, vae_load_time, vae_decode_time
        finally:
            self._eject_vae(vae)

    def _eject_vae(self, vae: Any) -> None:
        if vae is None:
            return
        logger.debug("[Flux Telemetry] Ejecting VAE model and freeing VRAM cache.")
        try:
            patcher = getattr(vae, "patcher", None)
            if patcher is not None:
                resources.eject_model(patcher)
            else:
                resources.eject_model(vae)
        except Exception:
            patcher = getattr(vae, "patcher", None)
            detach = getattr(patcher, "detach", None) if patcher is not None else getattr(vae, "detach", None)
            if callable(detach):
                detach()
        gc.collect()
        try:
            resources.soft_empty_cache(force=True)
        except Exception:
            pass
