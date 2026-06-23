from __future__ import annotations

import gc
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend import resources
from backend.flux_fill_v2.vae_loader import load_flux_ae
from backend.flux_fill_v2.contracts import FluxFillRequest, FluxLatentArtifactBundle


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
        f"mode:{request.mode}|"
        f"blend:{request.blend_mode}|"
        f"mp:{request.target_megapixels}"
    )
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()


def _encode_vae_latents(
    vae: Any, image: np.ndarray, mask: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from modules.core import numpy_to_pytorch, encode_vae
    from backend import resources

    resources.load_models_gpu([vae.patcher])
    vae_device = getattr(vae.patcher, "current_loaded_device", lambda: vae.patcher.load_device)()
    move_model = getattr(vae.first_stage_model, "to", None)
    if callable(move_model):
        move_model(device=vae_device, dtype=torch.float32)

    # 1. Encode unmasked source image
    orig_pixels = numpy_to_pytorch(image)
    source_latent = encode_vae(vae=vae, pixels=orig_pixels)["samples"]

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

    # Encode gray-masked to get concat_latent (bypass encode.py double-normalization)
    pixels_for_vae = (numpy_to_pytorch(bb_image_for_concat).movedim(-1, 1) * 2.0) - 1.0
    if pixels_for_vae.ndim == 3:
        pixels_for_vae = pixels_for_vae.unsqueeze(0)

    vae_param = next(vae.first_stage_model.parameters(), None)
    vae_input_device = vae.patcher.load_device
    vae_input_dtype = torch.float32
    if isinstance(vae_param, torch.Tensor):
        vae_input_device = vae_param.device
        vae_input_dtype = vae_param.dtype

    pixels_for_vae = pixels_for_vae.to(device=vae_input_device, dtype=vae_input_dtype)
    raw_latent = vae.first_stage_model.encode(pixels_for_vae)
    if hasattr(raw_latent, "sample"):
        raw_latent = raw_latent.sample()
    concat_latent = raw_latent.cpu()

    vae.patcher.detach()
    resources.soft_empty_cache()

    # 3. Build denoise_mask
    mask_t = torch.from_numpy(mask).float() / 255.0
    if mask_t.ndim == 3:
        mask_t = mask_t[:, :, 0]
    mask_t = mask_t[None, None, :, :]
    denoise_mask = torch.nn.functional.max_pool2d(mask_t, kernel_size=8)
    denoise_mask = (denoise_mask > 0.5).float()

    return source_latent, concat_latent, denoise_mask


def _decode_vae_latents(vae: Any, latent: torch.Tensor) -> np.ndarray:
    from modules import core
    from backend import resources
    import gc

    gc.collect()
    resources.soft_empty_cache()

    patcher = getattr(vae, "patcher", None)
    if patcher is not None:
        patch_model = getattr(patcher, "patch_model", None)
        if callable(patch_model):
            patch_model(device_to=vae.patcher.load_device)

    decoded = core.decode_vae(vae, {"samples": latent.detach().cpu()}, tiled=False)

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


class FluxTransientVAEPosture:
    """ Authoritative Greenfield Transient VAE Posture Contract.
    
    Loads VAE dynamically, performs operation, and immediately ejects.
    """
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request

    def compute_fingerprint(self) -> str:
        return compute_artifact_fingerprint(self.request)

    def prepare_artifacts(self, device: torch.device) -> FluxLatentArtifactBundle:
        """Loads VAE transiently, runs encode steps, builds artifacts, and ejects VAE."""
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(self.request.ae_path, load_device=device, offload_device="cpu")
        vae_load_time = time.perf_counter() - vae_load_start

        try:
            encode_start = time.perf_counter()
            source_latent, concat_latent, denoise_mask = _encode_vae_latents(
                vae, self.request.image, self.request.mask, device
            )
            vae_encode_time = time.perf_counter() - encode_start
            
            fingerprint = self.compute_fingerprint()
            return FluxLatentArtifactBundle(
                source_latent=source_latent,
                concat_latent=concat_latent,
                denoise_mask=denoise_mask,
                fingerprint=fingerprint,
                vae_load_time=vae_load_time,
                vae_encode_time=vae_encode_time,
            )
        finally:
            self._eject_vae(vae)

    def decode(self, samples: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float, float]:
        """Loads VAE transiently, decodes samples, and ejects VAE."""
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(self.request.ae_path, load_device=device, offload_device="cpu")
        vae_load_time = time.perf_counter() - vae_load_start

        try:
            decode_start = time.perf_counter()
            output_image = _decode_vae_latents(vae, samples)
            vae_decode_time = time.perf_counter() - decode_start
            return output_image, vae_load_time, vae_decode_time
        finally:
            self._eject_vae(vae)

    def _eject_vae(self, vae: Any) -> None:
        if vae is None:
            return
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
        resources.soft_empty_cache()
