from __future__ import annotations

import gc
import sys
import time
from typing import Any

import numpy as np
import torch
from backend import resources
from backend.flux_fill_v2.conditioning_loader import load_flux_empty_conditioning_cache
from backend.flux_fill_v2.contracts import (
    FluxFillRequest,
    FluxFillResult,
    FluxRuntimeIdentity,
    UNetSpineKind,
    VAEPostureKind,
)
from backend.flux_fill_v2.streaming_spine import FluxStreamingUNetSpine
from backend.flux_fill_v2.resident_spine import FluxResidentUNetSpine
from backend.flux_fill_v2.runtime_state import (
    acquire_resident_spine,
    release_active_flux_resident_spine,
)
from backend.flux_fill_v2.vae_loader import load_flux_ae

class FluxDispatcher:
    """Minimal greenfield dispatcher shell that coordinates posture assembly."""

    def __init__(self) -> None:
        pass

    def execute(self, request: FluxFillRequest, callback: Any | None = None) -> FluxFillResult:
        request.validate_dispatch_ready(require_existing_assets=True)
        device = torch.device(request.device) if request.device else resources.get_torch_device()
        resident_spine_reused = False
        retain_resident_spine = (request.unet_spine == UNetSpineKind.RESIDENT)

        # Select spine based on requested unet_spine kind
        if retain_resident_spine:
            spine: Any
            runtime_identity = FluxRuntimeIdentity(
                unet_spine=UNetSpineKind.RESIDENT,
                t5_posture=None,
                vae_posture=VAEPostureKind.TRANSIENT,
            )
        else:
            spine = FluxStreamingUNetSpine(request)
            runtime_identity = FluxRuntimeIdentity(
                unet_spine=UNetSpineKind.STREAMING,
                t5_posture=None,
                vae_posture=VAEPostureKind.TRANSIENT,
            )

        timings: dict[str, float] = {}

        # 1. Load VAE to encode source and concat images (transient VAE posture)
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(request.ae_path, load_device=device, offload_device="cpu")
        timings["vae_load_encode"] = time.perf_counter() - vae_load_start

        try:
            # Prepare inputs
            encode_start = time.perf_counter()
            source_latent, concat_latent, denoise_mask = self._prepare_latents(
                vae, request.image, request.mask, device
            )
            timings["vae_encode"] = time.perf_counter() - encode_start
        finally:
            # Eject transient VAE immediately after encoding to free memory
            try:
                resources.eject_model(vae)
            except Exception:
                detach = getattr(vae.patcher, "detach", None)
                if callable(detach):
                    detach()
            vae = None
            gc.collect()
            resources.soft_empty_cache()

        # 2. Start UNet spine and run denoise
        unet_start = time.perf_counter()
        if retain_resident_spine:
            spine, resident_spine_reused = acquire_resident_spine(request)
        else:
            spine.start()
        timings["unet_start"] = time.perf_counter() - unet_start

        try:
            empty_cond = load_flux_empty_conditioning_cache(request.conditioning_cache_path)

            denoise_start = time.perf_counter()
            samples, sigmas = spine.denoise(
                source_latent, concat_latent, denoise_mask, empty_cond, callback=callback
            )
            timings["unet_denoise"] = time.perf_counter() - denoise_start
        except Exception:
            if retain_resident_spine:
                release_active_flux_resident_spine(reason="dispatcher_denoise_failed")
            raise
        finally:
            if not retain_resident_spine:
                spine.end()

        # 3. Reload transient VAE to decode the denoised samples
        vae_load_start = time.perf_counter()
        vae = load_flux_ae(request.ae_path, load_device=device, offload_device="cpu")
        timings["vae_load_decode"] = time.perf_counter() - vae_load_start

        try:
            decode_start = time.perf_counter()
            output_image = self._decode_latents(vae, samples)
            timings["vae_decode"] = time.perf_counter() - decode_start
        finally:
            try:
                resources.eject_model(vae)
            except Exception:
                detach = getattr(vae.patcher, "detach", None)
                if callable(detach):
                    detach()
            vae = None
            gc.collect()
            resources.soft_empty_cache()

        # Stitch result back if requested
        if request.blend_mode == "morphological" and request.image is not None and request.mask is not None:
            stitch_start = time.perf_counter()
            output_image = self._stitch_image(request.image, request.mask, output_image)
            timings["stitch"] = time.perf_counter() - stitch_start

        return FluxFillResult(
            output_image=output_image,
            seed=request.seed,
            width=output_image.shape[1],
            height=output_image.shape[0],
            runtime_identity=runtime_identity,
            timings=timings,
            metadata={
                "runtime_identity": runtime_identity.as_dict(),
                "conditioning_contract": "empty_conditioning_only",
                "resident_spine_reused": resident_spine_reused,
                "resident_spine_retained": retain_resident_spine,
            },
        )

    def _prepare_latents(
        self, vae: Any, image: np.ndarray, mask: np.ndarray, device: torch.device
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

    def _decode_latents(self, vae: Any, latent: torch.Tensor) -> np.ndarray:
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

    def _stitch_image(self, original_image: np.ndarray, mask: np.ndarray, generated_image: np.ndarray) -> np.ndarray:
        import cv2
        canvas = original_image.copy().astype(np.float32)
        generated = generated_image.astype(np.float32)

        raw_mask = mask
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        morph_mask = cv2.dilate(raw_mask, kernel, iterations=2)
        blur_mask = cv2.GaussianBlur(morph_mask, (63, 63), 0)
        alpha = (blur_mask / 255.0)[..., np.newaxis]

        merged = (generated * alpha) + (canvas * (1.0 - alpha))
        return np.clip(merged, 0, 255).astype(np.uint8)
