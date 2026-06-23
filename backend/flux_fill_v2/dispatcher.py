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
    T5PostureKind,
)
from backend.flux_fill_v2.streaming_spine import FluxStreamingUNetSpine
from backend.flux_fill_v2.resident_spine import FluxResidentUNetSpine
from backend.flux_fill_v2.runtime_state import (
    acquire_resident_spine,
    release_active_flux_resident_spine,
    get_cached_latent_artifact_bundle,
    set_cached_latent_artifact_bundle,
)
from backend.flux_fill_v2.vae_loader import load_flux_ae
from backend.flux_fill_v2.t5_posture import acquire_t5_posture
from backend.flux_fill_v2.vae_posture import FluxTransientVAEPosture

class FluxDispatcher:
    """Minimal greenfield dispatcher shell that coordinates posture assembly."""

    def __init__(self) -> None:
        pass

    def execute(self, request: FluxFillRequest, callback: Any | None = None) -> FluxFillResult:
        request.validate_dispatch_ready(require_existing_assets=True)
        device = torch.device(request.device) if request.device else resources.get_torch_device()
        resident_spine_reused = False
        retain_resident_spine = (request.unet_spine == UNetSpineKind.RESIDENT)

        # Resolve T5 posture kind
        if request.t5_posture is not None:
            t5_posture_kind = request.t5_posture
        else:
            from backend.flux_fill_v2.activation import (
                resolve_flux_fill_t5_posture,
                resolve_flux_fill_total_ram_gb,
            )

            t5_posture_kind = resolve_flux_fill_t5_posture(
                request.unet_spine,
                resolve_flux_fill_total_ram_gb(request),
                low_ram_override=bool(request.flux_fill_t5_low_ram),
            )

        # Select spine based on requested unet_spine kind
        if retain_resident_spine:
            spine: Any
            runtime_identity = FluxRuntimeIdentity(
                unet_spine=UNetSpineKind.RESIDENT,
                t5_posture=t5_posture_kind,
                vae_posture=VAEPostureKind.TRANSIENT,
            )
        else:
            spine = FluxStreamingUNetSpine(request)
            runtime_identity = FluxRuntimeIdentity(
                unet_spine=UNetSpineKind.STREAMING,
                t5_posture=t5_posture_kind,
                vae_posture=VAEPostureKind.TRANSIENT,
            )

        timings: dict[str, float] = {}

        # 1. Prepare transient VAE posture and resolve cache
        vae_posture = FluxTransientVAEPosture(request)
        fingerprint = vae_posture.compute_fingerprint()

        bundle = get_cached_latent_artifact_bundle(fingerprint)
        if bundle is not None:
            timings["vae_load_encode"] = 0.0
            timings["vae_encode"] = 0.0
        else:
            bundle = vae_posture.prepare_artifacts(device)
            timings["vae_load_encode"] = bundle.vae_load_time
            timings["vae_encode"] = bundle.vae_encode_time
            set_cached_latent_artifact_bundle(bundle)

        # 2. Start UNet spine and run denoise
        unet_start = time.perf_counter()
        if retain_resident_spine:
            spine, resident_spine_reused = acquire_resident_spine(request)
        else:
            spine.start()
        timings["unet_start"] = time.perf_counter() - unet_start

        try:
            t5_posture = acquire_t5_posture(t5_posture_kind, request)
            empty_cond = t5_posture.get_conditioning(request)

            denoise_start = time.perf_counter()
            samples, sigmas = spine.denoise(
                bundle, empty_cond, callback=callback
            )
            timings["unet_denoise"] = time.perf_counter() - denoise_start
        except Exception:
            if retain_resident_spine:
                release_active_flux_resident_spine(reason="dispatcher_denoise_failed")
            if t5_posture_kind == T5PostureKind.CPU_FP16_RESIDENT:
                from backend.flux_fill_v2.runtime_state import release_active_flux_resident_t5
                release_active_flux_resident_t5()
            raise
        finally:
            if not retain_resident_spine:
                spine.end()

        # 3. Reload transient VAE to decode the denoised samples
        output_image, vae_load_decode, vae_decode = vae_posture.decode(samples, device)
        timings["vae_load_decode"] = vae_load_decode
        timings["vae_decode"] = vae_decode

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
                "conditioning_contract": "prompt_conditioning" if request.prompt else "empty_conditioning_only",
                "resident_spine_reused": resident_spine_reused,
                "resident_spine_retained": retain_resident_spine,
            },
        )

    def _prepare_latents(
        self, vae: Any, image: np.ndarray, mask: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from backend.flux_fill_v2.vae_posture import _encode_vae_latents
        return _encode_vae_latents(vae, image, mask, device)

    def _decode_latents(self, vae: Any, latent: torch.Tensor) -> np.ndarray:
        from backend.flux_fill_v2.vae_posture import _decode_vae_latents
        return _decode_vae_latents(vae, latent)

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
