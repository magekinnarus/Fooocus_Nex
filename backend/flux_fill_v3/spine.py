from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any
import logging

import torch
from backend import resources
from backend.flux_fill_v3.contracts import FluxFillPreviewContext, FluxFillRequest, FluxLatentArtifactBundle
from backend.flux_fill_v2.streaming_loader import (
    _sample_flux_fill_direct_streaming,
    load_flux_fill_unet_streaming,
)

logger = logging.getLogger(__name__)


def _shape_of_tensor(value: torch.Tensor | None) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    return None


def _mask_fill_ratio(mask: torch.Tensor | None) -> float | None:
    if not isinstance(mask, torch.Tensor):
        return None
    return float((mask.detach().float() > 0.5).float().mean().item())


@dataclass(frozen=True)
class FluxFillConditioningPayloads:
    positive: list[list[Any]]
    negative: list[list[Any]]
    latent_image: torch.Tensor
    denoise_mask: torch.Tensor
    guidance: float
    batch_size: int


def build_flux_fill_conditioning_payloads(
    empty_conditioning: Any,
    source_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    *,
    concat_latent: torch.Tensor | None = None,
    guidance: float = 15.0,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FluxFillConditioningPayloads:
    if guidance <= 0:
        raise ValueError(f"Guidance must be > 0, got {guidance}.")
    if not isinstance(source_latent, torch.Tensor) or source_latent.ndim != 4 or int(source_latent.shape[1]) != 16:
        raise ValueError("source_latent must have shape [B, 16, H, W].")
    if not isinstance(denoise_mask, torch.Tensor) or denoise_mask.ndim != 4 or int(denoise_mask.shape[1]) != 1:
        raise ValueError("denoise_mask must have shape [B, 1, H, W].")
    if source_latent.shape[0] != denoise_mask.shape[0] or source_latent.shape[-2:] != denoise_mask.shape[-2:]:
        raise ValueError(
            f"denoise_mask shape {list(denoise_mask.shape)} does not match source_latent {list(source_latent.shape)}."
        )

    cond_image = concat_latent if concat_latent is not None else source_latent

    batch = int(batch_size or source_latent.shape[0])
    cross_attn, pooled_output = empty_conditioning.repeat(batch, device=device, dtype=dtype)
    source = source_latent.detach().to(device=device or source_latent.device, dtype=dtype or source_latent.dtype)
    cond_img = cond_image.detach().to(device=device or cond_image.device, dtype=dtype or cond_image.dtype)
    mask = denoise_mask.detach().to(device=device or denoise_mask.device, dtype=dtype or denoise_mask.dtype)
    if int(source.shape[0]) != batch:
        if int(source.shape[0]) != 1:
            raise ValueError(f"Cannot repeat source_latent batch {source.shape[0]} to {batch}.")
        source = source.repeat(batch, 1, 1, 1)
    if int(cond_img.shape[0]) != batch:
        if int(cond_img.shape[0]) != 1:
            raise ValueError(f"Cannot repeat concat_latent batch {cond_img.shape[0]} to {batch}.")
        cond_img = cond_img.repeat(batch, 1, 1, 1)
    if int(mask.shape[0]) != batch:
        if int(mask.shape[0]) != 1:
            raise ValueError(f"Cannot repeat denoise_mask batch {mask.shape[0]} to {batch}.")
        mask = mask.repeat(batch, 1, 1, 1)

    payload = {
        "pooled_output": pooled_output,
        "guidance": float(guidance),
        "concat_latent_image": cond_img,
        "denoise_mask": mask,
        "concat_mask": mask,
    }
    positive = [[cross_attn, payload.copy()]]
    negative_payload = payload.copy()
    negative_payload["pooled_output"] = torch.zeros_like(pooled_output)
    negative = [[torch.zeros_like(cross_attn), negative_payload]]
    return FluxFillConditioningPayloads(
        positive=positive,
        negative=negative,
        latent_image=source,
        denoise_mask=mask,
        guidance=float(guidance),
        batch_size=batch,
    )


class StreamingUnetSpine:
    """Greenfields Streaming UNet Spine.

    This class is the authoritative state owner for the streaming inference lane.
    It owns the model lifecycle and prefetch options.
    """
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request
        self.device = torch.device(request.device) if request.device else resources.get_torch_device()
        self.unet_patcher: Any | None = None
        self.started: bool = False

    def start(self) -> None:
        self.request.validate_static(require_existing_assets=True)
        if self.unet_patcher is None:
            logger.debug(
                "[Flux Telemetry] Loading pinned CPU Flux UNet spine path=%s prefetch_depth=%s prefetch_chunk_mb=%s",
                self.request.unet_path,
                getattr(self.request, "prefetch_depth", None),
                getattr(self.request, "prefetch_chunk_mb", None),
            )
            self.unet_patcher = load_flux_fill_unet_streaming(
                self.request.unet_path,
                load_device="cpu",  # staged on CPU host memory for streaming
                offload_device="cpu",
                execution_class="standard_streaming",
                prefetch_depth=self.request.prefetch_depth,
                prefetch_chunk_mb=self.request.prefetch_chunk_mb,
            )
        else:
            logger.debug(
                "[Flux Telemetry] Reusing already-loaded pinned CPU Flux UNet spine path=%s",
                self.request.unet_path,
            )
        self.started = True

    def end(self) -> None:
        if self.unet_patcher is not None:
            logger.debug(
                "[Flux Telemetry] Releasing pinned CPU Flux UNet spine path=%s",
                self.request.unet_path,
            )
            try:
                resources.eject_model(self.unet_patcher)
            except Exception:
                detach = getattr(self.unet_patcher, "detach", None)
                if callable(detach):
                    detach()
            try:
                if getattr(self.unet_patcher, "can_runtime_release", lambda: False)():
                    self.unet_patcher.release_weights_to_meta()
            except Exception:
                pass
            finally:
                self.unet_patcher = None
        self.started = False
        gc.collect()
        try:
            resources.soft_empty_cache(force=True)
        except Exception:
            pass

    def _create_seeded_noise(self, source: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.request.seed))
        noise = torch.randn(
            tuple(source.shape),
            generator=generator,
            dtype=source.dtype,
            device="cpu",
        )
        return noise.to(device=self.device, dtype=source.dtype)

    def get_preview_context(self) -> FluxFillPreviewContext:
        patcher_model = getattr(self.unet_patcher, "model", None)
        latent_format = getattr(patcher_model, "latent_format", None)
        if latent_format is None:
            latent_format = getattr(getattr(patcher_model, "model", None), "latent_format", None)

        if latent_format is None:
            from ldm_patched.modules import latent_formats
            latent_format = latent_formats.Flux()
        return FluxFillPreviewContext(latent_format, self.device)

    def denoise(
        self,
        source_or_bundle: torch.Tensor | FluxLatentArtifactBundle,
        concat_latent: torch.Tensor | None = None,
        denoise_mask: torch.Tensor | None = None,
        empty_conditioning: Any | None = None,
        callback: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.started:
            self.start()

        if isinstance(source_or_bundle, FluxLatentArtifactBundle):
            source = source_or_bundle.source_latent
            concat = source_or_bundle.concat_latent
            mask = source_or_bundle.denoise_mask
            if empty_conditioning is None:
                empty_conditioning = concat_latent
        else:
            source = source_or_bundle
            concat = concat_latent
            mask = denoise_mask

        device = self.device
        source_device = source.to(device=device, dtype=torch.float32)
        concat_device = concat.to(device=device, dtype=torch.float32)
        mask_device = mask.to(device=device, dtype=torch.float32)
        noise = self._create_seeded_noise(source_device)

        logger.debug(
            "[Flux Telemetry] Spine denoise payload category=%s source_latent=%s concat_latent=%s "
            "denoise_mask=%s latent_mask_fill=%.4f device=%s seed=%s steps=%s guidance=%.2f "
            "sampler=%s scheduler=%s",
            self.request.category,
            _shape_of_tensor(source_device),
            _shape_of_tensor(concat_device),
            _shape_of_tensor(mask_device),
            _mask_fill_ratio(mask_device) or 0.0,
            device,
            self.request.seed,
            self.request.steps,
            self.request.guidance,
            self.request.sampler,
            self.request.scheduler,
        )

        # Attach/reset prefetch scheduler options if available
        flux_options = getattr(self.unet_patcher, "model_options", {}).get("flux_fill", {})
        streaming_scheduler = flux_options.get("streaming_scheduler")
        if streaming_scheduler is not None and hasattr(streaming_scheduler, "reset_run"):
            streaming_scheduler.reset_run()

        payloads = build_flux_fill_conditioning_payloads(
            empty_conditioning,
            source_device,
            mask_device,
            concat_latent=concat_device,
            guidance=self.request.guidance,
            batch_size=int(source_device.shape[0]),
            device=device,
            dtype=source_device.dtype,
        )

        samples, sigmas = _sample_flux_fill_direct_streaming(
            unet_patcher=self.unet_patcher,
            noise=noise,
            positive=payloads.positive,
            negative=payloads.negative,
            latent_image=payloads.latent_image,
            denoise_mask=payloads.denoise_mask,
            steps=self.request.steps,
            device=device,
            sampler_name=self.request.sampler,
            scheduler_name=self.request.scheduler,
            denoise=1.0,
            cfg=1.0,
            seed=self.request.seed,
            callback=callback,
            disable_pbar=True,
        )
        return samples, sigmas
