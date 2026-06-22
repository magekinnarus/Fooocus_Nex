from __future__ import annotations

import gc
from typing import Any

import torch
from backend import resources
from backend.flux_fill_v2.contracts import FluxFillPreviewContext, FluxFillRequest
from backend.flux_fill_v2.loader import load_flux_fill_unet
from backend.flux_fill_v2.streaming_spine import build_flux_fill_conditioning_payloads


def _sample_flux_fill_direct_resident(
    *,
    unet_patcher: Any,
    noise: torch.Tensor,
    positive: Any,
    negative: Any,
    latent_image: torch.Tensor,
    denoise_mask: torch.Tensor,
    steps: int,
    device: torch.device,
    sampler_name: str,
    scheduler_name: str,
    denoise: float,
    cfg: float,
    seed: int,
    callback: Any | None,
    disable_pbar: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from backend import sampling

    model_options = getattr(unet_patcher, "model_options", {})
    sampler_instance = sampling.KSampler(
        unet_patcher,
        steps,
        device,
        sampler_name,
        scheduler_name,
        denoise,
        model_options=model_options,
    )
    guider = sampling.prepare_sampler_conds(
        unet_patcher,
        noise,
        positive,
        negative,
        cfg,
        sampler_name=sampler_name,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        seed=seed,
        model_options=model_options,
        quality=getattr(sampler_instance, "quality", {}),
        inner_model=getattr(unet_patcher, "model", None),
    )
    samples = sampling.sample_prepared_sdxl(
        guider,
        noise,
        sampler_instance.sigmas,
        sampler=sampling.ksampler(sampler_name),
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        attach_model=False,
    )
    return samples, sampler_instance.sigmas.detach().cpu()


class FluxResidentUNetSpine:
    """ Greenfield Resident UNet Spine.
    
    This class is the authoritative state owner for the resident inference lane.
    It owns the model lifecycle and guarantees model residency in GPU VRAM
    without any CPU shadow copies.
    """
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request
        self.device = torch.device(request.device) if request.device else resources.get_torch_device()
        self.unet_patcher: Any | None = None
        self.started: bool = False

    def start(self) -> None:
        self.request.validate_static(require_existing_assets=True)
        if self.unet_patcher is None:
            self.unet_patcher = load_flux_fill_unet(
                self.request.unet_path,
                load_device=self.device,
                offload_device=self.device,
                execution_class="standard_resident",
                runtime_posture="resident",
                resident_load_strategy="sticky_no_cpu_shadow",
            )
            # Patch and load weights onto GPU immediately
            resources.load_models_gpu([self.unet_patcher])
        self.started = True

    def end(self) -> None:
        if self.unet_patcher is not None:
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
        
        # Fallback to Flux format if none detected
        if latent_format is None:
            from ldm_patched.modules import latent_formats
            latent_format = latent_formats.Flux()
        return FluxFillPreviewContext(latent_format, self.device)

    def denoise(
        self,
        source: torch.Tensor,
        concat: torch.Tensor,
        mask: torch.Tensor,
        empty_conditioning: Any,
        callback: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.started:
            self.start()

        device = self.device
        source_device = source.to(device=device, dtype=torch.float32)
        concat_device = concat.to(device=device, dtype=torch.float32)
        mask_device = mask.to(device=device, dtype=torch.float32)
        noise = self._create_seeded_noise(source_device)

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

        samples, sigmas = _sample_flux_fill_direct_resident(
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
