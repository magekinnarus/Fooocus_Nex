from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from backend import cond_utils, conditioning, decode as backend_decode, loader, precision, resources, sampling
from ldm_patched.modules import latent_formats


@dataclass
class DirectSDXLGGUFRunConfig:
    unet_path: str
    clip_l_path: str
    clip_g_path: str
    vae_path: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    clip_layer: int = -2
    denoise: float = 1.0
    batch_size: int = 1
    quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class DirectSDXLGGUFPreparedInputs:
    encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]]
    adm_pair: Dict[str, torch.Tensor]
    positive: Any
    negative: Any
    noise: torch.Tensor
    latent: torch.Tensor


@dataclass
class DirectSDXLGGUFDenoiseResult:
    samples: torch.Tensor
    cond_prepare_duration: float
    sampler_model_attach: float
    denoise_wall: float
    denoise_cpu_proc: float
    gguf_trace_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectSDXLGGUFRunResult:
    images: torch.Tensor
    latents: torch.Tensor
    benchmark: Dict[str, Any]


class DirectSDXLGGUFRuntime:
    """
    Narrow SDXL GGUF runtime state for the direct txt2img path.

    The runtime owns the W06 contract directly: component load, prompt encode,
    ADM build, latent/noise setup, direct UNet attach, explicit condition prep,
    denoise, direct VAE attach/decode, and benchmark-friendly return values.
    """

    route_label = "direct_sdxl_gguf"

    def __init__(
        self,
        config: DirectSDXLGGUFRunConfig,
        *,
        device: Optional[torch.device] = None,
        unet_budget_mb: Optional[int] = None,
    ) -> None:
        self.config = config
        self.device = device or resources.get_torch_device()
        self.unet_budget_mb = unet_budget_mb

        self.unet = None
        self.clip = None
        self.vae = None
        self._loaded = False
        self._cold_model_load_cpu = 0.0

    def load_components(self) -> float:
        if self._loaded:
            return 0.0

        start = time.perf_counter()
        self.unet = loader.load_sdxl_unet(self.config.unet_path, dtype=torch.float16)
        self.clip = loader.load_sdxl_clip(
            self.config.clip_l_path,
            self.config.clip_g_path,
            dtype=torch.float16,
        )
        self.clip.clip_layer(self.config.clip_layer)
        self.vae = loader.load_vae(
            self.config.vae_path,
            dtype=torch.float32,
            latent_format=latent_formats.SDXL(),
        )

        if self.config.quality:
            loader.patch_unet_for_quality(self.unet, self.config.quality)

        self._cold_model_load_cpu = time.perf_counter() - start
        self._loaded = True
        return self._cold_model_load_cpu

    def _clean_unet_budget_bytes(self) -> int:
        if self.unet_budget_mb is not None:
            return self.unet_budget_mb * 1024 * 1024
        return int(resources.maximum_vram_for_weights(self.device))

    def attach_clip_direct(self) -> None:
        self.load_components()
        self.clip.patcher.patch_model(device_to=self.device, lowvram_model_memory=0)

    def detach_clip_direct(self) -> None:
        if self.clip is not None:
            self.clip.patcher.detach()

    def attach_unet_direct(self) -> None:
        self.load_components()
        budget = self._clean_unet_budget_bytes()
        model_size = int(self.unet.model_size())
        lowvram_model_memory = 0 if budget >= model_size else budget
        self.unet.patch_model(device_to=self.device, lowvram_model_memory=lowvram_model_memory)

    def detach_unet_direct(self) -> None:
        if self.unet is not None:
            self.unet.detach()

    def attach_vae_direct(self) -> None:
        self.load_components()
        self.vae.patcher.patch_model(device_to=self.device, lowvram_model_memory=0)
        self.vae.first_stage_model.to(device=self.device, dtype=torch.float32)

    def detach_vae_direct(self) -> None:
        if self.vae is not None:
            self.vae.patcher.detach()

    def encode_prompt_pair_direct(self) -> tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
        attach_start = time.perf_counter()
        self.attach_clip_direct()
        clip_residency_attach = time.perf_counter() - attach_start

        encode_start = time.perf_counter()
        try:
            encoded_pair = conditioning.encode_prompt_pair_sdxl(
                self.clip,
                self.config.prompt,
                self.config.negative_prompt,
                use_explicit_residency=True,
            )
        finally:
            offload_start = time.perf_counter()
            self.detach_clip_direct()
            clip_residency_offload = time.perf_counter() - offload_start
        clip_encode = time.perf_counter() - encode_start

        return encoded_pair, {
            "clip_residency_attach": clip_residency_attach,
            "clip_residency_offload": clip_residency_offload,
            "clip_encode": clip_encode,
        }

    def build_adm_pair(
        self,
        encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        return conditioning.build_sdxl_adm_pair(
            encoded_prompt_pair,
            self.config.width,
            self.config.height,
            target_width=self.config.width,
            target_height=self.config.height,
            adm_scale_positive=self.config.quality.get("adm_scale_positive", 1.0),
            adm_scale_negative=self.config.quality.get("adm_scale_negative", 1.0),
        )

    def create_latent_and_noise(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.load_components()
        latent_h = self.config.height // 8
        latent_w = self.config.width // 8
        dtype = self.unet.model.get_dtype()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed)

        noise = torch.randn(
            (self.config.batch_size, 4, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=dtype,
        )
        latent = torch.zeros(
            (self.config.batch_size, 4, latent_h, latent_w),
            device=self.device,
            dtype=dtype,
        )
        return latent, noise

    def prepare_inputs(self) -> tuple[DirectSDXLGGUFPreparedInputs, Dict[str, float]]:
        encoded_prompt_pair, encode_metrics = self.encode_prompt_pair_direct()

        adm_start = time.perf_counter()
        adm_pair = self.build_adm_pair(encoded_prompt_pair)
        adm_build = time.perf_counter() - adm_start

        latent_noise_start = time.perf_counter()
        latent, noise = self.create_latent_and_noise()
        latent_noise_prep = time.perf_counter() - latent_noise_start

        positive = [[
            encoded_prompt_pair["positive"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["positive"]["pooled"],
                "model_conds": {"y": adm_pair["positive"]},
            },
        ]]
        negative = [[
            encoded_prompt_pair["negative"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["negative"]["pooled"],
                "model_conds": {"y": adm_pair["negative"]},
            },
        ]]

        return (
            DirectSDXLGGUFPreparedInputs(
                encoded_prompt_pair=encoded_prompt_pair,
                adm_pair=adm_pair,
                positive=positive,
                negative=negative,
                noise=noise,
                latent=latent,
            ),
            {
                **encode_metrics,
                "adm_build": adm_build,
                "latent_noise_prep": latent_noise_prep,
            },
        )

    def _prepare_cfg_guider(self, prepared_inputs: DirectSDXLGGUFPreparedInputs) -> sampling.CFGGuider:
        self.load_components()
        cfg_guider = sampling.CFGGuider(self.unet)
        cfg_guider.set_conds(prepared_inputs.positive, prepared_inputs.negative)
        cfg_guider.set_cfg(self.config.cfg, cfg_pp="_cfg_pp" in self.config.sampler)
        cfg_guider.set_quality(self.config.quality)
        cfg_guider.ensure_inner_model(self.unet.model)
        cfg_guider.clone_original_conds()

        cond_start = time.perf_counter()
        cfg_guider.conds = cond_utils.process_conds(
            self.unet.model,
            prepared_inputs.noise,
            cfg_guider.conds,
            self.device,
            latent_image=prepared_inputs.latent,
            denoise_mask=None,
            seed=self.config.seed,
        )
        cfg_guider.cond_prep_duration = time.perf_counter() - cond_start
        cfg_guider.prepared = True
        return cfg_guider

    def denoise_prepared_inputs(
        self,
        prepared_inputs: DirectSDXLGGUFPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> DirectSDXLGGUFDenoiseResult:
        self.load_components()

        sampler_instance = sampling.KSampler(
            self.unet,
            self.config.steps,
            self.device,
            self.config.sampler,
            self.config.scheduler,
            self.config.denoise,
            model_options={"quality": self.config.quality},
        )
        guider = self._prepare_cfg_guider(prepared_inputs)
        sampler_kernel = sampling.ksampler(self.config.sampler)

        attach_start = time.perf_counter()
        self.attach_unet_direct()
        sampler_model_attach = time.perf_counter() - attach_start

        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        try:
            with torch.inference_mode(), precision.autocast_context(self.device):
                samples = sampling.sample_prepared_sdxl(
                    guider,
                    prepared_inputs.noise,
                    sampler_instance.sigmas,
                    sampler=sampler_kernel,
                    latent_image=prepared_inputs.latent,
                    denoise_mask=None,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=self.config.seed,
                    attach_model=False,
                )
        finally:
            self.detach_unet_direct()

        denoise_wall = time.perf_counter() - denoise_start
        denoise_cpu_proc = time.process_time() - denoise_cpu_start

        return DirectSDXLGGUFDenoiseResult(
            samples=samples,
            cond_prepare_duration=guider.cond_prep_duration,
            sampler_model_attach=sampler_model_attach,
            denoise_wall=denoise_wall,
            denoise_cpu_proc=denoise_cpu_proc,
            gguf_trace_stats=dict(getattr(guider, "last_gguf_trace_stats", {}) or {}),
        )

    def decode_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        attach_start = time.perf_counter()
        self.attach_vae_direct()
        vae_attach = time.perf_counter() - attach_start

        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                images = backend_decode.decode_preloaded_vae(self.vae, latent)
        finally:
            self.detach_vae_direct()
        vae_decode = time.perf_counter() - decode_start
        return images, vae_attach, vae_decode

    def run(self) -> DirectSDXLGGUFRunResult:
        total_start = time.perf_counter()
        cold_model_load_cpu = self.load_components()

        prepared_inputs, prep_metrics = self.prepare_inputs()
        denoise_result = self.denoise_prepared_inputs(prepared_inputs)
        images, vae_attach, vae_decode = self.decode_latent(denoise_result.samples)
        total_wall = time.perf_counter() - total_start

        return DirectSDXLGGUFRunResult(
            images=images,
            latents=denoise_result.samples,
            benchmark={
                "route_label": self.route_label,
                "cold_model_load_cpu": cold_model_load_cpu,
                "clip_residency_attach": prep_metrics["clip_residency_attach"],
                "clip_residency_offload": prep_metrics["clip_residency_offload"],
                "clip_encode": prep_metrics["clip_encode"],
                "adm_build": prep_metrics["adm_build"],
                "latent_noise_prep": prep_metrics["latent_noise_prep"],
                "sampler_model_attach": denoise_result.sampler_model_attach,
                "cond_prepare_explicit": denoise_result.cond_prepare_duration,
                "denoise_wall": denoise_result.denoise_wall,
                "denoise_s_per_it": denoise_result.denoise_wall / max(1, self.config.steps),
                "denoise_cpu_proc": denoise_result.denoise_cpu_proc,
                "gguf_dequant": float(denoise_result.gguf_trace_stats.get("dequant_seconds", 0.0)),
                "gguf_dequant_cpu_proc": float(denoise_result.gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0)),
                "vae_attach": vae_attach,
                "vae_decode": vae_decode,
                "total_wall": total_wall,
            },
        )

    def close(self) -> None:
        self.detach_unet_direct()
        self.detach_vae_direct()
        self.detach_clip_direct()
        resources.soft_empty_cache(force=True)
