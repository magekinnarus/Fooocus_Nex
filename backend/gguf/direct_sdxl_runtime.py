from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from backend import (
    anisotropic,
    cond_utils,
    conditioning,
    k_diffusion,
    loader,
    precision,
    resources,
    sampling,
    utils as backend_utils,
)
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

    def _convert_sampler_cond(self, cond: Any) -> list[Dict[str, Any]]:
        out = []
        if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
            for entry in cond:
                converted = entry.copy()
                converted["uuid"] = converted.get("uuid", uuid.uuid4())
                out.append(converted)
            return out

        for cross_attn, payload in cond:
            converted = payload.copy()
            if cross_attn is not None:
                converted["cross_attn"] = cross_attn
            converted["model_conds"] = converted.get("model_conds", {})
            converted["uuid"] = uuid.uuid4()
            out.append(converted)
        return out

    def _prepare_direct_conds(
        self,
        prepared_inputs: DirectSDXLGGUFPreparedInputs,
    ) -> tuple[Dict[str, Any], float]:
        conds = {
            "positive": self._convert_sampler_cond(prepared_inputs.positive),
            "negative": self._convert_sampler_cond(prepared_inputs.negative),
        }
        cond_start = time.perf_counter()
        processed = cond_utils.process_conds(
            self.unet.model,
            prepared_inputs.noise,
            conds,
            self.device,
            latent_image=prepared_inputs.latent,
            denoise_mask=None,
            seed=self.config.seed,
        )
        return processed, time.perf_counter() - cond_start

    def _calculate_sigmas(self) -> torch.Tensor:
        sampler_instance = sampling.KSampler(
            self.unet,
            self.config.steps,
            self.device,
            self.config.sampler,
            self.config.scheduler,
            self.config.denoise,
            model_options={"quality": self.config.quality},
        )
        return sampler_instance.sigmas

    def _resolve_sampler_function(self) -> Callable[..., torch.Tensor]:
        sampler_name = self.config.sampler
        if sampler_name == "dpm_fast":
            def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_fast(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    len(sigmas) - 1,
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_fast_function

        if sampler_name == "dpm_adaptive":
            def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_adaptive(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_adaptive_function

        func_name = f"sample_{sampler_name.replace('_cfg_pp', '')}"
        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {sampler_name} not implemented in k_diffusion as {func_name}")
        return sampler_function

    def _begin_gguf_trace_capture(self) -> Any:
        try:
            from backend.gguf import ops as gguf_ops
        except Exception:
            return None

        gguf_ops.reset_trace_stats()
        return gguf_ops

    def _consume_gguf_trace_stats(self, gguf_ops: Any) -> Dict[str, Any]:
        if gguf_ops is None:
            return {}
        try:
            return dict(gguf_ops.consume_trace_stats())
        except Exception:
            return {}

    def _calc_fullframe_cond_batch(
        self,
        conds: list[Optional[list[Dict[str, Any]]]],
        x_in: torch.Tensor,
        timestep: torch.Tensor,
    ) -> list[torch.Tensor]:
        out_conds = [torch.zeros_like(x_in) for _ in conds]
        out_counts = [torch.ones_like(x_in) * 1e-37 for _ in conds]
        to_run = []

        for cond_index, cond in enumerate(conds):
            if cond is None:
                continue
            for cond_entry in cond:
                prepared = cond_utils.get_area_and_mult(cond_entry, x_in, timestep)
                if prepared is None:
                    continue
                if prepared.area is not None or prepared.input_x.shape != x_in.shape:
                    raise ValueError("Direct SDXL GGUF denoise only supports full-frame txt2img conditions.")
                to_run.append((prepared, cond_index))

        while len(to_run) > 0:
            first = to_run[0]
            to_batch = []
            for index in range(len(to_run)):
                if cond_utils.can_concat_cond(to_run[index][0], first[0]):
                    to_batch.append(index)

            batch_items = [to_run[index] for index in to_batch]
            for index in sorted(to_batch, reverse=True):
                to_run.pop(index)

            batch_input_x = [prepared.input_x for prepared, _ in batch_items]
            batch_mult = [prepared.mult for prepared, _ in batch_items]
            batch_conditioning = [prepared.conditioning for prepared, _ in batch_items]
            batch_cond_indices = [cond_index for _, cond_index in batch_items]
            input_x = torch.cat(batch_input_x)
            conditioning_batch = cond_utils.cond_cat(batch_conditioning)
            timestep_batch = torch.cat([timestep] * len(batch_cond_indices))
            outputs = self.unet.model.apply_model(input_x, timestep_batch, **conditioning_batch).chunk(len(batch_cond_indices))

            for output, cond_index, mult in zip(outputs, batch_cond_indices, batch_mult):
                out_conds[cond_index] += output * mult
                out_counts[cond_index] += mult

        for index in range(len(out_conds)):
            out_conds[index] /= out_counts[index]
        return out_conds

    def _apply_direct_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
    ) -> torch.Tensor:
        model_sampling = self.unet.model.model_sampling
        t = model_sampling.timestep(timestep).float()
        diffusion_progress = max(0.0, min(1.0, 1.0 - float(t.reshape(-1)[0].item()) / 999.0))

        sharpness = float(self.config.quality.get("sharpness", 0.0))
        if sharpness > 0.0:
            alpha = 0.001 * sharpness * diffusion_progress
            if alpha >= 0.01:
                positive_eps = x - cond_pred
                degraded_eps = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=cond_pred)
                positive_eps_weighted = degraded_eps * alpha + positive_eps * (1.0 - alpha)
                cond_pred = x - positive_eps_weighted

        adaptive_cfg = float(self.config.quality.get("adaptive_cfg", 0.0))
        if adaptive_cfg > 0.0 and self.config.cfg > adaptive_cfg:
            cond_eps = x - cond_pred
            uncond_eps = x - uncond_pred
            real_eps = uncond_eps + self.config.cfg * (cond_eps - uncond_eps)
            mimic_eps = uncond_eps + adaptive_cfg * (cond_eps - uncond_eps)
            final_eps = real_eps * diffusion_progress + mimic_eps * (1.0 - diffusion_progress)
            return x - final_eps

        if "_cfg_pp" in self.config.sampler:
            return cond_pred + (self.config.cfg - 1.0) * (cond_pred - uncond_pred)
        return uncond_pred + (cond_pred - uncond_pred) * self.config.cfg

    def _build_direct_model_callable(
        self,
        processed_conds: Dict[str, Any],
    ) -> Callable[..., torch.Tensor]:
        model_options = getattr(self.unet, "model_options", {}) or {}
        disable_cfg1_optimization = bool(model_options.get("disable_cfg1_optimization", False))

        def model_fn(x: torch.Tensor, sigma: torch.Tensor, **_: Any) -> torch.Tensor:
            negative_conds = processed_conds.get("negative")
            if math.isclose(self.config.cfg, 1.0) and not disable_cfg1_optimization:
                negative_conds = None
            cond_pred, uncond_pred = self._calc_fullframe_cond_batch(
                [processed_conds.get("positive"), negative_conds],
                x,
                sigma,
            )
            return self._apply_direct_cfg(x, sigma, cond_pred, uncond_pred)

        return model_fn

    def _direct_denoise(
        self,
        prepared_inputs: DirectSDXLGGUFPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> DirectSDXLGGUFDenoiseResult:
        self.load_components()
        processed_conds, cond_prepare_duration = self._prepare_direct_conds(prepared_inputs)
        sigmas = self._calculate_sigmas()
        if sigmas.shape[-1] == 0:
            return DirectSDXLGGUFDenoiseResult(
                samples=prepared_inputs.latent,
                cond_prepare_duration=cond_prepare_duration,
                sampler_model_attach=0.0,
                denoise_wall=0.0,
                denoise_cpu_proc=0.0,
                gguf_trace_stats={},
            )

        sampler_function = self._resolve_sampler_function()

        attach_start = time.perf_counter()
        self.attach_unet_direct()
        sampler_model_attach = time.perf_counter() - attach_start

        gguf_ops = self._begin_gguf_trace_capture()
        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        gguf_trace_stats = {}
        try:
            with torch.inference_mode(), precision.autocast_context(self.device):
                model_sampling = self.unet.model.model_sampling
                max_sigma = float(model_sampling.sigma_max)
                sigma = float(sigmas[0])
                max_denoise = math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma
                scaled_noise = model_sampling.noise_scaling(
                    sigmas[0],
                    prepared_inputs.noise,
                    prepared_inputs.latent,
                    max_denoise,
                )

                total_steps = len(sigmas) - 1
                k_callback = None
                if callback is not None:
                    k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps, x.get("denoised", None))

                samples = sampler_function(
                    self._build_direct_model_callable(processed_conds),
                    scaled_noise,
                    sigmas,
                    extra_args={"denoise_mask": None},
                    callback=k_callback,
                    disable=disable_pbar,
                )
                samples = model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        finally:
            denoise_wall = time.perf_counter() - denoise_start
            denoise_cpu_proc = time.process_time() - denoise_cpu_start
            gguf_trace_stats = self._consume_gguf_trace_stats(gguf_ops)
            self.detach_unet_direct()

        return DirectSDXLGGUFDenoiseResult(
            samples=samples,
            cond_prepare_duration=cond_prepare_duration,
            sampler_model_attach=sampler_model_attach,
            denoise_wall=denoise_wall,
            denoise_cpu_proc=denoise_cpu_proc,
            gguf_trace_stats=gguf_trace_stats,
        )

    def denoise_prepared_inputs(
        self,
        prepared_inputs: DirectSDXLGGUFPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> DirectSDXLGGUFDenoiseResult:
        return self._direct_denoise(
            prepared_inputs,
            callback=callback,
            disable_pbar=disable_pbar,
        )

    def _normalize_decoded_output(self, image: torch.Tensor) -> torch.Tensor:
        return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

    def _decode_tiled_local(
        self,
        scaled_latent: torch.Tensor,
        *,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
    ) -> torch.Tensor:
        decode_dtype = next(self.vae.first_stage_model.parameters()).dtype
        decode_fn = lambda a: self.vae.first_stage_model.decode(a.to(device=self.device, dtype=decode_dtype)).float()

        p3 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x, tile_y, overlap, upscale_amount=8, output_device='cpu')
        p1 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=8, output_device='cpu')
        p2 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=8, output_device='cpu')

        return self._normalize_decoded_output((p1 + p2 + p3) / 3.0).movedim(1, -1)

    def _decode_direct_or_tiled(self, latent: torch.Tensor) -> torch.Tensor:
        scaled_latent = self.vae.latent_format.process_out(latent)
        decode_dtype = next(self.vae.first_stage_model.parameters()).dtype

        try:
            direct_latent = scaled_latent.to(device=self.device, dtype=decode_dtype)
            pixels = self.vae.first_stage_model.decode(direct_latent).float()
            return self._normalize_decoded_output(pixels).movedim(1, -1).cpu()
        except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
            resources.soft_empty_cache(force=True)

        tile_attempts = [(64, 64), (32, 32), (16, 16)]
        last_error = None
        for tile_x, tile_y in tile_attempts:
            try:
                return self._decode_tiled_local(scaled_latent, tile_x=tile_x, tile_y=tile_y)
            except (resources.OOM_EXCEPTION, torch.OutOfMemoryError) as exc:
                last_error = exc
                resources.soft_empty_cache(force=True)

        if last_error is not None:
            raise last_error
        raise RuntimeError('Direct VAE decode failed without producing an output.')

    def decode_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        attach_start = time.perf_counter()
        self.attach_vae_direct()
        vae_attach = time.perf_counter() - attach_start

        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                images = self._decode_direct_or_tiled(latent)
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




