from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from backend import decode as backend_decode
from backend import loader as backend_loader
from backend import resources, sampling
from backend.flux.flux_fill_pipeline import FluxEmptyConditioning, load_flux_empty_conditioning_cache
from backend.gguf.loader import gguf_sd_loader
from backend.gguf.ops import GGMLOps
from backend.gguf.patcher import GGUFModelPatcher
from ldm_patched.modules import latent_formats, model_detection


def _trace_summary(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    return dict(stats)


def load_flux_dev_unet(
    unet_path: str | Path,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    handle_prefix: str | None = "model.diffusion_model.",
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux-dev UNet path does not exist: {path}")

    load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()

    state_dict, arch = gguf_sd_loader(str(path), handle_prefix=handle_prefix, return_arch=True)
    if arch != "flux":
        raise ValueError(f"Expected Flux GGUF arch for {path}, got {arch!r}.")

    model_config = model_detection.model_config_from_unet(state_dict, "", use_base_if_no_match=False)
    if model_config is None:
        raise ValueError(f"Unable to infer a Flux model config from {path}.")

    model = model_config.get_model(
        state_dict,
        "",
        device=offload_device,
        model_options={"custom_operations": GGMLOps},
    )
    model.load_model_weights(state_dict, "")
    del state_dict
    gc.collect()

    patcher = GGUFModelPatcher(model, load_device=load_device, offload_device=offload_device)
    patcher.model_options["flux_dev"] = {
        "path": str(path),
        "arch": arch,
    }
    return patcher


@dataclass
class FluxDevGGUFRunConfig:
    unet_path: Path | str
    positive_conditioning_path: Path | str
    negative_conditioning_path: Path | str
    vae_path: Path | str
    width: int = 512
    height: int = 512
    steps: int = 8
    cfg: float = 1.0
    guidance: float = 3.5
    sampler: str = "euler"
    scheduler: str = "karras"
    seed: int = 12345
    batch_size: int = 1
    denoise: float = 1.0
    output_path: Path | str | None = None

    def validate_static(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}.")
        if self.cfg <= 0:
            raise ValueError(f"cfg must be > 0, got {self.cfg}.")
        if self.guidance <= 0:
            raise ValueError(f"guidance must be > 0, got {self.guidance}.")
        if self.width < 1 or self.height < 1:
            raise ValueError(f"width/height must be positive, got {self.width}x{self.height}.")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise ValueError(f"width/height must be divisible by 8, got {self.width}x{self.height}.")
        if not str(self.sampler or "").strip():
            raise ValueError("sampler must be a non-empty string.")
        if not str(self.scheduler or "").strip():
            raise ValueError("scheduler must be a non-empty string.")
        if self.denoise <= 0:
            raise ValueError(f"denoise must be > 0, got {self.denoise}.")
        for label, value in (
            ("UNet", self.unet_path),
            ("positive conditioning", self.positive_conditioning_path),
            ("negative conditioning", self.negative_conditioning_path),
            ("AE", self.vae_path),
        ):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"{label} path does not exist: {path}")


@dataclass
class FluxDevGGUFDenoiseResult:
    samples: torch.Tensor
    sigmas: torch.Tensor
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    gguf_trace_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class FluxDevGGUFRunResult:
    output_image: np.ndarray
    output_path: Path | None
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    denoise_gguf_trace_stats: dict[str, Any] = field(default_factory=dict)


class FluxDevGGUFRuntime:
    route_label = "flux_dev_gguf"

    def __init__(self, config: FluxDevGGUFRunConfig, *, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or resources.get_torch_device()
        self.positive_conditioning: FluxEmptyConditioning | None = None
        self.negative_conditioning: FluxEmptyConditioning | None = None
        self.unet = None
        self.vae = None
        self._loaded = False

    def load_components(self) -> dict[str, float]:
        if self._loaded:
            return {
                "load_components": 0.0,
                "conditioning_load": 0.0,
                "unet_load": 0.0,
                "vae_load": 0.0,
            }

        timings: dict[str, float] = {}
        load_start = time.perf_counter()

        stage_start = time.perf_counter()
        self.positive_conditioning = load_flux_empty_conditioning_cache(
            self.config.positive_conditioning_path,
            map_location="cpu",
        )
        self.negative_conditioning = load_flux_empty_conditioning_cache(
            self.config.negative_conditioning_path,
            map_location="cpu",
        )
        timings["conditioning_load"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        self.unet = load_flux_dev_unet(
            self.config.unet_path,
            load_device=self.device,
            offload_device=resources.unet_offload_device(),
        )
        gc.collect()
        timings["unet_load"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        self.vae = backend_loader.load_vae(
            str(self.config.vae_path),
            load_device=self.device,
            offload_device=resources.vae_offload_device(),
            latent_format=latent_formats.Flux(),
        )
        gc.collect()
        timings["vae_load"] = time.perf_counter() - stage_start

        timings["load_components"] = time.perf_counter() - load_start
        self._loaded = True
        return timings

    def _attach_unet(self) -> float:
        if self.unet is None:
            raise RuntimeError("Flux-dev UNet has not been loaded.")
        return resources.prepare_models_for_stage(
            [self.unet],
            stage_name="flux_dev_denoise",
            target_phase=resources.MemoryPhase.DIFFUSION,
            force_full_load=False,
        )

    def _detach_unet(self) -> float:
        if self.unet is None:
            return 0.0
        start = time.perf_counter()
        resources.eject_model(self.unet)
        return time.perf_counter() - start

    def _attach_vae(self) -> float:
        if self.vae is None:
            raise RuntimeError("Flux-dev VAE has not been loaded.")
        return resources.prepare_models_for_stage(
            [self.vae.patcher],
            stage_name="flux_dev_decode",
            target_phase=resources.MemoryPhase.DECODE,
            force_full_load=False,
        )

    def _detach_vae(self) -> float:
        if self.vae is None:
            return 0.0
        start = time.perf_counter()
        resources.eject_model(self.vae.patcher)
        return time.perf_counter() - start

    def _build_conditioning_pair(self) -> tuple[list[list[Any]], list[list[Any]], dict[str, float]]:
        if self.positive_conditioning is None or self.negative_conditioning is None:
            raise RuntimeError("Flux-dev conditioning caches have not been loaded.")

        timings: dict[str, float] = {}
        build_start = time.perf_counter()
        build_cpu_start = time.process_time()
        positive_cross, positive_pooled = self.positive_conditioning.repeat(
            int(self.config.batch_size),
            device=torch.device("cpu"),
        )
        negative_cross, negative_pooled = self.negative_conditioning.repeat(
            int(self.config.batch_size),
            device=torch.device("cpu"),
        )
        timings["conditioning_prepare"] = time.perf_counter() - build_start
        timings["conditioning_prepare_cpu_proc"] = time.process_time() - build_cpu_start

        guidance_tensor = torch.tensor([self.config.guidance], dtype=torch.float32)
        positive = [[
            positive_cross,
            {
                "model_conds": {
                    "pooled_output": positive_pooled,
                    "guidance": guidance_tensor,
                }
            },
        ]]
        negative = [[
            negative_cross,
            {
                "model_conds": {
                    "pooled_output": negative_pooled,
                    "guidance": guidance_tensor.clone(),
                }
            },
        ]]
        return positive, negative, timings

    def _create_latent_and_noise(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.unet is None:
            raise RuntimeError("Flux-dev UNet has not been loaded.")
        latent_h = self.config.height // 8
        latent_w = self.config.width // 8
        dtype = self.unet.model.get_dtype()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.config.seed))
        noise = torch.randn(
            (self.config.batch_size, 16, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=dtype,
        )
        latent = torch.zeros_like(noise)
        return latent, noise

    def denoise(self, *, callback: Any | None = None, disable_pbar: bool = True) -> FluxDevGGUFDenoiseResult:
        if self.unet is None:
            raise RuntimeError("Flux-dev UNet has not been loaded.")

        positive, negative, prep_timings = self._build_conditioning_pair()
        latent_start = time.perf_counter()
        latent_cpu_start = time.process_time()
        latent, noise = self._create_latent_and_noise()
        prep_timings["latent_noise_prep"] = time.perf_counter() - latent_start
        prep_timings["latent_noise_prep_cpu_proc"] = time.process_time() - latent_cpu_start

        sampler = sampling.KSampler(
            self.unet,
            self.config.steps,
            self.device,
            self.config.sampler,
            self.config.scheduler,
            self.config.denoise,
            model_options=getattr(self.unet, "model_options", {}),
        )

        guider = sampling.prepare_sampler_conds(
            self.unet,
            noise,
            positive,
            negative,
            self.config.cfg,
            sampler_name=self.config.sampler,
            latent_image=latent,
            denoise_mask=None,
            seed=self.config.seed,
            model_options=getattr(self.unet, "model_options", {}),
            quality={},
            inner_model=self.unet.model,
        )

        timings: dict[str, float] = {}
        denoise_attach = 0.0
        denoise_offload = 0.0
        attach_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            denoise_attach = self._attach_unet()
            with torch.inference_mode():
                samples = sampling.sample_prepared_sdxl(
                    guider,
                    noise,
                    sampler.sigmas,
                    sampler=sampling.ksampler(self.config.sampler),
                    latent_image=latent,
                    denoise_mask=None,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=self.config.seed,
                    attach_model=False,
                )
        finally:
            denoise_offload = self._detach_unet()
            timings["denoise_wall"] = time.perf_counter() - attach_start
            timings["denoise_cpu_proc"] = time.process_time() - cpu_start
            timings["sampler_model_attach"] = denoise_attach
            timings["sampler_model_offload"] = denoise_offload

        gguf_trace_stats = dict(getattr(guider, "last_gguf_trace_stats", {}) or {})
        timings["denoise_gguf_dequant"] = float(gguf_trace_stats.get("dequant_seconds", 0.0))
        timings["denoise_gguf_dequant_cpu_proc"] = float(gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0))
        timings["denoise_gguf_forward"] = float(gguf_trace_stats.get("forward_seconds", 0.0))
        timings["denoise_gguf_forward_cpu_proc"] = float(gguf_trace_stats.get("forward_cpu_process_seconds", 0.0))
        timings["denoise_s_per_it"] = timings["denoise_wall"] / max(1, self.config.steps)
        timings.update(prep_timings)

        return FluxDevGGUFDenoiseResult(
            samples=samples.detach().cpu(),
            sigmas=sampler.sigmas.detach().cpu(),
            timings=timings,
            metadata={
                "guidance": float(self.config.guidance),
                "cfg": float(self.config.cfg),
                "sampler": self.config.sampler,
                "scheduler": self.config.scheduler,
                "seed": int(self.config.seed),
                "conditioning_mode": "pt_cache",
            },
            gguf_trace_stats=gguf_trace_stats,
        )

    def decode(self, samples: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        if self.vae is None:
            raise RuntimeError("Flux-dev VAE has not been loaded.")

        timings: dict[str, float] = {}
        vae_attach = self._attach_vae()
        timings["vae_attach"] = vae_attach
        decode_start = time.perf_counter()
        decode_cpu_start = time.process_time()
        try:
            with torch.inference_mode():
                decoded = backend_decode.decode_preloaded_vae(self.vae, samples, tiled=False)
        finally:
            timings["vae_decode"] = time.perf_counter() - decode_start
            timings["vae_decode_cpu_proc"] = time.process_time() - decode_cpu_start
            timings["vae_offload"] = self._detach_vae()
            timings["vae_model_attach"] = vae_attach

        if isinstance(decoded, torch.Tensor):
            decoded_np = decoded.detach().cpu().numpy()
        else:
            decoded_np = np.asarray(decoded)
        if decoded_np.ndim != 4:
            raise RuntimeError(f"Flux-dev VAE decode produced unexpected shape: {decoded_np.shape}.")

        output = np.clip(decoded_np[0] * 255.0, 0, 255).astype(np.uint8)
        return output, timings

    def run(self, *, callback: Any | None = None, disable_pbar: bool = True) -> FluxDevGGUFRunResult:
        self.config.validate_static()
        total_start = time.perf_counter()
        load_timings = self.load_components()

        denoise_result = self.denoise(callback=callback, disable_pbar=disable_pbar)
        decoded_image, decode_timings = self.decode(denoise_result.samples)

        output_path = Path(self.config.output_path) if self.config.output_path is not None else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(decoded_image).save(output_path)

        timings = {
            **load_timings,
            **denoise_result.timings,
            **decode_timings,
            "total_wall": time.perf_counter() - total_start,
        }
        metadata = {
            "route_label": self.route_label,
            "unet_path": str(self.config.unet_path),
            "positive_conditioning_path": str(self.config.positive_conditioning_path),
            "negative_conditioning_path": str(self.config.negative_conditioning_path),
            "vae_path": str(self.config.vae_path),
            "width": int(self.config.width),
            "height": int(self.config.height),
            "steps": int(self.config.steps),
            "cfg": float(self.config.cfg),
            "guidance": float(self.config.guidance),
            "sampler": self.config.sampler,
            "scheduler": self.config.scheduler,
            "seed": int(self.config.seed),
            "batch_size": int(self.config.batch_size),
            "device": str(self.device),
            "bootstrapped": True,
        }
        return FluxDevGGUFRunResult(
            output_image=decoded_image,
            output_path=output_path,
            timings=timings,
            metadata=metadata,
            denoise_gguf_trace_stats=_trace_summary(denoise_result.gguf_trace_stats),
        )

    def close(self) -> None:
        if self.unet is not None:
            try:
                resources.eject_model(self.unet)
            except Exception:
                detach = getattr(self.unet, "detach", None)
                if callable(detach):
                    detach()
            self.unet = None

        if self.vae is not None:
            try:
                resources.eject_model(self.vae.patcher)
            except Exception:
                detach = getattr(self.vae.patcher, "detach", None)
                if callable(detach):
                    detach()
            self.vae = None

        self.positive_conditioning = None
        self.negative_conditioning = None

        self._loaded = False
        resources.soft_empty_cache(force=True)
