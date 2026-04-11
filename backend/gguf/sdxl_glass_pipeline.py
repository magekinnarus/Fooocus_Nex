from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
class GlassSDXLGGUFRunConfig:
    """Configuration owned by the glass SDXL GGUF harness."""

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
    latent_mode: str = "txt2img"
    ancestral_eta: float = 1.0
    ancestral_s_noise: float = 1.0
    ancestral_noise_policy: str = "direct_compatible"


@dataclass
class GlassSDXLGGUFLatentSource:
    """
    Explicit latent/noise source contract.

    M09 implements only txt2img. The extra fields keep the seam visible for
    later img2img and inpaint support without pretending those modes exist yet.
    """

    mode: str
    latent_image: torch.Tensor
    noise: torch.Tensor
    denoise_mask: Optional[torch.Tensor] = None
    concat_latent_image: Optional[torch.Tensor] = None


@dataclass
class GlassSDXLGGUFPreparedInputs:
    """Prepared control-plane inputs for the future glass denoise loop."""

    encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]]
    adm_pair: Dict[str, torch.Tensor]
    raw_positive: Any
    raw_negative: Any
    processed_positive: Any
    processed_negative: Any
    latent_source: GlassSDXLGGUFLatentSource
    sigmas: torch.Tensor
    scaled_initial_latent: torch.Tensor


@dataclass
class GlassSDXLGGUFDenoiseResult:
    """Denoise output for the glass pipeline."""

    samples: torch.Tensor
    cond_prepare_duration: float
    sampler_model_attach: float
    denoise_wall: float
    denoise_cpu_proc: float
    gguf_trace_stats: Dict[str, Any] = field(default_factory=dict)
    checkpoint_records: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GlassSDXLGGUFCheckpointConfig:
    """Optional checkpoint capture policy for the glass pipeline."""

    enabled: bool = False
    persist_full_tensors: bool = False
    tensor_output_dir: Optional[str] = None
    persist_steps: Optional[list[int]] = None


@dataclass
class GlassSDXLGGUFCheckpointRecord:
    """One checkpoint evidence item captured at a glass boundary."""

    name: str
    kind: str
    step_index: Optional[int] = None
    digest: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    tensor_stats: Dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None


@dataclass
class GlassSDXLGGUFPreparationMetrics:
    """Timing buckets for the W00 preparation boundary."""

    clip_residency_attach: float = 0.0
    clip_residency_offload: float = 0.0
    clip_encode: float = 0.0
    adm_build: float = 0.0
    latent_noise_prep: float = 0.0
    cond_prepare: float = 0.0
    sigma_prepare: float = 0.0
    initial_noise_scaling: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "clip_residency_attach": self.clip_residency_attach,
            "clip_residency_offload": self.clip_residency_offload,
            "clip_encode": self.clip_encode,
            "adm_build": self.adm_build,
            "latent_noise_prep": self.latent_noise_prep,
            "cond_prepare": self.cond_prepare,
            "sigma_prepare": self.sigma_prepare,
            "initial_noise_scaling": self.initial_noise_scaling,
        }


class GlassSDXLGGUFPipeline:
    """Reusable SDXL GGUF glass pipeline shell.

    W00 defines the packet surface first. Each preparation seam is added in
    approved steps so the control plane stays reviewable.
    """

    route_label = "glass_sdxl_gguf"

    def __init__(
        self,
        config: GlassSDXLGGUFRunConfig,
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

    def attach_clip_direct(self) -> None:
        self.load_components()
        self.clip.patcher.patch_model(device_to=self.device, lowvram_model_memory=0)

    def _clean_unet_budget_bytes(self) -> int:
        if self.unet_budget_mb is not None:
            return self.unet_budget_mb * 1024 * 1024
        return int(resources.maximum_vram_for_weights(self.device))

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

    @staticmethod
    def _json_digest(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _tensor_digest(tensor: Optional[torch.Tensor]) -> str:
        if tensor is None:
            return ""
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()

    @staticmethod
    def _tensor_stats(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
        if tensor is None:
            return {}
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        stats: Dict[str, Any] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "device": str(tensor.device),
        }
        if cpu_tensor.numel() > 0 and cpu_tensor.dtype.is_floating_point:
            as_float = cpu_tensor.float()
            stats.update({
                "mean": float(as_float.mean().item()),
                "std": float(as_float.std(unbiased=False).item()) if as_float.numel() > 1 else 0.0,
                "min": float(as_float.min().item()),
                "max": float(as_float.max().item()),
            })
        return stats

    def _checkpoint_record(
        self,
        name: str,
        payload: Any,
        *,
        kind: str,
        step_index: Optional[int] = None,
        checkpoint_config: Optional[GlassSDXLGGUFCheckpointConfig] = None,
        tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        summary = payload if isinstance(payload, dict) else {"value": payload}
        record = GlassSDXLGGUFCheckpointRecord(
            name=name,
            kind=kind,
            step_index=step_index,
            digest=self._tensor_digest(tensor) if tensor is not None else self._json_digest(summary),
            summary=summary,
            tensor_stats=self._tensor_stats(tensor),
        )

        should_persist = (
            checkpoint_config is not None
            and checkpoint_config.enabled
            and checkpoint_config.persist_full_tensors
            and tensor is not None
            and checkpoint_config.tensor_output_dir is not None
        )
        if should_persist:
            persist_steps = set(checkpoint_config.persist_steps or [])
            if not persist_steps or step_index is None or step_index in persist_steps:
                output_dir = Path(checkpoint_config.tensor_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = name.replace(".", "_")
                if step_index is not None:
                    filename = f"{step_index:03d}_{filename}"
                artifact_path = output_dir / f"{filename}.pt"
                torch.save(tensor.detach().cpu(), artifact_path)
                record.artifact_path = str(artifact_path)

        return {
            "name": record.name,
            "kind": record.kind,
            "step_index": record.step_index,
            "digest": record.digest,
            "summary": record.summary,
            "tensor_stats": record.tensor_stats,
            "artifact_path": record.artifact_path,
        }

    def encode_prompt_pair(self) -> tuple[Dict[str, Dict[str, torch.Tensor]], GlassSDXLGGUFPreparationMetrics]:
        metrics = GlassSDXLGGUFPreparationMetrics()

        attach_start = time.perf_counter()
        self.attach_clip_direct()
        metrics.clip_residency_attach = time.perf_counter() - attach_start

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
            metrics.clip_residency_offload = time.perf_counter() - offload_start
        metrics.clip_encode = time.perf_counter() - encode_start

        return encoded_pair, metrics

    def build_adm_pair(
        self,
        encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]],
    ) -> tuple[Dict[str, torch.Tensor], float]:
        adm_start = time.perf_counter()
        adm_pair = conditioning.build_sdxl_adm_pair(
            encoded_prompt_pair,
            self.config.width,
            self.config.height,
            target_width=self.config.width,
            target_height=self.config.height,
            adm_scale_positive=self.config.quality.get("adm_scale_positive", 1.0),
            adm_scale_negative=self.config.quality.get("adm_scale_negative", 1.0),
        )
        return adm_pair, time.perf_counter() - adm_start

    def build_latent_source(self) -> tuple[GlassSDXLGGUFLatentSource, float]:
        start = time.perf_counter()
        mode = self.config.latent_mode
        if mode != "txt2img":
            raise NotImplementedError(
                f"Glass SDXL GGUF latent mode '{mode}' is not implemented in P4-M09-W00. "
                "Future img2img/inpaint support must enter through this latent-source seam."
            )

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
        latent_image = torch.zeros(
            (self.config.batch_size, 4, latent_h, latent_w),
            device=self.device,
            dtype=dtype,
        )

        latent_source = GlassSDXLGGUFLatentSource(
            mode="txt2img",
            latent_image=latent_image,
            noise=noise,
            denoise_mask=None,
            concat_latent_image=None,
        )
        return latent_source, time.perf_counter() - start

    def build_raw_conditioning_payloads(
        self,
        encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]],
        adm_pair: Dict[str, torch.Tensor],
    ) -> tuple[Any, Any]:
        raw_positive = [[
            encoded_prompt_pair["positive"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["positive"]["pooled"],
                "model_conds": {"y": adm_pair["positive"]},
            },
        ]]
        raw_negative = [[
            encoded_prompt_pair["negative"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["negative"]["pooled"],
                "model_conds": {"y": adm_pair["negative"]},
            },
        ]]
        return raw_positive, raw_negative

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

    def process_conditioning_payloads(
        self,
        raw_positive: Any,
        raw_negative: Any,
        latent_source: GlassSDXLGGUFLatentSource,
    ) -> tuple[Dict[str, Any], float]:
        self.load_components()
        conds = {
            "positive": self._convert_sampler_cond(raw_positive),
            "negative": self._convert_sampler_cond(raw_negative),
        }
        cond_start = time.perf_counter()
        processed = cond_utils.process_conds(
            self.unet.model,
            latent_source.noise,
            conds,
            self.device,
            latent_image=latent_source.latent_image,
            denoise_mask=latent_source.denoise_mask,
            seed=self.config.seed,
        )
        return processed, time.perf_counter() - cond_start

    def calculate_sigmas(self) -> tuple[torch.Tensor, float]:
        self.load_components()
        sigma_start = time.perf_counter()
        sampler_instance = sampling.KSampler(
            self.unet,
            self.config.steps,
            self.device,
            self.config.sampler,
            self.config.scheduler,
            self.config.denoise,
            model_options={"quality": self.config.quality},
        )
        return sampler_instance.sigmas, time.perf_counter() - sigma_start

    def scale_initial_latent(
        self,
        latent_source: GlassSDXLGGUFLatentSource,
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        self.load_components()
        if sigmas.shape[-1] == 0:
            return latent_source.latent_image, 0.0

        scaling_start = time.perf_counter()
        model_sampling = self.unet.model.model_sampling
        max_sigma = float(model_sampling.sigma_max)
        sigma = float(sigmas[0])
        max_denoise = math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma
        scaled_initial_latent = model_sampling.noise_scaling(
            sigmas[0],
            latent_source.noise,
            latent_source.latent_image,
            max_denoise,
        )
        return scaled_initial_latent, time.perf_counter() - scaling_start

    def prepare_inputs(self) -> tuple[GlassSDXLGGUFPreparedInputs, GlassSDXLGGUFPreparationMetrics]:
        encoded_prompt_pair, metrics = self.encode_prompt_pair()

        adm_pair, metrics.adm_build = self.build_adm_pair(encoded_prompt_pair)
        latent_source, metrics.latent_noise_prep = self.build_latent_source()
        raw_positive, raw_negative = self.build_raw_conditioning_payloads(
            encoded_prompt_pair,
            adm_pair,
        )
        processed_conds, metrics.cond_prepare = self.process_conditioning_payloads(
            raw_positive,
            raw_negative,
            latent_source,
        )
        sigmas, metrics.sigma_prepare = self.calculate_sigmas()
        scaled_initial_latent, metrics.initial_noise_scaling = self.scale_initial_latent(
            latent_source,
            sigmas,
        )

        prepared = GlassSDXLGGUFPreparedInputs(
            encoded_prompt_pair=encoded_prompt_pair,
            adm_pair=adm_pair,
            raw_positive=raw_positive,
            raw_negative=raw_negative,
            processed_positive=processed_conds.get("positive"),
            processed_negative=processed_conds.get("negative"),
            latent_source=latent_source,
            sigmas=sigmas,
            scaled_initial_latent=scaled_initial_latent,
        )
        return prepared, metrics

    @staticmethod
    def _noise_sampler(x: torch.Tensor, seed: Optional[int]):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)

        def sample_noise(_: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
            return torch.randn(
                x.size(),
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
                generator=generator,
            )

        return sample_noise

    def _resolve_ancestral_noise_seed(self) -> Optional[int]:
        policy = self.config.ancestral_noise_policy
        if policy == "direct_compatible":
            return None
        if policy == "seeded":
            return self.config.seed
        raise ValueError(
            f"Unsupported ancestral noise policy '{policy}'. "
            "Use 'direct_compatible' or 'seeded'."
        )

    def _negative_conds_for_cfg(self, negative_conds: Any) -> Any:
        model_options = getattr(self.unet, "model_options", {}) or {}
        disable_cfg1_optimization = bool(model_options.get("disable_cfg1_optimization", False))
        if math.isclose(float(self.config.cfg), 1.0) and not disable_cfg1_optimization:
            return None
        return negative_conds

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
                    raise ValueError("Glass SDXL GGUF denoise only supports full-frame txt2img conditions.")
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

    def _apply_sharpness(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond_pred: torch.Tensor,
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
                return x - positive_eps_weighted
        return cond_pred

    def _apply_glass_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
    ) -> torch.Tensor:
        model_sampling = self.unet.model.model_sampling
        t = model_sampling.timestep(timestep).float()
        diffusion_progress = max(0.0, min(1.0, 1.0 - float(t.reshape(-1)[0].item()) / 999.0))

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

    def _build_glass_model_callable(
        self,
        processed_conds: Dict[str, Any],
    ):
        def model_fn(x: torch.Tensor, sigma: torch.Tensor, **_: Any) -> torch.Tensor:
            cond_pred, uncond_pred = self._calc_fullframe_cond_batch(
                [
                    processed_conds.get("positive"),
                    self._negative_conds_for_cfg(processed_conds.get("negative")),
                ],
                x,
                sigma,
            )
            return self._apply_glass_cfg(x, sigma, self._apply_sharpness(x, sigma, cond_pred), uncond_pred)

        return model_fn

    def run_prepared_inputs(
        self,
        prepared: GlassSDXLGGUFPreparedInputs,
        *,
        prepare_metrics: Optional[GlassSDXLGGUFPreparationMetrics] = None,
        checkpoint_config: Optional[GlassSDXLGGUFCheckpointConfig] = None,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> GlassSDXLGGUFDenoiseResult:
        if self.config.sampler != "euler_ancestral":
            raise NotImplementedError(
                f"Glass SDXL GGUF sampler '{self.config.sampler}' is not implemented in W01. "
                "Only euler_ancestral is supported for the first explicit loop."
            )

        self.load_components()
        checkpoint_records: list[Dict[str, Any]] = []
        checkpoint_enabled = bool(checkpoint_config and checkpoint_config.enabled)
        prepared_summary = self.summarize_prepared_inputs(prepared)

        if checkpoint_enabled:
            checkpoint_records.append(self._checkpoint_record(
                "prepared.processed_conditioning",
                {
                    "processed_positive": prepared_summary["processed_positive"],
                    "processed_negative": prepared_summary["processed_negative"],
                },
                kind="summary",
                checkpoint_config=checkpoint_config,
            ))
            checkpoint_records.append(self._checkpoint_record(
                "prepared.latent_source",
                prepared_summary["latent_source"],
                kind="summary",
                checkpoint_config=checkpoint_config,
            ))
            checkpoint_records.append(self._checkpoint_record(
                "prepared.sigma_schedule",
                {"sigma_count": prepared_summary["sigma_count"]},
                kind="tensor",
                tensor=prepared.sigmas,
                checkpoint_config=checkpoint_config,
            ))
            checkpoint_records.append(self._checkpoint_record(
                "prepared.initial_latent",
                {"role": "latent_image"},
                kind="tensor",
                tensor=prepared.latent_source.latent_image,
                checkpoint_config=checkpoint_config,
            ))
            checkpoint_records.append(self._checkpoint_record(
                "prepared.initial_noise",
                {"role": "noise"},
                kind="tensor",
                tensor=prepared.latent_source.noise,
                checkpoint_config=checkpoint_config,
            ))
            checkpoint_records.append(self._checkpoint_record(
                "prepared.scaled_initial_latent",
                {"role": "scaled_initial_latent"},
                kind="tensor",
                tensor=prepared.scaled_initial_latent,
                checkpoint_config=checkpoint_config,
            ))

        if prepared.sigmas.shape[-1] == 0:
            if checkpoint_enabled:
                checkpoint_records.append(self._checkpoint_record(
                    "final.latent",
                    {"reason": "empty_sigma_schedule"},
                    kind="tensor",
                    tensor=prepared.scaled_initial_latent,
                    checkpoint_config=checkpoint_config,
                ))
            return GlassSDXLGGUFDenoiseResult(
                samples=prepared.scaled_initial_latent,
                cond_prepare_duration=prepare_metrics.cond_prepare if prepare_metrics is not None else 0.0,
                sampler_model_attach=0.0,
                denoise_wall=0.0,
                denoise_cpu_proc=0.0,
                checkpoint_records=checkpoint_records,
            )

        attach_start = time.perf_counter()
        self.attach_unet_direct()
        sampler_model_attach = time.perf_counter() - attach_start

        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        samples = prepared.scaled_initial_latent
        try:
            with torch.inference_mode(), precision.autocast_context(self.device):
                model_sampling = self.unet.model.model_sampling
                noise_sampler = self._noise_sampler(samples, self._resolve_ancestral_noise_seed())
                negative_conds = self._negative_conds_for_cfg(prepared.processed_negative)

                for i in range(len(prepared.sigmas) - 1):
                    sigma = prepared.sigmas[i:i + 1]
                    sigma_next = prepared.sigmas[i + 1:i + 2]
                    x_in = samples
                    positive_prediction, negative_prediction = self._calc_fullframe_cond_batch(
                        [prepared.processed_positive, negative_conds],
                        x_in,
                        sigma,
                    )
                    post_sharpness_positive_prediction = self._apply_sharpness(
                        x_in,
                        sigma,
                        positive_prediction,
                    )
                    post_cfg_denoised = self._apply_glass_cfg(
                        x_in,
                        sigma,
                        post_sharpness_positive_prediction,
                        negative_prediction,
                    )
                    sigma_down, sigma_up = k_diffusion.get_ancestral_step(
                        float(sigma),
                        float(sigma_next),
                        eta=self.config.ancestral_eta,
                    )
                    d = (x_in - post_cfg_denoised) / sigma.view(-1, *([1] * (x_in.ndim - 1)))
                    ancestral_noise = None
                    if sigma_down == 0:
                        samples = post_cfg_denoised
                    else:
                        ancestral_noise = noise_sampler(sigma, sigma_next)
                        dt = sigma_down - float(sigma)
                        samples = x_in + d * dt + ancestral_noise * self.config.ancestral_s_noise * sigma_up

                    if callback is not None:
                        callback({
                            "x": x_in,
                            "i": i,
                            "sigma": sigma,
                            "sigma_hat": sigma,
                            "denoised": post_cfg_denoised,
                        })

                    if checkpoint_enabled:
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.x_in",
                            {"step_index": i},
                            kind="tensor",
                            tensor=x_in,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.apply_model_positive_prediction",
                            {"step_index": i},
                            kind="tensor",
                            tensor=positive_prediction,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.apply_model_negative_prediction",
                            {"step_index": i},
                            kind="tensor",
                            tensor=negative_prediction,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.post_sharpness_positive_prediction",
                            {"step_index": i},
                            kind="tensor",
                            tensor=post_sharpness_positive_prediction,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.post_cfg_denoised",
                            {"step_index": i},
                            kind="tensor",
                            tensor=post_cfg_denoised,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.d",
                            {"step_index": i},
                            kind="tensor",
                            tensor=d,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.sigma_down_up",
                            {
                                "step_index": i,
                                "sigma_down": float(sigma_down),
                                "sigma_up": float(sigma_up),
                            },
                            kind="summary",
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))
                        if ancestral_noise is not None:
                            checkpoint_records.append(self._checkpoint_record(
                                f"step_{i:03d}.ancestral_noise",
                                {"step_index": i},
                                kind="tensor",
                                tensor=ancestral_noise,
                                step_index=i,
                                checkpoint_config=checkpoint_config,
                            ))
                        checkpoint_records.append(self._checkpoint_record(
                            f"step_{i:03d}.post_step_latent",
                            {"step_index": i},
                            kind="tensor",
                            tensor=samples,
                            step_index=i,
                            checkpoint_config=checkpoint_config,
                        ))

                samples = model_sampling.inverse_noise_scaling(prepared.sigmas[-1], samples)
                if checkpoint_enabled:
                    checkpoint_records.append(self._checkpoint_record(
                        "final.latent",
                        {"reason": "inverse_noise_scaling"},
                        kind="tensor",
                        tensor=samples,
                        checkpoint_config=checkpoint_config,
                    ))
        finally:
            denoise_wall = time.perf_counter() - denoise_start
            denoise_cpu_proc = time.process_time() - denoise_cpu_start
            self.detach_unet_direct()

        return GlassSDXLGGUFDenoiseResult(
            samples=samples,
            cond_prepare_duration=prepare_metrics.cond_prepare if prepare_metrics is not None else 0.0,
            sampler_model_attach=sampler_model_attach,
            denoise_wall=denoise_wall,
            denoise_cpu_proc=denoise_cpu_proc,
            checkpoint_records=checkpoint_records,
        )

    def attach_vae_direct(self) -> None:
        self.load_components()
        self.vae.patcher.patch_model(device_to=self.device, lowvram_model_memory=0)
        self.vae.first_stage_model.to(device=self.device, dtype=torch.float32)

    def detach_vae_direct(self) -> None:
        if self.vae is not None:
            self.vae.patcher.detach()

    @staticmethod
    def _normalize_decoded_output(image: torch.Tensor) -> torch.Tensor:
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

        p3 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x, tile_y, overlap, upscale_amount=8, output_device="cpu")
        p1 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=8, output_device="cpu")
        p2 = backend_utils.tiled_scale(scaled_latent, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=8, output_device="cpu")

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
        raise RuntimeError("Glass VAE decode failed without producing an output.")

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

    @staticmethod
    def _tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
        if tensor is None:
            return None
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }

    @classmethod
    def _condition_list_summary(cls, conds: Any) -> Dict[str, Any]:
        entries = list(conds or [])
        summaries = []
        for entry in entries:
            if not isinstance(entry, dict):
                summaries.append({"type": type(entry).__name__})
                continue
            model_conds = entry.get("model_conds", {}) or {}
            summaries.append({
                "keys": sorted(str(key) for key in entry.keys()),
                "model_cond_keys": sorted(str(key) for key in model_conds.keys()),
                "has_cross_attn": "cross_attn" in entry,
                "cross_attn": cls._tensor_summary(entry.get("cross_attn")),
            })
        return {"count": len(entries), "entries": summaries}

    @classmethod
    def summarize_prepared_inputs(cls, prepared: GlassSDXLGGUFPreparedInputs) -> Dict[str, Any]:
        return {
            "latent_source": {
                "mode": prepared.latent_source.mode,
                "latent_image": cls._tensor_summary(prepared.latent_source.latent_image),
                "noise": cls._tensor_summary(prepared.latent_source.noise),
                "denoise_mask": cls._tensor_summary(prepared.latent_source.denoise_mask),
                "concat_latent_image": cls._tensor_summary(prepared.latent_source.concat_latent_image),
            },
            "sigmas": cls._tensor_summary(prepared.sigmas),
            "sigma_count": int(prepared.sigmas.shape[-1]) if prepared.sigmas is not None else 0,
            "scaled_initial_latent": cls._tensor_summary(prepared.scaled_initial_latent),
            "raw_positive_count": len(prepared.raw_positive or []),
            "raw_negative_count": len(prepared.raw_negative or []),
            "processed_positive": cls._condition_list_summary(prepared.processed_positive),
            "processed_negative": cls._condition_list_summary(prepared.processed_negative),
            "encoded_prompt_pair": {
                "positive_cond": cls._tensor_summary(prepared.encoded_prompt_pair["positive"]["cond"]),
                "positive_pooled": cls._tensor_summary(prepared.encoded_prompt_pair["positive"]["pooled"]),
                "negative_cond": cls._tensor_summary(prepared.encoded_prompt_pair["negative"]["cond"]),
                "negative_pooled": cls._tensor_summary(prepared.encoded_prompt_pair["negative"]["pooled"]),
            },
            "adm_pair": {
                "positive": cls._tensor_summary(prepared.adm_pair["positive"]),
                "negative": cls._tensor_summary(prepared.adm_pair["negative"]),
            },
        }

    @staticmethod
    def compare_prepared_input_summaries(
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> Dict[str, Any]:
        keys_to_compare = (
            ("latent_source", "mode"),
            ("latent_source", "latent_image"),
            ("latent_source", "noise"),
            ("latent_source", "denoise_mask"),
            ("sigmas",),
            ("sigma_count",),
            ("scaled_initial_latent",),
            ("raw_positive_count",),
            ("raw_negative_count",),
        )
        mismatches = []
        for path in keys_to_compare:
            left_value: Any = left
            right_value: Any = right
            for key in path:
                left_value = left_value.get(key) if isinstance(left_value, dict) else None
                right_value = right_value.get(key) if isinstance(right_value, dict) else None
            if left_value != right_value:
                mismatches.append({
                    "path": ".".join(path),
                    "left": left_value,
                    "right": right_value,
                })
        return {"matches": len(mismatches) == 0, "mismatches": mismatches}
