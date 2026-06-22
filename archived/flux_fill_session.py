from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from backend import resources
from backend.flux.flux_runtime import FluxFillPipeline, FluxFillPipelineConfig, FluxFillPipelineResult
from backend.flux.flux_fill_pipeline import FluxEmptyConditioning, FluxFillLatentSource, load_flux_empty_conditioning_cache
from backend.flux.flux_fill_loader import load_flux_ae, load_flux_fill_unet


@dataclass
class FluxPromptConditioningCache:
    resolve_conditioning: Callable[[str], FluxEmptyConditioning] | None = None
    resolve_cache_path: Callable[[str], str | Path] | None = None
    _cache: dict[str, FluxEmptyConditioning] = field(default_factory=dict)
    load_count: int = 0
    hit_count: int = 0
    miss_count: int = 0

    def load(
        self,
        prompt: str | None = None,
        *,
        conditioning: FluxEmptyConditioning | None = None,
        conditioning_cache_path: str | Path | None = None,
        fallback_path: str | Path | None = None,
        progress: bool = True,
    ) -> tuple[FluxEmptyConditioning, dict[str, Any]]:
        prompt_text = str(prompt or "").strip()
        if conditioning is not None:
            return conditioning, {
                "stage": "conditioning_payload",
                "prompt": prompt_text,
                "cache_path": None,
                "cache_hit": False,
                "load_count": self.load_count,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "source": "explicit_payload",
            }

        if prompt_text and self.resolve_conditioning is not None and conditioning_cache_path is None:
            cache_key = f"prompt::{prompt_text}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.hit_count += 1
                return cached, {
                    "stage": "conditioning_payload",
                    "prompt": prompt_text,
                    "cache_path": None,
                    "cache_hit": True,
                    "load_count": self.load_count,
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "source": "live_prompt_cache",
                }
            try:
                live_conditioning = self.resolve_conditioning(prompt_text)
            except Exception:
                live_conditioning = None
            else:
                self._cache[cache_key] = live_conditioning
                self.load_count += 1
                self.miss_count += 1
                return live_conditioning, {
                    "stage": "conditioning_payload",
                    "prompt": prompt_text,
                    "cache_path": None,
                    "cache_hit": False,
                    "load_count": self.load_count,
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "source": "live_prompt_encoder",
                }

        if conditioning_cache_path is not None:
            cache_path = Path(conditioning_cache_path)
        elif prompt_text and self.resolve_cache_path is not None:
            cache_path = Path(self.resolve_cache_path(prompt_text))
        elif fallback_path is not None:
            cache_path = Path(fallback_path)
        else:
            raise ValueError("A conditioning cache path is required.")

        cache_key = str(cache_path)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.hit_count += 1
            return cached, {
                "stage": "conditioning_cache",
                "prompt": prompt_text,
                "cache_path": cache_key,
                "cache_hit": True,
                "load_count": self.load_count,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "source": "pt_cache",
            }

        conditioning = load_flux_empty_conditioning_cache(cache_path, map_location="cpu")
        self._cache[cache_key] = conditioning
        self.load_count += 1
        self.miss_count += 1
        return conditioning, {
            "stage": "conditioning_cache",
            "prompt": prompt_text,
            "cache_path": cache_key,
            "cache_hit": False,
            "load_count": self.load_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "source": "pt_cache",
        }

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class FluxFillStreamingRuntime:
    config: FluxFillPipelineConfig
    device: torch.device
    unet_patcher: Any | None = None
    started: bool = False
    start_count: int = 0
    end_count: int = 0
    unet_load_count: int = 0

    def start(self) -> None:
        self.config.validate_static(require_existing_assets=True)
        self.start_count += 1

        if self.unet_patcher is None:
            self.unet_patcher = load_flux_fill_unet(
                self.config.unet_path,
                load_device=self.device,
                offload_device=None,
                execution_class=self.config.execution_class,
                runtime_family=self.config.runtime_family,
                runtime_posture=self.config.runtime_posture,
                streaming_profile=self.config.streaming_profile,
                prefetch_depth=self.config.prefetch_depth,
                prefetch_chunk_mb=self.config.prefetch_chunk_mb,
                resident_load_strategy=self.config.resident_load_strategy,
            )
            self.unet_load_count += 1

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
        self.end_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "started": bool(self.started),
            "start_count": int(self.start_count),
            "end_count": int(self.end_count),
            "unet_load_count": int(self.unet_load_count),
            "unet_loaded": self.unet_patcher is not None,
            "unet_id": id(self.unet_patcher) if self.unet_patcher is not None else None,
        }


@dataclass
class FluxFillResidentRuntime:
    config: FluxFillPipelineConfig
    device: torch.device
    unet_patcher: Any | None = None
    started: bool = False
    start_count: int = 0
    end_count: int = 0
    unet_load_count: int = 0

    def start(self) -> None:
        self.config.validate_static(require_existing_assets=True)
        self.start_count += 1

        if self.unet_patcher is None:
            self.unet_patcher = load_flux_fill_unet(
                self.config.unet_path,
                load_device=self.device,
                offload_device=None,
                execution_class=self.config.execution_class,
                runtime_family=self.config.runtime_family,
                runtime_posture=self.config.runtime_posture,
                streaming_profile=self.config.streaming_profile,
                resident_load_strategy=self.config.resident_load_strategy,
            )
            self.unet_load_count += 1

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
        self.end_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "started": bool(self.started),
            "start_count": int(self.start_count),
            "end_count": int(self.end_count),
            "unet_load_count": int(self.unet_load_count),
            "unet_loaded": self.unet_patcher is not None,
            "unet_id": id(self.unet_patcher) if self.unet_patcher is not None else None,
        }


@dataclass
class FluxFillSession:
    config: FluxFillPipelineConfig
    device: torch.device | str | None = None
    conditioning_provider: FluxPromptConditioningCache | None = None
    unet_runtime: Any = None
    vae: Any | None = None
    vae_load_count: int = 0
    generation_count: int = 0
    conditioning_cache_count: int = 0
    latent_cache_hit_count: int = 0
    latent_cache_miss_count: int = 0
    latent_source_fingerprint: str | None = None
    latent_source: FluxFillLatentSource | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device(self.config.device) if self.config.device else resources.get_torch_device()
        else:
            self.device = torch.device(self.device)

        if self.unet_runtime is None:
            posture = str(self.config.runtime_posture or "").strip().lower()
            if posture == "streaming":
                self.unet_runtime = FluxFillStreamingRuntime(self.config, self.device)
            else:
                self.unet_runtime = FluxFillResidentRuntime(self.config, self.device)

    # Delegated properties for active UNet runtime
    @property
    def unet_patcher(self) -> Any:
        return self.unet_runtime.unet_patcher if self.unet_runtime else None

    @unet_patcher.setter
    def unet_patcher(self, value: Any) -> None:
        if self.unet_runtime:
            self.unet_runtime.unet_patcher = value

    @property
    def started(self) -> bool:
        return bool(self.unet_runtime.started) if self.unet_runtime else False

    @started.setter
    def started(self, value: bool) -> None:
        if self.unet_runtime:
            self.unet_runtime.started = value

    @property
    def start_count(self) -> int:
        return int(self.unet_runtime.start_count) if self.unet_runtime else 0

    @start_count.setter
    def start_count(self, value: int) -> None:
        if self.unet_runtime:
            self.unet_runtime.start_count = value

    @property
    def end_count(self) -> int:
        return int(self.unet_runtime.end_count) if self.unet_runtime else 0

    @end_count.setter
    def end_count(self, value: int) -> None:
        if self.unet_runtime:
            self.unet_runtime.end_count = value

    @property
    def unet_load_count(self) -> int:
        return int(self.unet_runtime.unet_load_count) if self.unet_runtime else 0

    @unet_load_count.setter
    def unet_load_count(self, value: int) -> None:
        if self.unet_runtime:
            self.unet_runtime.unet_load_count = value

    def _keep_vae_resident(self) -> bool:
        keep_vae_resident = getattr(self.config, "keep_vae_resident", None)
        if keep_vae_resident is None:
            return True
        return bool(keep_vae_resident)

    def _release_vae(self) -> None:
        if self.vae is None:
            return
        vae_patcher = getattr(self.vae, "patcher", None)
        if vae_patcher is not None:
            try:
                resources.eject_model(vae_patcher)
            except Exception:
                detach = getattr(vae_patcher, "detach", None)
                if callable(detach):
                    detach()
            try:
                if getattr(vae_patcher, "can_runtime_release", lambda: False)():
                    vae_patcher.release_weights_to_meta()
            except Exception:
                pass
        else:
            try:
                resources.eject_model(self.vae)
            except Exception:
                detach = getattr(self.vae, "detach", None)
                if callable(detach):
                    detach()
            try:
                if getattr(self.vae, "can_runtime_release", lambda: False)():
                    self.vae.release_weights_to_meta()
            except Exception:
                pass
        self.vae = None

    def _clear_latent_source_cache(self) -> None:
        self.latent_source_fingerprint = None
        self.latent_source = None

    def _supports_latent_source_cache(self, mode: str | None) -> bool:
        return str(mode or self.config.mode or "").strip().lower() == "baseline"

    def _build_latent_source_fingerprint(self, image: Any, mask: Any, *, mode: str | None) -> str:
        image_np = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        mask_np = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
        digest = hashlib.sha256()
        digest.update(b"flux_fill_latent_source/v1")
        digest.update(str(mode or self.config.mode or "").strip().lower().encode("utf-8"))
        digest.update(str(self.config.ae_path).encode("utf-8"))
        for label, value in (("image", image_np), ("mask", mask_np)):
            digest.update(label.encode("utf-8"))
            digest.update(str(value.shape).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _prepare_latent_source(
        self,
        pipeline: FluxFillPipeline,
        image: Any,
        mask: Any,
        *,
        vae: Any,
    ) -> FluxFillLatentSource:
        source_pixels, _source_pixels_summary = pipeline.prepare_source_pixels(image)
        concat_pixels, _concat_pixels_summary = pipeline.prepare_concat_pixels(image, mask)
        source_latent, _source_summary = pipeline.encode_source_latent(source_pixels, vae=vae, cleanup_vae=False)
        concat_latent, _concat_summary = pipeline.encode_concat_latent(concat_pixels, vae=vae, cleanup_vae=False)
        if tuple(source_latent.shape) != tuple(concat_latent.shape):
            raise ValueError(
                f"Flux latent source prep shape mismatch: {list(source_latent.shape)} vs {list(concat_latent.shape)}."
            )
        denoise_mask, _denoise_summary = pipeline.prepare_denoise_mask(mask, source_latent.shape)
        return FluxFillLatentSource(
            context=None,
            source_latent=source_latent.detach().cpu(),
            concat_latent=concat_latent.detach().cpu(),
            denoise_mask=denoise_mask.detach().cpu(),
            width=int(np.asarray(image).shape[1]),
            height=int(np.asarray(image).shape[0]),
        )

    def start(self) -> dict[str, Any]:
        if self.unet_runtime:
            self.unet_runtime.start()
        if self._keep_vae_resident() and self.vae is None:
            self.vae = load_flux_ae(self.config.ae_path, load_device=self.device, offload_device=None)
            self.vae_load_count += 1
        elif not self._keep_vae_resident():
            self._release_vae()
        return self.snapshot()

    def _resolve_conditioning(
        self,
        *,
        prompt: str | None = None,
        conditioning: FluxEmptyConditioning | None = None,
        conditioning_cache_path: str | Path | None = None,
        progress: bool = True,
    ) -> tuple[FluxEmptyConditioning, dict[str, Any]]:
        fallback_path = conditioning_cache_path or self.config.conditioning_cache_path
        if self.conditioning_provider is not None:
            conditioning, summary = self.conditioning_provider.load(
                prompt,
                conditioning=conditioning,
                conditioning_cache_path=conditioning_cache_path,
                fallback_path=fallback_path,
                progress=progress,
            )
        else:
            if conditioning is not None:
                summary = {
                    "stage": "conditioning_payload",
                    "prompt": str(prompt or "").strip(),
                    "cache_path": None,
                    "cache_hit": False,
                    "load_count": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "source": "explicit_payload",
                }
            else:
                conditioning = load_flux_empty_conditioning_cache(fallback_path, map_location="cpu")
                summary = {
                    "stage": "conditioning_cache",
                    "prompt": str(prompt or "").strip(),
                    "cache_path": str(fallback_path),
                    "cache_hit": False,
                    "load_count": 1,
                    "hit_count": 0,
                    "miss_count": 1,
                    "source": "pt_cache",
                }
        self.conditioning_cache_count += 1
        return conditioning, summary

    def _run_pipeline(
        self,
        image: Any,
        mask: Any,
        *,
        prompt: str | None = None,
        conditioning: FluxEmptyConditioning | None = None,
        conditioning_cache_path: str | Path | None = None,
        seed: int | None = None,
        steps: int | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        guidance: float | None = None,
        mode: str | None = None,
        blend_mode: str | None = None,
        callback: Any | None = None,
        disable_pbar: bool = True,
        progress: bool = True,
    ) -> FluxFillPipelineResult:
        self.start()
        conditioning, conditioning_summary = self._resolve_conditioning(
            prompt=prompt,
            conditioning=conditioning,
            conditioning_cache_path=conditioning_cache_path,
            progress=progress,
        )

        session_config = replace(
            self.config,
            seed=int(self.config.seed if seed is None else seed),
            steps=int(self.config.steps if steps is None else steps),
            sampler=self.config.sampler if sampler is None else str(sampler),
            scheduler=self.config.scheduler if scheduler is None else str(scheduler),
            guidance=float(self.config.guidance if guidance is None else guidance),
            mode=self.config.mode if mode is None else mode,
            blend_mode=self.config.blend_mode if blend_mode is None else blend_mode,
            conditioning_cache_path=conditioning_cache_path or self.config.conditioning_cache_path,
        )
        pipeline = FluxFillPipeline(session_config, device=self.device)
        keep_vae_resident = self._keep_vae_resident()
        runtime_vae = self.vae
        cleanup_vae = False
        if not keep_vae_resident:
            runtime_vae = load_flux_ae(self.config.ae_path, load_device=self.device, offload_device=None)
            self.vae_load_count += 1
            cleanup_vae = True
        latent_source = None
        latent_source_summary = {
            "cacheable": False,
            "cache_hit": False,
            "input_fingerprint": None,
            "cached": self.latent_source is not None,
        }
        if self._supports_latent_source_cache(session_config.mode):
            input_fingerprint = self._build_latent_source_fingerprint(image, mask, mode=session_config.mode)
            cache_hit = self.latent_source is not None and self.latent_source_fingerprint == input_fingerprint
            if cache_hit:
                latent_source = self.latent_source
                self.latent_cache_hit_count += 1
            else:
                latent_source = self._prepare_latent_source(pipeline, image, mask, vae=runtime_vae)
                self.latent_source = latent_source
                self.latent_source_fingerprint = input_fingerprint
                self.latent_cache_miss_count += 1
            latent_source_summary = {
                "cacheable": True,
                "cache_hit": cache_hit,
                "input_fingerprint": input_fingerprint,
                "cached": self.latent_source is not None,
            }
        result = pipeline.run(
            image,
            mask,
            disable_pbar=disable_pbar,
            unet_patcher=self.unet_patcher,
            vae=runtime_vae,
            cleanup_vae=cleanup_vae,
            empty_conditioning=conditioning,
            callback=callback,
            latent_source=latent_source,
        )
        result.debug_summary.setdefault("session", {})
        result.debug_summary["session"].update(
            {
                "started": self.started,
                "start_count": self.start_count,
                "unet_load_count": self.unet_load_count,
                "vae_load_count": self.vae_load_count,
                "keep_vae_resident": keep_vae_resident,
                "generation_count": self.generation_count + 1,
                "conditioning": conditioning_summary,
                "latent_source": latent_source_summary,
            }
        )
        self.generation_count += 1
        return result

    def generate_removal(
        self,
        image: Any,
        mask: Any,
        *,
        prompt: str | None = None,
        conditioning: FluxEmptyConditioning | None = None,
        conditioning_cache_path: str | Path | None = None,
        seed: int | None = None,
        steps: int | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        guidance: float | None = None,
        mode: str | None = None,
        blend_mode: str | None = None,
        callback: Any | None = None,
        disable_pbar: bool = True,
        progress: bool = True,
    ) -> FluxFillPipelineResult:
        return self._run_pipeline(
            image,
            mask,
            prompt=prompt,
            conditioning=conditioning,
            conditioning_cache_path=conditioning_cache_path,
            seed=seed,
            steps=steps,
            sampler=sampler,
            scheduler=scheduler,
            guidance=guidance,
            mode=mode,
            blend_mode=blend_mode,
            callback=callback,
            disable_pbar=disable_pbar,
            progress=progress,
        )

    def generate_inpaint(
        self,
        image: Any,
        mask: Any,
        *,
        prompt: str | None = None,
        conditioning: FluxEmptyConditioning | None = None,
        conditioning_cache_path: str | Path | None = None,
        seed: int | None = None,
        steps: int | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        guidance: float | None = None,
        mode: str | None = None,
        blend_mode: str | None = None,
        callback: Any | None = None,
        disable_pbar: bool = True,
        progress: bool = True,
    ) -> FluxFillPipelineResult:
        prompt_text = str(prompt or "").strip()
        if prompt_text == "" and conditioning is None and conditioning_cache_path is None:
            raise ValueError(
                "Flux Fill inpaint requires a non-empty prompt or explicit prompt-conditioned payload/cache; "
                "empty conditioning is removal-only."
            )
        return self._run_pipeline(
            image,
            mask,
            prompt=prompt,
            conditioning=conditioning,
            conditioning_cache_path=conditioning_cache_path,
            seed=seed,
            steps=steps,
            sampler=sampler,
            scheduler=scheduler,
            guidance=guidance,
            mode=mode,
            blend_mode=blend_mode,
            callback=callback,
            disable_pbar=disable_pbar,
            progress=progress,
        )

    def end(self) -> dict[str, Any]:
        if self.unet_runtime:
            self.unet_runtime.end()

        self._release_vae()

        self._clear_latent_source_cache()
        if self.conditioning_provider is not None:
            self.conditioning_provider.clear()
        gc.collect()
        try:
            resources.soft_empty_cache(force=True)
        except Exception:
            pass
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        unet_snap = self.unet_runtime.snapshot() if self.unet_runtime else {}
        return {
            "started": bool(unet_snap.get("started", False)),
            "start_count": int(unet_snap.get("start_count", 0)),
            "end_count": int(unet_snap.get("end_count", 0)),
            "unet_load_count": int(unet_snap.get("unet_load_count", 0)),
            "vae_load_count": int(self.vae_load_count),
            "generation_count": int(self.generation_count),
            "conditioning_cache_count": int(self.conditioning_cache_count),
            "latent_cache_hit_count": int(self.latent_cache_hit_count),
            "latent_cache_miss_count": int(self.latent_cache_miss_count),
            "unet_loaded": bool(unet_snap.get("unet_loaded", False)),
            "keep_vae_resident": self._keep_vae_resident(),
            "vae_loaded": self.vae is not None,
            "unet_id": unet_snap.get("unet_id"),
            "vae_id": id(self.vae) if self.vae is not None else None,
            "latent_source_cached": self.latent_source is not None,
            "latent_source_fingerprint": self.latent_source_fingerprint,
            "conditioning_provider": type(self.conditioning_provider).__name__ if self.conditioning_provider is not None else None,
        }
