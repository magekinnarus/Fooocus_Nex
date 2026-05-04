from __future__ import annotations

import gc
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import torch

from backend import resources
from backend.flux.flux_runtime import FluxFillPipeline, FluxFillPipelineConfig, FluxFillPipelineResult
from backend.flux.flux_fill_pipeline import FluxEmptyConditioning, load_flux_ae, load_flux_empty_conditioning_cache, load_flux_fill_unet


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

        if prompt_text and self.resolve_conditioning is not None:
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
class FluxFillSession:
    config: FluxFillPipelineConfig
    device: torch.device | str | None = None
    conditioning_provider: FluxPromptConditioningCache | None = None
    unet_patcher: Any | None = None
    vae: Any | None = None
    started: bool = False
    start_count: int = 0
    end_count: int = 0
    unet_load_count: int = 0
    vae_load_count: int = 0
    generation_count: int = 0
    conditioning_cache_count: int = 0

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device(self.config.device) if self.config.device else resources.get_torch_device()
        else:
            self.device = torch.device(self.device)

    def start(self) -> dict[str, Any]:
        self.config.validate_static(require_existing_assets=True)
        self.start_count += 1

        if self.unet_patcher is None:
            self.unet_patcher = load_flux_fill_unet(self.config.unet_path, load_device=self.device, offload_device=None)
            self.unet_load_count += 1
        if self.vae is None:
            self.vae = load_flux_ae(self.config.ae_path, load_device=self.device, offload_device=None)
            self.vae_load_count += 1

        self.started = True
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
        mode: str | None = None,
        blend_mode: str | None = None,
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
            mode=self.config.mode if mode is None else mode,
            blend_mode=self.config.blend_mode if blend_mode is None else blend_mode,
            conditioning_cache_path=conditioning_cache_path or self.config.conditioning_cache_path,
        )
        pipeline = FluxFillPipeline(session_config, device=self.device)
        result = pipeline.run(
            image,
            mask,
            disable_pbar=disable_pbar,
            unet_patcher=self.unet_patcher,
            vae=self.vae,
            empty_conditioning=conditioning,
        )
        result.debug_summary.setdefault("session", {})
        result.debug_summary["session"].update(
            {
                "started": self.started,
                "start_count": self.start_count,
                "unet_load_count": self.unet_load_count,
                "vae_load_count": self.vae_load_count,
                "generation_count": self.generation_count + 1,
                "conditioning": conditioning_summary,
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
        mode: str | None = None,
        blend_mode: str | None = None,
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
            mode=mode,
            blend_mode=blend_mode,
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
        mode: str | None = None,
        blend_mode: str | None = None,
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
            mode=mode,
            blend_mode=blend_mode,
            disable_pbar=disable_pbar,
            progress=progress,
        )

    def end(self) -> dict[str, Any]:
        if self.unet_patcher is not None:
            try:
                resources.eject_model(self.unet_patcher)
            except Exception:
                detach = getattr(self.unet_patcher, "detach", None)
                if callable(detach):
                    detach()
            finally:
                self.unet_patcher = None

        if self.vae is not None:
            try:
                resources.eject_model(self.vae.patcher)
            except Exception:
                detach = getattr(self.vae.patcher, "detach", None)
                if callable(detach):
                    detach()
            finally:
                self.vae = None

        self.started = False
        self.end_count += 1
        if self.conditioning_provider is not None:
            self.conditioning_provider.clear()
        gc.collect()
        try:
            resources.soft_empty_cache(force=True)
        except Exception:
            pass
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "started": bool(self.started),
            "start_count": int(self.start_count),
            "end_count": int(self.end_count),
            "unet_load_count": int(self.unet_load_count),
            "vae_load_count": int(self.vae_load_count),
            "generation_count": int(self.generation_count),
            "conditioning_cache_count": int(self.conditioning_cache_count),
            "unet_loaded": self.unet_patcher is not None,
            "vae_loaded": self.vae is not None,
            "unet_id": id(self.unet_patcher) if self.unet_patcher is not None else None,
            "vae_id": id(self.vae) if self.vae is not None else None,
            "conditioning_provider": type(self.conditioning_provider).__name__ if self.conditioning_provider is not None else None,
        }
