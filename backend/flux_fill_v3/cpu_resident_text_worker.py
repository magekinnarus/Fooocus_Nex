from __future__ import annotations
import gc
import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from backend.flux_fill_v3.contracts import (
    FluxFillRequest,
)
from backend.flux_fill_v3.conditioning_loader import (
    FluxEmptyConditioning,
    load_flux_empty_conditioning_cache,
    format_flux_conditioning_memory_summary,
)

from backend.flux_fill_v3.t5_worker import (
    _resolve_request_conditioning_cache_path,
    _describe_cache_path,
    save_flux_prompt_conditioning_cache,
)

logger = logging.getLogger(__name__)
_EXPECTED_EAGER_T5_RESIDUAL_KEYS = frozenset({"encoder.embed_tokens.weight"})


def _filter_expected_eager_t5_unexpected_keys(unexpected_keys: list[str]) -> list[str]:
    return [key for key in unexpected_keys if key not in _EXPECTED_EAGER_T5_RESIDUAL_KEYS]


def load_t5_state_dict_zero_copy(model: torch.nn.Module, sd: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Popping sequence to load state dict one parameter at a time.
    Replaces param.data references directly to avoid duplicate CPU RAM allocation (shadow copy).
    """
    missing_keys = []
    
    # Process named parameters
    for name, param in model.named_parameters():
        if name in sd:
            val = sd.pop(name)
            if hasattr(val, "load") and callable(val.load):
                val = val.load()
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            
            target_dtype = param.dtype
            cast_val = val.to(device=param.device, dtype=target_dtype)
            
            # Direct data assignment to reuse safetensors memory
            param.data = cast_val
        else:
            missing_keys.append(name)
            
    # Process named buffers
    for name, buf in model.named_buffers():
        if name in sd:
            val = sd.pop(name)
            if hasattr(val, "load") and callable(val.load):
                val = val.load()
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            
            target_dtype = buf.dtype
            cast_val = val.to(device=buf.device, dtype=target_dtype)
            
            buf.copy_(cast_val)
        else:
            missing_keys.append(name)
            
    unexpected_keys = list(sd.keys())
    sd.clear()
    gc.collect()
    
    return missing_keys, unexpected_keys


def load_flux_prompt_text_encoder_eager(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
) -> Any:
    """Eagerly loads the CLIP-L and T5 prompt text encoder into CPU RAM."""
    from backend.flux_fill_v3.t5_worker import (
        _load_text_encoder_state_dict,
        _detect_t5_dtype,
        FluxClipModel,
        FluxTokenizer,
        FluxPromptTextEncoder,
    )
    import backend.patching as patching
    from backend import precision
    from ldm_patched.modules import utils as comfy_utils

    clip_l_path = Path(clip_l_path)
    t5_path = Path(t5_path)
    
    # 1. Load CLIP-L state dict (~235MB)
    clip_l_sd, clip_l_options = _load_text_encoder_state_dict(clip_l_path)
    
    # 2. Load T5 safetensors state dict (~9.3GB) into CPU RAM
    logger.info(f"[Flux Telemetry] Loading eager T5 safetensors from: {t5_path}")
    t5_sd = comfy_utils.load_torch_file(str(t5_path), safe_load=True)
    detected_t5_dtype = _detect_t5_dtype(t5_sd)

    load_device = torch.device("cpu")
    offload_device = torch.device("cpu")
    dtype = precision.text_encoder_dtype(load_device)
    
    initial_device = torch.device("cpu")
    model_options = {
        "initial_device": initial_device,
    }
    model_options.update(clip_l_options)

    cond_stage_model = FluxClipModel(
        dtype_t5=detected_t5_dtype,
        device=initial_device,
        dtype=dtype,
        model_options=model_options,
    )
    tokenizer = FluxTokenizer(embedding_directory=embedding_directory)
    patcher = patching.NexModelPatcher(cond_stage_model, load_device=load_device, offload_device=offload_device)

    # 3. Load CLIP-L into model
    missing, unexpected = cond_stage_model.load_sd(clip_l_sd)
    if missing:
        logger.debug("Flux CLIP-L missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux CLIP-L unexpected keys: %s", unexpected)

    # 4. Zero-copy load of T5 into model to prevent shadow copy/double peak memory
    missing, unexpected = load_t5_state_dict_zero_copy(cond_stage_model.t5xxl.transformer, t5_sd)
    unexpected = _filter_expected_eager_t5_unexpected_keys(unexpected)
    if missing:
        logger.debug("Flux T5 missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux T5 unexpected keys: %s", unexpected)

    encoder = FluxPromptTextEncoder(cond_stage_model=cond_stage_model, tokenizer=tokenizer, patcher=patcher)
    
    setattr(
        encoder,
        "_nex_load_metadata",
        {
            "clip_l_path": str(clip_l_path),
            "t5_path": str(t5_path),
            "t5_loader_policy": "eager",
            "t5_detected_dtype": str(detected_t5_dtype) if detected_t5_dtype is not None else None,
            "t5_source_kind": "safetensors_eager",
            "t5_full_state_dict_materialized": True,
            "t5_stream_runtime": False,
            "t5_lazy_runtime": False,
        },
    )
    return encoder


def _log_cpu_resident_encoder_state(
    *,
    event: str,
    clip_l_path: Path | None = None,
    t5_path: Path | None = None,
    reused: bool | None = None,
    level: int = logging.DEBUG,
) -> None:
    details: list[str] = [f"event={event}"]
    if clip_l_path is not None:
        details.append(f"clip_l={clip_l_path}")
    if t5_path is not None:
        details.append(f"t5={t5_path}")
    if reused is not None:
        details.append(f"reused={reused}")
    details.append(format_flux_conditioning_memory_summary(tag=event))
    logger.log(
        level,
        "[Flux Telemetry] CPU-resident text encoder state %s",
        " ".join(details),
    )


class CpuResidentTextEncoderCache:
    """Thread-safe registry for acquiring, caching, and tearing down the eager T5 text encoder."""
    _lock = RLock()
    _encoder: Any = None
    _clip_l_path: Path | None = None
    _t5_path: Path | None = None

    @classmethod
    def acquire(cls, clip_l_path: Path, t5_path: Path) -> tuple[Any, bool]:
        clip_l_path = Path(clip_l_path)
        t5_path = Path(t5_path)
        with cls._lock:
            if cls._encoder is not None and cls._clip_l_path == clip_l_path and cls._t5_path == t5_path:
                logger.debug("[Flux Telemetry] Reusing cached CPU-resident text encoder.")
                _log_cpu_resident_encoder_state(
                    event="cpu_resident_encoder_reuse",
                    clip_l_path=clip_l_path,
                    t5_path=t5_path,
                    reused=True,
                )
                return cls._encoder, True
            
            cls.teardown()

            cls._encoder = load_flux_prompt_text_encoder_eager(
                clip_l_path=clip_l_path,
                t5_path=t5_path,
            )
            cls._clip_l_path = clip_l_path
            cls._t5_path = t5_path
            _log_cpu_resident_encoder_state(
                event="cpu_resident_encoder_ready",
                clip_l_path=clip_l_path,
                t5_path=t5_path,
                reused=False,
                level=logging.INFO,
            )
            return cls._encoder, False

    @classmethod
    def teardown(cls) -> bool:
        with cls._lock:
            if cls._encoder is None:
                return False
            _log_cpu_resident_encoder_state(
                event="cpu_resident_encoder_teardown_begin",
                clip_l_path=cls._clip_l_path,
                t5_path=cls._t5_path,
            )
            logger.debug("[Flux Telemetry] Tearing down and releasing CPU-resident text encoder from RAM.")
            cls._encoder = None
            cls._clip_l_path = None
            cls._t5_path = None

        gc.collect()
        try:
            from backend import resources
            resources.soft_empty_cache(force=True)
        except Exception:
            pass
        _log_cpu_resident_encoder_state(event="cpu_resident_encoder_teardown_complete")
        return True


class CpuResidentTextWorker:
    """Greenfield CPU Resident T5 Posture/Worker Contract."""
    def __init__(self, request: FluxFillRequest) -> None:
        self.request = request

    def get_conditioning(self) -> FluxEmptyConditioning:
        if not self.request.prompt or not str(self.request.prompt).strip():
            logger.debug(
                "[Flux Telemetry] Empty prompt, loading empty conditioning cache. %s",
                format_flux_conditioning_memory_summary(tag="cpu_resident_empty_prompt"),
            )
            return load_flux_empty_conditioning_cache(self.request.conditioning_cache_path)

        return self._generate_conditioning_resident()

    def _generate_conditioning_resident(self) -> FluxEmptyConditioning:

        clip_l_path = Path(self.request.clip_l_path)
        t5_path = Path(self.request.t5_path)
        prompt_text = str(self.request.prompt or "").strip()

        cache_path = _resolve_request_conditioning_cache_path(self.request)
        logger.debug(
            "[Flux Telemetry] T5 conditioning begin posture=cpu_resident prompt_len=%d cache_path=%s %s %s",
            len(prompt_text),
            cache_path,
            _describe_cache_path(cache_path),
            format_flux_conditioning_memory_summary(tag="conditioning_begin"),
        )
        logger.debug(f"[Flux Telemetry] Checking prompt conditioning cache at: {cache_path}")
        if cache_path.exists():
            logger.debug(f"[Flux Telemetry] Prompt conditioning cache HIT for path: {cache_path}")
            try:
                return load_flux_empty_conditioning_cache(cache_path)
            except Exception:
                logger.exception(
                    "[Flux Telemetry] Prompt conditioning cache reuse failed path=%s posture=cpu_resident %s",
                    cache_path,
                    format_flux_conditioning_memory_summary(tag="conditioning_cache_reuse_failed"),
                )
                raise

        logger.debug(
            "[Flux Telemetry] Prompt conditioning cache MISS. Acquiring CPU-resident text encoder clip_l=%s t5=%s",
            clip_l_path,
            t5_path,
        )

        encoder, reused = CpuResidentTextEncoderCache.acquire(clip_l_path, t5_path)
        
        start_time = time.perf_counter()
        cond, pooled = encoder.encode(prompt_text)
        duration = time.perf_counter() - start_time
        logger.debug(f"[Flux Telemetry] CPU-resident T5 prompt encoding completed in {duration:.3f} seconds.")

        metadata = {
            "prompt": prompt_text,
            "duration": duration,
            "posture": "cpu_resident",
            "reused": reused,
        }
        return save_flux_prompt_conditioning_cache(cache_path, cross_attn=cond, pooled_output=pooled, metadata=metadata)
