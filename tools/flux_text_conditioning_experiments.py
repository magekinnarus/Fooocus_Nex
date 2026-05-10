from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from backend import patching, resources
from backend.flux import text_conditioning as tc
from backend.gguf.ops import GGMLOps
from backend.gguf.patcher import GGUFModelPatcher
from ldm_patched.modules import model_management


_RESIDENT_ENCODER_CACHE: dict[tuple[str, str, str | None, str | None, str | None], tc.FluxPromptTextEncoder] = {}


def _resident_encoder_key(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        str(Path(clip_l_path)),
        str(Path(t5_path)),
        str(Path(embedding_directory)) if embedding_directory is not None else None,
        str(torch.device(load_device)) if load_device is not None else None,
        str(torch.device(offload_device)) if offload_device is not None else None,
    )


def clear_flux_prompt_text_encoder_cache() -> None:
    cached = list(_RESIDENT_ENCODER_CACHE.values())
    _RESIDENT_ENCODER_CACHE.clear()
    for encoder in cached:
        try:
            del encoder
        except Exception:
            pass
    try:
        resources.soft_empty_cache(force=True)
    except Exception:
        pass


def load_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
) -> tc.FluxPromptTextEncoder:
    clip_l_path = Path(clip_l_path)
    t5_path = Path(t5_path)
    clip_l_sd, clip_l_options = tc._load_text_encoder_state_dict(clip_l_path)
    t5_sd, t5_options = tc._load_text_encoder_state_dict(t5_path)

    load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    dtype = model_management.text_encoder_dtype(load_device)
    model_options: dict[str, Any] = {}
    model_options.update(clip_l_options)
    model_options.update(t5_options)
    initial_device = torch.device("cpu")
    model_options["initial_device"] = initial_device

    cond_stage_model = tc.FluxClipModel(
        dtype_t5=tc._detect_t5_dtype(t5_sd),
        device=initial_device,
        dtype=dtype,
        model_options=model_options,
    )
    tokenizer = tc.FluxTokenizer(embedding_directory=embedding_directory)
    patcher_cls = GGUFModelPatcher if model_options.get("custom_operations") is GGMLOps else patching.NexModelPatcher
    patcher = patcher_cls(cond_stage_model, load_device=load_device, offload_device=offload_device)

    missing, unexpected = cond_stage_model.load_sd(clip_l_sd)
    if missing:
        tc.logger.debug("Flux CLIP-L missing keys: %s", missing)
    if unexpected:
        tc.logger.debug("Flux CLIP-L unexpected keys: %s", unexpected)

    missing, unexpected = cond_stage_model.load_sd(t5_sd)
    if missing:
        tc.logger.debug("Flux T5 missing keys: %s", missing)
    if unexpected:
        tc.logger.debug("Flux T5 unexpected keys: %s", unexpected)

    return tc.FluxPromptTextEncoder(cond_stage_model=cond_stage_model, tokenizer=tokenizer, patcher=patcher)


def get_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
) -> tc.FluxPromptTextEncoder:
    if not keep_resident:
        return load_flux_prompt_text_encoder(
            clip_l_path=clip_l_path,
            t5_path=t5_path,
            embedding_directory=embedding_directory,
            load_device=load_device,
            offload_device=offload_device,
        )

    key = _resident_encoder_key(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
    )
    cached = _RESIDENT_ENCODER_CACHE.get(key)
    if cached is not None:
        return cached

    clear_flux_prompt_text_encoder_cache()
    encoder = load_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
    )
    _RESIDENT_ENCODER_CACHE[key] = encoder
    return encoder


def encode_flux_prompt_conditioning(
    prompt: str,
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
) -> tc.FluxEmptyConditioning:
    prompt_text = str(prompt or "").strip()
    if prompt_text == "":
        raise ValueError("Flux prompt conditioning requires a non-empty prompt.")

    encoder = get_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        keep_resident=keep_resident,
    )
    try:
        cross_attn, pooled_output = encoder.encode(prompt_text)
        return tc.FluxEmptyConditioning(
            cross_attn=cross_attn.to(device="cpu"),
            pooled_output=pooled_output.to(device="cpu"),
            metadata={
                "prompt": prompt_text,
                "clip_l_path": str(clip_l_path),
                "t5_path": str(t5_path),
                "t5_format": "gguf" if str(t5_path).lower().endswith(".gguf") else "safetensors",
                "generator": "tools/flux_text_conditioning_experiments.py",
                "conditioning_kind": "prompt",
                "transport": "memory",
                "text_encoder_resident": bool(keep_resident),
                "load_device": str(torch.device(load_device)) if load_device is not None else "cpu",
                "offload_device": str(torch.device(offload_device)) if offload_device is not None else "cpu",
            },
        )
    finally:
        if not keep_resident:
            del encoder
            gc.collect()
            try:
                resources.soft_empty_cache()
            except Exception:
                pass


def save_flux_prompt_conditioning_cache(
    prompt: str,
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    output_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
) -> tc.FluxEmptyConditioning:
    conditioning = encode_flux_prompt_conditioning(
        prompt,
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        keep_resident=keep_resident,
    )
    return tc.save_flux_empty_conditioning_cache(
        output_path,
        cross_attn=conditioning.cross_attn,
        pooled_output=conditioning.pooled_output,
        metadata=dict(conditioning.metadata, transport="pt_cache"),
    )
