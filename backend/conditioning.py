import math
from dataclasses import asdict, dataclass
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import hashlib
import torch

from backend import lora_artifacts, resources


_PROMPT_CONDITIONING_CACHE: OrderedDict[str, Dict[str, Dict[str, torch.Tensor]]] = OrderedDict()
_PROMPT_CONDITIONING_CACHE_LIMIT = 32


@dataclass(frozen=True)
class StageFingerprint:
    stage_name: str
    residency_class: str
    components: tuple[tuple[str, Any], ...]

    def as_key(self) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        return self.stage_name, self.residency_class, self.components

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.as_key()).encode("utf-8"))
        return digest.hexdigest()


def build_stage_fingerprint(
    stage_name: str,
    *,
    residency_class: str | None = None,
    **inputs: Any,
) -> StageFingerprint:
    normalized_residency = resources.normalize_sdxl_residency_class(residency_class)
    components = tuple(
        (str(key), _freeze_stage_value(value))
        for key, value in sorted(inputs.items(), key=lambda item: str(item[0]))
    )
    return StageFingerprint(
        stage_name=str(stage_name),
        residency_class=normalized_residency,
        components=components,
    )


def build_sdxl_text_conditioning_fingerprint(
    *,
    prompt: str,
    negative_prompt: str,
    positive_texts: Any = None,
    negative_texts: Any = None,
    positive_top_k: Any = None,
    negative_top_k: Any = None,
    model_identity: Any,
    text_encoder_identity: Any,
    clip_patch_uuid: Any,
    clip_layer_idx: Any,
    lora_artifacts_state: Any,
    route_family_reconciliation_signature: Any,
    residency_class: str | None = None,
    route_family: str | None = None,
    execution_family: Any = None,
    clip_residency_mode: Any = None,
) -> StageFingerprint:
    return build_stage_fingerprint(
        "sdxl_text_conditioning",
        residency_class=residency_class,
        prompt=prompt,
        negative_prompt=negative_prompt,
        positive_texts=positive_texts,
        negative_texts=negative_texts,
        positive_top_k=positive_top_k,
        negative_top_k=negative_top_k,
        model_identity=model_identity,
        text_encoder_identity=text_encoder_identity,
        clip_patch_uuid=clip_patch_uuid,
        clip_layer_idx=clip_layer_idx,
        lora_signature=lora_artifacts.artifact_registry_signature(lora_artifacts_state),
        route_family_reconciliation_signature=route_family_reconciliation_signature,
        route_family=route_family,
        execution_family=execution_family,
        clip_residency_mode=clip_residency_mode,
    )


def build_sdxl_prepared_payload_fingerprint(
    stage_name: str,
    *,
    residency_class: str | None = None,
    model_identity: Any,
    route_family_reconciliation_signature: Any,
    prepared_artifact_signature: Any,
    **inputs: Any,
) -> StageFingerprint:
    payload = dict(inputs)
    payload.update(
        model_identity=model_identity,
        route_family_reconciliation_signature=route_family_reconciliation_signature,
        prepared_artifact_signature=prepared_artifact_signature,
    )
    return build_stage_fingerprint(
        stage_name,
        residency_class=residency_class,
        **payload,
    )


def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _freeze_stage_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _freeze_stage_value(asdict(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", hashlib.sha256(bytes(value)).hexdigest())
    if isinstance(value, dict):
        return tuple((str(key), _freeze_stage_value(item)) for key, item in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_stage_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_stage_value(item) for item in value))
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            return (
                "ndarray",
                tuple(int(dim) for dim in contiguous.shape),
                str(contiguous.dtype),
                hashlib.sha256(contiguous.tobytes()).hexdigest(),
            )
    except Exception:
        pass
    if isinstance(value, torch.Tensor):
        contiguous = value.detach().cpu().contiguous()
        return (
            "tensor",
            tuple(int(dim) for dim in contiguous.shape),
            str(contiguous.dtype),
            hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        )
    return value


def _clone_prompt_conditioning_payload(value: Any, *, device: str | torch.device = "cpu") -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: _clone_prompt_conditioning_payload(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_prompt_conditioning_payload(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_prompt_conditioning_payload(item, device=device) for item in value)
    return value


def load_prompt_conditioning_from_cache(fingerprint: StageFingerprint | None) -> Dict[str, Dict[str, torch.Tensor]] | None:
    if fingerprint is None:
        return None

    cache_key = fingerprint.digest()
    cached = _PROMPT_CONDITIONING_CACHE.get(cache_key)
    if cached is None:
        return None

    _PROMPT_CONDITIONING_CACHE.move_to_end(cache_key)
    return _clone_prompt_conditioning_payload(cached, device="cpu")


def remember_prompt_conditioning_cache(
    fingerprint: StageFingerprint | None,
    payload: Dict[str, Dict[str, torch.Tensor]] | None,
) -> None:
    if fingerprint is None or payload is None:
        return

    cache_key = fingerprint.digest()
    _PROMPT_CONDITIONING_CACHE[cache_key] = _clone_prompt_conditioning_payload(payload, device="cpu")
    _PROMPT_CONDITIONING_CACHE.move_to_end(cache_key)
    while len(_PROMPT_CONDITIONING_CACHE) > _PROMPT_CONDITIONING_CACHE_LIMIT:
        _PROMPT_CONDITIONING_CACHE.popitem(last=False)


def clear_prompt_conditioning_cache() -> None:
    _PROMPT_CONDITIONING_CACHE.clear()


def encode_tokens_sdxl(clip: Any, tokens: Any, *, use_explicit_residency: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes pre-tokenized SDXL text.
    When use_explicit_residency is True, the caller owns CLIP attach/detach.
    """
    if use_explicit_residency and hasattr(clip, "encode_from_tokens_resident"):
        cond, pooled = clip.encode_from_tokens_resident(tokens, return_pooled=True)
    else:
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond, pooled


def encode_text_sdxl(clip: Any, text: str, *, use_explicit_residency: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes text using the provided CLIP model (SDXL specific, dual encoders).
    Returns (cond, pooled).
    """
    tokens = clip.tokenize(text)
    return encode_tokens_sdxl(clip, tokens, use_explicit_residency=use_explicit_residency)


def encode_text_list_sdxl(
    clip: Any,
    texts: Sequence[str],
    *,
    pool_top_k: int = 1,
    use_explicit_residency: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a prompt workload list using Fooocus-style SDXL pooling semantics.

    The returned conditioning concatenates every text block along the sequence
    dimension, while the pooled output sums the first ``pool_top_k`` entries.
    """
    items = [str(text) for text in list(texts or []) if str(text) != ""]
    if len(items) == 0:
        raise ValueError("encode_text_list_sdxl requires at least one prompt text.")

    cond_list: list[torch.Tensor] = []
    pooled_acc: torch.Tensor | None = None
    pooled_template: torch.Tensor | None = None
    top_k = max(0, min(int(pool_top_k), len(items)))
    for index, text in enumerate(items):
        cond, pooled = encode_text_sdxl(
            clip,
            text,
            use_explicit_residency=use_explicit_residency,
        )
        cond_list.append(cond)
        if pooled_template is None:
            pooled_template = pooled
        if index < top_k:
            pooled_acc = pooled if pooled_acc is None else pooled_acc + pooled

    if pooled_acc is None:
        pooled_acc = torch.zeros_like(pooled_template)
    return torch.cat(cond_list, dim=1), pooled_acc


def encode_prompt_pair_sdxl(
    clip: Any,
    positive_text: str,
    negative_text: str,
    *,
    positive_texts: Sequence[str] | None = None,
    negative_texts: Sequence[str] | None = None,
    positive_top_k: int = 1,
    negative_top_k: int = 1,
    use_explicit_residency: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Encodes positive and negative prompt text through a shared SDXL CLIP instance.
    """
    positive_items = list(positive_texts or [positive_text])
    negative_items = list(negative_texts or [negative_text])
    positive_cond, positive_pooled = encode_text_list_sdxl(
        clip,
        positive_items,
        pool_top_k=positive_top_k,
        use_explicit_residency=use_explicit_residency,
    )
    negative_cond, negative_pooled = encode_text_list_sdxl(
        clip,
        negative_items,
        pool_top_k=negative_top_k,
        use_explicit_residency=use_explicit_residency,
    )
    return {
        "positive": {"cond": positive_cond, "pooled": positive_pooled},
        "negative": {"cond": negative_cond, "pooled": negative_pooled},
    }


def timed_adm(y: torch.Tensor, timestep: torch.Tensor, model: Any, adm_scaler_end: float = 0.3) -> torch.Tensor:
    """
    Swaps from 'emphasized' ADM to 'consistent' ADM after a certain progress.
    y is expected to be [B, 5632] if scaling is enabled.
    """
    if y.shape[1] == 5632:
        model_sampling = model.model_sampling
        t = model_sampling.timestep(timestep)
        threshold = 999.0 * (1.0 - adm_scaler_end)

        y_mask = (t > threshold).to(y.dtype)

        y_emphasized = y[:, :2816]
        y_consistent = y[:, 2816:]

        if len(y_mask.shape) == 1:
            y_mask = y_mask.view(-1, 1)

        return y_emphasized * y_mask + y_consistent * (1.0 - y_mask)
    return y


def build_sdxl_adm(
    pooled: torch.Tensor,
    width: int,
    height: int,
    crop_w: int = 0,
    crop_h: int = 0,
    target_width: int = None,
    target_height: int = None,
    prompt_type: str = "positive",
    adm_scale_positive: float = 1.5,
    adm_scale_negative: float = 0.8,
) -> torch.Tensor:
    """
    Generates a single SDXL ADM embedding tensor with support for Fooocus-style scaling.
    Returns [B, 2816] if scaling disabled, or [B, 5632] if scaling enabled.
    """
    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    device = pooled.device
    dtype = pooled.dtype
    batch_size = pooled.shape[0]

    if prompt_type == "positive":
        width = int(float(width) * adm_scale_positive)
        height = int(float(height) * adm_scale_positive)
    elif prompt_type == "negative":
        width = int(float(width) * adm_scale_negative)
        height = int(float(height) * adm_scale_negative)

    def get_embs(h, w, th, tw):
        params = [h, w, crop_h, crop_w, th, tw]
        embs = []
        for p in params:
            t = torch.tensor([p], device=device)
            emb = get_timestep_embedding(t, 256)
            embs.append(emb)
        return torch.cat(embs, dim=1).repeat(batch_size, 1).to(dtype)

    if (adm_scale_positive != 1.0 and prompt_type == "positive") or \
       (adm_scale_negative != 1.0 and prompt_type == "negative"):
        flat_emphasized = get_embs(height, width, target_height, target_width)
        flat_consistent = get_embs(target_height, target_width, target_height, target_width)

        adm_emphasized = torch.cat([pooled, flat_emphasized], dim=1)
        adm_consistent = torch.cat([pooled, flat_consistent], dim=1)

        return torch.cat([adm_emphasized, adm_consistent], dim=1)

    flat_params = get_embs(height, width, target_height, target_width)
    return torch.cat([pooled, flat_params], dim=1)


def build_sdxl_adm_pair(
    encoded_prompt_pair: Dict[str, Dict[str, torch.Tensor]],
    width: int,
    height: int,
    crop_w: int = 0,
    crop_h: int = 0,
    target_width: int = None,
    target_height: int = None,
    adm_scale_positive: float = 1.5,
    adm_scale_negative: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    Builds positive and negative SDXL ADM tensors from an encoded prompt pair.
    """
    return {
        "positive": build_sdxl_adm(
            encoded_prompt_pair["positive"]["pooled"],
            width,
            height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
            prompt_type="positive",
            adm_scale_positive=adm_scale_positive,
            adm_scale_negative=adm_scale_negative,
        ),
        "negative": build_sdxl_adm(
            encoded_prompt_pair["negative"]["pooled"],
            width,
            height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
            prompt_type="negative",
            adm_scale_positive=adm_scale_positive,
            adm_scale_negative=adm_scale_negative,
        ),
    }


def get_adm_embeddings_sdxl(
    pooled: torch.Tensor,
    width: int,
    height: int,
    crop_w: int = 0,
    crop_h: int = 0,
    target_width: int = None,
    target_height: int = None,
    prompt_type: str = "positive",
    adm_scale_positive: float = 1.5,
    adm_scale_negative: float = 0.8,
) -> torch.Tensor:
    """
    Compatibility wrapper for callers still using the older ADM helper name.
    """
    return build_sdxl_adm(
        pooled,
        width,
        height,
        crop_w=crop_w,
        crop_h=crop_h,
        target_width=target_width,
        target_height=target_height,
        prompt_type=prompt_type,
        adm_scale_positive=adm_scale_positive,
        adm_scale_negative=adm_scale_negative,
    )
