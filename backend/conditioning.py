import math
from typing import Any, Dict, Tuple

import torch


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


def encode_prompt_pair_sdxl(
    clip: Any,
    positive_text: str,
    negative_text: str,
    *,
    use_explicit_residency: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Encodes positive and negative prompt text through a shared SDXL CLIP instance.
    """
    positive_cond, positive_pooled = encode_text_sdxl(
        clip,
        positive_text,
        use_explicit_residency=use_explicit_residency,
    )
    negative_cond, negative_pooled = encode_text_sdxl(
        clip,
        negative_text,
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
