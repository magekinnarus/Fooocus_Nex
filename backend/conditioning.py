import torch
import math
from typing import Any, Tuple, Union

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

def encode_text_sdxl(clip: Any, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes text using the provided CLIP model (SDXL specific, dual encoders).
    Returns (cond, pooled)
    """
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond, pooled

def timed_adm(y: torch.Tensor, timestep: torch.Tensor, model: Any, adm_scaler_end: float = 0.3) -> torch.Tensor:
    """
    Swaps from 'emphasized' ADM to 'consistent' ADM after a certain progress.
    y is expected to be [B, 5632] if scaling is enabled.
    """
    if y.shape[1] == 5632:
        model_sampling = model.model_sampling
        t = model_sampling.timestep(timestep)
        # progress = 1.0 - t / 999.0
        # Swap after progress > adm_scaler_end
        # which means t < 999 * (1.0 - adm_scaler_end)
        threshold = 999.0 * (1.0 - adm_scaler_end)
        
        y_mask = (t > threshold).to(y.dtype)
        # y_mask is 1.0 (True) during the first phase (emphasized)
        # y_mask is 0.0 (False) during the second phase (consistent)
        
        y_emphasized = y[:, :2816]
        y_consistent = y[:, 2816:]
        
        # Reshape y_mask for broadcasting if needed? t is usually a single value or batch
        if len(y_mask.shape) == 1:
            y_mask = y_mask.view(-1, 1)
        
        return y_emphasized * y_mask + y_consistent * (1.0 - y_mask)
    return y

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
    adm_scale_negative: float = 0.8
) -> torch.Tensor:
    """
    Generates ADM embeddings for SDXL with support for Fooocus-style scaling.
    Returns [B, 2816] if scaling disabled, or [B, 5632] if scaling enabled.
    """
    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    device = pooled.device
    dtype = pooled.dtype
    batch_size = pooled.shape[0]

    # Scaling logic
    orig_width, orig_height = width, height
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

    # If scaling is effective, return concatenated vector for timed_adm
    if (adm_scale_positive != 1.0 and prompt_type == "positive") or \
       (adm_scale_negative != 1.0 and prompt_type == "negative"):
        # Emphasized (scaled)
        flat_emphasized = get_embs(height, width, target_height, target_width)
        # Consistent (original)
        # Note: Fooocus uses target_height/target_width for both source and target in consistent mode
        flat_consistent = get_embs(target_height, target_width, target_height, target_width)
        
        # [B, 2816] x 2 concatenated
        adm_emphasized = torch.cat([pooled, flat_emphasized], dim=1)
        adm_consistent = torch.cat([pooled, flat_consistent], dim=1)
        
        return torch.cat([adm_emphasized, adm_consistent], dim=1) # [B, 5632]

    # Standard path
    flat_params = get_embs(height, width, target_height, target_width)
    return torch.cat([pooled, flat_params], dim=1) # [B, 2816]
