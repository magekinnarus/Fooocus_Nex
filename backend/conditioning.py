import torch
import torch.nn as nn
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

def get_adm_embeddings_sdxl(
    pooled: torch.Tensor,
    width: int,
    height: int,
    crop_w: int = 0,
    crop_h: int = 0,
    target_width: int = None,
    target_height: int = None
) -> torch.Tensor:
    """
    Generates ADM (Adaptive Domain Mixing) embeddings for SDXL.
    Concatenates pooled CLIP output with resolution and crop embeddings.
    """
    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    device = pooled.device
    dtype = pooled.dtype
    batch_size = pooled.shape[0]

    # Sinusoidal embeddings (256-dim each)
    # Order: height, width, crop_h, crop_w, target_height, target_width
    params = [height, width, crop_h, crop_w, target_height, target_width]
    embs = []
    
    for p in params:
        t = torch.tensor([p], device=device)
        emb = get_timestep_embedding(t, 256) # [1, 256]
        embs.append(emb)

    # Concatenate param embeddings: [1, 1536]
    flat_params = torch.cat(embs, dim=1)
    
    # Repeat for batch size and match dtype: [B, 1536]
    flat_params = flat_params.repeat(batch_size, 1).to(dtype)

    # Final ADM embedding: concat(pooled, flat_params) -> [B, 1280 + 1536] = [B, 2816]
    return torch.cat([pooled, flat_params], dim=1)
