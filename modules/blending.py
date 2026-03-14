import torch
import numpy as np
import math

def sin_blend_1d(length: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Source of truth for sin2 blending.
    Produces a 1D tensor of 'length' transitioning from 0 to 1 following 0.5 * (1 - cos(pi * t))
    """
    if length <= 0: 
        return torch.tensor([], device=device, dtype=dtype)
    t = torch.linspace(0, math.pi, length, device=device, dtype=dtype)
    return 0.5 * (1.0 - torch.cos(t))

def sin_bell_1d(length: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Produces a 1D tensor of 'length' transitioning from 0 to 1 and back to 0.
    Follows sin^2(t) from 0 to pi.
    """
    if length <= 0: 
        return torch.tensor([], device=device, dtype=dtype)
    t = torch.linspace(0, math.pi, length, device=device, dtype=dtype)
    return torch.sin(t) ** 2

def sin_blend_2d(width: int, height: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Produces a 2D weight map (height, width) with 0 at edges and 1 at center.
    """
    w_x = sin_bell_1d(width, device, dtype)
    w_y = sin_bell_1d(height, device, dtype)
    return w_y.view(-1, 1) * w_x.view(1, -1)

def apply_sin2_curve(x):
    """
    Remaps [0, 1] linear weights to sin2 curve: 0.5 * (1 - cos(pi * x))
    Works with both numpy arrays and torch tensors.
    """
    if isinstance(x, np.ndarray):
        return 0.5 * (1.0 - np.cos(np.pi * x))
    
    # Handle torch tensor
    return 0.5 * (1.0 - torch.cos(math.pi * x))
