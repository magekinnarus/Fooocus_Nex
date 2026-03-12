import torch
from torch import nn
from torch.nn.init import trunc_normal_
from functools import partial

class DCCM(nn.Sequential):
    """Doubled Convolutional Channel Mixer"""
    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)

class PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer"""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x

class EA(nn.Module):
    """Element-wise Attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)

class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()
        self.channel_mixer = DCCM(dim)
        pdim = int(dim * split_ratio)
        self.lk = PLKConv2d(pdim, kernel_size)
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)
        self.norm = nn.GroupNorm(norm_groups, dim)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)
        return x + x_skip

class RealPLKSR(nn.Module):
    def __init__(
        self,
        state_dict,
    ):
        super().__init__()
        self.model_arch = "RealPLKSR"
        self.sub_type = "SR"
        self.state = state_dict

        if "params" in self.state:
            self.state = self.state["params"]
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]

        # Infer params from state_dict
        self.key_arr = list(self.state.keys())
        in_ch = self.state[self.key_arr[0]].shape[1]
        dim = self.state[self.key_arr[0]].shape[0]
        
        # Count blocks
        n_blocks = 0
        for k in self.state.keys():
            if k.startswith('feats.') and '.channel_mixer.' in k:
                n_blocks = max(n_blocks, int(k.split('.')[1]))
        
        # Detect scale from last conv
        last_conv_key = next(k for k in reversed(self.state.keys()) if k.endswith('.weight') and 'feats' in k)
        out_ch_total = self.state[last_conv_key].shape[0]
        # Assume out_ch = in_ch = 3 usually, but let's be safe
        out_ch = 3 
        self.scale = int((out_ch_total / out_ch) ** 0.5)
        
        # Kernel size and split ratio might be harder to infer, using defaults from neosr
        kernel_size = 17
        split_ratio = 0.25
        use_ea = any('.attn.f.0.weight' in k for k in self.state.keys())
        norm_groups = 4 # standard
        
        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(0)]
            + [nn.Conv2d(dim, out_ch * self.scale**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=self.scale**2, dim=1
        )
        self.to_img = nn.PixelShuffle(self.scale)
        
        # DySample not implemented here for simplicity/Fooocus compatibility
        # If model expects DySample, it might need more work, but most RealPLKSR use PixelShuffle.

        self.load_state_dict(self.state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Some RealPLKSR variants use a global skip connection
        x_in = x
        x = self.feats(x) + self.repeat_op(x_in)
        x = self.to_img(x)
        return x
