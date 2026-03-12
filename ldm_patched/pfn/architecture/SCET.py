import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class BackBoneBlock(nn.ModuleList):
    def __init__(self, num, fm, **kwargs):
        super().__init__([fm(**kwargs) for _ in range(num)])
    def forward(self, x):
        for block in self:
            x = block(x)
        return x

class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    def forward(self, x):
        y = self.sigmoid(self.k2(x))
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out

class SCPA(nn.Module):
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.k1 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.PAConv = PAConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        residual = x
        out_a = self.lrelu(self.conv1_a(x))
        out_b = self.lrelu(self.conv1_b(x))
        out_a = self.lrelu(self.k1(out_a))
        out_b = self.lrelu(self.PAConv(out_b))
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out

class SCET(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.model_arch = "SCET (SCHAT)"
        self.sub_type = "SR"
        self.state = state_dict
        if "params" in self.state: self.state = self.state["params"]
        if "params_ema" in self.state: self.state = self.state["params_ema"]

        # Infer params
        hiddenDim = self.state["conv3.weight"].shape[0]
        # Detect scale from path2.1.weight
        out_ch_total = self.state["path2.1.weight"].shape[0]
        out_ch = 3
        self.scale = int((out_ch_total / out_ch) ** 0.5)
        
        num_heads = 7 if self.scale == 3 else 8
        
        self.conv3 = nn.Conv2d(3, hiddenDim, kernel_size=3, padding=1)
        self.path1 = nn.Sequential(
            BackBoneBlock(16, SCPA, nf=hiddenDim, reduction=2, stride=1, dilation=1),
            BackBoneBlock(1, TransformerBlock, dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale),
            nn.Conv2d(hiddenDim // (self.scale ** 2), 3, kernel_size=3, padding=1),
        )
        self.path2 = nn.Sequential(
            nn.PixelShuffle(self.scale),
            nn.Conv2d(hiddenDim // (self.scale ** 2), 3, kernel_size=3, padding=1),
        )
        self.load_state_dict(self.state, strict=False)

    def forward(self, x):
        x = self.conv3(x)
        return self.path1(x) + self.path2(x)
