#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion / Flux latent previews.
"""
import torch
import torch.nn as nn

import ldm_patched.modules.utils
import ldm_patched.modules.ops


def conv(n_in, n_out, **kwargs):
    return ldm_patched.modules.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = ldm_patched.modules.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1 = lambda c_in, c_out: ldm_patched.modules.ops.disable_weight_init.Conv2d(c_in, c_out, 1, bias=False)
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                conv1x1(n_in, n_gn),
                nn.GroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                conv1x1(n_gn, n_in),
            )

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def F32Encoder(latent_channels=32):
    return nn.Sequential(
        conv(3, 32, stride=2), nn.ReLU(inplace=True), conv(32, 64, stride=2), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def F32Decoder(latent_channels=32):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=None, arch_variant=None):
        super().__init__()

        if latent_channels is None:
            latent_channels, guessed_arch_variant = self.guess_latent_channels_and_arch(
                str(encoder_path or decoder_path or "")
            )
            if arch_variant is None:
                arch_variant = guessed_arch_variant

        use_midblock_gn = arch_variant in {"flux_2"}
        self.encoder = Encoder(latent_channels, use_midblock_gn=use_midblock_gn)
        self.decoder = Decoder(latent_channels, use_midblock_gn=use_midblock_gn)
        if arch_variant == "f32":
            self.encoder = F32Encoder(latent_channels)
            self.decoder = F32Decoder(latent_channels)

        # Backward-compatible aliases used elsewhere in the repo.
        self.taesd_encoder = self.encoder
        self.taesd_decoder = self.decoder
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))

        if encoder_path is not None:
            self.encoder.load_state_dict(ldm_patched.modules.utils.load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.decoder.load_state_dict(ldm_patched.modules.utils.load_torch_file(decoder_path, safe_load=True))

    def guess_latent_channels(self, encoder_path):
        return self.guess_latent_channels_and_arch(encoder_path)[0]

    def guess_latent_channels_and_arch(self, encoder_path):
        name = str(encoder_path or "").lower()
        if "taef1" in name:
            return 16, None
        if "taef2" in name:
            return 32, "flux_2"
        if "taesd3" in name:
            return 16, None
        if "taesana" in name:
            return 32, "f32"
        return 4, None

    @staticmethod
    def scale_latents(x):
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        x_sample = self.decoder(x * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        return self.encoder(x * 0.5 + 0.5) / self.vae_scale
