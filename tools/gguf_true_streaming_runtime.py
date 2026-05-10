from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from backend import loader as backend_loader
from backend import resources
from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRuntime
from backend.gguf.loader import gguf_sd_loader
from backend.gguf.ops import GGMLOps
from backend.gguf.patcher import GGUFModelPatcher
from backend.patching import NexModelPatcher
from ldm_patched.modules import latent_formats, model_base


class TrueStreamingGGUFModelPatcher(GGUFModelPatcher):
    """
    SDXL GGUF patcher variant that keeps the original mmap-backed source alive.

    The production GGUF patcher intentionally forces a detach/release pass after
    the first load. W05 wants the opposite behavior, so this class reuses the
    GGUF weight handling but routes `load()` through the parent ModelPatcher
    implementation instead of the GGUF override.
    """

    def load(self, *args, **kwargs):
        return NexModelPatcher.load(self, *args, **kwargs)


def load_true_streaming_sdxl_unet(
    source: str,
    *,
    load_device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    quality: Optional[Dict[str, Any]] = None,
) -> TrueStreamingGGUFModelPatcher:
    load_device = load_device or resources.get_torch_device()
    offload_device = offload_device or resources.unet_offload_device()
    effective_dtype = dtype or torch.float16

    sd = gguf_sd_loader(source)
    model = model_base.SDXL(
        model_config=backend_loader.ModelConfig(
            backend_loader.sdxl_def.UNET_CONFIG,
            latent_formats.SDXL(),
        ),
        operations=GGMLOps,
    )
    model.diffusion_model.to(device=load_device, dtype=effective_dtype)
    model.diffusion_model.load_state_dict(sd, strict=False)

    patcher = TrueStreamingGGUFModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        runtime_reload=None,
        runtime_release_to_meta=False,
    )

    if quality:
        backend_loader.patch_unet_for_quality(patcher, quality)

    return patcher


class TrueStreamingSDXLGGUFRuntime(DirectSDXLGGUFRuntime):
    route_label = "tools_only_true_streaming"

    def load_components(self) -> float:
        if self._loaded:
            return self._cold_model_load_cpu

        start = time.perf_counter()
        self.unet = load_true_streaming_sdxl_unet(
            self.config.unet_path,
            load_device=self.device,
            offload_device=resources.unet_offload_device(),
            dtype=torch.float16,
            quality=self.config.quality,
        )
        self.clip = backend_loader.load_sdxl_clip(
            self.config.clip_l_path,
            self.config.clip_g_path,
            load_device=resources.text_encoder_load_device(),
            offload_device=resources.text_encoder_offload_device(),
            dtype=torch.float16,
        )
        self.clip.clip_layer(self.config.clip_layer)
        self.vae = backend_loader.load_vae(
            self.config.vae_path,
            load_device=resources.get_torch_device(),
            offload_device=resources.vae_offload_device(),
            dtype=torch.float32,
            latent_format=latent_formats.SDXL(),
        )

        self._cold_model_load_cpu = time.perf_counter() - start
        self._loaded = True
        return self._cold_model_load_cpu
