import os
import torch
import numpy as np
from PIL import Image
from ldm_patched.modules.args_parser import args, LatentPreviewMethod
from ldm_patched.taesd.taesd import TAESD
import ldm_patched.utils.path_utils

MAX_PREVIEW_RESOLUTION = 512


class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)


class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].detach()
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)
        return Image.fromarray(x_sample)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview(self, x0):
        latent_image = x0[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors
        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()
        return Image.fromarray(latents_ubyte.numpy())


def _latent_preview_image_to_numpy(preview_image):
    if isinstance(preview_image, Image.Image):
        return np.asarray(preview_image.convert("RGB"))
    try:
        preview_array = np.asarray(preview_image)
    except Exception:
        return None
    return preview_array if getattr(preview_array, "ndim", 0) == 3 else None


def resolve_taesd_previewer(device, latent_format, vae_approx_path=None):
    """Resolve a TAESD previewer for the given latent format.

    Args:
        device: torch device to load the TAESD decoder onto.
        latent_format: latent format object with ``taesd_decoder_name`` attribute.
        vae_approx_path: Optional explicit directory where TAESD decoder files
            live (e.g. ``config.path_vae_approx``).
    """
    if latent_format is None or getattr(latent_format, "taesd_decoder_name", None) is None:
        return None

    taesd_decoder_name = latent_format.taesd_decoder_name
    taesd_decoder_path = None

    # --- Fast path: direct lookup in the known download directory ---
    if vae_approx_path is not None:
        candidate = os.path.join(vae_approx_path, f"{taesd_decoder_name}.pth")
        if os.path.isfile(candidate):
            taesd_decoder_path = candidate

    # --- Slow fallback: directory walk via path_utils (legacy callers) ---
    if taesd_decoder_path is None:
        taesd_decoder_file = next(
            (
                fn
                for fn in ldm_patched.utils.path_utils.get_filename_list("vae_approx")
                if fn == f"{taesd_decoder_name}.pth" or fn == f"{taesd_decoder_name}.pt"
            ),
            "",
        )
        if taesd_decoder_file:
            taesd_decoder_path = ldm_patched.utils.path_utils.get_full_path("vae_approx", taesd_decoder_file)

    if not taesd_decoder_path:
        return None

    try:
        return TAESDPreviewerImpl(TAESD(None, taesd_decoder_path).to(device))
    except Exception:
        return None


def decode_latent_preview(previewer, latent_format, x0):
    if previewer is None:
        return None

    preview_latent = x0.detach() if hasattr(x0, "detach") else x0

    try:
        preview_image = previewer.decode_latent_to_preview(preview_latent)
    except Exception:
        return None

    return _latent_preview_image_to_numpy(preview_image)
