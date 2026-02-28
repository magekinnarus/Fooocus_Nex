import torch
import numpy as np
import cv2
from dataclasses import dataclass
from PIL import Image
from modules.util import resample_image, set_image_shape_ceil, get_image_shape_ceil
from modules.upscaler import perform_upscale
from modules.core import numpy_to_pytorch
import modules.core as core
import modules.default_pipeline as pipeline

@dataclass
class InpaintContext:
    """Carries all state between inpaint stages. No globals."""
    original_image: np.ndarray      # Full original image for final compositing
    original_mask: np.ndarray       # Full original mask
    interested_area: tuple           # (a, b, c, d) bounding box
    interested_image: np.ndarray     # Cropped + upscaled region
    interested_fill: np.ndarray      # Fill version for latent init
    interested_mask: np.ndarray      # Processed mask for encoding
    context_mask: np.ndarray         # Full-image soft mask for color correction
    
    latent_fill: torch.Tensor = None        # Encoded fill latent
    latent_mask: torch.Tensor = None        # Downsampled latent-space mask
    latent_swap: torch.Tensor = None        # SD1.5 swap latent (if applicable)
    inpaint_head_feature: torch.Tensor = None

class InpaintHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device='cpu'))

    def __call__(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)

class InpaintPipeline:
    def __init__(self):
        self._inpaint_head_model = None

    def _box_blur(self, x, k):
        kernel_size = 2 * k + 1
        return cv2.blur(x, (kernel_size, kernel_size))

    def _max_filter_opencv(self, x, ksize=3):
        return cv2.dilate(x, np.ones((ksize, ksize), dtype=np.int16))

    def _morphological_open(self, x):
        x_int16 = np.zeros_like(x, dtype=np.int16)
        x_int16[x > 127] = 256
        for _ in range(32):
            maxed = self._max_filter_opencv(x_int16, ksize=3) - 8
            x_int16 = np.maximum(maxed, x_int16)
        return np.clip(x_int16, 0, 255).astype(np.uint8)

    def _up255(self, x, t=0):
        y = np.zeros_like(x).astype(np.uint8)
        y[x > t] = 255
        return y

    def _regulate_abcd(self, x, a, b, c, d):
        H, W = x.shape[:2]
        return int(max(0, min(a, H))), int(max(0, min(b, H))), int(max(0, min(c, W))), int(max(0, min(d, W)))

    def _compute_initial_abcd(self, x):
        indices = np.where(x)
        if len(indices[0]) == 0:
            return 0, x.shape[0], 0, x.shape[1]
        a, b, c, d = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
        abp, abm = (b + a) // 2, (b - a) // 2
        cdp, cdm = (d + c) // 2, (d - c) // 2
        l = int(max(abm, cdm) * 1.15)
        a, b, c, d = abp - l, abp + l + 1, cdp - l, cdp + l + 1
        return self._regulate_abcd(x, a, b, c, d)

    def _solve_abcd(self, x, a, b, c, d, k):
        k = float(k)
        H, W = x.shape[:2]
        if k >= 1.0:
            return 0, H, 0, W
        while True:
            if b - a >= H * k and d - c >= W * k:
                break
            add_h = (b - a) < (d - c) or d - c == W
            if b - a == H: add_h = False
            if add_h: a, b = a - 1, b + 1
            else: c, d = c - 1, d + 1
            a, b, c, d = self._regulate_abcd(x, a, b, c, d)
        return a, b, c, d

    def _fooocus_fill(self, image, mask):
        current_image = image.copy()
        area = np.where(mask < 127)
        store = image[area]
        for k, repeats in [(512, 2), (256, 2), (128, 4), (64, 4), (33, 8), (15, 8), (5, 16), (3, 16)]:
            for _ in range(repeats):
                current_image = self._box_blur(current_image, k)
                current_image[area] = store
        return current_image

    def prepare(self, image, mask, k=0.618, use_fill=True) -> InpaintContext:
        """Bounding Box computation -> crop -> upscale -> mask processing -> fill."""
        a, b, c, d = self._compute_initial_abcd(mask > 0)
        a, b, c, d = self._solve_abcd(mask, a, b, c, d, k=k)

        interested_mask = mask[a:b, c:d]
        interested_image = image[a:b, c:d]

        # Super resolution if too small
        if get_image_shape_ceil(interested_image) <= 512:
            print(f'[InpaintPipeline] Image is too small ({interested_image.shape}), applying AI upscaling ...')
            interested_image = perform_upscale(interested_image)
        else:
            print(f'[InpaintPipeline] Image shape is {interested_image.shape}, skipping AI upscaling.')

        # Resize to make images ready for diffusion
        interested_image = set_image_shape_ceil(interested_image, 1024)
        H, W, _ = interested_image.shape

        # Process mask
        processed_mask = self._up255(resample_image(interested_mask, W, H), t=127)

        # Compute filling
        interested_fill = interested_image.copy()
        if use_fill:
            interested_fill = self._fooocus_fill(interested_image, processed_mask)

        # Soft pixels for stitching
        context_mask = self._morphological_open(mask)

        return InpaintContext(
            original_image=image,
            original_mask=mask,
            interested_area=(a, b, c, d),
            interested_image=interested_image,
            interested_fill=interested_fill,
            interested_mask=processed_mask,
            context_mask=context_mask
        )

    def encode(self, context: InpaintContext, vae, vae_swap=None, progressbar_callback=None, task_state=None) -> InpaintContext:
        """VAE encode interested region + fill -> populate context latents."""
        inpaint_pixel_fill = numpy_to_pytorch(context.interested_fill)
        inpaint_pixel_image = numpy_to_pytorch(context.interested_image)
        inpaint_pixel_mask = numpy_to_pytorch(context.interested_mask)

        if progressbar_callback and task_state:
            task_state.current_progress += 1
            progressbar_callback(task_state, task_state.current_progress, 'VAE Inpaint encoding ...')

        latent_inpaint, latent_mask = core.encode_vae_inpaint(
            mask=inpaint_pixel_mask,
            vae=vae,
            pixels=inpaint_pixel_image
        )

        latent_swap = None
        if vae_swap is not None:
            if progressbar_callback and task_state:
                task_state.current_progress += 1
                progressbar_callback(task_state, task_state.current_progress, 'VAE SD15 encoding ...')
            latent_swap = core.encode_vae(vae=vae_swap, pixels=inpaint_pixel_fill)['samples']

        if progressbar_callback and task_state:
            task_state.current_progress += 1
            progressbar_callback(task_state, task_state.current_progress, 'VAE encoding ...')

        latent_fill = core.encode_vae(vae=vae, pixels=inpaint_pixel_fill)['samples']
        
        context.latent_fill = latent_fill
        context.latent_mask = latent_mask
        context.latent_inpaint = latent_inpaint
        context.latent_swap = latent_swap
        return context

    def patch_model(self, context: InpaintContext, unet, head_model_path) -> torch.nn.Module:
        """Apply inpaint head feature patch to UNet."""
        if self._inpaint_head_model == None:
            self._inpaint_head_model = InpaintHead()
            sd = torch.load(head_model_path, map_location='cpu', weights_only=True)
            self._inpaint_head_model.load_state_dict(sd)
        
        feed = torch.cat([
            context.latent_mask,
            unet.model.process_latent_in(context.latent_inpaint)
        ], dim=1)

        self._inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        inpaint_head_feature = self._inpaint_head_model(feed)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                h = h + inpaint_head_feature.to(h)
            return h

        m = unet.clone()
        m.set_model_input_block_patch(input_block_patch)
        return m

    def stitch(self, context: InpaintContext, generated_image) -> np.ndarray:
        """Rescale generated BB -> paste into original -> color correct."""
        a, b, c, d = context.interested_area
        
        # FIX: explicitly use original BB dimensions to avoid 256x256 bug
        target_width = d - c
        target_height = b - a
        
        print(f"[InpaintPipeline] Stitching back to {target_width}x{target_height}")
        
        content = resample_image(generated_image, target_width, target_height)
        result = context.original_image.copy()
        result[a:b, c:d] = content
        
        # Color correction
        fg = result.astype(np.float32)
        bg = context.original_image.astype(np.float32)
        w = context.context_mask[:, :, None].astype(np.float32) / 255.0
        y = fg * w + bg * (1 - w)
        return y.clip(0, 255).astype(np.uint8)
