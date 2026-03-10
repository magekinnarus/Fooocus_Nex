import os
import einops
import torch
import numpy as np
import ldm_patched.modules.model_detection
from backend import loader, resources, sampling, conditioning, decode, lora, utils as backend_utils
import ldm_patched.modules.model_patcher
import ldm_patched.modules.controlnet
import ldm_patched.modules.latent_formats

from ldm_patched.modules.sd import load_checkpoint_guess_config
import math
import json
import ldm_patched.modules.model_management
import ldm_patched.modules.model_sampling
import ldm_patched.modules.sd
import ldm_patched.modules.latent_formats

from modules.util import get_file_from_folder_list
from backend.lora import match_lora, model_lora_keys_unet, model_lora_keys_clip
import modules.config

# Inlined from ldm_patched.contrib.external
class VAEDecode:
    def decode(self, vae, samples):
        resources.load_models_gpu([vae.patcher])
        result = (vae.decode(samples["samples"]), )
        resources.eject_model(vae.patcher)
        return result

class VAEDecodeTiled:
    def decode(self, vae, samples, tile_size):
        resources.load_models_gpu([vae.patcher])
        result = (vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, ), )
        resources.eject_model(vae.patcher)
        return result

def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels

class VAEEncode:
    def encode(self, vae, pixels):
        resources.load_models_gpu([vae.patcher])
        pixels = vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:,:,:,:3])
        result = ({"samples":t}, )
        resources.eject_model(vae.patcher)
        return result

class VAEEncodeTiled:
    def encode(self, vae, pixels, tile_size):
        resources.load_models_gpu([vae.patcher])
        pixels = vae_encode_crop_pixels(pixels)
        t = vae.encode_tiled(pixels[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, )
        result = ({"samples":t}, )
        resources.eject_model(vae.patcher)
        return result

class EmptyLatentImage:
    def __init__(self):
        self.device = ldm_patched.modules.model_management.intermediate_device()

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )

class ControlNetApplyAdvanced:
    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

# Inlined from ldm_patched.contrib.external_model_advanced
def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

class ModelSamplingDiscrete:
    def patch(self, model, sampling, zsnr):
        m = model.clone()
        sampling_base = ldm_patched.modules.model_sampling.ModelSamplingDiscrete
        if sampling in ["eps", "lcm", "tcd"]:
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION
        else:
            raise NotImplementedError(f"Sampling type {sampling} not inlined.")

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        if zsnr:
            model_sampling.set_sigmas(rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))

        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

class ModelSamplingContinuousEDM:
    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()
        latent_format = None
        sigma_data = 1.0
        if sampling == "eps":
            sampling_type = ldm_patched.modules.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = ldm_patched.modules.model_sampling.V_PREDICTION
        elif sampling == "edm_playground_v2.5":
            sampling_type = ldm_patched.modules.model_sampling.EDM
            sigma_data = 0.5
            latent_format = ldm_patched.modules.latent_formats.SDXL_Playground_2_5()

        class ModelSamplingAdvanced(ldm_patched.modules.model_sampling.ModelSamplingContinuousEDM, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        if latent_format is not None:
            m.add_object_patch("latent_format", latent_format)
        return (m, )

opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()



class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras):
        assert isinstance(loras, list)

        if self.visited_loras == str(loras):
            return

        self.visited_loras = str(loras)

        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        loras_to_load = []

        for filename, weight in loras:
            if filename == 'None':
                continue

            if os.path.exists(filename):
                lora_filename = filename
            else:
                lora_filename = get_file_from_folder_list(filename, modules.config.paths_loras)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            loras_to_load.append((lora_filename, weight))

        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        for lora_filename, weight in loras_to_load:
            lora_sd = backend_utils.load_torch_file(lora_filename)
            
            # Map keys using backend functions
            lora_keys_unet = lora.model_lora_keys_unet(self.unet.model)
            lora_keys_clip = lora.model_lora_keys_clip(self.clip.cond_stage_model)
            
            # Load patches
            lora_unet = lora.load_lora(lora_sd, lora_keys_unet, log_missing=False)
            lora_clip = []
            if self.clip is not None:
                lora_clip = lora.load_lora(lora_sd, lora_keys_clip, log_missing=False)

            if self.unet_with_lora is not None and len(lora_unet) > 0:
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                print(f'Loaded LoRA [{lora_filename}] for UNet [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')

            if self.clip_with_lora is not None and len(lora_clip) > 0:
                loaded_keys = self.clip_with_lora.add_patches(lora_clip, weight)
                print(f'Loaded LoRA [{lora_filename}] for CLIP [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return ldm_patched.modules.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
        image=image, strength=strength, start_percent=start_percent, end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, vae_filename=None, clip_filename=None):
    # Check file existence first
    if not os.path.isfile(ckpt_filename):
        print(f'[Nex Error] Model file not found: {ckpt_filename}')
        return StableDiffusionModel(filename=ckpt_filename)

    basename = os.path.basename(ckpt_filename).lower()

    unet = None
    clip = None
    vae = None

    if basename.endswith('.gguf'):
        # GGUF UNet-only file — load via backend GGUF path
        # CLIP and VAE must be provided separately via UI dropdowns
        print(f'[Nex] Loading GGUF UNet: {basename}')
        try:
            unet = loader.load_sdxl_unet(ckpt_filename)
            print(f'[Nex] GGUF UNet loaded successfully.')
            print(f'[Nex Warning] GGUF is UNet-only. Please select CLIP and VAE in the Models tab.')
        except Exception as e:
            print(f'[Nex Error] Failed to load GGUF UNet: {e}')
    elif basename.startswith("xl_") or "sdxl" in basename or basename.startswith("xl-"):
        unet, clip, vae = loader.load_sdxl_checkpoint(ckpt_filename)
    else:
        unet, clip, vae = loader.load_sd15_checkpoint(ckpt_filename)

    is_sdxl_base = basename.endswith('.gguf') or basename.startswith("xl_") or "sdxl" in basename or basename.startswith("xl-")

    # Support for separate CLIP if provided
    if clip_filename is not None and clip_filename != 'None':
        clip_filename_abs = get_file_from_folder_list(clip_filename, modules.config.paths_clips)
        if os.path.exists(clip_filename_abs):
            try:
                if is_sdxl_base:
                    # GGUF and SDXL use the same CLIP loader pattern
                    clip = loader.load_sdxl_clip(clip_filename_abs, clip_filename_abs)
                else:
                    clip = loader.load_sd15_clip(clip_filename_abs)
                print(f'[Nex] Force CLIP loaded: {clip_filename}')
            except Exception as e:
                print(f'[Nex Error] Failed to load Force CLIP [{clip_filename}]: {e}')

    # Support for separate VAE if provided
    if vae_filename is not None and vae_filename != 'None':
        vae_filename_abs = get_file_from_folder_list(vae_filename, modules.config.path_vae)
        if os.path.exists(vae_filename_abs):
            try:
                vae = loader.load_vae(vae_filename_abs)
                print(f'[Nex] External VAE loaded: {vae_filename}')
            except Exception as e:
                print(f'[Nex Error] Failed to load VAE [{vae_filename}]: {e}')

    # Warn about missing components
    if unet is None:
        print(f'[Nex Warning] No UNet loaded. Generation will not work.')
    if clip is None:
        print(f'[Nex Warning] No CLIP loaded. Please select a CLIP model in Advanced > Models.')
    if vae is None:
        print(f'[Nex Warning] No VAE loaded. Please select a VAE in Advanced > Models.')

    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, filename=ckpt_filename, vae_filename=vae_filename)


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    resources.load_models_gpu([vae.patcher])
    result = vae.decode(latent_image["samples"], tiled=tiled)
    resources.eject_model(vae.patcher)
    return result


@torch.inference_mode()
def encode_vae(vae, pixels):
    resources.load_models_gpu([vae.patcher])
    result = vae.encode(pixels)
    resources.eject_model(vae.patcher)
    return result


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    assert mask.ndim == 3 and pixels.ndim == 4
    assert mask.shape[-1] == pixels.shape[-2]
    assert mask.shape[-2] == pixels.shape[-3]

    w = mask.round()[..., None]
    pixels = pixels * (1 - w) + 0.5 * w

    resources.load_models_gpu([vae.patcher])
    latent = vae.encode(pixels)['samples']
    resources.eject_model(vae.patcher)
    
    B, C, H, W = latent.shape

    latent_mask = mask[:, None, :, :]
    latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
    latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent)

    return latent, latent_mask


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_models = {}


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models

    from modules.config import path_vae_approx
    is_sdxl = isinstance(model.model.latent_format, ldm_patched.modules.latent_formats.SDXL)
    vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        sd = torch.load(vae_approx_filename, map_location='cpu', weights_only=True)
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if resources.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(resources.get_torch_device())
        VAE_approx_models[vae_approx_filename] = VAE_approx_model

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None,
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False,
             quality=None):

    device = resources.get_torch_device()
    latent_image = latent["samples"].to(device)

    denoise_mask = latent.get("noise_mask", None)
    if denoise_mask is not None:
        denoise_mask = denoise_mask.to(device)

    # Prep noise
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=device)
    else:
        torch.manual_seed(seed)
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=device)

    # Setup Previewer
    previewer = get_previewer(model) if not disable_preview else None
    
    def wrapped_callback(step, x0, x, total_steps, denoised=None):
        preview_image = None
        if previewer is not None and denoised is not None:
            preview_image = previewer(denoised, step, total_steps)
        if callback_function is not None:
            callback_function(step, x0, x, total_steps, preview_image)

    # Route through backend sampling
    # NOTE: autocast is already applied inside KSampler.sample() — no need to wrap here.
    samples = sampling.sample_sdxl(
        model,
        noise,
        positive,
        negative,
        cfg=cfg,
        steps=steps,
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        seed=seed,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        callback=wrapped_callback,
        model_options={"quality": quality or {}}
    )

    out = latent.copy()
    out["samples"] = samples
    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
