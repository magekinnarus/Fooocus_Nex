import modules.core as core
import os
import torch
import modules.patch
import modules.config
import modules.flags
from backend import resources, schedulers, lora
import extras.vae_interpose as vae_interpose


from ldm_patched.modules.model_base import SDXL
from modules.util import get_file_from_folder_list, get_enabled_loras


model_base = core.StableDiffusionModel()
final_unet = None
final_clip = None
final_vae = None

loaded_ControlNets = {}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets = cache
    return


@torch.inference_mode()
def assert_model_integrity():
    if model_base.unet_with_lora is None:
        return True

    from ldm_patched.modules.model_base import BaseModel, SDXL
    if not isinstance(model_base.unet_with_lora.model, BaseModel):
        raise NotImplementedError('Unknown model type loaded.')

    if not isinstance(model_base.unet_with_lora.model, SDXL):
        print('[Nex Warning] Non-SDXL base model loaded. Some features may not work.')

    return True


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, vae_name=None, clip_name=None):
    global model_base

    if name == 'None':
        print('Skipping base model load (name is None)')
        return

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    if model_base.filename == filename and model_base.vae_filename == vae_filename and getattr(model_base, 'clip_filename', None) == clip_name:
        return

    model_base = core.load_model(filename, vae_filename, clip_name)
    model_base.clip_filename = clip_name
    
    # Only print if we actually performed a load (check above handled the early return)
    print(f'Base model loaded: {model_base.filename}')
    if model_base.vae_filename:
        print(f'VAE loaded: {model_base.vae_filename}')
    if clip_name:
        print(f'Force CLIP: {clip_name}')
    return




@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None):
    global model_base

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    model_base.refresh_loras(loras + base_model_additional_loras)

    return


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[text] = result
    if verbose:
        print(f'[CLIP Encoded] {text}')
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1):
    global final_clip

    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0

    for i, text in enumerate(texts):
        cond, pooled = clip_encode_single(final_clip, text)
        cond_list.append(cond)
        if i < pool_top_k:
            pooled_acc += pooled

    return [[torch.cat(cond_list, dim=1), {"pooled_output": pooled_acc}]]


@torch.no_grad()
@torch.inference_mode()
def set_clip_skip(clip_skip: int):
    global final_clip

    if final_clip is None:
        return

    final_clip.clip_layer(-abs(clip_skip))
    return

@torch.inference_mode()
def clear_all_caches():
    if final_clip is not None:
        final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    
    if final_clip is None:
        return

    resources.load_models_gpu([final_clip.patcher])
    return


refresh_state = {
    'base_model_name': None,
    'loras': None,
    'base_model_additional_loras': None,
    'vae_name': None,
    'clip_name': None
}


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(base_model_name, loras,
                       base_model_additional_loras=None, vae_name=None, clip_name=None):
    global final_unet, final_clip, final_vae, refresh_state

    # Sort loras to ensure consistent comparison
    loras = sorted(loras) if loras is not None else []
    base_model_additional_loras = sorted(base_model_additional_loras) if base_model_additional_loras is not None else []

    current_state = {
        'base_model_name': base_model_name,
        'loras': loras,
        'base_model_additional_loras': base_model_additional_loras,
        'vae_name': vae_name,
        'clip_name': clip_name
    }

    if refresh_state == current_state and final_unet is not None:
        return

    final_unet = None
    final_clip = None
    final_vae = None

    refresh_base_model(base_model_name, vae_name, clip_name)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    prepare_text_encoder(async_call=True)
    clear_all_caches()

    refresh_state = current_state
    return


try:
    refresh_everything(
        base_model_name=modules.config.default_base_model_name,
        loras=get_enabled_loras(modules.config.default_loras),
        vae_name=modules.config.default_vae,
        clip_name=modules.config.default_clip
    )
except Exception as e:
    print(f'[Nex Warning] Failed to load default model at startup: {e}')
    print('[Nex Warning] The UI will launch without a model. Select one from Advanced > Models.')




@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    model_sampling = model.get_model_object("model_sampling")
    sigmas = schedulers.calculate_sigmas(model_sampling, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def get_candidate_vae(steps, denoise=1.0):
    return final_vae, None


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, disable_preview=False, quality=None):
    target_unet, target_vae, target_clip = final_unet, final_vae, final_clip

    if target_unet is None:
        print('Error: Base model is not loaded. Please select a model in the Models tab.')
        return

    if latent is None:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        initial_latent = latent

    # Backend sampling handles noise internally. 
    # BrownianTreeNoiseSamplerPatched monkey-patch is skipped.
    
    sampled_latent = core.ksampler(
        model=target_unet,
        positive=positive_cond,
        negative=negative_cond,
        latent=initial_latent,
        steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
        seed=image_seed,
        denoise=denoise,
        callback_function=callback,
        cfg=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        previewer_start=0,
        previewer_end=steps,
        disable_preview=disable_preview,
        quality=quality
    )

    # Phase: Sampling \u2192 Decoding (Immediate Cleanup)
    import gc
    from backend import resources
    gc.collect()
    resources.soft_empty_cache()
    print(f'[Nex-Memory] Phase: Sampling \u2192 Decoding')

    decoded_latent = core.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)

    images = core.pytorch_to_numpy(decoded_latent)
    return images
