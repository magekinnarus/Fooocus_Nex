import modules.core as core
import os
import gc
import torch
import modules.patch
import modules.config
import modules.flags
import modules.model_taxonomy
from backend import conditioning, resources, schedulers, lora
from backend import environment_profile
from backend import sdxl_runtime_policy
import extras.vae_interpose as vae_interpose


from ldm_patched.modules.model_base import SDXL
from modules.util import get_file_from_folder_list, get_enabled_loras


model_base = core.StableDiffusionModel()
final_unet = None
final_clip = None
final_vae = None

loaded_ControlNets = {}


def _resolved_memory_profile():
    return getattr(modules.config, 'resolved_memory_environment_profile', None)


def _should_skip_eager_pipeline_preload() -> bool:
    return environment_profile.should_skip_eager_model_preload(_resolved_memory_profile())


def _controlnet_residency_summary():
    return {'cached_paths': len(loaded_ControlNets)}


def _offload_controlnet(model):
    if model is None:
        return
    patcher = getattr(model, 'control_model_wrapped', None)
    if patcher is not None:
        try:
            patcher.detach()
        except Exception:
            pass


def _destroy_controlnet(model):
    if model is None:
        return
    _offload_controlnet(model)
    try:
        model.cleanup()
    except Exception:
        pass


def apply_controlnet_residency(mode='offload'):
    global loaded_ControlNets

    actions = {'mode': mode, 'count': len(loaded_ControlNets)}
    if mode == 'destroy':
        stale = list(loaded_ControlNets.values())
        loaded_ControlNets = {}
        for model in stale:
            _destroy_controlnet(model)
    else:
        for model in loaded_ControlNets.values():
            _offload_controlnet(model)
    return actions


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    requested_paths = {p for p in model_paths if p is not None}
    stale_paths = [p for p in loaded_ControlNets.keys() if p not in requested_paths]

    for stale_path in stale_paths:
        _destroy_controlnet(loaded_ControlNets.pop(stale_path, None))

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


def _is_sdxl_model_base() -> bool:
    return getattr(model_base, 'architecture', None) == modules.model_taxonomy.ARCHITECTURE_SDXL


def _policy_signature(policy) -> tuple:
    if policy is None:
        return ()
    return (
        getattr(policy, 'execution_family', None),
        getattr(policy, 'residency_class', None),
        getattr(policy, 'clip_residency_mode', None),
        getattr(policy, 'vae_encode_mode', None),
        bool(getattr(policy, 'keep_clip_loaded', False)),
    )


def _apply_sdxl_policy_to_model_base(policy) -> None:
    setattr(model_base, 'sdxl_execution_policy', policy)
    setattr(model_base, 'sdxl_execution_family', getattr(policy, 'execution_family', None))
    setattr(model_base, 'sdxl_residency_class', getattr(policy, 'residency_class', None))
    setattr(model_base, 'sdxl_clip_residency_mode', getattr(policy, 'clip_residency_mode', None))
    setattr(model_base, 'sdxl_vae_encode_mode', getattr(policy, 'vae_encode_mode', None))
    setattr(model_base, 'sdxl_keep_clip_loaded', bool(getattr(policy, 'keep_clip_loaded', False)))
    clip_load_device = resources.get_torch_device() if bool(getattr(policy, 'prefer_clip_gpu', False)) else torch.device('cpu')
    clip_offload_device = torch.device('cpu')
    for component in (
        getattr(model_base, 'clip', None),
        getattr(model_base, 'clip_with_lora', None),
    ):
        if component is not None:
            setattr(component, 'runtime_policy', policy)
            patcher = getattr(component, 'patcher', None)
            if patcher is not None:
                patcher.load_device = clip_load_device
                patcher.offload_device = clip_offload_device
    vae = getattr(model_base, 'vae', None)
    if vae is not None:
        setattr(vae, 'runtime_policy', policy)


def _clip_device_args_for_policy(policy) -> dict:
    if not getattr(policy, 'prefer_clip_gpu', False):
        return {}
    return {
        'clip_load_device': resources.get_torch_device(),
        'clip_offload_device': torch.device('cpu'),
    }


@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, vae_name=None, clip_name=None, sdxl_policy=None):
    global model_base

    if name == 'None':
        print('Skipping base model load (name is None)')
        return

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    vae_filename = None
    if vae_name is not None and vae_name != modules.flags.default_vae:
        vae_filename = get_file_from_folder_list(vae_name, modules.config.path_vae)

    current_clip_name = getattr(model_base, 'clip_filename', None)
    if model_base.filename == filename and model_base.vae_filename == vae_filename and current_clip_name == clip_name:
        _apply_sdxl_policy_to_model_base(sdxl_policy)
        return

    previous_model_filename = getattr(model_base, 'filename', None)
    if previous_model_filename is not None:
        def release_previous_model_state():
            global model_base
            previous_model = model_base
            model_base = core.StableDiffusionModel()
            del previous_model
            gc.collect()

        resources.prepare_for_checkpoint_switch(
            current_model=previous_model_filename,
            next_model=filename,
            release_callback=release_previous_model_state,
            notes={
                'current_vae': getattr(model_base, 'vae_filename', None),
                'next_vae': vae_filename,
                'current_clip': current_clip_name,
                'next_clip': clip_name,
            },
        )

    model_base = core.load_model(
        filename,
        vae_filename,
        clip_name,
        sdxl_policy=sdxl_policy,
        **_clip_device_args_for_policy(sdxl_policy),
    )
    model_base.clip_filename = clip_name
    _apply_sdxl_policy_to_model_base(sdxl_policy)

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


def _resolve_sdxl_policy(policy=None):
    resolved = policy
    if resolved is None:
        resolved = getattr(model_base, 'sdxl_execution_policy', None)
    return resolved


def _resolve_sdxl_execution_family(execution_family=None, policy=None):
    if execution_family is not None:
        return execution_family
    resolved_policy = _resolve_sdxl_policy(policy)
    return getattr(resolved_policy, 'execution_family', None)


def _resolve_sdxl_clip_residency_mode(clip_residency_mode=None, policy=None):
    if clip_residency_mode is not None:
        return clip_residency_mode
    resolved_policy = _resolve_sdxl_policy(policy)
    return getattr(resolved_policy, 'clip_residency_mode', None)


def _resolve_sdxl_residency_class(residency_class=None):
    resolved = residency_class
    if resolved is None:
        resolved = getattr(model_base, 'sdxl_residency_class', None)
    if resolved is None:
        resolved = getattr(model_base, 'residency_class', None)
    return resources.normalize_sdxl_residency_class(resolved)


def _build_sdxl_text_conditioning_fingerprint(clip, text, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
    text_encoder_identity = (
        type(getattr(clip, 'model', clip)).__name__,
        getattr(clip, 'layer_idx', None),
    )
    return conditioning.build_sdxl_text_conditioning_fingerprint(
        prompt=text,
        negative_prompt='',
        model_identity=getattr(model_base, 'filename', None),
        text_encoder_identity=text_encoder_identity,
        clip_patch_uuid=resources.model_reconciliation_signature(clip.patcher),
        clip_layer_idx=getattr(clip, 'layer_idx', None),
        lora_artifacts_state=getattr(model_base, 'lora_artifact_registry', ()),
        route_family_reconciliation_signature=route_family or getattr(model_base, 'compatibility_family', None),
        residency_class=_resolve_sdxl_residency_class(residency_class),
        route_family=route_family,
        execution_family=_resolve_sdxl_execution_family(execution_family),
        clip_residency_mode=_resolve_sdxl_clip_residency_mode(clip_residency_mode),
    )


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
    if _is_sdxl_model_base():
        cache_key = _build_sdxl_text_conditioning_fingerprint(
            clip,
            text,
            route_family=route_family,
            residency_class=residency_class,
            execution_family=execution_family,
            clip_residency_mode=clip_residency_mode,
        ).digest()
    else:
        cache_key = (text, clip.layer_idx, resources.model_reconciliation_signature(clip.patcher))
    cached = clip.fcs_cond_cache.get(cache_key, None)
    if cached is not None:
        if verbose:
            print(f'[CLIP Cached] {text}')
        return cached
    tokens = clip.tokenize(text)
    result = clip.encode_from_tokens(tokens, return_pooled=True)
    clip.fcs_cond_cache[cache_key] = result
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
def clip_encode(texts, pool_top_k=1, *, route_family=None, residency_class=None, execution_family=None, clip_residency_mode=None):
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
        cond, pooled = clip_encode_single(
            final_clip,
            text,
            route_family=route_family,
            residency_class=residency_class,
            execution_family=execution_family,
            clip_residency_mode=clip_residency_mode,
        )
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
        if _should_skip_eager_pipeline_preload():
            return
    assert_model_integrity()
    
    if final_clip is None:
        return

    resources.prepare_models_for_stage(
        [final_clip.patcher],
        stage_name="text_encode",
        target_phase=resources.MemoryPhase.PROMPT_ENCODE,
    )
    return


refresh_state = {
    'base_model_name': None,
    'loras': None,
    'base_model_additional_loras': None,
    'vae_name': None,
    'clip_name': None,
    'sdxl_policy': None,
}


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(base_model_name, loras,
                       base_model_additional_loras=None, vae_name=None, clip_name=None, sdxl_policy=None):
    global final_unet, final_clip, final_vae, refresh_state

    # Sort loras to ensure consistent comparison
    loras = sorted(loras) if loras is not None else []
    base_model_additional_loras = sorted(base_model_additional_loras) if base_model_additional_loras is not None else []

    current_state = {
        'base_model_name': base_model_name,
        'loras': loras,
        'base_model_additional_loras': base_model_additional_loras,
        'vae_name': vae_name,
        'clip_name': clip_name,
        'sdxl_policy': _policy_signature(sdxl_policy),
    }

    if refresh_state == current_state and final_unet is not None:
        _apply_sdxl_policy_to_model_base(sdxl_policy)
        return

    print(f'[Nex-Pipeline] Reconciling model state (LoRAs: {len(loras)} slots, Additional: {len(base_model_additional_loras)} slots)')

    final_unet = None
    final_clip = None
    final_vae = None

    refresh_base_model(base_model_name, vae_name, clip_name, sdxl_policy=sdxl_policy)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras)
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    prepare_text_encoder(async_call=True)
    clear_all_caches()

    refresh_state = current_state
    return


if _should_skip_eager_pipeline_preload():
    print('[Startup] Skipping eager default SDXL preload for Colab Free memory profile.')
else:
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
def process_diffusion(positive_cond, negative_cond, steps, width, height, image_seed, callback, sampler_name, scheduler_name, latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, disable_preview=False, quality=None, task_state=None):
    target_unet, target_vae, target_clip = final_unet, final_vae, final_clip

    if target_unet is None:
        print('Error: Base model is not loaded. Please select a model in the Models tab.')
        return

    with resources.memory_phase_scope(
        resources.MemoryPhase.DIFFUSION,
        task=task_state,
        notes={
            'steps': steps,
            'sampler': sampler_name,
            'scheduler': scheduler_name,
            'tiled': tiled,
        },
        end_notes={'completed': True},
    ):
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

    # Phase: Sampling -> Decoding
    resources.cleanup_memory('sampling_to_decode', notes={'tiled': tiled}, target_phase=resources.MemoryPhase.DECODE, task=task_state)
    print('[Nex-Memory] Phase: Sampling -> Decoding')

    with resources.memory_phase_scope(
        resources.MemoryPhase.DECODE,
        task=task_state,
        notes={
            'tiled': tiled,
            'latent_provided': latent is not None,
        },
        end_notes={'completed': True},
    ):
        decoded_latent = core.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)
        result = core.pytorch_to_numpy(decoded_latent)

    resources.cleanup_memory(
        'decode_complete',
        notes={'tiled': tiled},
        target_phase=resources.MemoryPhase.FINALIZE,
        task=task_state,
    )
    return result
