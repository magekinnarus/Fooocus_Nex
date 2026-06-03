import time
import torch
import numpy as np
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import modules.config as config
import modules.model_taxonomy as model_taxonomy
import backend.resources as resources
import backend.loader as loader
from backend import sdxl_runtime_policy
from modules.util import get_file_from_folder_list
from modules.pipeline.output import save_and_log, yield_result


def get_sampling_callback(task_state, progressbar_callback, current_task_id, total_count, preparation_steps, all_steps, preview_transform=None):
    """
    Returns a callback function for the diffusion sampler to report progress.
    """
    def callback(step, x0, x, total_steps, y):
        resources.throw_exception_if_processing_interrupted()
        if step == 0:
            task_state.callback_steps = 0
        task_state.callback_steps += (100 - preparation_steps) / float(all_steps)

        progress_val = int(preparation_steps + task_state.callback_steps)
        status_text = f'Sampling step {step + 1}/{total_steps}, image {current_task_id + 1}/{total_count} ...'

        if preview_transform is not None and y is not None:
            y = preview_transform(y)

        if (
            y is not None
            and isinstance(y, np.ndarray)
            and hasattr(task_state, 'inpaint_context')
            and task_state.inpaint_context is not None
        ):
            # y is an RGB preview array from the previewer.
            from modules.pipeline.inpaint import InpaintPipeline
            inpaint = InpaintPipeline()
            y = inpaint.stitch(task_state.inpaint_context, y)
        elif not isinstance(y, np.ndarray):
            y = None

        task_state.yields.append(['preview', (progress_val, status_text, y)])

    return callback





def _resolve_unified_checkpoint_path(task_state):
    model_name = str(getattr(task_state, 'base_model_name', '') or '').strip()
    if model_name == '':
        raise ValueError('Unified SDXL runtime requires a selected base model.')
    return get_file_from_folder_list(model_name, config.paths_checkpoints)


def _resolve_unified_vae_path(task_state):
    vae_name = str(getattr(task_state, 'vae_name', '') or '').strip()
    if vae_name in {'', flags.default_vae}:
        return None
    return get_file_from_folder_list(vae_name, config.path_vae)


def _ensure_supported_unified_runtime_request(task_state):
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is None or not bool(getattr(policy, 'enabled', False)):
        raise RuntimeError('Unified SDXL runtime requires an active SDXL execution policy; legacy shared diffusion path is gutted.')

    checkpoint_path = _resolve_unified_checkpoint_path(task_state)
    resolved_taxonomy = config.resolve_model_taxonomy(checkpoint_path)
    if sdxl_runtime_policy.is_legacy_sdxl_gguf_selection(
        architecture=resolved_taxonomy.architecture,
        base_model_name=checkpoint_path,
    ):
        raise RuntimeError(
            'SDXL GGUF base models are deprecated and no longer supported. '
            'Select an SDXL checkpoint base model instead.'
        )
    if resolved_taxonomy.architecture != model_taxonomy.ARCHITECTURE_SDXL:
        raise RuntimeError('SD 1.5 execution is no longer supported.')
    return checkpoint_path


def _resolve_unified_sdxl_lora_specs(task_state, *, loras=None, base_model_additional_loras=None):
    resolved_loras = list(getattr(task_state, 'loras_processed', None) or loras or getattr(task_state, 'loras', []) or [])
    if base_model_additional_loras is None:
        base_model_additional_loras = getattr(task_state, 'base_model_additional_loras', []) or []
    resolved_additional_loras = list(base_model_additional_loras or [])
    return tuple((str(path), float(weight)) for path, weight in (resolved_loras + resolved_additional_loras))


def resolve_unified_sdxl_process_key(task_state, *, loras=None, base_model_additional_loras=None):
    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is None or not bool(getattr(policy, 'enabled', False)):
        return None

    return pipeline._sdxl_process_key(
        base_model_name=_resolve_unified_checkpoint_path(task_state),
        vae_name=_resolve_unified_vae_path(task_state),
        clip_name=getattr(task_state, 'clip_model_name', None),
        sdxl_policy=policy,
        loras=list(
            _resolve_unified_sdxl_lora_specs(
                task_state,
                loras=loras,
                base_model_additional_loras=base_model_additional_loras,
            )
        ),
    )


def _build_unified_spatial_kwargs(task_state, image_input_result=None):
    image_input_result = image_input_result or {}
    resolved_spatial_context = getattr(task_state, 'inpaint_context', None)
    goals = set(getattr(task_state, 'goals', []) or [])
    if 'outpaint' in goals:
        return {
            'source_pixels': image_input_result.get('outpaint_image'),
            'source_mask': image_input_result.get('outpaint_mask'),
            'spatial_mode': 'outpaint',
            'resolved_spatial_context': resolved_spatial_context,
            'outpaint_direction': getattr(task_state, 'outpaint_direction', None),
            'outpaint_expansion_size': int(getattr(task_state, 'inpaint_outpaint_expansion_size', 384) or 384),
            'outpaint_pixelate': bool(getattr(task_state, 'inpaint_pixelate_primer', True)),
        }
    if 'inpaint' in goals:
        return {
            'source_pixels': image_input_result.get('inpaint_image'),
            'source_mask': getattr(task_state, 'context_mask', None) or image_input_result.get('inpaint_mask'),
            'spatial_mode': 'inpaint',
            'resolved_spatial_context': resolved_spatial_context,
        }
    return {}


def _run_unified_sdxl_task(
    task_state,
    task_dict,
    current_task_id,
    total_count,
    all_steps,
    preparation_steps,
    denoising_strength,
    final_scheduler_name,
    *,
    loras,
    base_model_additional_loras=None,
    controlnet_paths=None,
    contextual_assets=None,
    image_input_result=None,
    progressbar_callback=None,
):
    from backend.sdxl_unified_runtime import UnifiedSDXLRuntime, UnifiedSDXLRuntimeConfig

    policy = getattr(task_state, 'sdxl_execution_policy', None)
    checkpoint_path = _ensure_supported_unified_runtime_request(task_state)
    stream_budget = float(getattr(policy, 'stream_budget_mb', 256.0))

    merged_loras = _resolve_unified_sdxl_lora_specs(
        task_state,
        loras=loras,
        base_model_additional_loras=base_model_additional_loras,
    )

    quality = {
        "sharpness": float(getattr(task_state, 'sharpness', 2.0)),
        "adaptive_cfg": float(getattr(task_state, 'adaptive_cfg', 7.0)),
        "adm_scaler_positive": float(getattr(task_state, 'adm_scaler_positive', 1.5)),
        "adm_scaler_negative": float(getattr(task_state, 'adm_scaler_negative', 0.8)),
        "adm_scaler_end": float(getattr(task_state, 'adm_scaler_end', 0.3)),
        "controlnet_softness": float(getattr(task_state, 'controlnet_softness', 0.25)),
    }
    callback = get_sampling_callback(
        task_state,
        progressbar_callback,
        current_task_id,
        total_count,
        preparation_steps,
        all_steps,
    )
    config_kwargs = dict(
        model_variant='sdxl',
        execution_class=(
            getattr(policy, 'execution_class', None)
            or getattr(task_state, 'sdxl_execution_family', None)
            or getattr(policy, 'execution_family', None)
            or 'standard_sdxl'
        ),
        streamlike_budget_mb=stream_budget,
        quality=quality,
        checkpoint_path=checkpoint_path,
        vae_path=_resolve_unified_vae_path(task_state),
        prompt=str(task_dict.get('task_prompt', task_state.prompt) or ''),
        negative_prompt=str(task_dict.get('task_negative_prompt', task_state.negative_prompt) or ''),
        positive_texts=tuple(str(item) for item in (task_dict.get('positive') or [task_dict.get('task_prompt', task_state.prompt)])),
        negative_texts=tuple(str(item) for item in (task_dict.get('negative') or [task_dict.get('task_negative_prompt', task_state.negative_prompt)])),
        positive_top_k=int(task_dict.get('positive_top_k', 1) or 1),
        negative_top_k=int(task_dict.get('negative_top_k', 1) or 1),
        width=int(task_state.width),
        height=int(task_state.height),
        steps=int(task_state.steps),
        cfg=float(task_state.cfg_scale),
        sampler=str(task_state.sampler_name),
        scheduler=str(final_scheduler_name),
        seed=int(task_dict['task_seed']),
        clip_layer=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
        batch_size=1,
        lora_specs=merged_loras,
        structural_tasks={
            cn_type: tuple(tuple(task) for task in list(tasks))
            for cn_type, tasks in (getattr(task_state, 'prepared_structural_cn_tasks', {}) or {}).items()
            if tasks
        },
        controlnet_paths=dict(controlnet_paths or {}),
        controlnet_quality=quality,
        contextual_tasks={
            cn_type: tuple(tuple(task) for task in list(tasks))
            for cn_type, tasks in (getattr(task_state, 'prepared_contextual_cn_tasks', {}) or {}).items()
            if tasks
        },
        contextual_assets=dict(contextual_assets or {}),
        runtime_policy=policy,
        initial_latent=getattr(task_state, 'initial_latent', None),
        disable_initial_latent=bool(getattr(task_state, 'inpaint_disable_initial_latent', False)),
        denoise_strength=float(denoising_strength) if denoising_strength is not None else None,
        original_scheduler_name=str(task_state.scheduler_name),
    )
    config_kwargs.update(_build_unified_spatial_kwargs(task_state, image_input_result=image_input_result))

    runtime = UnifiedSDXLRuntime(UnifiedSDXLRuntimeConfig(**config_kwargs))
    try:
        prepared_inputs, _ = runtime.prepare_inputs()
        active_process_key = resolve_unified_sdxl_process_key(
            task_state,
            loras=loras,
            base_model_additional_loras=base_model_additional_loras,
        )
        if active_process_key is not None:
            from backend import process_transition

            process_transition.set_active_process_key(active_process_key)
        denoise_result = runtime.denoise_prepared_inputs(
            prepared_inputs,
            callback=callback,
            disable_pbar=True,
        )
        decoded_images, _, _ = runtime.decode_latent(denoise_result.samples, tiled=bool(getattr(task_state, 'tiled', False)))
        return core.pytorch_to_numpy(decoded_images)
    finally:
        runtime.close()


def process_task(task_state, task_dict, current_task_id, total_count, all_steps,
                 preparation_steps, denoising_strength, final_scheduler_name, loras,
                 controlnet_paths=None,
                 progressbar_callback=None, yield_result_callback=None,
                 route_family=None, contextual_assets=None,
                 base_model_additional_loras=None, image_input_result=None):
    """
    Executes a single generation task (one image) using the unified SDXL runtime.
    """
    if task_state.last_stop is not False:
        resources.interrupt_current_processing()

    controlnet_paths = controlnet_paths or {}
    _ensure_supported_unified_runtime_request(task_state)
    imgs = _run_unified_sdxl_task(
        task_state,
        task_dict,
        current_task_id,
        total_count,
        all_steps,
        preparation_steps,
        denoising_strength,
        final_scheduler_name,
        loras=loras,
        base_model_additional_loras=base_model_additional_loras,
        controlnet_paths=controlnet_paths,
        contextual_assets=contextual_assets,
        image_input_result=image_input_result,
        progressbar_callback=progressbar_callback,
    )

    current_progress = int(preparation_steps + (100 - preparation_steps) / float(all_steps) * task_state.steps)

    if progressbar_callback:
        progressbar_callback(task_state, current_progress, f'Saving image {current_task_id + 1}/{total_count} to system ...')

    img_paths = save_and_log(
        task_state, task_state.height, task_state.width, imgs,
        task_dict, task_state.use_expansion, loras
    )

    if yield_result_callback:
        show_results = not task_state.disable_intermediate_results
        yield_result_callback(task_state, img_paths, current_progress, do_not_show_finished_images=not show_results)

    return imgs, img_paths, current_progress
