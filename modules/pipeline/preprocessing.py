import math
import random
from collections import OrderedDict
import backend.resources as resources
import modules.config as config
import modules.constants as constants
import modules.core as core
import modules.default_pipeline as pipeline
from backend import conditioning
import modules.util as util
from modules.sdxl_styles import apply_style, get_random_style, apply_arrays, random_style_name
from modules.util import safe_str, remove_empty_str, parse_lora_references_from_prompt


_PROMPT_TASK_CACHE: OrderedDict[str, dict] = OrderedDict()
_PROMPT_TASK_CACHE_LIMIT = 16


def _uses_unified_sdxl_runtime_owner(task_state) -> bool:
    return str(getattr(task_state, 'sdxl_runtime_owner', '') or '').strip().lower() == 'unified'


def _resolve_residency_class(task_state, residency_class=None):
    if residency_class is not None:
        return resources.normalize_sdxl_residency_class(residency_class)
    return resources.normalize_sdxl_residency_class(getattr(task_state, 'sdxl_residency_class', None))


def _clone_prompt_task(task):
    cloned = dict(task)
    if cloned.get('c') is not None:
        cloned['c'] = pipeline.clone_cond(cloned['c'])
    if cloned.get('uc') is not None:
        cloned['uc'] = pipeline.clone_cond(cloned['uc'])
    for key in ('positive', 'negative', 'styles'):
        if isinstance(cloned.get(key), list):
            cloned[key] = list(cloned[key])
    return cloned


def _clone_prompt_tasks(tasks):
    return [_clone_prompt_task(task) for task in tasks]


def _freeze_prompt_tasks(tasks):
    return tuple(
        tuple(sorted((key, value) for key, value in task.items() if key not in {'c', 'uc', 'task_seed'}))
        for task in tasks
    )


def _build_prompt_task_fingerprint(task_state, tasks, *, route_family=None, residency_class=None):
    residency = _resolve_residency_class(task_state, residency_class=residency_class)
    prompt_blueprint = _freeze_prompt_tasks(tasks)
    clip = pipeline.final_clip
    execution_policy = getattr(task_state, 'sdxl_execution_policy', None)
    if _uses_unified_sdxl_runtime_owner(task_state):
        return conditioning.build_stage_fingerprint(
            'sdxl_prompt_encode',
            residency_class=residency,
            model_identity=str(getattr(task_state, 'base_model_name', None) or ''),
            text_encoder_identity=(
                'unified_runtime_clip',
                -abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
            ),
            clip_patch_uuid=tuple(getattr(task_state, 'loras_processed', ()) or getattr(task_state, 'loras', ()) or ()),
            clip_layer_idx=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
            lora_artifacts_state=tuple(getattr(task_state, 'loras_processed', ()) or ()),
            route_family_reconciliation_signature=(
                route_family or getattr(task_state, 'current_tab', None),
                'unified',
            ),
            route_family=route_family or getattr(task_state, 'current_tab', None),
            execution_family=getattr(execution_policy, 'execution_family', None),
            clip_residency_mode='runtime_owned',
            prompt_blueprint=prompt_blueprint,
        )
    return conditioning.build_stage_fingerprint(
        'sdxl_prompt_encode',
        residency_class=residency,
        model_identity=getattr(pipeline.model_base, 'filename', None),
        text_encoder_identity=(
            type(getattr(clip, 'model', clip)).__name__ if clip is not None else None,
            getattr(clip, 'layer_idx', None) if clip is not None else None,
        ),
        clip_patch_uuid=resources.model_reconciliation_signature(clip.patcher) if clip is not None else None,
        clip_layer_idx=getattr(clip, 'layer_idx', None) if clip is not None else None,
        lora_artifacts_state=getattr(pipeline.model_base, 'lora_artifact_registry', ()),
        route_family_reconciliation_signature=route_family or getattr(task_state, 'current_tab', None),
        route_family=route_family or getattr(task_state, 'current_tab', None),
        execution_family=getattr(execution_policy, 'execution_family', None),
        clip_residency_mode=getattr(execution_policy, 'clip_residency_mode', None),
        clip_skip=getattr(task_state, 'clip_skip', None),
        prompt_blueprint=prompt_blueprint,
    )


def _remember_prompt_tasks(fingerprint, tasks):
    cache_key = fingerprint.digest()
    _PROMPT_TASK_CACHE[cache_key] = {
        'fingerprint': fingerprint,
        'tasks': _clone_prompt_tasks(tasks),
    }
    _PROMPT_TASK_CACHE.move_to_end(cache_key)
    while len(_PROMPT_TASK_CACHE) > _PROMPT_TASK_CACHE_LIMIT:
        _PROMPT_TASK_CACHE.popitem(last=False)


def _load_prompt_tasks_from_cache(fingerprint):
    cached = _PROMPT_TASK_CACHE.get(fingerprint.digest())
    if cached is None:
        return None
    _PROMPT_TASK_CACHE.move_to_end(fingerprint.digest())
    return _clone_prompt_tasks(cached['tasks'])


def apply_overrides(task_state):
    """
    Applies user-defined overrides for width and height.
    Steps are now controlled directly by task_state.steps.
    """
    steps = task_state.steps
    width = task_state.width
    height = task_state.height

    if task_state.overwrite_width > 0:
        width = task_state.overwrite_width
    if task_state.overwrite_height > 0:
        height = task_state.overwrite_height
    
    task_state.width = width
    task_state.height = height
    return steps, width, height


def patch_discrete(unet, scheduler_name):
    return core.opModelSamplingDiscrete.patch(unet, scheduler_name, False)[0]


def patch_edm(unet, scheduler_name):
    return core.opModelSamplingContinuousEDM.patch(unet, scheduler_name, 120.0, 0.002)[0]


def patch_samplers(task_state):
    """
    Patches the UNet for specific schedulers like LCM, TCD, or EDM.
    Returns the final scheduler name to use in the sampler.
    """
    final_scheduler_name = task_state.scheduler_name

    if task_state.scheduler_name in ['lcm', 'tcd']:
        final_scheduler_name = 'sgm_uniform'
        if pipeline.final_unet is not None:
            pipeline.final_unet = patch_discrete(pipeline.final_unet, task_state.scheduler_name)

    elif task_state.scheduler_name == 'edm_playground_v2.5':
        final_scheduler_name = 'karras'
        if pipeline.final_unet is not None:
            pipeline.final_unet = patch_edm(pipeline.final_unet, task_state.scheduler_name)

    return final_scheduler_name




def process_prompt(task_state, base_model_additional_loras, progressbar_callback=None, *, route_context=None, route_family=None, residency_class=None):
    """
    Gathers prompts, styles, and LoRAs. Encodes prompts via CLIP.
    """
    prompt = task_state.prompt
    negative_prompt = task_state.negative_prompt
    image_number = task_state.image_number
    disable_seed_increment = task_state.disable_seed_increment
    use_expansion = task_state.use_expansion
    use_style = task_state.use_style

    prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
    negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')
    prompt = prompts[0]
    negative_prompt = negative_prompts[0]
    
    # Masked-edit additional prompt handling
    edit_additional_prompt = ''
    if 'inpaint' in task_state.goals and task_state.inpaint_additional_prompt != '':
        edit_additional_prompt = task_state.inpaint_additional_prompt
    elif 'outpaint' in task_state.goals and getattr(task_state, 'outpaint_additional_prompt', '') != '':
        edit_additional_prompt = task_state.outpaint_additional_prompt

    if edit_additional_prompt != '':
        if prompt == '':
            prompt = edit_additional_prompt
        else:
            # Concatenate to the beginning so it's prioritized by CLIP
            prompt = edit_additional_prompt + '\n' + prompt
    
    if prompt == '':
        use_expansion = False
    
    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')

    loras, prompt = parse_lora_references_from_prompt(prompt, task_state.loras,
                                                      config.default_max_lora_number)
    task_state.loras_processed = loras
    unified_runtime_owner = _uses_unified_sdxl_runtime_owner(task_state)

    sdxl_policy = getattr(task_state, 'sdxl_execution_policy', None)

    if not unified_runtime_owner:
        # Legacy shared-pipeline fallback: refresh the inherited SDXL bridge and encode CLIP here.
        with resources.memory_phase_scope(
            resources.MemoryPhase.MODEL_REFRESH,
            task=task_state,
            notes={
                'base_model': task_state.base_model_name,
                'vae': task_state.vae_name,
                'clip': task_state.clip_model_name,
            },
            end_notes={'completed': True},
        ):
            pipeline.refresh_everything(base_model_name=task_state.base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        vae_name=task_state.vae_name,
                                        clip_name=task_state.clip_model_name,
                                        sdxl_policy=sdxl_policy)
            pipeline.set_clip_skip(task_state.clip_skip)

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'Processing prompts ...')

    tasks = []
    task_rng = random.Random(task_state.seed)

    for i in range(image_number):
        if disable_seed_increment:
            task_seed = task_state.seed % (constants.MAX_SEED + 1)
        else:
            task_seed = (task_state.seed + i) % (constants.MAX_SEED + 1)

        task_prompt = apply_arrays(prompt, i)
        task_negative_prompt = negative_prompt
        task_extra_positive_prompts = extra_positive_prompts
        task_extra_negative_prompts = extra_negative_prompts

        positive_basic_workloads = []
        negative_basic_workloads = []

        task_styles = task_state.style_selections.copy()
        if use_style:
            for j, s in enumerate(task_styles):
                if s == random_style_name:
                    s = get_random_style(task_rng)
                    task_styles[j] = s
                p, n, _ = apply_style(s, positive=task_prompt)
                positive_basic_workloads = positive_basic_workloads + p
                negative_basic_workloads = negative_basic_workloads + n

            positive_basic_workloads = [task_prompt] + positive_basic_workloads
            negative_basic_workloads = [task_negative_prompt] + negative_basic_workloads
        else:
            positive_basic_workloads.append(task_prompt)
            negative_basic_workloads.append(task_negative_prompt)

        positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

        
        tasks.append(dict(
            task_seed=task_seed,
            task_prompt=task_prompt,
            task_negative_prompt=task_negative_prompt,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            c=None,
            uc=None,
            positive_top_k=len(positive_basic_workloads),
            negative_top_k=len(negative_basic_workloads),
            log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
            log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
            styles=task_styles
        ))

    prompt_fingerprint = _build_prompt_task_fingerprint(
        task_state,
        tasks,
        route_family=route_family or getattr(route_context, 'route_family', None),
        residency_class=residency_class,
    )
    cached_tasks = _load_prompt_tasks_from_cache(prompt_fingerprint)
    if cached_tasks is not None:
        task_state.use_expansion = use_expansion
        if len(cached_tasks) > 0 and not unified_runtime_owner:
            task_state.positive_cond = cached_tasks[0]['c']
            task_state.negative_cond = cached_tasks[0]['uc']
        if (not unified_runtime_owner) and pipeline.final_clip is not None and not bool(getattr(sdxl_policy, 'keep_clip_loaded', False)):
            resources.eject_model(pipeline.final_clip.patcher)
        if route_context is not None:
            route_context.set_route_artifact('prompt_encode', cached_tasks, fingerprint=prompt_fingerprint)
        return cached_tasks

    if unified_runtime_owner:
        # Unified SDXL owns prompt execution later; keep only the prompt task blueprint here.
        task_state.use_expansion = use_expansion
        task_state.positive_cond = None
        task_state.negative_cond = None
        _remember_prompt_tasks(prompt_fingerprint, tasks)
        if route_context is not None:
            route_context.set_route_artifact('prompt_encode', tasks, fingerprint=prompt_fingerprint)
        return tasks

    with resources.memory_phase_scope(
        resources.MemoryPhase.PROMPT_ENCODE,
        task=task_state,
        notes={'image_number': image_number, 'cfg_scale': float(task_state.cfg_scale)},
        end_notes={'tasks_encoded': len(tasks)},
    ):
        if progressbar_callback:
            task_state.current_progress += 1
            for i, t in enumerate(tasks):
                progressbar_callback(task_state, task_state.current_progress, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(
                    texts=t['positive'],
                    pool_top_k=t['positive_top_k'],
                    route_family=route_family or getattr(route_context, 'route_family', None),
                    residency_class=residency_class,
                    execution_family=getattr(sdxl_policy, 'execution_family', None),
                    clip_residency_mode=getattr(sdxl_policy, 'clip_residency_mode', None),
                )
            
            task_state.current_progress += 1
            for i, t in enumerate(tasks):
                if abs(float(task_state.cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar_callback(task_state, task_state.current_progress, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(
                        texts=t['negative'],
                        pool_top_k=t['negative_top_k'],
                        route_family=route_family or getattr(route_context, 'route_family', None),
                        residency_class=residency_class,
                        execution_family=getattr(sdxl_policy, 'execution_family', None),
                        clip_residency_mode=getattr(sdxl_policy, 'clip_residency_mode', None),
                    )
        else:
            for i, t in enumerate(tasks):
                t['c'] = pipeline.clip_encode(
                    texts=t['positive'],
                    pool_top_k=t['positive_top_k'],
                    route_family=route_family or getattr(route_context, 'route_family', None),
                    residency_class=residency_class,
                    execution_family=getattr(sdxl_policy, 'execution_family', None),
                    clip_residency_mode=getattr(sdxl_policy, 'clip_residency_mode', None),
                )
                if abs(float(task_state.cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    t['uc'] = pipeline.clip_encode(
                        texts=t['negative'],
                        pool_top_k=t['negative_top_k'],
                        route_family=route_family or getattr(route_context, 'route_family', None),
                        residency_class=residency_class,
                        execution_family=getattr(sdxl_policy, 'execution_family', None),
                        clip_residency_mode=getattr(sdxl_policy, 'clip_residency_mode', None),
                    )

        # Offload CLIP after encoding is finished for all tasks
        if pipeline.final_clip is not None and not bool(getattr(sdxl_policy, 'keep_clip_loaded', False)):
            resources.eject_model(pipeline.final_clip.patcher)

    task_state.use_expansion = use_expansion
    _remember_prompt_tasks(prompt_fingerprint, tasks)
    if route_context is not None:
        route_context.set_route_artifact('prompt_encode', tasks, fingerprint=prompt_fingerprint)
    
    # For pipeline components (e.g. upscaler) that need conditioning
    if len(tasks) > 0:
        task_state.positive_cond = tasks[0]['c']
        task_state.negative_cond = tasks[0]['uc']
        
    return tasks
