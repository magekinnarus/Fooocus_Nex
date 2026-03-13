import math
import random
import modules.config as config
import modules.constants as constants
import modules.core as core
import modules.default_pipeline as pipeline
import modules.util as util
from modules.sdxl_styles import apply_style, get_random_style, apply_arrays, random_style_name
from modules.util import safe_str, remove_empty_str, parse_lora_references_from_prompt


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




def process_prompt(task_state, base_model_additional_loras, progressbar_callback=None):
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
    
    # Inpaint Additional Prompt handling
    if task_state.current_tab == 'inpaint' and task_state.inpaint_additional_prompt != '':
        if prompt == '':
            prompt = task_state.inpaint_additional_prompt
        else:
            # Concatenate to the beginning so it's prioritized by CLIP
            prompt = task_state.inpaint_additional_prompt + '\n' + prompt
    
    if prompt == '':
        use_expansion = False
    
    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')

    loras, prompt = parse_lora_references_from_prompt(prompt, task_state.loras,
                                                      config.default_max_lora_number)

    pipeline.refresh_everything(base_model_name=task_state.base_model_name,
                                loras=loras, base_model_additional_loras=base_model_additional_loras,
                                vae_name=task_state.vae_name,
                                clip_name=task_state.clip_model_name)
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

    if progressbar_callback:
        task_state.current_progress += 1
        for i, t in enumerate(tasks):
            progressbar_callback(task_state, task_state.current_progress, f'Encoding positive #{i + 1} ...')
            t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])
        
        task_state.current_progress += 1
        for i, t in enumerate(tasks):
            if abs(float(task_state.cfg_scale) - 1.0) < 1e-4:
                t['uc'] = pipeline.clone_cond(t['c'])
            else:
                progressbar_callback(task_state, task_state.current_progress, f'Encoding negative #{i + 1} ...')
                t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])
    
    # Offload CLIP after encoding is finished for all tasks
    if pipeline.final_clip is not None:
        import backend.resources as resources
        resources.eject_model(pipeline.final_clip.patcher)

    task_state.use_expansion = use_expansion
    task_state.loras_processed = loras # Storing back to state if needed
    
    # For pipeline components (e.g. upscaler) that need conditioning
    if len(tasks) > 0:
        task_state.positive_cond = tasks[0]['c']
        task_state.negative_cond = tasks[0]['uc']
        
    return tasks
