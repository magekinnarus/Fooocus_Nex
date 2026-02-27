import time
import torch
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import modules.inpaint_worker as inpaint_worker
import backend.resources as resources
import backend.loader as loader
from modules.pipeline.output import save_and_log, yield_result


def get_sampling_callback(task_state, progressbar_callback, current_task_id, total_count, preparation_steps, all_steps):
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
        
        task_state.yields.append(['preview', (progress_val, status_text, y)])

    return callback


def process_task(task_state, task_dict, current_task_id, total_count, all_steps, 
                 preparation_steps, denoising_strength, final_scheduler_name, loras, 
                 controlnet_canny_path, controlnet_cpds_path, inpaint_head_model_path,
                 progressbar_callback=None, yield_result_callback=None):
    """
    Executes a single generation task (one image).
    """
    if task_state.last_stop is not False:
        resources.interrupt_current_processing()
            
    quality = {
        "sharpness": task_state.sharpness,
        "adaptive_cfg": task_state.adaptive_cfg,
        "adm_scaler_positive": task_state.adm_scaler_positive,
        "adm_scaler_negative": task_state.adm_scaler_negative,
        "adm_scaler_end": task_state.adm_scaler_end,
        "controlnet_softness": task_state.controlnet_softness
    }

    positive_cond = task_dict['c']
    negative_cond = task_dict['uc']

    if 'cn' in task_state.goals:
        for cn_flag, cn_path in [
            (flags.cn_canny, controlnet_canny_path),
            (flags.cn_cpds, controlnet_cpds_path)
        ]:
            for cn_task in task_state.cn_tasks[cn_flag]:
                cn_img, cn_stop, cn_weight = cn_task
                cn_net = pipeline.loaded_ControlNets[cn_path]
                loader.patch_controlnet_for_quality(cn_net, quality)
                positive_cond, negative_cond = core.apply_controlnet(
                    positive_cond, negative_cond,
                    cn_net, cn_img, cn_weight, 0, cn_stop)

    callback = get_sampling_callback(
        task_state, progressbar_callback, current_task_id, total_count, 
        preparation_steps, all_steps
    )

    imgs = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=task_state.steps,
        width=task_state.width,
        height=task_state.height,
        image_seed=task_dict['task_seed'],
        callback=callback,
        sampler_name=task_state.sampler_name,
        scheduler_name=final_scheduler_name,
        latent=task_state.initial_latent,
        denoise=denoising_strength,
        tiled=task_state.tiled,
        cfg_scale=task_state.cfg_scale,
        disable_preview=task_state.disable_preview,
        quality=quality
    )
    
    del positive_cond, negative_cond  # Save memory
    
    if inpaint_worker.current_task is not None:
        imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
    
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
