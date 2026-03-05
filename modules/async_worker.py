import gc
import os
import time
import traceback
import threading
import torch
import numpy as np
import backend.resources as resources
import modules.config as config
import modules.default_pipeline as pipeline
import modules.flags as flags
from modules.task_state import TaskState
from modules.pipeline import (
    apply_overrides,
    patch_samplers,
    set_hyper_sd_defaults,
    set_lightning_defaults,
    set_lcm_defaults,
    process_prompt,
    apply_image_input,
    apply_control_nets,
    apply_vary,
    apply_outpaint,
    apply_upscale,
    apply_inpaint,
    process_task,
    yield_result,
    build_image_wall,
    EarlyReturnException
)


class AsyncTask:
    callback_steps: float = 0.0

    def __init__(self, args):
        from modules.flags import Performance, MetadataScheme
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.state = TaskState()
        self.yields = self.state.yields
        self.results = self.state.results # Shared reference
        self.args = args.copy()
        
        if len(args) == 0:
            return

        args.reverse()
        s = self.state
        s.generate_image_grid = args.pop()
        s.prompt = args.pop()
        s.negative_prompt = args.pop()
        s.style_selections = args.pop()

        s.performance_selection = Performance(args.pop())
        s.steps = s.performance_selection.steps()
        s.original_steps = s.steps

        s.aspect_ratios_selection = args.pop()
        s.image_number = args.pop()
        s.output_format = args.pop()
        s.seed = int(args.pop())
        _ = args.pop()  # read_wildcards_in_order (removed)
        s.sharpness = args.pop()
        s.cfg_scale = args.pop()
        s.base_model_name = args.pop()
        s.vae_name = args.pop()
        s.clip_model_name = args.pop()
        s.loras = get_enabled_loras([(bool(args.pop()), str(args.pop()), float(args.pop())) for _ in
                                        range(default_max_lora_number)])
        s.input_image_checkbox = args.pop()
        s.current_tab = args.pop()
        s.uov_method = args.pop()
        s.uov_input_image = args.pop()
        
        s.outpaint_selections = args.pop()
        s.outpaint_input_image = args.pop()
        s.outpaint_mask_image = args.pop()
        
        s.inpaint_input_image = args.pop()
        s.inpaint_additional_prompt = args.pop()
        s.inpaint_mask_image = args.pop()
        s.inpaint_bb_image = args.pop()

        s.disable_preview = args.pop()
        s.disable_intermediate_results = args.pop()
        s.disable_seed_increment = args.pop()
        s.adm_scaler_positive = args.pop()
        s.adm_scaler_negative = args.pop()
        s.adm_scaler_end = args.pop()
        s.adaptive_cfg = args.pop()
        s.clip_skip = args.pop()
        s.sampler_name = args.pop()
        s.scheduler_name = args.pop()
        s.overwrite_step = args.pop()
        s.overwrite_width = args.pop()
        s.overwrite_height = args.pop()
        s.overwrite_vary_strength = args.pop()
        s.overwrite_upscale_strength = args.pop()
        s.mixing_image_prompt_and_vary_upscale = args.pop()
        s.mixing_image_prompt_and_inpaint = args.pop()
        s.debugging_cn_preprocessor = args.pop()
        s.skipping_cn_preprocessor = args.pop()
        s.canny_low_threshold = args.pop()
        s.canny_high_threshold = args.pop()
        s.controlnet_softness = args.pop()
        
        s.debugging_inpaint_preprocessor = args.pop()
        s.inpaint_disable_initial_latent = args.pop()
        s.inpaint_engine = args.pop()
        s.inpaint_strength = args.pop()
        s.inpaint_advanced_masking_checkbox = args.pop()
        s.invert_mask_checkbox = args.pop()
        s.inpaint_erode_or_dilate = args.pop()
        s.inpaint_step2_checkbox = args.pop()
        
        s.outpaint_engine = args.pop()
        s.outpaint_strength = args.pop()
        s.outpaint_advanced_masking_checkbox = args.pop()
        s.outpaint_invert_mask_checkbox = args.pop()
        
        inpaint_outpaint_expansion_size_val = args.pop()
        if inpaint_outpaint_expansion_size_val is None or inpaint_outpaint_expansion_size_val == '':
            s.inpaint_outpaint_expansion_size = config.default_outpaint_expansion_size
        else:
            s.inpaint_outpaint_expansion_size = int(inpaint_outpaint_expansion_size_val)
            
        s.outpaint_step2_checkbox = args.pop()

        s.save_metadata_to_images = args.pop() if not args_manager.args.disable_image_log else False
        s.metadata_scheme = MetadataScheme(
            args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        for _ in range(config.default_controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                s.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

    @property
    def generate_image_grid(self): return self.state.generate_image_grid
    @property
    def last_stop(self): return self.state.last_stop
    @last_stop.setter
    def last_stop(self, value): self.state.last_stop = value
    @property
    def processing(self): return self.state.processing
    @processing.setter
    def processing(self, value): self.state.processing = value


async_tasks = []


def progressbar(task_state, number, text):
    resources.throw_exception_if_processing_interrupted()
    print(f'[Fooocus] {text}')
    task_state.yields.append(['preview', (number, text, None)])


@torch.no_grad()
@torch.inference_mode()
def handler(async_task: AsyncTask):
    s = async_task.state
    preparation_start_time = time.perf_counter()
    s.processing = True
    s.current_progress = 0

    if s.performance_selection == flags.Performance.EXTREME_SPEED:
        set_lcm_defaults(s, progressbar)
    elif s.performance_selection == flags.Performance.LIGHTNING:
        set_lightning_defaults(s, progressbar)
    elif s.performance_selection == flags.Performance.HYPER_SD:
        set_hyper_sd_defaults(s, progressbar)

    print(f'[Parameters] Seed = {s.seed}')
    
    width_str, height_str = s.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
    s.width, s.height = int(width_str), int(height_str)

    base_model_additional_loras = []
    
    res = {}
    if s.input_image_checkbox:
        res = apply_image_input(s, base_model_additional_loras, progressbar)
        base_model_additional_loras = res['base_model_additional_loras']
        
    s.current_progress = 1
    progressbar(s, s.current_progress, 'Loading ControlNets ...')
    pipeline.refresh_controlnets([res.get('controlnet_canny_path'), res.get('controlnet_cpds_path')] if s.input_image_checkbox else [])
    
    import extras.ip_adapter as ip_adapter
    ip_adapter.load_ip_adapter(res.get('clip_vision_path'), res.get('ip_negative_path'), res.get('ip_adapter_path'))
    ip_adapter.load_ip_adapter(res.get('clip_vision_path'), res.get('ip_negative_path'), res.get('ip_adapter_face_path'))

    apply_overrides(s)
    
    tasks = []
    if not res.get('skip_prompt_processing', False):
        tasks = process_prompt(s, base_model_additional_loras, progressbar)

    if len(s.goals) > 0:
        s.current_progress += 1
        progressbar(s, s.current_progress, 'Image processing ...')

    if 'vary' in s.goals:
        apply_vary(s, progressbar)
    
    if 'upscale' in s.goals:
        direct_return = apply_upscale(s, progressbar)
        if direct_return:
            from modules.pipeline.output import save_and_log
            progressbar(s, 100, 'Saving image to system ...')
            img_paths = save_and_log(s, s.height, s.width, [s.uov_input_image], {'log_positive_prompt': s.prompt, 'log_negative_prompt': s.negative_prompt, 'styles': s.style_selections, 'task_seed': s.seed}, s.use_expansion, s.loras)
            yield_result(s, img_paths, 100, do_not_show_finished_images=True)
            return

    if 'inpaint' in s.goals:
        try:
            inpaint_image, inpaint_mask = apply_outpaint(s, res['inpaint_image'], res['inpaint_mask'])
            apply_inpaint(s, inpaint_image, inpaint_mask, progressbar, yield_result)
        except EarlyReturnException as e:
            # Step 1 outpaint: save expanded canvas + mask for the user to use in Step 2
            if e.payload is not None:
                inpaint_image, inpaint_mask = e.payload
                from modules.pipeline.output import save_and_log
                progressbar(s, 100, 'Saving composite image and mask ...')
                
                # Ensure mask is 3 channel for saving
                if inpaint_mask.ndim == 2:
                    inpaint_mask_save = np.stack([inpaint_mask]*3, axis=-1)
                else:
                    inpaint_mask_save = inpaint_mask
                
                img_paths = save_and_log(
                    s, s.height, s.width, [inpaint_image, inpaint_mask_save], 
                    {
                        'log_positive_prompt': s.prompt, 
                        'log_negative_prompt': s.negative_prompt, 
                        'positive': s.prompt,
                        'negative': s.negative_prompt,
                        'styles': s.style_selections, 
                        'task_seed': s.seed,
                        'description': 'Phase 1 Outpaint Prep'
                    }, 
                    False, s.loras)
                yield_result(s, img_paths, 100, do_not_show_finished_images=True)
                return

    if 'cn' in s.goals:
        apply_control_nets(s, res['ip_adapter_face_path'], res['ip_adapter_path'], yield_result)
        if s.debugging_cn_preprocessor:
            return

    steps, _, _ = apply_overrides(s)
    all_steps = max(steps * s.image_number, 1)
    
    preparation_steps = s.current_progress
    final_scheduler_name = patch_samplers(s)
    
    s.yields.append(['preview', (s.current_progress, 'Moving model to GPU ...', None)])
    processing_start_time = time.perf_counter()

    for i, t in enumerate(tasks):
        progressbar(s, s.current_progress, f'Preparing task {i + 1}/{s.image_number} ...')
        execution_start_time = time.perf_counter()
        interrupted_action = None
        
        try:
            process_task(
                s, t, i, s.image_number, all_steps, preparation_steps, 
                s.denoising_strength, final_scheduler_name, s.loras,
                res.get('controlnet_canny_path'), res.get('controlnet_cpds_path'),
                progressbar, yield_result
            )
        except resources.InterruptProcessingException:
            if s.last_stop == 'skip':
                print('User skipped')
                s.last_stop = False
                interrupted_action = 'skip'
            else:
                print('User stopped')
                interrupted_action = 'stop'
        finally:
            if 'c' in t:
                del t['c']
            if 'uc' in t:
                del t['uc']
            resources.soft_empty_cache()

        if interrupted_action == 'skip':
            continue
        if interrupted_action == 'stop':
            break

        print(f'Task {i+1} time: {time.perf_counter() - execution_start_time:.2f}s')

    s.processing = False
    print(f'Total processing time: {time.perf_counter() - processing_start_time:.2f}s')

    # Free runtime state that the original closure pattern naturally released
    s.initial_latent = None
    s.positive_cond = None
    s.negative_cond = None
    s.uov_input_image = None
    s.inpaint_input_image = None
    s.inpaint_mask_image_upload = None


def worker():
    pid = os.getpid()
    print(f'Started worker with PID {pid}')
    
    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            try:
                handler(task)
                if task.state.generate_image_grid:
                    build_image_wall(task.state)
                task.yields.append(['finish', task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                gc.collect()
                resources.soft_empty_cache()


threading.Thread(target=worker, daemon=True).start()
