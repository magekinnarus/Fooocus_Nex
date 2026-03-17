import gc
import os
import time
import traceback
import threading
import re
import torch
import numpy as np
import backend.resources as resources
import modules.config as config
import modules.default_pipeline as pipeline
import modules.flags as flags
import modules.bgr_engine as bgr_engine
import modules.objr_engine as objr_engine
from modules.task_state import TaskState
from modules.pipeline import (
    apply_overrides,
    patch_samplers,
    process_prompt,
    apply_image_input,
    apply_control_nets,
    apply_outpaint_expansion,
    apply_outpaint_inference_setup,
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
        from modules.flags import MetadataScheme
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.state = TaskState()
        self.yields = self.state.yields
        self.results = self.state.results # Shared reference
        self.is_valid = len(args) > 0
        
        if not self.is_valid:
            return

        if isinstance(args, list):
            raise TypeError("AsyncTask received a positional args list instead of a named dictionary. Clear your browser cache and restart.")

        s = self.state

        import modules.parameter_registry as registry
        for param in registry.PARAM_REGISTRY:
            if param.task_field is None:
                continue
            
            val = args.get(param.name, param.default)
            if param.transform and val is not None:
                try:
                    val = param.transform(val)
                except (ValueError, TypeError):
                    val = param.default
            setattr(s, param.task_field, val)

        s.original_steps = s.steps

        lora_data = []
        for i in range(default_max_lora_number):
            enabled = bool(args.get(f'lora_{i}_enabled', False))
            name = str(args.get(f'lora_{i}_model', 'None'))
            weight = float(args.get(f'lora_{i}_weight', 1.0))
            lora_data.append((enabled, name, weight))
        s.loras = get_enabled_loras(lora_data)

        if not args_manager.args.disable_metadata:
            s.save_metadata_to_images = args.get('save_metadata_to_images', False)
            scheme_val = args.get('metadata_scheme', 'fooocus')
            try:
                s.metadata_scheme = MetadataScheme(scheme_val)
            except ValueError:
                s.metadata_scheme = MetadataScheme.FOOOCUS
        else:
            s.save_metadata_to_images = False
            s.metadata_scheme = MetadataScheme.FOOOCUS

        def has_controlnet_input(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ''
            if isinstance(value, dict):
                for key in ['image', 'mask', 'background']:
                    item = value.get(key)
                    if isinstance(item, str) and item.strip() != '':
                        return True
                    if item is not None and not isinstance(item, str):
                        return True
                return False
            return True

        from modules.config import default_controlnet_image_count
        for i in range(default_controlnet_image_count):
            cn_img = args.get(f'cn_{i}_image')
            cn_stop = args.get(f'cn_{i}_stop', 1.0)
            cn_weight = args.get(f'cn_{i}_weight', 1.0)
            cn_type = args.get(f'cn_{i}_type')
            if has_controlnet_input(cn_img) and cn_type in s.cn_tasks:
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


    print(f'[Parameters] Seed = {s.seed}')
    dims = re.findall(r'\d+', str(s.aspect_ratios_selection))
    if len(dims) < 2:
        raise ValueError(f'Invalid aspect ratio selection: {s.aspect_ratios_selection!r}')
    s.width, s.height = int(dims[0]), int(dims[1])

    base_model_additional_loras = []

    if flags.remove_bg in s.goals or flags.remove_obj in s.goals:
        # User requested removals. Clean up diffusion models first to free ~2GB VRAM.
        progressbar(s, 5, "Clearing VRAM for Removal Models...")
        resources.soft_empty_cache()
        resources.unload_all_models()

        if flags.remove_bg in s.goals:
            progressbar(s, 10, "Background Removal Starting...")
            char_path, mask_path = bgr_engine.remove_background_from_file(
                filepath=s.remove_base_image,
                threshold=s.bgr_threshold,
                jit=s.bgr_jit
            )
            bgr_engine.unload_model()
            
            # Record result
            yield_result(s, [char_path, mask_path], 50 if flags.remove_obj in s.goals else 100, do_not_show_finished_images=True)
            
            # Handoff for sequential cleanup
            if flags.remove_obj in s.goals:
                s.remove_mask_image = mask_path

        if flags.remove_obj in s.goals:
            progressbar(s, 60 if flags.remove_bg in s.goals else 10, "Object Removal Starting...")
            res_path = objr_engine.remove_object_from_file(
                image_path=s.remove_base_image,
                mask_path=s.remove_mask_image,
                seed=s.seed,
                mask_dilate=s.objr_mask_dilate
            )
            objr_engine.unload_model()
            
            yield_result(s, [res_path], 100, do_not_show_finished_images=True)

        s.processing = False
        return
    
    save_step1_result = None
    try:
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

        if s.input_image_checkbox:
            if 'outpaint' in s.goals:
                # Phase 2: Inference setup on the expanded canvas
                # Note: res['outpaint_image/mask'] contains the result of Step 1 if it finished, 
                # or the original inputs if Step 2 was already selected.
                apply_outpaint_inference_setup(s, res['outpaint_image'], res['outpaint_mask'], progressbar, yield_result)

            if 'inpaint' in s.goals:
                # Phase 1/2 Inpaint: Manual dual-masking and inference setup
                apply_inpaint(s, res['inpaint_image'], res['inpaint_mask'], progressbar, yield_result)
    except EarlyReturnException as e:
        save_step1_result = e.payload
        if save_step1_result is None:
            s.yields.append(['finish', s.results])
            s.processing = False
            return

    if save_step1_result is not None:
        # Handle variable number of images (Outpaint: [canvas, bb], Inpaint: [bb])
        if not isinstance(save_step1_result, (list, tuple)):
            save_step1_result = [save_step1_result]
            
        images_to_save = []
        for img in save_step1_result:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                images_to_save.append(np.stack([img]*3, axis=-1))
            else:
                images_to_save.append(img)
                
        from modules.pipeline.output import save_and_log
        
        description = 'Phase 1 Outpaint Expansion' if 'outpaint' in s.goals else 'Phase 1 Inpaint BB'
        progressbar(s, 100, f'Saving {description} ...')
        
        img_paths = save_and_log(
            s, s.height, s.width, images_to_save, 
            {
                'log_positive_prompt': s.prompt, 
                'log_negative_prompt': s.negative_prompt, 
                'positive': [],
                'negative': [],
                'styles': s.style_selections, 
                'task_seed': s.seed,
                'description': description
            }, 
            False, s.loras)
        yield_result(s, img_paths, 100, do_not_show_finished_images=True)
        s.yields.append(['finish', s.results])
        s.processing = False
        return

    apply_overrides(s)
    
    tasks = []
    if not res.get('skip_prompt_processing', False):
        tasks = process_prompt(s, base_model_additional_loras, progressbar)
        
        # Phase Boundary: Encoding -> Diffusion
        print('[Nex-Memory] Phase: Encoding → Diffusion')
        gc.collect()
        resources.soft_empty_cache()

    if len(s.goals) > 0:
        s.current_progress += 1
        progressbar(s, s.current_progress, 'Image processing ...')

    if 'upscale' in s.goals:
        direct_return = apply_upscale(s, progressbar)
        
        # If apply_upscale returns False, it means we need refinement (Super-Upscale)
        if not direct_return:
            from modules.pipeline.tiled_refinement import apply_tiled_diffusion_refinement
            s.uov_input_image = apply_tiled_diffusion_refinement(s, s.uov_input_image, progressbar)
            
        # Final Save and Return for either Upscale or Super-Upscale
        from modules.pipeline.output import save_and_log
        progressbar(s, 100, 'Saving image to system ...')
        img_paths = save_and_log(s, s.height, s.width, [s.uov_input_image], {'log_positive_prompt': s.prompt, 'log_negative_prompt': s.negative_prompt, 'positive': [], 'negative': [], 'styles': s.style_selections, 'task_seed': s.seed}, s.use_expansion, s.loras)
        yield_result(s, img_paths, 100, do_not_show_finished_images=True)
        s.yields.append(['finish', s.results])
        s.processing = False
        return


    if 'cn' in s.goals:
        apply_control_nets(s, res['ip_adapter_face_path'], res['ip_adapter_path'], yield_result)
        if s.debugging_cn_preprocessor:
            s.yields.append(['finish', s.results])
            s.processing = False
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
    s.inpaint_mask_image = None


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

