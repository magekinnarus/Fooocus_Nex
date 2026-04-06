import numpy as np
import modules.config as config
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import extras.preprocessors as preprocessors
import backend.ip_adapter as contextual_ip_adapter
import backend.preprocessors as structural_preprocessors
import backend.pulid_runtime as pulid_runtime
import backend.resources as resources
from modules.util import (HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil, 
                          get_shape_ceil, resample_image, erode_or_dilate)
from modules.upscaler import perform_upscale
import modules.mask_processing as mask_proc


class EarlyReturnException(BaseException):
    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload




def apply_outpaint_inference_setup(task_state, inpaint_image, inpaint_mask, 
                                  progressbar_callback=None, yield_result_callback=None):
    """
    Sets up the outpainting worker, patches the UNet, and encodes the initial latent.
    Exclusively using OutpaintPipeline.
    """
    inpaint_disable_initial_latent = getattr(task_state, 'inpaint_disable_initial_latent', False)

    from modules.pipeline.outpaint import OutpaintPipeline
    outpaint = OutpaintPipeline()
    
    # Use UI outpaint_strength (default 1.0, user usually lowers it for sketching)
    denoising_strength = getattr(task_state, 'outpaint_strength', 1.0)
        
    outpaint_direction = getattr(task_state, 'outpaint_direction', None)
    if isinstance(outpaint_direction, list) and len(outpaint_direction) > 0:
        outpaint_direction = outpaint_direction[0].lower()
        
    ctx = outpaint.prepare(
        image=inpaint_image,
        mask=inpaint_mask,
        outpaint_direction=outpaint_direction,
        extend_factor=1.2
    )
    
    # --- Resolved BB Image and BB Mask Support ---
    # If the user has provided an edited BB image or a manual BB mask, override the context.
    import modules.mask_processing as mask_proc
    
    raw_bb_image = getattr(task_state, 'outpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'outpaint_mask_image', None) # Upload slot
    # Hidden mask field from brush drawing on BB image
    brush_mask_data = getattr(task_state, 'outpaint_bb_mask_data', '')
    
    bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
    if bb_img_data is not None:
        ctx.bb_image = bb_img_data
        ctx.target_h, ctx.target_w = bb_img_data.shape[:2]
        
    # Combine uploaded mask with brush-drawn mask
    manual_mask = mask_proc.unpack_gradio_data(raw_bb_mask)
    brush_mask = mask_proc.unpack_gradio_data(brush_mask_data)
    combined_bb_mask = mask_proc.combine_masks(manual_mask, brush_mask)
    if combined_bb_mask is not None:
        # Ensure mask matches BB image resolution
        combined_bb_mask = resample_image(combined_bb_mask, width=ctx.target_w, height=ctx.target_h)
        ctx.bb_mask = combined_bb_mask
        ctx.bb_image = outpaint.pixelate_mask_area(ctx.bb_image, combined_bb_mask)
    # Rebuild the full-image blend mask from the final BB mask so stitch-back
    # follows the user-edited Outpaint mask instead of the earlier auto mask.
    y1, y2, x1, x2 = ctx.bb
    full_mask = np.zeros_like(ctx.original_image[:, :, 0])
    H, W = full_mask.shape
    patch_mask_resized = resample_image(ctx.bb_mask, width=x2-x1, height=y2-y1)

    iy1, iy2 = max(0, y1), min(H, y2)
    ix1, ix2 = max(0, x1), min(W, x2)
    cy1, cy2 = iy1 - y1, iy2 - y1
    cx1, cx2 = ix1 - x1, ix2 - x1

    full_mask[iy1:iy2, ix1:ix2] = patch_mask_resized[cy1:cy2, cx1:cx2]
    ctx.blend_mask = outpaint._morphological_open(full_mask)

    
    with resources.memory_phase_scope(
        resources.MemoryPhase.VAE_ENCODE,
        task=task_state,
        notes={'route': 'outpaint', 'denoise': float(denoising_strength)},
        end_notes={'completed': True},
    ):
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=task_state.steps,
            denoise=denoising_strength
        )
        latent_dict = outpaint.encode(ctx, candidate_vae)
    
    task_state.inpaint_context = ctx
    task_state.width = ctx.target_w
    task_state.height = ctx.target_h
    
    if not inpaint_disable_initial_latent:
        task_state.initial_latent = latent_dict
        
    task_state.denoising_strength = denoising_strength
    
    final_height, final_width = ctx.original_image.shape[:2]
    print(f'Outpaint setup: BB resolution {ctx.target_w}x{ctx.target_h}, Original resolution {final_width}x{final_height}.')


def apply_inpaint(task_state, inpaint_image, inpaint_mask, 
                  progressbar_callback=None, yield_result_callback=None):
    """
    Resolves the required inpaint assets, patches the UNet, and encodes the initial latent.
    Inference always runs from the resolved Full Image, Context Mask, BB Image, and BB Mask set.
    """
    denoising_strength = getattr(task_state, 'inpaint_strength', 1.0)
    inpaint_disable_initial_latent = getattr(task_state, 'inpaint_disable_initial_latent', False)

    raw_input_image = getattr(task_state, 'inpaint_input_image', None)
    raw_context_mask = getattr(task_state, 'inpaint_context_mask_image', None)
    raw_bb_image = getattr(task_state, 'inpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'inpaint_mask_image', None)

    input_image = mask_proc.unpack_gradio_data(raw_input_image) if raw_input_image is not None else inpaint_image
    prepared_context_mask = getattr(task_state, 'context_mask', None)
    if raw_context_mask is not None:
        context_mask = mask_proc.unpack_gradio_data(raw_context_mask)
    elif prepared_context_mask is not None:
        context_mask = prepared_context_mask
    else:
        context_mask = inpaint_mask
    if context_mask is not None:
        context_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(context_mask))

    bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
    if bb_img_data is not None:
        bb_img_data = HWC3(bb_img_data)
    bb_mask_2d = mask_proc.combine_masks(
        mask_proc.unpack_gradio_data(raw_bb_mask),
        mask_proc.extract_mask_from_layers(raw_bb_image) if isinstance(raw_bb_image, dict) else None
    )

    from modules.pipeline.inpaint import InpaintPipeline
    inpaint = InpaintPipeline()

    if input_image is not None and context_mask is not None:
        ctx = inpaint.prepare(
            image=input_image,
            mask=context_mask,
            extend_factor=1.2
        )
        print(f"[Debug] Context derived from {input_image.shape[1]}x{input_image.shape[0]} image via context mask.")
    else:
        ctx = inpaint.prepare(
            image=inpaint_image,
            mask=inpaint_mask,
            extend_factor=1.2
        )
        print(f"[Debug] Context derived from standard inpaint inputs.")

    if bb_img_data is not None:
        ctx.bb_image = bb_img_data
        ctx.target_h, ctx.target_w = bb_img_data.shape[:2]
        print(f"[Debug] Using resolved BB image: {ctx.target_w}x{ctx.target_h}")

    if bb_mask_2d is not None:
        ctx.bb_mask = bb_mask_2d
        print(f"[Debug] Using resolved BB mask.")

    if ctx.bb_image is None:
        raise ValueError('Inpaint BB image is required before inference')
    if ctx.bb_mask is None:
        raise ValueError('Inpaint BB mask is required before inference')

    ctx.bb_image = HWC3(mask_proc.ensure_numpy(ctx.bb_image))
    ctx.bb_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(ctx.bb_mask))
    ctx.bb_image = resample_image(ctx.bb_image, width=ctx.target_w, height=ctx.target_h)
    ctx.bb_mask = resample_image(ctx.bb_mask, width=ctx.target_w, height=ctx.target_h)

    y1, y2, x1, x2 = ctx.bb
    full_mask = np.zeros_like(ctx.original_image[:, :, 0])
    H, W = full_mask.shape
    patch_mask_resized = resample_image(ctx.bb_mask, width=x2-x1, height=y2-y1)

    iy1, iy2 = max(0, y1), min(H, y2)
    ix1, ix2 = max(0, x1), min(W, x2)
    cy1, cy2 = iy1 - y1, iy2 - y1
    cx1, cx2 = ix1 - x1, ix2 - x1

    full_mask[iy1:iy2, ix1:ix2] = patch_mask_resized[cy1:cy2, cx1:cx2]
    ctx.blend_mask = inpaint._morphological_open(full_mask)

    task_state.width = ctx.target_w
    task_state.height = ctx.target_h

    if getattr(task_state, 'debugging_inpaint_preprocessor', False):
        if yield_result_callback:
            yield_result_callback(task_state, [ctx.bb_image, ctx.bb_mask], 100, do_not_show_finished_images=True)
        raise EarlyReturnException

    with resources.memory_phase_scope(
        resources.MemoryPhase.VAE_ENCODE,
        task=task_state,
        notes={'route': 'inpaint', 'denoise': float(denoising_strength)},
        end_notes={'completed': True},
    ):
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=task_state.steps,
            denoise=denoising_strength
        )

        latent_dict = inpaint.encode(ctx, candidate_vae)
    task_state.inpaint_context = ctx
    task_state.width = ctx.target_w
    task_state.height = ctx.target_h

    if not inpaint_disable_initial_latent:
        task_state.initial_latent = latent_dict

    task_state.denoising_strength = denoising_strength

    final_height, final_width = ctx.original_image.shape[:2]
    print(f'Inpaint setup: BB resolution {ctx.target_w}x{ctx.target_h}, Original resolution {final_width}x{final_height}.')


def apply_upscale(task_state, progressbar_callback=None):
    """
    Performs image upscaling and sets up the latent for the diffusion pass.
    """
    uov_input_image = task_state.uov_input_image
    uov_method = task_state.uov_method.lower()
    uov_input_image = mask_proc.ensure_numpy(uov_input_image)
    H, W, C = uov_input_image.shape
    
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, f'Upscaling image from {str((W, H))} ...')
    
    # Pre-upscale memory cleanup: Clear UNet/VAE/CLIP to make room for GAN
    from backend import resources
    resources.unload_all_models()
    resources.soft_empty_cache()
    import gc
    gc.collect()

    # 1. GAN Upscale with new multi-model engine
    from modules.upscaler import perform_upscale, clear_model_cache
    
    # Super-Upscale should use the lightest default model (Nomos2) to save memory
    upscale_model_to_use = task_state.upscale_model
    if uov_method == 'super-upscale':
        upscale_model_to_use = '4xNomos2_otf_esrgan.pth'
        print(f'Super-Upscale detected: Forcing light model {upscale_model_to_use} for initial pass.')

    import gc
    from backend import resources
    
    # Pre-upscale cleanup: Offload everything to make room for GAN model
    resources.unload_all_models()
    gc.collect()
    resources.soft_empty_cache()

    uov_input_image = perform_upscale(
        uov_input_image, 
        model_name=upscale_model_to_use if upscale_model_to_use != "None" else None,
        scale_override=task_state.upscale_scale_override if task_state.upscale_scale_override > 0 else None
    )
    print(f'Image upscaled via GAN to {str(uov_input_image.shape[:2])}.')

    # Post-upscale cleanup: Purge GAN model and ensure GPU is clear
    clear_model_cache()
    gc.collect()
    resources.soft_empty_cache()

    # 2. Handle "Upscale" (Light) or "Super-Upscale" (Stage 1)
    if uov_method == 'upscale':
        task_state.uov_input_image = uov_input_image
        task_state.width = uov_input_image.shape[1]
        task_state.height = uov_input_image.shape[0]
        print('Upscale (Light) completed.')
        return True

    if uov_method == 'super-upscale':
        # Prepare for sequential tiled refinement. NO VAE encode here to save VRAM.
        task_state.uov_input_image = uov_input_image
        task_state.width = uov_input_image.shape[1]
        task_state.height = uov_input_image.shape[0]
        print('Super-Upscale Stage 1 (GAN) completed. Passing to Tiled Refinement.')
        return False # False triggers refinement in worker


def prepare_upscale(task_state, progressbar_callback=None):
    """
    Determines if upscale is needed and sets the appropriate goals.
    """
    task_state.uov_input_image = HWC3(mask_proc.ensure_numpy(task_state.uov_input_image))
    uov_method = task_state.uov_method.lower()
    
    skip_prompt_processing = False
    if 'upscale' in uov_method:
        task_state.goals.append('upscale')
        
        # Validate selected model exists (if not "None")
        if task_state.upscale_model != "None":
            from modules.upscaler import list_available_models
            available = list_available_models()
            if task_state.upscale_model not in available:
                print(f"[Warning] Selected upscale model {task_state.upscale_model} not found. Fallback will be used.")
        
        if uov_method == 'upscale':
            skip_prompt_processing = True
            task_state.steps = 0
            # Note: bypass_alignment is implicit since skip_prompt_processing avoids SDXL specific steps
        else: # Super-Upscale
             # Use the current steps from state (user can still tweak them in Settings)
             # But for UoV it usually defaults to something reasonable.
             pass
    
    return skip_prompt_processing


def apply_image_input(task_state: 'TaskState', base_model_additional_loras, progressbar_callback=None):
    """
    Orchestrates the image input stage, handling UoV, Inpaint/Outpaint, and Image Prompt goals.
    """
    inpaint_image = None
    inpaint_mask = None
    outpaint_image = None
    outpaint_mask = None
    inpaint_patch_model_path = None
    controlnet_paths = {}
    structural_preprocessor_paths = {}
    contextual_assets = {
        'clip_vision_path': None,
        'ip_negative_path': None,
        'contextual_model_paths': {},
        'insightface_model_names': [],
        'eva_clip_path': None,
    }
    skip_prompt_processing = False

    # UoV handling
    if task_state.current_tab == 'uov' \
            and task_state.uov_method != flags.disabled.casefold() and task_state.uov_input_image is not None:
        skip_prompt_processing = prepare_upscale(task_state, progressbar_callback)

    mixed_cn_inpaint_workflow = task_state.current_tab == 'ip' and task_state.mixing_image_prompt_and_inpaint
    mixed_cn_outpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_outpaint', False)
    has_mixed_outpaint_request = mixed_cn_outpaint_workflow and task_state.outpaint_input_image is not None and (
        getattr(task_state, 'outpaint_step2_checkbox', False)
        or bool(getattr(task_state, 'outpaint_selections', []))
        or getattr(task_state, 'outpaint_mask_image', None) is not None
    )
    has_mixed_inpaint_request = mixed_cn_inpaint_workflow and task_state.inpaint_input_image is not None

    # Outpaint UI Parsing & setup
    if (task_state.current_tab == 'outpaint' or has_mixed_outpaint_request) and task_state.outpaint_input_image is not None:
        if isinstance(task_state.outpaint_input_image, dict):
            if 'background' in task_state.outpaint_input_image:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['background']))
                outpaint_mask = mask_proc.extract_mask_from_layers(task_state.outpaint_input_image)
            else:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['image']))
                raw_mask = task_state.outpaint_input_image.get('mask')
                outpaint_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(raw_mask))
        else:
            outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image))
            outpaint_mask = None

        if outpaint_mask is None:
            outpaint_mask = np.zeros(outpaint_image.shape[:2], dtype=np.uint8)

        merged_upload = mask_proc.combine_image_and_mask(task_state.outpaint_mask_image)
        if merged_upload is not None:
            H, W, C = outpaint_image.shape
            upload_mask = mask_proc.to_binary_mask(resample_image(merged_upload, width=W, height=H))
            outpaint_mask = mask_proc.combine_masks(outpaint_mask, upload_mask)

        if len(task_state.outpaint_selections) > 0:
            task_state.outpaint_direction = task_state.outpaint_selections[0].lower()

        task_state.inpaint_pixelate_primer = False
        task_state.goals.append('outpaint')

    # Inpaint UI Parsing & setup
    elif (task_state.current_tab == 'inpaint' or (has_mixed_inpaint_request and not has_mixed_outpaint_request)) \
            and task_state.inpaint_input_image is not None:
        if isinstance(task_state.inpaint_input_image, dict):
            if 'background' in task_state.inpaint_input_image:
                inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image['background'])
            else:
                inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image['image'])
        else:
            inpaint_image = mask_proc.ensure_numpy(task_state.inpaint_input_image)

        inpaint_mask = np.zeros(inpaint_image.shape[:2], dtype=np.uint8)
        context_mask = np.zeros(inpaint_image.shape[:2], dtype=np.uint8)

        task_state.context_mask = context_mask

        if not getattr(task_state, 'inpaint_step2_checkbox', False):
            merged_upload = mask_proc.combine_image_and_mask(task_state.inpaint_mask_image)
            if merged_upload is not None:
                H, W, C = inpaint_image.shape
                merged_upload = resample_image(merged_upload, width=W, height=H)
                upload_mask = mask_proc.to_binary_mask(merged_upload)
                task_state.context_mask = mask_proc.combine_masks(task_state.context_mask, upload_mask)

        if int(task_state.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(inpaint_mask, task_state.inpaint_erode_or_dilate)

        inpaint_image = HWC3(inpaint_image)
        task_state.goals.append('inpaint')

    if ('inpaint' in task_state.goals or 'outpaint' in task_state.goals) and not skip_prompt_processing:
        working_image = outpaint_image if 'outpaint' in task_state.goals else inpaint_image
        working_mask = outpaint_mask if 'outpaint' in task_state.goals else inpaint_mask

        if isinstance(working_image, np.ndarray) and isinstance(working_mask, np.ndarray):
            if progressbar_callback:
                progressbar_callback(task_state, 1, 'Initializing inpainter ...')

            engine = getattr(task_state, 'outpaint_engine', 'None') if 'outpaint' in task_state.goals \
                else getattr(task_state, 'inpaint_engine', 'None')

            if engine != 'None':
                if progressbar_callback:
                    progressbar_callback(task_state, 1, 'Downloading inpainter ...')
                inpaint_patch_model_path = config.downloading_inpaint_models(engine)
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            else:
                inpaint_patch_model_path = None

    # ControlNet handling
    if task_state.current_tab == 'ip' or task_state.mixing_image_prompt_and_inpaint or getattr(task_state, 'mixing_image_prompt_and_outpaint', False):
        task_state.goals.append('cn')
        if progressbar_callback:
            progressbar_callback(task_state, 1, 'Downloading control models ...')

        structural_tasks = task_state.get_cn_tasks_for_channel(flags.cn_structural)
        contextual_tasks = task_state.get_cn_tasks_for_channel(flags.cn_contextual)

        from modules import model_registry

        for cn_type in flags.cn_structural_types:
            if len(structural_tasks.get(cn_type, [])) == 0:
                continue

            controlnet_asset_id = structural_preprocessors.STRUCTURAL_CONTROLNET_ASSETS.get(cn_type)
            if controlnet_asset_id is not None:
                controlnet_paths[cn_type] = model_registry.ensure_asset(controlnet_asset_id)

            if not task_state.skipping_cn_preprocessor:
                preprocessor_asset_id = structural_preprocessors.STRUCTURAL_PREPROCESSOR_ASSETS.get(cn_type)
                if preprocessor_asset_id is not None:
                    structural_preprocessor_paths[cn_type] = model_registry.ensure_asset(preprocessor_asset_id)

        if any(len(contextual_tasks.get(cn_type, [])) > 0 for cn_type in flags.cn_contextual_types):
            if len(contextual_tasks.get(flags.cn_ip, [])) > 0 or len(contextual_tasks.get(flags.cn_faceid, [])) > 0:
                contextual_assets['clip_vision_path'] = model_registry.ensure_asset('contextual.shared.clip_vision')

            if len(contextual_tasks.get(flags.cn_ip, [])) > 0:
                contextual_assets['ip_negative_path'] = model_registry.ensure_asset('contextual.shared.ip_negative')
                contextual_assets['contextual_model_paths'][flags.cn_ip] = model_registry.ensure_asset('contextual.image_prompt.adapter')

            if len(contextual_tasks.get(flags.cn_faceid, [])) > 0:
                contextual_assets['contextual_model_paths'][flags.cn_faceid] = model_registry.ensure_asset('contextual.faceid.adapter')
                faceid_lora_path = model_registry.ensure_asset('contextual.faceid.lora')
                if (faceid_lora_path, 1.0) not in base_model_additional_loras:
                    base_model_additional_loras += [(faceid_lora_path, 1.0)]
                model_registry.ensure_asset('contextual.insightface.antelopev2')
                model_registry.ensure_asset('contextual.insightface.buffalo_l')
                contextual_assets['insightface_model_names'] = ['antelopev2', 'buffalo_l']

            if len(contextual_tasks.get(flags.cn_pulid, [])) > 0:
                contextual_assets['contextual_model_paths'][flags.cn_pulid] = model_registry.ensure_asset('contextual.pulid.model')
                model_registry.ensure_asset('contextual.insightface.antelopev2')
                contextual_assets['eva_clip_path'] = model_registry.ensure_asset('contextual.pulid.eva_clip')
                if 'antelopev2' not in contextual_assets['insightface_model_names']:
                    contextual_assets['insightface_model_names'].append('antelopev2')

    if task_state.current_tab == 'enhance' and task_state.enhance_input_image is not None:
        task_state.goals.append('enhance')
        skip_prompt_processing = True
        task_state.enhance_input_image = HWC3(mask_proc.ensure_numpy(task_state.enhance_input_image))

    return {
        'base_model_additional_loras': base_model_additional_loras,
        'clip_vision_path': contextual_assets.get('clip_vision_path'),
        'contextual_assets': contextual_assets,
        'controlnet_paths': controlnet_paths,
        'controlnet_canny_path': controlnet_paths.get(flags.cn_canny),
        'controlnet_cpds_path': controlnet_paths.get(flags.cn_cpds),
        'inpaint_image': inpaint_image,
        'inpaint_mask': inpaint_mask,
        'outpaint_image': outpaint_image,
        'outpaint_mask': outpaint_mask,
        'ip_adapter_path': contextual_assets.get('contextual_model_paths', {}).get(flags.cn_ip),
        'ip_negative_path': contextual_assets.get('ip_negative_path'),
        'skip_prompt_processing': skip_prompt_processing,
        'structural_preprocessor_paths': structural_preprocessor_paths
    }

def apply_control_nets(task_state, contextual_assets=None, structural_preprocessor_paths=None):
    """
    Applies Structural preprocessors and patches the UNet for contextual guidance.
    """
    width, height = task_state.width, task_state.height
    structural_tasks = task_state.get_cn_tasks_for_channel(flags.cn_structural)
    contextual_tasks = task_state.get_cn_tasks_for_channel(flags.cn_contextual)
    structural_preprocessor_paths = structural_preprocessor_paths or {}
    contextual_assets = contextual_assets or {}
    contextual_model_paths = contextual_assets.get('contextual_model_paths', {})
    clip_vision_path = contextual_assets.get('clip_vision_path')
    ip_negative_path = contextual_assets.get('ip_negative_path')
    insightface_model_names = contextual_assets.get('insightface_model_names') or ['antelopev2', 'buffalo_l']
    eva_clip_path = contextual_assets.get('eva_clip_path')

    def unpack_cn_image(raw_img, label):
        cn_img = mask_proc.unpack_gradio_data(raw_img)
        if cn_img is None:
            print(f'[ControlNet] Skipping {label} task with empty or invalid image input.')
            return None
        return HWC3(cn_img)

    def save_structural_preprocessor_output(cn_img, cn_type, slot_index):
        prefix = f"{cn_type.lower().replace(' ', '_')}_slot{slot_index}"
        saved_path = mask_proc.save_to_temp_png(cn_img)
        if saved_path is not None:
            print(f'[ControlNet] Saved {cn_type} preprocessor output to temp: {saved_path}')

    def preprocess_structural_tasks(cn_type, tasks, processor=None):
        valid_tasks = []
        for slot_index, task in enumerate(tasks, start=1):
            raw_img, cn_stop, cn_weight = task
            cn_img = unpack_cn_image(raw_img, cn_type)
            if cn_img is None:
                continue
            cn_img = resize_image(cn_img, width=width, height=height)
            if not task_state.skipping_cn_preprocessor and processor is not None:
                model_path = structural_preprocessor_paths.get(cn_type)
                try:
                    cn_img = processor(cn_type, cn_img, model_path)
                except Exception as exc:
                    print(f'[ControlNet] Failed to preprocess {cn_type} slot {slot_index}: {exc}')
                    continue
                save_structural_preprocessor_output(cn_img, cn_type, slot_index)
            cn_img = HWC3(cn_img)
            task[0] = core.numpy_to_pytorch(cn_img)
            valid_tasks.append(task)
        task_state.set_cn_tasks(cn_type, valid_tasks)

    def preprocess_contextual_tasks(cn_type, tasks, resize_to=None):
        valid_tasks = []
        model_path = contextual_model_paths.get(cn_type)
        if len(tasks) > 0 and model_path is None:
            print(f'[ControlNet] {cn_type} is missing its contextual model path. Skipping these tasks for now.')
            task_state.set_cn_tasks(cn_type, [])
            return

        for slot_index, task in enumerate(tasks, start=1):
            raw_img, cn_stop, cn_weight = task
            cn_img = unpack_cn_image(raw_img, cn_type)
            if cn_img is None:
                continue
            if resize_to is not None:
                cn_img = resize_image(cn_img, width=resize_to, height=resize_to, resize_mode=0)
            try:
                if cn_type == flags.cn_pulid:
                    task[0] = pulid_runtime.preprocess(
                        cn_img,
                        model_path=model_path,
                        eva_clip_path=eva_clip_path,
                        insightface_model_names=insightface_model_names,
                    )
                else:
                    task[0] = contextual_ip_adapter.preprocess(
                        cn_img,
                        model_path=model_path,
                        clip_vision_path=clip_vision_path,
                        ip_negative_path=ip_negative_path,
                        insightface_model_names=insightface_model_names,
                    )
            except Exception as exc:
                print(f'[ControlNet] Failed to preprocess {cn_type} slot {slot_index}: {exc}')
                continue
            valid_tasks.append(task)
        task_state.set_cn_tasks(cn_type, valid_tasks)

    with resources.memory_phase_scope(
        resources.MemoryPhase.STRUCTURAL_PREPROCESS,
        task=task_state,
        notes={'task_count': sum(len(tasks) for tasks in structural_tasks.values())},
        end_notes={'completed': True},
    ):
        preprocess_structural_tasks(
            flags.cn_canny,
            structural_tasks.get(flags.cn_canny, []),
            lambda _cn_type, cn_img, _model_path: preprocessors.canny_pyramid(cn_img, task_state.canny_low_threshold, task_state.canny_high_threshold)
        )
        preprocess_structural_tasks(
            flags.cn_cpds,
            structural_tasks.get(flags.cn_cpds, []),
            lambda _cn_type, cn_img, _model_path: preprocessors.cpds(cn_img)
        )
        preprocess_structural_tasks(
            flags.cn_depth,
            structural_tasks.get(flags.cn_depth, []),
            structural_preprocessors.run_structural_preprocessor
        )
        preprocess_structural_tasks(
            flags.cn_mistoline,
            structural_tasks.get(flags.cn_mistoline, []),
            structural_preprocessors.run_structural_preprocessor
        )
        preprocess_structural_tasks(
            flags.cn_mlsd,
            structural_tasks.get(flags.cn_mlsd, []),
            structural_preprocessors.run_structural_preprocessor
        )
        structural_preprocessors.offload_cached_preprocessors()

    with resources.memory_phase_scope(
        resources.MemoryPhase.CONTEXTUAL_PREPROCESS,
        task=task_state,
        notes={'task_count': sum(len(tasks) for tasks in contextual_tasks.values())},
        end_notes={'completed': True},
    ):
        preprocess_contextual_tasks(flags.cn_ip, contextual_tasks.get(flags.cn_ip, []), resize_to=224)
        preprocess_contextual_tasks(flags.cn_faceid, contextual_tasks.get(flags.cn_faceid, []))
        preprocess_contextual_tasks(flags.cn_pulid, contextual_tasks.get(flags.cn_pulid, []))

    all_contextual_tasks = []
    for cn_type in [flags.cn_ip, flags.cn_faceid]:
        all_contextual_tasks.extend(list(task_state.cn_tasks[cn_type]))

    pulid_tasks = list(task_state.cn_tasks[flags.cn_pulid])

    with resources.memory_phase_scope(
        resources.MemoryPhase.CONTROL_APPLY,
        task=task_state,
        notes={
            'contextual_patch_tasks': len(all_contextual_tasks),
            'pulid_patch_tasks': len(pulid_tasks),
        },
        end_notes={'completed': True},
    ):
        if len(all_contextual_tasks) > 0:
            pipeline.final_unet = contextual_ip_adapter.patch_model(pipeline.final_unet, all_contextual_tasks)

        if len(pulid_tasks) > 0:
            pipeline.final_unet = pulid_runtime.patch_model(pipeline.final_unet, pulid_tasks)




