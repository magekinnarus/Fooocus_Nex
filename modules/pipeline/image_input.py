import copy
import numpy as np
import modules.config as config
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
import extras.face_crop as face_crop
from modules.util import (HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil, 
                          get_shape_ceil, resample_image, erode_or_dilate)
from modules.upscaler import perform_upscale
import modules.mask_processing as mask_proc


class EarlyReturnException(BaseException):
    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload


LAST_INPAINT_STEP1_CONTEXT = None


def apply_vary(task_state, progressbar_callback=None):
    """
    Sets up variation parameters and encodes the initial latent.
    """
    uov_method = task_state.uov_method
    denoising_strength = task_state.denoising_strength
    uov_input_image = task_state.uov_input_image
    uov_input_image = mask_proc.ensure_numpy(uov_input_image)

    if 'subtle' in uov_method:
        denoising_strength = 0.5
    if 'strong' in uov_method:
        denoising_strength = 0.85
    if task_state.overwrite_vary_strength > 0:
        denoising_strength = task_state.overwrite_vary_strength
    
    shape_ceil = get_image_shape_ceil(uov_input_image)
    if shape_ceil < 1024:
        print(f'[Vary] Image is resized because it is too small.')
        shape_ceil = 1024
    elif shape_ceil > 2048:
        print(f'[Vary] Image is resized because it is too big.')
        shape_ceil = 2048
    
    uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)
    initial_pixels = core.numpy_to_pytorch(uov_input_image)
    
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'VAE encoding ...')
    
    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=task_state.steps,
        denoise=denoising_strength
    )
    initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
    B, C, H, W = initial_latent['samples'].shape
    
    task_state.uov_input_image = uov_input_image
    task_state.denoising_strength = denoising_strength
    task_state.initial_latent = initial_latent
    task_state.width = W * 8
    task_state.height = H * 8
    print(f'Final resolution is {str((task_state.width, task_state.height))}.')


def apply_outpaint_expansion(task_state, inpaint_image, inpaint_mask):
    """
    Phase 1 Outpaint: Pads the image and returns BOTH the expanded canvas and 
    the auto-calculated BB image for Step 2 external editing.
    """
    if task_state.current_tab == 'outpaint' and task_state.outpaint_direction is not None:
        direction = task_state.outpaint_direction
        if isinstance(direction, list) and len(direction) > 0:
            direction = direction[0].lower()
        
        if getattr(task_state, 'inpaint_pixelate_primer', False):
            return inpaint_image, inpaint_mask
            
        from modules.pipeline.outpaint import OutpaintPipeline
        outpaint = OutpaintPipeline()
        
        # 1. Expand canvas
        expanded_image, generated_mask = outpaint.prepare_outpaint_canvas_only(
            inpaint_image, direction, expansion_size=task_state.inpaint_outpaint_expansion_size, pixelate=False
        )
        
        # 2. Combine with existing mask
        expanded_mask = np.maximum(
            resample_image(inpaint_mask, width=expanded_image.shape[1], height=expanded_image.shape[0]), 
            generated_mask
        )

        # 3. Pre-calculate the BB image that Step 2 inference will use
        ctx = outpaint.prepare(
            image=expanded_image,
            mask=expanded_mask,
            outpaint_direction=direction,
            extend_factor=1.2
        )

        # 4. Save to temp PNGs for Filepath Invariant
        from modules.mask_processing import save_to_temp_png
        canvas_path = save_to_temp_png(expanded_image)
        bb_path = save_to_temp_png(ctx.bb_image)
        
        # Payload is a list of [Canvas, BB] for the Gradio Gallery
        raise EarlyReturnException(payload=[canvas_path, bb_path])
        
    return inpaint_image, inpaint_mask


def apply_outpaint_inference_setup(task_state, inpaint_image, inpaint_mask, 
                                  progressbar_callback=None, yield_result_callback=None):
    """
    Sets up the outpainting worker, patches the UNet, and encodes the initial latent.
    Exclusively using OutpaintPipeline.
    """
    inpaint_disable_initial_latent = getattr(task_state, 'inpaint_disable_initial_latent', False)

    from modules.pipeline.outpaint import OutpaintPipeline
    outpaint = OutpaintPipeline()

    # Apply Pixelation Primer if requested (Critical for Step 2 color guidance)
    if getattr(task_state, 'inpaint_pixelate_primer', False):
        inpaint_image = outpaint.pixelate_mask_area(inpaint_image, inpaint_mask)
    
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
    
    # --- Step 2 Refactor: Explicit BB and BB Mask Support ---
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
        print(f"[Debug] Outpaint Step 2: Using explicit BB image from slot: {ctx.target_w}x{ctx.target_h}")
        
    # Combine uploaded mask with brush-drawn mask
    manual_mask = mask_proc.unpack_gradio_data(raw_bb_mask)
    brush_mask = mask_proc.unpack_gradio_data(brush_mask_data)
    
    combined_bb_mask = mask_proc.combine_masks(manual_mask, brush_mask)
    if combined_bb_mask is not None:
        # Ensure mask matches BB image resolution
        combined_bb_mask = resample_image(combined_bb_mask, width=ctx.target_w, height=ctx.target_h)
        ctx.bb_mask = combined_bb_mask
        print(f"[Debug] Outpaint Step 2: Using explicit BB mask (manual or brush).")

    print(f"[Debug] outpaint.prepare() final BB size: {ctx.bb_image.shape}")
    
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
    Sets up the inpainting worker, patches the UNet, and encodes the initial latent.
    Exclusively using InpaintPipeline.
    """
    global LAST_INPAINT_STEP1_CONTEXT

    denoising_strength = getattr(task_state, 'inpaint_strength', 1.0)
    inpaint_disable_initial_latent = getattr(task_state, 'inpaint_disable_initial_latent', False)

    # 1. Identify explicit 4-slot inputs
    raw_input_image = getattr(task_state, 'inpaint_input_image', None)
    raw_context_mask = getattr(task_state, 'inpaint_context_mask_image', None)
    raw_bb_image = getattr(task_state, 'inpaint_bb_image', None)
    raw_bb_mask = getattr(task_state, 'inpaint_mask_image', None)
    
    # 2. Extract usable arrays
    input_image = mask_proc.unpack_gradio_data(raw_input_image) if raw_input_image is not None else inpaint_image
    context_mask = mask_proc.unpack_gradio_data(raw_context_mask) if raw_context_mask is not None else inpaint_mask
    
    from modules.pipeline.inpaint import InpaintPipeline
    inpaint = InpaintPipeline()
    
    # 3. Always derive coordinate system from Full Image + Context Mask if possible
    # This prevents "BB Inception" (cropping within a crop).
    if input_image is not None and context_mask is not None:
        ctx = inpaint.prepare(
            image=input_image,
            mask=context_mask,
            extend_factor=1.2
        )
        print(f"[Debug] Context derived from {input_image.shape[1]}x{input_image.shape[0]} image via context mask.")
    else:
        # Fallback to standard preparation if explicit slots are missing
        ctx = inpaint.prepare(
            image=inpaint_image,
            mask=inpaint_mask,
            extend_factor=1.2
        )
        print(f"[Debug] Context derived from standard inpaint inputs.")

    is_step1_inpaint = task_state.current_tab == 'inpaint' and not getattr(task_state, 'inpaint_step2_checkbox', False)
    is_step2_inpaint = task_state.current_tab == 'inpaint' and getattr(task_state, 'inpaint_step2_checkbox', False)

    if is_step1_inpaint:
        LAST_INPAINT_STEP1_CONTEXT = copy.deepcopy(ctx)
        # Use a list to ensure Gradio Gallery receives a proper list of images
        raise EarlyReturnException(payload=[ctx.bb_image])

    if is_step2_inpaint:
        # 4. Integrate explicit Step 2 slots (BB image and BB mask)
        # We use the coordinates (y1, y2, x1, x2) from the Full Context (ctx.bb)
        # to ensure stitching is perfectly aligned.
        
        bb_img_data = mask_proc.unpack_gradio_data(raw_bb_image)
        # Handle mask from both the explicit slot and potential drawing on the BB image
        bb_mask_2d = mask_proc.combine_masks(
            mask_proc.unpack_gradio_data(raw_bb_mask),
            mask_proc.extract_mask_from_layers(raw_bb_image) if isinstance(raw_bb_image, dict) else None
        )

        if bb_img_data is not None:
            ctx.bb_image = bb_img_data
            ctx.target_h, ctx.target_w = bb_img_data.shape[:2]
            print(f"[Debug] Using explicit BB image from slot: {ctx.target_w}x{ctx.target_h}")
        
        if bb_mask_2d is not None:
            # Ensure mask matches BB image resolution
            bb_mask_2d = resample_image(bb_mask_2d, width=ctx.target_w, height=ctx.target_h)
            ctx.bb_mask = bb_mask_2d
            print(f"[Debug] Using explicit BB mask.")
            
        # 5. Re-run morphological blend calculation for stitching
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
    uov_method = task_state.uov_method
    uov_input_image = mask_proc.ensure_numpy(uov_input_image)
    H, W, C = uov_input_image.shape
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, f'Upscaling image from {str((W, H))} ...')
    
    uov_input_image = perform_upscale(uov_input_image)
    print(f'Image upscaled.')
    
    if '1.5x' in uov_method:
        f = 1.5
    elif '2x' in uov_method:
        f = 2.0
    else:
        f = 1.0
    
    shape_ceil = get_shape_ceil(H * f, W * f)
    if shape_ceil < 1024:
        print(f'[Upscale] Image is resized because it is too small.')
        uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
        shape_ceil = 1024
    else:
        uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)
    
    image_is_super_large = shape_ceil > 2800
    direct_return = False
    if 'fast' in uov_method:
        direct_return = True
    elif image_is_super_large:
        print('Image is too large. Directly returned the SR image.')
        direct_return = True
    
    if direct_return:
        task_state.uov_input_image = uov_input_image
        return direct_return

    task_state.tiled = True
    denoising_strength = 0.382
    if task_state.overwrite_upscale_strength > 0:
        denoising_strength = task_state.overwrite_upscale_strength
    
    initial_pixels = core.numpy_to_pytorch(uov_input_image)
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'VAE encoding ...')
    
    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=task_state.steps,
        denoise=denoising_strength
    )
    initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
    B, C, H, W = initial_latent['samples'].shape
    
    task_state.uov_input_image = uov_input_image
    task_state.denoising_strength = denoising_strength
    task_state.initial_latent = initial_latent
    task_state.width = W * 8
    task_state.height = H * 8
    print(f'Final resolution is {str((task_state.width, task_state.height))}.')
    return direct_return


def prepare_upscale(task_state, progressbar_callback=None):
    """
    Determines if vary or upscale is needed and sets the appropriate goals.
    """
    task_state.uov_input_image = HWC3(mask_proc.ensure_numpy(task_state.uov_input_image))
    uov_method = task_state.uov_method
    
    skip_prompt_processing = False
    if 'vary' in uov_method:
        task_state.goals.append('vary')
    elif 'upscale' in uov_method:
        task_state.goals.append('upscale')
        if 'fast' in uov_method:
            skip_prompt_processing = True
            task_state.steps = 0
        else:
            task_state.steps = task_state.performance_selection.steps_uov()

        if progressbar_callback:
            task_state.current_progress += 1
            progressbar_callback(task_state, task_state.current_progress, 'Downloading upscale models ...')
        config.downloading_upscale_model()
    
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
    controlnet_canny_path = None
    controlnet_cpds_path = None
    clip_vision_path = None
    ip_negative_path = None
    ip_adapter_path = None
    ip_adapter_face_path = None
    skip_prompt_processing = False

    # UoV handling
    if task_state.current_tab == 'uov' \
            and task_state.uov_method != flags.disabled.casefold() and task_state.uov_input_image is not None:
        skip_prompt_processing = prepare_upscale(task_state, progressbar_callback)

    # Outpaint UI Parsing & setup
    if task_state.current_tab == 'outpaint' and task_state.outpaint_input_image is not None:
        if isinstance(task_state.outpaint_input_image, dict):
            if 'background' in task_state.outpaint_input_image:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['background']))
                outpaint_mask = mask_proc.extract_mask_from_layers(task_state.outpaint_input_image)
            else:
                outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image['image']))
                raw_mask = task_state.outpaint_input_image.get('mask')
                outpaint_mask = mask_proc.to_binary_mask(mask_proc.ensure_numpy(raw_mask))
        else:
            # Direct path or numpy
            outpaint_image = HWC3(mask_proc.ensure_numpy(task_state.outpaint_input_image))
            outpaint_mask = None
            
        if outpaint_mask is None:
            outpaint_mask = np.zeros(outpaint_image.shape[:2], dtype=np.uint8)
        
        merged_upload = mask_proc.combine_image_and_mask(task_state.outpaint_mask_image)
        if merged_upload is not None:
            H, W, C = outpaint_image.shape
            upload_mask = mask_proc.to_binary_mask(resample_image(merged_upload, width=W, height=H))
            outpaint_mask = mask_proc.combine_masks(outpaint_mask, upload_mask)

        # Parse direction even for Step 2
        if len(task_state.outpaint_selections) > 0:
            task_state.outpaint_direction = task_state.outpaint_selections[0].lower()

        task_state.inpaint_pixelate_primer = getattr(task_state, 'outpaint_step2_checkbox', False)
        
        # Step 1 Outpaint expansion
        if not task_state.inpaint_pixelate_primer:
            outpaint_image, outpaint_mask = apply_outpaint_expansion(task_state, outpaint_image, outpaint_mask)
            skip_prompt_processing = True

        task_state.goals.append('outpaint')

    # Inpaint UI Parsing & setup
    elif (task_state.current_tab == 'inpaint' or (
            task_state.current_tab == 'ip' and task_state.mixing_image_prompt_and_inpaint)) \
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
            # Normal inpaint: check mask
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
        # Determine which image/mask pair to use for model downloading logic
        working_image = outpaint_image if 'outpaint' in task_state.goals else inpaint_image
        working_mask = outpaint_mask if 'outpaint' in task_state.goals else inpaint_mask
        
        if isinstance(working_image, np.ndarray) and isinstance(working_mask, np.ndarray):
            if progressbar_callback:
                progressbar_callback(task_state, 1, 'Downloading upscale models ...')
            config.downloading_upscale_model()
            
            engine = getattr(task_state, 'inpaint_engine', 'None')
            if task_state.current_tab == 'outpaint':
                engine = getattr(task_state, 'outpaint_engine', 'None')
                
            if engine != 'None':
                if progressbar_callback:
                    progressbar_callback(task_state, 1, 'Downloading inpainter ...')
                inpaint_patch_model_path = config.downloading_inpaint_models(engine)
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            else:
                inpaint_patch_model_path = None

    # ControlNet (IP-Adapter) handling
    if task_state.current_tab == 'ip' or task_state.mixing_image_prompt_and_inpaint:
        task_state.goals.append('cn')
        if progressbar_callback:
            progressbar_callback(task_state, 1, 'Downloading control models ...')
        
        if len(task_state.cn_tasks[flags.cn_canny]) > 0:
            controlnet_canny_path = config.downloading_controlnet_canny()
        if len(task_state.cn_tasks[flags.cn_cpds]) > 0:
            controlnet_cpds_path = config.downloading_controlnet_cpds()
        if len(task_state.cn_tasks[flags.cn_ip]) > 0:
            clip_vision_path, ip_negative_path, ip_adapter_path = config.downloading_ip_adapters('ip')
        if len(task_state.cn_tasks[flags.cn_ip_face]) > 0:
            clip_vision_path, ip_negative_path, ip_adapter_face_path = config.downloading_ip_adapters('face')

    # Enhance handling
    if task_state.current_tab == 'enhance' and task_state.enhance_input_image is not None:
        task_state.goals.append('enhance')
        skip_prompt_processing = True
        task_state.enhance_input_image = HWC3(mask_proc.ensure_numpy(task_state.enhance_input_image))

    return {
        'base_model_additional_loras': base_model_additional_loras,
        'clip_vision_path': clip_vision_path,
        'controlnet_canny_path': controlnet_canny_path,
        'controlnet_cpds_path': controlnet_cpds_path,
        'inpaint_image': inpaint_image,
        'inpaint_mask': inpaint_mask,
        'outpaint_image': outpaint_image,
        'outpaint_mask': outpaint_mask,
        'ip_adapter_face_path': ip_adapter_face_path,
        'ip_adapter_path': ip_adapter_path,
        'ip_negative_path': ip_negative_path,
        'skip_prompt_processing': skip_prompt_processing
    }


def apply_control_nets(task_state, ip_adapter_face_path, ip_adapter_path, yield_result_callback=None):
    """
    Applies ControlNet preprocessors and patches the UNet for IP-Adapters.
    """
    width, height = task_state.width, task_state.height
    
    for task in task_state.cn_tasks[flags.cn_canny]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(HWC3(mask_proc.ensure_numpy(cn_img)), width=width, height=height)
        if not task_state.skipping_cn_preprocessor:
            cn_img = preprocessors.canny_pyramid(cn_img, task_state.canny_low_threshold, task_state.canny_high_threshold)
        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_cpds]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(HWC3(mask_proc.ensure_numpy(cn_img)), width=width, height=height)
        if not task_state.skipping_cn_preprocessor:
            cn_img = preprocessors.cpds(cn_img)
        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_ip]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(mask_proc.ensure_numpy(cn_img))
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_ip_face]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(mask_proc.ensure_numpy(cn_img))
        if not task_state.skipping_cn_preprocessor:
            cn_img = face_crop.crop_image(cn_img)
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    all_ip_tasks = task_state.cn_tasks[flags.cn_ip] + task_state.cn_tasks[flags.cn_ip_face]
    if len(all_ip_tasks) > 0:
        pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)


