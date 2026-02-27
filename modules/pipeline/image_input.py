import numpy as np
import modules.config as config
import modules.core as core
import modules.default_pipeline as pipeline
import modules.flags as flags
import modules.inpaint_worker as inpaint_worker
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
import extras.face_crop as face_crop
from modules.util import (HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil, 
                          get_shape_ceil, resample_image, erode_or_dilate)
from modules.upscaler import perform_upscale


class EarlyReturnException(BaseException):
    pass


def apply_vary(task_state, progressbar_callback=None):
    """
    Sets up variation parameters and encodes the initial latent.
    """
    uov_method = task_state.uov_method
    denoising_strength = task_state.denoising_strength
    uov_input_image = task_state.uov_input_image

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


def apply_outpaint(task_state, inpaint_image, inpaint_mask):
    """
    Pads the image and mask for outpainting.
    """
    if len(task_state.outpaint_selections) > 0:
        H, W, C = inpaint_image.shape
        outpaint_selections = [o.lower() for o in task_state.outpaint_selections]
        
        if 'top' in outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant', constant_values=255)
        if 'bottom' in outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant', constant_values=255)

        H, W, C = inpaint_image.shape
        if 'left' in outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant', constant_values=255)
        if 'right' in outpaint_selections:
            inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant', constant_values=255)

        inpaint_image = np.ascontiguousarray(inpaint_image.copy())
        inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
        task_state.inpaint_strength = 1.0
        task_state.inpaint_respective_field = 1.0
        
    return inpaint_image, inpaint_mask


def apply_inpaint(task_state, inpaint_head_model_path, inpaint_image, inpaint_mask, 
                  progressbar_callback=None, yield_result_callback=None):
    """
    Sets up the inpainting worker, patches the UNet, and encodes the initial latent.
    """
    inpaint_parameterized = task_state.inpaint_engine != 'None'
    denoising_strength = task_state.inpaint_strength
    inpaint_respective_field = task_state.inpaint_respective_field
    inpaint_disable_initial_latent = task_state.inpaint_disable_initial_latent

    # Outpaint is handled before calling this in the original code, but we keep it here for modularity if needed
    # or just assume it was called.
    
    inpaint_worker.current_task = inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=denoising_strength > 0.99,
        k=inpaint_respective_field
    )
    
    if task_state.debugging_inpaint_preprocessor:
        if yield_result_callback:
            yield_result_callback(task_state, inpaint_worker.current_task.visualize_mask_processing(), 100, do_not_show_finished_images=True)
        raise EarlyReturnException

    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'VAE Inpaint encoding ...')
    
    inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
    inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
    inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)
    
    candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
        steps=task_state.steps,
        denoise=denoising_strength
    )
    latent_inpaint, latent_mask = core.encode_vae_inpaint(
        mask=inpaint_pixel_mask,
        vae=candidate_vae,
        pixels=inpaint_pixel_image)
    
    latent_swap = None
    if candidate_vae_swap is not None:
        if progressbar_callback:
            task_state.current_progress += 1
            progressbar_callback(task_state, task_state.current_progress, 'VAE SD15 encoding ...')
        latent_swap = core.encode_vae(vae=candidate_vae_swap, pixels=inpaint_pixel_fill)['samples']
    
    if progressbar_callback:
        task_state.current_progress += 1
        progressbar_callback(task_state, task_state.current_progress, 'VAE encoding ...')
    
    latent_fill = core.encode_vae(vae=candidate_vae, pixels=inpaint_pixel_fill)['samples']
    inpaint_worker.current_task.load_latent(latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)
    
    if inpaint_parameterized:
        pipeline.final_unet = inpaint_worker.current_task.patch(
            inpaint_head_model_path=inpaint_head_model_path,
            inpaint_latent=latent_inpaint,
            inpaint_latent_mask=latent_mask,
            model=pipeline.final_unet
        )
    
    if not inpaint_disable_initial_latent:
        task_state.initial_latent = {'samples': latent_fill}
    
    B, C, H, W = latent_fill.shape
    task_state.height, task_state.width = H * 8, W * 8
    final_height, final_width = inpaint_worker.current_task.image.shape[:2]
    print(f'Final resolution is {str((final_width, final_height))}, latent is {str((task_state.width, task_state.height))}.')


def apply_upscale(task_state, progressbar_callback=None):
    """
    Performs image upscaling and sets up the latent for the diffusion pass.
    """
    uov_input_image = task_state.uov_input_image
    uov_method = task_state.uov_method
    
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
    task_state.uov_input_image = HWC3(task_state.uov_input_image)
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


def apply_image_input(task_state, base_model_additional_loras, progressbar_callback=None):
    """
    Orchestrates the image input stage, handling UoV, Inpaint/Outpaint, and Image Prompt goals.
    """
    inpaint_image = None
    inpaint_mask = None
    inpaint_head_model_path = None
    controlnet_canny_path = None
    controlnet_cpds_path = None
    clip_vision_path = None
    ip_negative_path = None
    ip_adapter_path = None
    ip_adapter_face_path = None
    skip_prompt_processing = False

    # UoV handling
    if (task_state.current_tab == 'uov' or (
            task_state.current_tab == 'ip' and task_state.mixing_image_prompt_and_vary_upscale)) \
            and task_state.uov_method != flags.disabled.casefold() and task_state.uov_input_image is not None:
        skip_prompt_processing = prepare_upscale(task_state, progressbar_callback)

    # Inpaint/Outpaint handling
    if (task_state.current_tab == 'inpaint' or (
            task_state.current_tab == 'ip' and task_state.mixing_image_prompt_and_inpaint)) \
            and isinstance(task_state.inpaint_input_image, dict):
        inpaint_image = task_state.inpaint_input_image['image']
        inpaint_mask = task_state.inpaint_input_image['mask'][:, :, 0]

        if task_state.inpaint_advanced_masking_checkbox:
            mask_upload = task_state.inpaint_mask_image_upload
            if isinstance(mask_upload, dict):
                if (isinstance(mask_upload['image'], np.ndarray)
                        and isinstance(mask_upload['mask'], np.ndarray)
                        and mask_upload['image'].ndim == 3):
                    mask_upload = np.maximum(mask_upload['image'], mask_upload['mask'])
            
            if isinstance(mask_upload, np.ndarray) and mask_upload.ndim == 3:
                H, W, C = inpaint_image.shape
                mask_upload = resample_image(mask_upload, width=W, height=H)
                mask_upload = np.mean(mask_upload, axis=2)
                mask_upload = (mask_upload > 127).astype(np.uint8) * 255
                inpaint_mask = np.maximum(inpaint_mask, mask_upload)

        if int(task_state.inpaint_erode_or_dilate) != 0:
            inpaint_mask = erode_or_dilate(inpaint_mask, task_state.inpaint_erode_or_dilate)

        if task_state.invert_mask_checkbox:
            inpaint_mask = 255 - inpaint_mask

        inpaint_image = HWC3(inpaint_image)
        if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                and (np.any(inpaint_mask > 127) or len(task_state.outpaint_selections) > 0):
            if progressbar_callback:
                progressbar_callback(task_state, 1, 'Downloading upscale models ...')
            config.downloading_upscale_model()
            
            if task_state.inpaint_engine != 'None':
                if progressbar_callback:
                    progressbar_callback(task_state, 1, 'Downloading inpainter ...')
                inpaint_head_model_path, inpaint_patch_model_path = config.downloading_inpaint_models(task_state.inpaint_engine)
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            else:
                inpaint_head_model_path = None
            
            if task_state.inpaint_additional_prompt != '':
                if task_state.prompt == '':
                    task_state.prompt = task_state.inpaint_additional_prompt
                else:
                    task_state.prompt = task_state.inpaint_additional_prompt + '\n' + task_state.prompt
            task_state.goals.append('inpaint')

    # Image Prompt (ControlNet/IP-Adapter) handling
    if task_state.current_tab == 'ip' or \
            task_state.mixing_image_prompt_and_vary_upscale or \
            task_state.mixing_image_prompt_and_inpaint:
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
        task_state.enhance_input_image = HWC3(task_state.enhance_input_image)

    return {
        'base_model_additional_loras': base_model_additional_loras,
        'clip_vision_path': clip_vision_path,
        'controlnet_canny_path': controlnet_canny_path,
        'controlnet_cpds_path': controlnet_cpds_path,
        'inpaint_head_model_path': inpaint_head_model_path,
        'inpaint_image': inpaint_image,
        'inpaint_mask': inpaint_mask,
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
        cn_img = resize_image(HWC3(cn_img), width=width, height=height)
        if not task_state.skipping_cn_preprocessor:
            cn_img = preprocessors.canny_pyramid(cn_img, task_state.canny_low_threshold, task_state.canny_high_threshold)
        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_cpds]:
        cn_img, cn_stop, cn_weight = task
        cn_img = resize_image(HWC3(cn_img), width=width, height=height)
        if not task_state.skipping_cn_preprocessor:
            cn_img = preprocessors.cpds(cn_img)
        cn_img = HWC3(cn_img)
        task[0] = core.numpy_to_pytorch(cn_img)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_ip]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(cn_img)
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    for task in task_state.cn_tasks[flags.cn_ip_face]:
        cn_img, cn_stop, cn_weight = task
        cn_img = HWC3(cn_img)
        if not task_state.skipping_cn_preprocessor:
            cn_img = face_crop.crop_image(cn_img)
        cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
        task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
        if task_state.debugging_cn_preprocessor and yield_result_callback:
            yield_result_callback(task_state, cn_img, task_state.current_progress, do_not_show_finished_images=True)
            
    all_ip_tasks = task_state.cn_tasks[flags.cn_ip] + task_state.cn_tasks[flags.cn_ip_face]
    if len(all_ip_tasks) > 0:
        pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)
