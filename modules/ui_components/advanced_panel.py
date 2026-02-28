import gradio as gr
import modules.config
import modules.flags as flags
import args_manager

def build_advanced_tab():
    """
    Builds the Advanced tab: guidance, sharpness, ADM, sampler, scheduler, etc.
    
    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    with gr.Row():
        results['guidance_scale'] = gr.Slider(
            label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
            value=modules.config.default_cfg_scale,
            info='Higher value means following prompt more strictly.'
        )
        results['sharpness'] = gr.Slider(
            label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.01,
            value=modules.config.default_sample_sharpness,
            info='Higher value means sharper edges.'
        )

    with gr.Row():
        results['adm_scaler_positive'] = gr.Slider(
            label='ADM Guidance Positive', minimum=0.0, maximum=3.0, step=0.01,
            value=modules.config.default_adms[0],
            info='Positive ADM Guidance.'
        )
        results['adm_scaler_negative'] = gr.Slider(
            label='ADM Guidance Negative', minimum=0.0, maximum=3.0, step=0.01,
            value=modules.config.default_adms[1],
            info='Negative ADM Guidance.'
        )
        results['adm_scaler_end'] = gr.Slider(
            label='ADM Guidance End At Step', minimum=0.0, maximum=1.0, step=0.01,
            value=modules.config.default_adms[2],
            info='ADM Guidance End At Step.'
        )

    results['adaptive_cfg'] = gr.Slider(
        label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
        value=modules.config.default_cfg_tsnr,
        info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR (effective when real CFG > mimicked CFG).'
    )
    
    results['clip_skip'] = gr.Slider(
        label='CLIP Skip', minimum=1, maximum=flags.clip_skip_max, step=1,
        value=modules.config.default_clip_skip,
        info='Bypass CLIP layers to avoid overfitting (use 1 to not skip any layers, 2 is recommended).'
    )
    
    results['sampler_name'] = gr.Dropdown(
        label='Sampler', choices=flags.sampler_list,
        value=modules.config.default_sampler
    )
    
    results['scheduler_name'] = gr.Dropdown(
        label='Scheduler', choices=flags.scheduler_list,
        value=modules.config.default_scheduler
    )

    results['generate_image_grid'] = gr.Checkbox(
        label='Generate Image Grid for Each Batch',
        info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
        value=False
    )

    results['overwrite_step'] = gr.Slider(
        label='Forced Overwrite of Sampling Step',
        minimum=-1, maximum=200, step=1,
        value=modules.config.default_overwrite_step,
        info='Set as -1 to disable. For developer debugging.'
    )
    
    results['overwrite_width'] = gr.Slider(
        label='Forced Overwrite of Generating Width',
        minimum=-1, maximum=2048, step=1, value=-1,
        info='Set as -1 to disable. For developer debugging. Results will be worse for non-standard numbers that SDXL is not trained on.'
    )
    
    results['overwrite_height'] = gr.Slider(
        label='Forced Overwrite of Generating Height',
        minimum=-1, maximum=2048, step=1, value=-1,
        info='Set as -1 to disable. For developer debugging. Results will be worse for non-standard numbers that SDXL is not trained on.'
    )
    
    results['overwrite_vary_strength'] = gr.Slider(
        label='Forced Overwrite of Denoising Strength of "Vary"',
        minimum=-1, maximum=1.0, step=0.001, value=-1,
        info='Set as negative number to disable. For developer debugging.'
    )
    
    results['overwrite_upscale_strength'] = gr.Slider(
        label='Forced Overwrite of Denoising Strength of "Upscale"',
        minimum=-1, maximum=1.0, step=0.001,
        value=modules.config.default_overwrite_upscale,
        info='Set as negative number to disable. For developer debugging.'
    )

    results['disable_preview'] = gr.Checkbox(
        label='Disable Preview', value=False,
        info='Disable preview during generation.'
    )
    
    results['disable_intermediate_results'] = gr.Checkbox(
        label='Disable Intermediate Results',
        value=flags.Performance.has_restricted_features(modules.config.default_performance),
        info='Disable intermediate results during generation, only show final gallery.'
    )

    results['disable_seed_increment'] = gr.Checkbox(
        label='Disable seed increment',
        info='Disable automatic seed increment when image number is > 1.',
        value=False
    )
    
    results['read_wildcards_in_order'] = gr.Checkbox(
        label="Read wildcards in order", value=False
    )

    if not args_manager.args.disable_metadata:
        results['save_metadata_to_images'] = gr.Checkbox(
            label='Save Metadata to Images', 
            value=modules.config.default_save_metadata_to_images,
            info='Adds parameters to generated images allowing manual regeneration.'
        )
        results['metadata_scheme'] = gr.Radio(
            label='Metadata Scheme', 
            choices=flags.metadata_scheme, 
            value=modules.config.default_metadata_scheme,
            info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
            visible=modules.config.default_save_metadata_to_images
        )
        
        results['save_metadata_to_images'].change(
            lambda x: gr.update(visible=x), 
            inputs=[results['save_metadata_to_images']], 
            outputs=[results['metadata_scheme']],
            queue=False, 
            show_progress=False
        )

    return results
