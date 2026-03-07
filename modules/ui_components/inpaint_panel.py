import gradio as gr
import modules.config
import modules.flags as flags

def build_inpaint_tab(inpaint_advanced_masking_checkbox, invert_mask_checkbox, 
                      inpaint_mask_image, inpaint_mask_generation_col, inpaint_input_image):
    """
    Builds the Inpaint tab (Advanced section): engine, strength, field, etc.
    
    Args:
        inpaint_advanced_masking_checkbox: Gradio component from Image Input section.
        invert_mask_checkbox: Gradio component from Image Input section.
        inpaint_mask_image: Gradio component from Image Input section.
        inpaint_mask_generation_col: Gradio component from Image Input section.
        inpaint_input_image: Gradio component from Image Input section.

    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    results['debugging_inpaint_preprocessor'] = gr.Checkbox(label='Debug Inpaint Preprocessing', value=False)
    results['inpaint_disable_initial_latent'] = gr.Checkbox(label='Disable initial latent in inpaint', value=False)
    results['inpaint_engine'] = gr.Dropdown(
        label='Inpaint Engine',
        value=modules.config.default_inpaint_engine_version,
        choices=flags.inpaint_engine_versions,
        info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.'
    )
    results['inpaint_strength'] = gr.Slider(
        label='Inpaint Denoising Strength',
        minimum=0.0, maximum=1.0, step=0.001, value=0.5,
        info='Same as the denoising strength in A1111 inpaint.'
    )
    results['inpaint_erode_or_dilate'] = gr.Slider(
        label='Mask Erode or Dilate',
        minimum=-64, maximum=64, step=1, value=0,
        info='Positive value will make white area in the mask larger, negative value will make white area smaller. (default is 0, always processed before any mask invert)'
    )


    # Event bindings that depend on components from other tabs
    inpaint_advanced_masking_checkbox.change(
        lambda x: [gr.update(visible=not x)] * 2,
        inputs=inpaint_advanced_masking_checkbox,
        outputs=[inpaint_mask_image, inpaint_mask_generation_col],
        queue=False, 
        show_progress=False
    )


    return results
