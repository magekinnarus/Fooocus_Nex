import gradio as gr
import modules.config
import modules.flags as flags

def build_inpaint_tab():
    """
    Builds the Inpaint tab (Advanced section): engine, strength, field, etc.
    
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
    # (Removed advanced masking toggle binding)


    return results
