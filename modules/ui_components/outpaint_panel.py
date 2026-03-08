import gradio as gr
import modules.config
import modules.flags as flags

def build_outpaint_tab():
    """
    Builds the Outpaint tab (Advanced section): engine, strength, field, etc.

    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    results['outpaint_engine'] = gr.Dropdown(
        label='Outpaint Engine',
        value=modules.config.default_outpaint_engine_version,
        choices=flags.inpaint_engine_versions,
        info='Version of Fooocus inpaint model. If set, use performance Quality or Speed (no performance LoRAs) for best results.'
    )
    results['outpaint_strength'] = gr.Slider(
        label='Outpaint Denoising Strength',
        minimum=0.0, maximum=1.0, step=0.001, value=0.75,
        info='Same as the denoising strength in A1111 inpaint.'
    )
    
    results['inpaint_outpaint_expansion_size'] = gr.Dropdown(
        label='Outpaint Expansion (Pixels)',
        choices=['384', '416', '448'],
        value=str(modules.config.default_outpaint_expansion_size),
        info='Number of pixels to add during outpainting. Default is 384.'
    )

    return results

