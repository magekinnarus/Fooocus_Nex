import gradio as gr

def build_control_tab():
    """
    Builds the Control tab: Canny thresholds, preprocessor toggles, mixing options.
    
    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    results['debugging_cn_preprocessor'] = gr.Checkbox(
        label='Debug Preprocessors', value=False,
        info='See the results from preprocessors.'
    )
    results['skipping_cn_preprocessor'] = gr.Checkbox(
        label='Skip Preprocessors', value=False,
        info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)'
    )

    results['mixing_image_prompt_and_inpaint'] = gr.Checkbox(
        label='Add Controlnet to Inpainting/outpainting',
        value=False
    )

    results['controlnet_softness'] = gr.Slider(
        label='Softness of ControlNet', minimum=0.0, maximum=1.0,
        step=0.001, value=0.25,
        info='Similar to the Control Mode in A1111 (use 0.0 to disable). '
    )

    with gr.Tab(label='Canny'):
        results['canny_low_threshold'] = gr.Slider(
            label='Canny Low Threshold', minimum=1, maximum=255,
            step=1, value=64
        )
        results['canny_high_threshold'] = gr.Slider(
            label='Canny High Threshold', minimum=1, maximum=255,
            step=1, value=128
        )

    return results
