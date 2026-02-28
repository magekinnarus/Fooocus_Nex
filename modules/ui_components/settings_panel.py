import gradio as gr
import args_manager
import modules.config
import modules.flags as flags

def build_settings_tab():
    """
    Builds the Settings tab contents: preset, performance, aspect ratios,
    image number, output format, negative prompt, seed, and history link.
    
    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    if not args_manager.args.disable_preset_selection:
        results['preset_selection'] = gr.Dropdown(
            label='Preset',
            choices=modules.config.available_presets,
            value=args_manager.args.preset if args_manager.args.preset else "initial",
            interactive=True
        )

    results['performance_selection'] = gr.Radio(
        label='Performance',
        choices=flags.Performance.values(),
        value=modules.config.default_performance,
        elem_classes=['performance_selection']
    )

    with gr.Accordion(label='Aspect Ratios', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
        results['aspect_ratios_accordion'] = aspect_ratios_accordion
        results['aspect_ratios_selection'] = gr.Radio(
            label='Aspect Ratios',
            show_label=False,
            choices=modules.config.available_aspect_ratios_labels,
            value=modules.config.default_aspect_ratio,
            info='width × height',
            elem_classes='aspect_ratios'
        )

    results['image_number'] = gr.Slider(
        label='Image Number',
        minimum=1,
        maximum=modules.config.default_max_image_number,
        step=1,
        value=modules.config.default_image_number
    )

    results['output_format'] = gr.Radio(
        label='Output Format',
        choices=flags.OutputFormat.list(),
        value=modules.config.default_output_format
    )

    results['negative_prompt'] = gr.Textbox(
        label='Negative Prompt',
        show_label=True,
        placeholder="Type prompt here.",
        info='Describing what you do not want to see.',
        lines=2,
        elem_id='negative_prompt',
        value=modules.config.default_prompt_negative
    )

    results['seed_random'] = gr.Checkbox(label='Random', value=True)
    results['image_seed'] = gr.Textbox(label='Seed', value=0, max_lines=1, visible=False)
    results['history_link'] = gr.HTML()

    return results
