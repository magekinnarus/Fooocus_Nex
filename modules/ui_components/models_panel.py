import gradio as gr
import modules.config
import modules.flags as flags
import modules.ui_components.styles_panel as styles_panel


def build_models_tab():
    """
    Builds the Models tab: base model, VAE, CLIP, LoRA rows, and refresh button.

    Returns:
        dict: Gradio components mapping name to instance,
              includes 'lora_ctrls' list.
    """
    results = {}

    base_model_choices = list(modules.config.model_filenames or [])
    if not base_model_choices:
        base_model_choices = ['None']
    base_model_value = modules.config.default_base_model_name if modules.config.default_base_model_name in base_model_choices else base_model_choices[0]

    with gr.Group():
        with gr.Row():
            results['base_model'] = gr.Dropdown(
                label='Base Model',
                choices=base_model_choices,
                value=base_model_value,
                show_label=True,
                elem_id='model_base_dropdown',
            )
            results['vae_model'] = gr.Dropdown(
                label='VAE',
                choices=[flags.default_vae] + modules.config.vae_filenames,
                value=modules.config.default_vae,
                show_label=True,
                elem_id='model_vae_dropdown',
            )

        results['clip_model'] = gr.Dropdown(
            label='Force CLIP',
            choices=['None'] + modules.config.clip_filenames,
            value=modules.config.default_clip,
            show_label=True,
            elem_id='model_clip_dropdown',
        )

    with gr.Accordion(label='Prompt Presets', open=False, elem_id='style_selections_accordion') as style_selections_accordion:
        results['style_selections_accordion'] = style_selections_accordion
        styles_result = styles_panel.build_styles_tab()
        results.update(styles_result)

    with gr.Group():
        lora_ctrls = []
        for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
            with gr.Row():
                lora_enabled = gr.Checkbox(
                    label='Enable', value=enabled,
                    elem_classes=['lora_enable', 'min_check'],
                    scale=1
                )
                lora_model = gr.Dropdown(
                    label=f'LoRA {i + 1}',
                    choices=['None'] + modules.config.lora_filenames,
                    value=filename,
                    allow_custom_value=True,
                    elem_classes='lora_model',
                    elem_id=f'lora_model_dropdown_{i + 1}',
                    scale=5
                )
                lora_weight = gr.Slider(
                    label='Weight',
                    minimum=modules.config.default_loras_min_weight,
                    maximum=modules.config.default_loras_max_weight,
                    step=0.01,
                    value=weight,
                    elem_classes='lora_weight',
                    scale=5
                )
                lora_ctrls += [lora_enabled, lora_model, lora_weight]
        results['lora_ctrls'] = lora_ctrls

    with gr.Row():
        results['refresh_files'] = gr.Button(
            value='\U0001f504 Refresh All Files',
            variant='secondary',
            elem_classes='refresh_button',
            elem_id='refresh_files_button',
        )


    return results


