import gradio as gr
import modules.config
import modules.flags as flags

def build_models_tab():
    """
    Builds the Models tab: base model, VAE, CLIP, LoRA rows, and refresh button.
    
    Returns:
        dict: Gradio components mapping name to instance, 
              includes 'lora_ctrls' list.
    """
    results = {}

    with gr.Group():
        with gr.Row():
            results['base_model'] = gr.Dropdown(
                label='Base Model', 
                choices=modules.config.model_filenames, 
                value=modules.config.default_base_model_name, 
                show_label=True
            )
            results['vae_model'] = gr.Dropdown(
                label='VAE', 
                choices=[flags.default_vae] + modules.config.vae_filenames, 
                value=modules.config.default_vae, 
                show_label=True
            )

        results['clip_model'] = gr.Dropdown(
            label='Force CLIP', 
            choices=['None'] + modules.config.clip_filenames, 
            value=modules.config.default_clip, 
            show_label=True
        )

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
                    elem_classes='lora_model', 
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
            label='Refresh', 
            value='\U0001f504 Refresh All Files', 
            variant='secondary', 
            elem_classes='refresh_button'
        )

    return results
