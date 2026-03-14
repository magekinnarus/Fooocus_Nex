import gradio as gr
import os

gr.set_static_paths(paths=["javascript", "css", f"sdxl_styles{os.sep}samples"])
import random
import os
import json
import time
import numpy as np
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import modules.ui_components.metadata_ui as metadata_ui
import modules.ui_components.settings_panel as settings_panel
import modules.ui_components.styles_panel as styles_panel
import modules.ui_components.models_panel as models_panel
import modules.ui_components.advanced_panel as advanced_panel
import modules.ui_components.control_panel as control_panel
import modules.ui_components.inpaint_panel as inpaint_panel
import modules.ui_components.outpaint_panel as outpaint_panel
import args_manager
import copy
from modules.setup_utils import download_models

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import javascript_html, css_html
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

def get_task(*args):
    global ctrls_keys
    named_args = dict(zip(ctrls_keys, args))
    named_args.pop('_currentTask', None)
    return worker.AsyncTask(args=named_args)

def generate_clicked(task: worker.AsyncTask, image_number, disable_preview):
    import backend.resources as resources

    with resources.interrupt_processing_mutex:
        resources.interrupt_processing = False
    # outputs=[progress_html, progress_window, gallery, preview_column, gallery_column]

    if not task.is_valid:
        return

    try:
        batch_size = int(image_number)
    except Exception:
        batch_size = 1
    preview_enabled = not bool(disable_preview)

    execution_start_time = time.perf_counter()
    finished = False
    has_results = False

    if preview_enabled:
        initial_preview_col = gr.update(visible=True)
        initial_gallery_col = gr.update(visible=False)
    else:
        initial_preview_col = gr.update(visible=False)
        initial_gallery_col = gr.update(visible=True)

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=True, columns=1), \
        initial_preview_col, \
        initial_gallery_col

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                # Coalesce consecutive preview updates so the UI sees the newest
                # progress state without dropping every preview under load.
                while len(task.yields) > 0 and task.yields[0][0] == 'preview':
                    next_percentage, next_title, next_image = task.yields.pop(0)[1]
                    percentage, title = next_percentage, next_title
                    if next_image is not None:
                        image = next_image
                if preview_enabled:
                    if has_results and batch_size >= 2:
                        preview_col = gr.update(visible=True)
                        gallery_col = gr.update(visible=True)
                    else:
                        preview_col = gr.update(visible=True)
                        gallery_col = gr.update(visible=False)
                else:
                    preview_col = gr.update(visible=False)
                    gallery_col = gr.update(visible=True)

                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(visible=True), \
                    gr.update(visible=True, columns=1), \
                    preview_col, \
                    gallery_col
            if flag == 'results':
                has_results = True
                if preview_enabled and batch_size >= 2:
                    preview_col = gr.update(visible=True)
                    gallery_col = gr.update(visible=True)
                elif preview_enabled and batch_size == 1:
                    preview_col = gr.update(visible=True)
                    gallery_col = gr.update(visible=False)
                else:
                    preview_col = gr.update(visible=False)
                    gallery_col = gr.update(visible=True)

                yield gr.update(visible=True), \
                    gr.update(), \
                    gr.update(visible=True, value=product, columns=1), \
                    preview_col, \
                    gallery_col
            if flag == 'finish':
                cols = max(1, int(np.ceil(np.sqrt(len(product))))) if len(product) > 0 else 1

                yield gr.update(visible=False), \
                    gr.update(visible=True, value=None), \
                    gr.update(visible=True, value=product, columns=cols), \
                    gr.update(visible=False), \
                    gr.update(visible=True)
                finished = True
                
                # Auto-populate mask if BGR was run
                if task.state.current_tab == 'remove' and 'remove_bg' in task.state.goals and len(product) > 1:
                    # product[0] = character, product[1] = mask
                    task.yields.append(('bgr_mask_update', product[1]))

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return




def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 0.5
    ]


def expand_mask(outpaint_selections, inpaint_mask_image):
    print(f"[Debug] Mask Expansion Requested. Direction: {outpaint_selections}")
    if inpaint_mask_image is None:
        print("[Debug] Mask Image is None. Aborting.")
        return gr.update()
    
    from modules.mask_processing import combine_image_and_mask, to_binary_mask, expand_mask_direction, extract_mask_from_layers
    
    # Handle ImageEditor EditorValue
    if isinstance(inpaint_mask_image, dict) and 'background' in inpaint_mask_image:
        merged = combine_image_and_mask(inpaint_mask_image)
    else:
        merged = combine_image_and_mask(inpaint_mask_image)
    if merged is None:
        return gr.update()
        
    print(f"[Debug Expand Mask] merged shape: {merged.shape}, max: {merged.max()}, min: {merged.min()}, mean: {merged.mean()}")
    
    new_mask = to_binary_mask(merged)
    print(f"[Debug Expand Mask] binary_mask shape: {new_mask.shape}, sum (white pixels): {new_mask.sum() // 255} out of {new_mask.size}")
    
    for direction in outpaint_selections:
        new_mask = expand_mask_direction(new_mask, direction, pixels=32)

    from PIL import Image
    import modules.util
    import os
    
    result_rgb = np.stack([new_mask]*3, axis=-1)
    result_img = Image.fromarray(result_rgb)
    
    _, temp_path, _ = modules.util.generate_temp_filename(folder=modules.config.path_temp_outputs, extension='png')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    result_img.save(temp_path)
    
    return temp_path

def uov_method_change(method):
    if method == 'Super-Upscale':
        # Force light model for Super-Upscale and disable dropdown
        return gr.update(visible=True), gr.update(interactive=False, value='4xNomos2_otf_esrgan.pth'), gr.update(visible=True)
    return gr.update(visible=False), gr.update(interactive=True), gr.update(visible=True)

def update_upscale_scale_info(image_path, model_name, scale_override):
    if image_path is None:
        return gr.update(value="<b>Scale:</b> No image uploaded.")
    
    if model_name == 'None' or model_name is None:
         return gr.update(value="<b>Scale:</b> No model selected.")
    
    import modules.upscaler as upscaler
    try:
        model = upscaler.load_model(model_name)
        native_scale = upscaler.get_model_scale(model)
        
        if scale_override > 0:
            return gr.update(value=f"<b>Scale:</b> {scale_override}x (Overridden)")
        else:
            return gr.update(value=f"<b>Scale:</b> {native_scale}x (Model default)")
    except Exception as e:
        return gr.update(value=f"<b>Scale:</b> Error detection: {str(e)}")

def refresh_upscale_models():
    import modules.upscaler as upscaler
    models = upscaler.list_available_models()
    default_model = 'None'
    if '4xNomos2_otf_esrgan.pth' in models:
        default_model = '4xNomos2_otf_esrgan.pth'
    elif len(models) > 0:
        default_model = models[0]
        
    return gr.update(choices=['None'] + models, value=default_model)
    
def stop_clicked(currentTask):
    import backend.resources as resources
    currentTask.last_stop = 'stop'
    resources.interrupt_current_processing()
    return currentTask

def skip_clicked(currentTask):
    import backend.resources as resources
    currentTask.last_stop = 'skip'
    resources.interrupt_current_processing()
    return currentTask

def outpaint_selection_change(choices):
    if len(choices) <= 1:
        return choices
    return [choices[-1]]

def trigger_metadata_preview(file):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

    results = {}
    if parameters is not None:
        results['parameters'] = parameters

    if isinstance(metadata_scheme, flags.MetadataScheme):
        results['metadata_scheme'] = metadata_scheme.value

    return results

def random_checked(r):
    return gr.update(visible=not r)

def refresh_seed(r, seed_string):
    if r:
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

def update_history_link():
    if args_manager.args.disable_image_log:
        return gr.update(value='')

    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

def update_style_label(selections):
    if not selections or len(selections) == 0:
        return gr.update(label='Prompt Presets')
    
    visible_styles = selections[:2]
    label = f"Presets: {', '.join(visible_styles)}"
    if len(selections) > 2:
        label += f" ... (+{len(selections) - 2} more)"
    
    return gr.update(label=label)

def refresh_files_clicked():
    modules.config.update_files()
    results = [gr.update(choices=modules.config.model_filenames)]
    results += [gr.update(choices=[modules.flags.default_vae] + modules.config.vae_filenames)]
    results += [gr.update(choices=['None'] + modules.config.clip_filenames)]
    if not args_manager.args.disable_preset_selection:
        results += [gr.update(choices=modules.config.available_presets)]
    for i in range(modules.config.default_max_lora_number):
        results += [gr.update(interactive=True),
                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
    return results

def preset_selection_change(preset, is_generating):
    preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
    preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

    default_model = preset_prepared.get('base_model')
    previous_default_models = preset_prepared.get('previous_default_models', [])
    checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
    embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
    lora_downloads = preset_prepared.get('lora_downloads', {})
    vae_downloads = preset_prepared.get('vae_downloads', {})

    preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = download_models(
        default_model, checkpoint_downloads, embeddings_downloads, lora_downloads,
        vae_downloads)

    if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
        del preset_prepared['prompt']

    return metadata_ui.load_parameter_button_click(json.dumps(preset_prepared), is_generating, modules.flags.inpaint_option_default)

def inpaint_engine_state_change(inpaint_engine_version):
    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version
    return gr.update(value=inpaint_engine_version)

def outpaint_engine_state_change(outpaint_engine_version):
    if outpaint_engine_version == 'empty':
        outpaint_engine_version = modules.config.default_outpaint_engine_version
    return gr.update(value=outpaint_engine_version)

def parse_meta(raw_prompt_txt, is_generating):
    loaded_json = None
    if is_json(raw_prompt_txt):
        loaded_json = json.loads(raw_prompt_txt)

    if loaded_json is None:
        if is_generating:
            return gr.update(), gr.update(), gr.update()
        else:
            return gr.update(), gr.update(visible=True), gr.update(visible=False)

    return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

def trigger_metadata_import(file, state_is_generating):
    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    if parameters is None:
        print('Could not find metadata in the image!')
        parsed_parameters = {}
    else:
        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
        parsed_parameters = metadata_parser.to_json(parameters)

    return metadata_ui.trigger_metadata_import(file, state_is_generating)





# MVC-Light Controller: ui_logic.py

def register_all_events(ctrls_dict, currentTask_component, ui_elements):
    # Unpack components for easy reference
    for name, component in ctrls_dict.items():
        globals()[name] = component
    
    for name, component in ui_elements.items():
        globals()[name] = component
    
    global currentTask
    currentTask = currentTask_component
    
    global ctrls_keys, ctrls
    ctrls_keys = ['_currentTask'] + list(ctrls_dict.keys())
    ctrls = [currentTask_component] + list(ctrls_dict.values())

    # Global/Shared states and components that are needed by logic
    # (These are passed in via ctrls_dict or accessible via shared)
    
    # Phase 3 UI Bindings
    global toggle_toolbar_js, switch_js, down_js
    toggle_toolbar_js = """
    () => {
        const wrap = document.querySelector('#inpaint_canvas');
        if(wrap){
            wrap.classList.toggle('hide-toolbar');
            if(!document.getElementById('inpaint-toolbar-style')){
                const style = document.createElement('style');
                style.id = 'inpaint-toolbar-style';
                style.innerHTML = `
                    #inpaint_canvas.hide-toolbar button[aria-label="Undo"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Clear"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Remove Image"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Draw"],
                    #inpaint_canvas.hide-toolbar button[aria-label="Erase"],
                    #inpaint_canvas.hide-toolbar .canvas-tooltip-info,
                    #inpaint_canvas.hide-toolbar .toolbar,
                    #inpaint_canvas.hide-toolbar input[type="range"] {
                        display: none !important;
                        opacity: 0 !important;
                        visibility: hidden !important;
                    }
                `;
                document.head.appendChild(style);
            }
        }
    }
    """
    
    switch_js = "(x) => {if(x){if(window.viewer_to_bottom){viewer_to_bottom(100);viewer_to_bottom(500);}}else{if(window.viewer_to_top){viewer_to_top();}} return x;}"
    down_js = "() => {if(window.viewer_to_bottom){viewer_to_bottom();}}"

    # Bindings start here
    inpaint_toggle_toolbar.click(lambda: None, queue=False, show_progress=False, js=toggle_toolbar_js)

    input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                outputs=image_input_panel, queue=False, show_progress=False, js=switch_js)

    outpaint_selections.change(outpaint_selection_change, inputs=outpaint_selections, outputs=outpaint_selections, queue=False, show_progress=False)

    uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    outpaint_tab.select(lambda: 'outpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    remove_tab.select(lambda: 'remove', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, js=down_js, show_progress=False)
    metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, js=down_js, show_progress=False)

    uov_method.change(uov_method_change, inputs=uov_method, outputs=[upscale_refinement_container, upscale_model, upscale_scale_override], queue=False, show_progress=False)
    
    uov_input_image.upload(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)
    uov_input_image.clear(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)
    upscale_model.change(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)
    upscale_scale_override.change(update_upscale_scale_info, inputs=[uov_input_image, upscale_model, upscale_scale_override], outputs=upscale_scale_info, queue=False, show_progress=False)

    shared.gradio_root.load(refresh_upscale_models, outputs=upscale_model, queue=False, show_progress=False)

    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')
    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')

    seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                       queue=False, show_progress=False)

    shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

    style_selections.change(update_style_label, inputs=style_selections, outputs=style_selections_accordion, queue=False, show_progress=False)

    shared.gradio_root.load(
        lambda: gr.update(
            choices=copy.deepcopy(style_sorter.all_styles),
            value=[x for x in modules.config.default_styles if x in style_sorter.all_styles]
        ),
        outputs=style_selections,
        queue=False,
        show_progress=False
    ).then(update_style_label, inputs=style_selections, outputs=style_selections_accordion, queue=False, show_progress=False).then(lambda: None, js='()=>{refresh_style_localization();}', queue=False, show_progress=False)

    style_search_bar.change(style_sorter.search_styles,
                            inputs=[style_selections, style_search_bar],
                            outputs=style_selections,
                            queue=False,
                            show_progress=False).then(
        lambda: None, js='()=>{refresh_style_localization();}')

    refresh_files_output = [base_model, vae_model, clip_model]
    if not args_manager.args.disable_preset_selection:
        refresh_files_output += [preset_selection]
    refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                        queue=False, show_progress=False)

    if not args_manager.args.disable_preset_selection:
        preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, js='()=>{refresh_style_localization();}')

    output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

    # load configured default_inpaint_method
    shared.gradio_root.load(inpaint_engine_state_change, inputs=[inpaint_engine_state], outputs=[
        inpaint_engine
    ], show_progress=False, queue=False)

    shared.gradio_root.load(outpaint_engine_state_change, inputs=[outpaint_engine_state], outputs=[
        outpaint_engine
    ], show_progress=False, queue=False)

    prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

    load_parameter_button.click(metadata_ui.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=False)

    metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
        .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

    import modules.mask_processing as mask_proc
    inpaint_context_mask_data.change(
        mask_proc.compute_inpaint_step1_context,
        inputs=[inpaint_input_image, inpaint_context_mask_data],
        outputs=[inpaint_context_mask_image, inpaint_bb_image, inpaint_context_mask_data],
        queue=False,
        show_progress=False
    )

    inpaint_bb_mask_data.change(
        mask_proc.compute_inpaint_step2_mask,
        inputs=[inpaint_bb_mask_data],
        outputs=[inpaint_mask_image, inpaint_bb_mask_data],
        queue=False,
        show_progress=False
    )

    outpaint_bb_mask_data.change(
        mask_proc.compute_outpaint_step2_mask,
        inputs=[outpaint_bb_mask_data],
        outputs=[outpaint_mask_image, outpaint_bb_mask_data],
        queue=False,
        show_progress=False
    )

    generate_button.click(
        lambda disable_preview_value: (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            gr.update(visible=False, interactive=False),
            gr.update(visible=True, columns=1),
            True,
            gr.update(visible=not disable_preview_value),
            gr.update(visible=disable_preview_value)
        ),
        inputs=[disable_preview],
        outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating, preview_column, gallery_column]
    ) \
        .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
        .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
        .then(fn=lambda t, tab, rbg, robj: (
            t.state.goals.append('remove_bg') if tab == 'remove' and rbg else None,
            t.state.goals.append('remove_obj') if tab == 'remove' and robj else None,
            t
        )[-1], inputs=[currentTask, current_tab, ctrls_dict['remove_bg_enabled'], ctrls_dict['remove_obj_enabled']], outputs=currentTask) \
        .then(fn=generate_clicked, inputs=[currentTask, image_number, disable_preview],
              outputs=[progress_html, progress_window, gallery, preview_column, gallery_column]) \
        .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
              outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
        .then(fn=update_history_link, outputs=history_link) \
        .then(fn=lambda: None, js='playNotification').then(fn=lambda: None, js='refresh_grid_delayed')


    reset_button.click(lambda: [
                                worker.AsyncTask(args=[]),
                                False,
                                gr.update(visible=True, interactive=True),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=True, value=None),
                                gr.update(visible=True, value=[], columns=2),
                                gr.update(visible=False),
                                gr.update(visible=True)
                            ],
                       outputs=[currentTask, state_is_generating, generate_button,
                                reset_button, stop_button, skip_button,
                                progress_html, progress_window, gallery, preview_column, gallery_column],
                       queue=False)

    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, js='cancelGenerateForever')
    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)

    example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)
    metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image, outputs=metadata_json, queue=False, show_progress=True)
    outpaint_mask_expansion_button.click(expand_mask, inputs=[outpaint_selections, outpaint_mask_image], outputs=[outpaint_mask_image], queue=False, show_progress=False)


