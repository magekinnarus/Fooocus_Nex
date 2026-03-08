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
    del named_args['_currentTask']
    return worker.AsyncTask(args=named_args)

def generate_clicked(task: worker.AsyncTask, image_number, disable_preview):
    import backend.resources as resources

    with resources.interrupt_processing_mutex:
        resources.interrupt_processing = False
    # outputs=[progress_html, progress_window, gallery, preview_column, gallery_column]

    if len(task.args) == 0:
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
    
    _, temp_path, _ = modules.util.generate_temp_filename(folder=modules.config.path_outputs, extension='png')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    result_img.save(temp_path)
    
    return [(temp_path, 'Expanded Mask')]




# reload_javascript() removed; handled via gr.Blocks(head=...)

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title, head=javascript_html() + css_html()).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    outpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column(scale=5, min_width=420, visible=False) as preview_column:
                    progress_window = gr.Image(label='Live Preview', show_label=True, interactive=False, visible=True,
                                               height=768, type='numpy',
                                               elem_classes=['main_view', 'preview_panel'])
                with gr.Column(scale=6, min_width=500, visible=True) as gallery_column:
                    gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain', visible=True, height=768,
                                         elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                         elem_id='final_gallery')
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    reset_button = gr.Button(value="Reconnect", elem_classes='type_row', elem_id='reset_button', visible=False)
                    load_parameter_button = gr.Button(value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(value="Skip", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import backend.resources as resources
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            resources.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import backend.resources as resources
                        currentTask.last_stop = 'skip'
                        if (currentTask.processing):
                            resources.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale or Variation', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = gr.Image(label='Image', sources='upload', type='filepath', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')
                    with gr.Tab(label='Image Prompt', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = gr.Image(label='Image', sources='upload', type='filepath', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='Advanced', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Documentation</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='Outpaint', id='outpaint_tab') as outpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                outpaint_input_image = gr.Image(label='Image', sources='upload', type='filepath', height=500, show_label=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=['Left'], label='Outpaint Direction')
                                outpaint_step2_checkbox = gr.Checkbox(label='Outpaint 2nd Step generation', value=False, elem_id='outpaint_step2_checkbox', info='Provides color guidance for outpaint by pixelating the blank area.')
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')

                            with gr.Column(visible=True) as outpaint_mask_generation_col:
                                outpaint_mask_image = gr.Image(label='Mask Upload', sources='upload', type='filepath', height=500, elem_id='outpaint_mask_canvas')
                                outpaint_mask_expansion_button = gr.Button(value='Expand Mask (32 pixels)')
                                
                                outpaint_mask_expansion_button.click(expand_mask, inputs=[outpaint_selections, outpaint_mask_image], outputs=[gallery], queue=False, show_progress=False)

                    with gr.Tab(label='Inpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = gr.Image(label='Image Upload', sources='upload', type='filepath', height=500, elem_id='inpaint_canvas', show_label=False)
                                inpaint_context_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_context_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                gr.HTML("""
<div id="inpaint-mask-tools" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">MASK WORKFLOW</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-context" title="Step 1: Paint Context">Context Mask</button>
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-bb" title="Step 2: Paint BB Patch">BB Mask</button>
    </div>
    <div style="width:1px; height:22px; background:rgba(128,128,128,0.3); margin:0 4px;"></div>
    <div style="display:flex; gap:8px;">
      <button type="button" class="mask-tool-btn" id="inpaint-mask-brush">Brush</button>
      <button type="button" class="mask-tool-btn" id="inpaint-mask-erase">Erase</button>
    </div>
    <button type="button" class="mask-tool-btn" id="inpaint-mask-clear" style="margin-left:auto; opacity:0.8;">Clear All</button>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="inpaint-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <span id="inpaint-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                inpaint_toggle_toolbar = gr.Button("Toggle Canvas Toolbar", size="sm", visible=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='Hide Advanced Masking Features', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=True)
                                inpaint_step2_checkbox = gr.Checkbox(label='2nd Step generation', value=False, elem_id='inpaint_step2_checkbox', info='Enable to use the uploaded edited BB patch for final generation.')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=True)
                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">Documentation</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)


                            with gr.Column(visible=not modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_context_mask_image = gr.Image(label='Step 1: Context Mask Upload', sources='upload', type='filepath', height=500, elem_id='inpaint_context_mask_canvas')
                                inpaint_bb_image = gr.Image(label='Step 2: Edited BB Image Upload', sources='upload', type='filepath', height=500, elem_id='inpaint_bb_canvas')
                                inpaint_bb_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_bb_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                inpaint_mask_image = gr.Image(label='Step 2: BB Mask Upload (Optional)', sources='upload', type='filepath', height=500, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='Invert Mask When Generating', value=modules.config.default_invert_mask_checkbox)



                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = gr.Image(label='For images created by Fooocus', sources='upload', type='pil')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            # Phase 3 UI Bindings

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
            inpaint_toggle_toolbar.click(lambda: None, queue=False, show_progress=False, js=toggle_toolbar_js)

            switch_js = "(x) => {if(x){if(window.viewer_to_bottom){viewer_to_bottom(100);viewer_to_bottom(500);}}else{if(window.viewer_to_top){viewer_to_top();}} return x;}"
            down_js = "() => {if(window.viewer_to_bottom){viewer_to_bottom();}}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, js=down_js)

            def outpaint_selection_change(choices):
                if len(choices) <= 1:
                    return choices
                return [choices[-1]]

            outpaint_selections.change(outpaint_selection_change, inputs=outpaint_selections, outputs=outpaint_selections, queue=False, show_progress=False)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            outpaint_tab.select(lambda: 'outpaint', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, js=down_js, show_progress=False)

        with gr.Column(scale=1, visible=True) as advanced_column:
            with gr.Tab(label='Settings'):
                settings_panel_result = settings_panel.build_settings_tab()
                if not args_manager.args.disable_preset_selection:
                    preset_selection = settings_panel_result['preset_selection']
                aspect_ratios_selection = settings_panel_result['aspect_ratios_selection']
                image_number = settings_panel_result['image_number']
                overwrite_step = settings_panel_result['overwrite_step']
                sampler_name = settings_panel_result['sampler_name']
                scheduler_name = settings_panel_result['scheduler_name']
                guidance_scale = settings_panel_result['guidance_scale']
                clip_skip = settings_panel_result['clip_skip']
                # output_format moved to debug_panel_result
                negative_prompt = settings_panel_result['negative_prompt']
                seed_random = settings_panel_result['seed_random']
                image_seed = settings_panel_result['image_seed']
                history_link = settings_panel_result['history_link']

                aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')
                shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, js='(x)=>{refresh_aspect_ratios_label(x);}')

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

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">\U0001F4DA History Log</a>')

                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)


            with gr.Tab(label='Models'):
                models_panel_result = models_panel.build_models_tab()
                base_model = models_panel_result['base_model']
                vae_model = models_panel_result['vae_model']
                clip_model = models_panel_result['clip_model']
                
                style_search_bar = models_panel_result['style_search_bar']
                style_selections = models_panel_result['style_selections']
                gradio_receiver_style_selections = models_panel_result['gradio_receiver_style_selections']
                style_selections_accordion = models_panel_result['style_selections_accordion']

                lora_ctrls = models_panel_result['lora_ctrls']
                refresh_files = models_panel_result['refresh_files']

                def update_style_label(selections):
                    if not selections or len(selections) == 0:
                        return gr.update(label='Styles')
                    
                    visible_styles = selections[:2]
                    label = f"Styles: {', '.join(visible_styles)}"
                    if len(selections) > 2:
                        label += f" ... (+{len(selections) - 2} more)"
                    
                    return gr.update(label=label)

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

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, js='()=>{refresh_style_localization();}')
            with gr.Tab(label='Advanced'):
                with gr.Column(visible=True) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        debug_panel_result = advanced_panel.build_debug_tab()
                        sharpness = debug_panel_result['sharpness']
                        output_format = debug_panel_result['output_format']
                        adm_scaler_positive = debug_panel_result['adm_scaler_positive']
                        adm_scaler_negative = debug_panel_result['adm_scaler_negative']
                        adm_scaler_end = debug_panel_result['adm_scaler_end']
                        adaptive_cfg = debug_panel_result['adaptive_cfg']
                        generate_image_grid = debug_panel_result['generate_image_grid']
                        overwrite_width = debug_panel_result['overwrite_width']
                        overwrite_height = debug_panel_result['overwrite_height']
                        overwrite_vary_strength = debug_panel_result['overwrite_vary_strength']
                        overwrite_upscale_strength = debug_panel_result['overwrite_upscale_strength']
                        disable_preview = debug_panel_result['disable_preview']
                        disable_intermediate_results = debug_panel_result['disable_intermediate_results']
                        disable_seed_increment = debug_panel_result['disable_seed_increment']
                        read_wildcards_in_order = debug_panel_result['read_wildcards_in_order']
                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = debug_panel_result['save_metadata_to_images']
                            metadata_scheme = debug_panel_result['metadata_scheme']

                    with gr.Tab(label='Control'):
                        control_panel_result = control_panel.build_control_tab()
                        debugging_cn_preprocessor = control_panel_result['debugging_cn_preprocessor']
                        skipping_cn_preprocessor = control_panel_result['skipping_cn_preprocessor']
                        mixing_image_prompt_and_vary_upscale = control_panel_result['mixing_image_prompt_and_vary_upscale']
                        mixing_image_prompt_and_inpaint = control_panel_result['mixing_image_prompt_and_inpaint']
                        controlnet_softness = control_panel_result['controlnet_softness']
                        canny_low_threshold = control_panel_result['canny_low_threshold']
                        canny_high_threshold = control_panel_result['canny_high_threshold']

                    with gr.Tab(label='Outpaint'):
                        outpaint_panel_result = outpaint_panel.build_outpaint_tab()
                        outpaint_engine = outpaint_panel_result['outpaint_engine']
                        outpaint_strength = outpaint_panel_result['outpaint_strength']
                        inpaint_outpaint_expansion_size = outpaint_panel_result['inpaint_outpaint_expansion_size']

                        outpaint_ctrls = [outpaint_engine, outpaint_strength,
                                          inpaint_outpaint_expansion_size, outpaint_step2_checkbox]

                    with gr.Tab(label='Inpaint'):
                        inpaint_panel_result = inpaint_panel.build_inpaint_tab(
                            inpaint_advanced_masking_checkbox, invert_mask_checkbox,
                            inpaint_mask_image, inpaint_mask_generation_col, inpaint_input_image
                        )
                        debugging_inpaint_preprocessor = inpaint_panel_result['debugging_inpaint_preprocessor']
                        inpaint_disable_initial_latent = inpaint_panel_result['inpaint_disable_initial_latent']
                        inpaint_engine = inpaint_panel_result['inpaint_engine']
                        inpaint_strength = inpaint_panel_result['inpaint_strength']
                        inpaint_erode_or_dilate = inpaint_panel_result['inpaint_erode_or_dilate']

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate,
                                         inpaint_step2_checkbox]



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

                refresh_files_output = [base_model, vae_model, clip_model]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

        state_is_generating = gr.State(False)

        load_data_outputs = [image_number, prompt, negative_prompt, style_selections,
                             overwrite_step, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip,
                             base_model, vae_model, clip_model, sampler_name, scheduler_name, 
                             seed_random, image_seed, outpaint_engine_state, inpaint_engine_state,
                             generate_button,
                             load_parameter_button] + lora_ctrls

        if not args_manager.args.disable_preset_selection:
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


        ctrls_dict = {
            'generate_image_grid': generate_image_grid,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'style_selections': style_selections,
            'aspect_ratios_selection': aspect_ratios_selection,
            'image_number': image_number,
            'output_format': output_format,
            'image_seed': image_seed,
            'read_wildcards_in_order': read_wildcards_in_order,
            'sharpness': sharpness,
            'guidance_scale': guidance_scale,
            'base_model': base_model,
            'vae_model': vae_model,
            'clip_model': clip_model,
        }

        for i in range(modules.config.default_max_lora_number):
            ctrls_dict[f'lora_{i}_enabled'] = lora_ctrls[i * 3]
            ctrls_dict[f'lora_{i}_model'] = lora_ctrls[i * 3 + 1]
            ctrls_dict[f'lora_{i}_weight'] = lora_ctrls[i * 3 + 2]

        ctrls_dict.update({
            'input_image_checkbox': input_image_checkbox,
            'current_tab': current_tab,
            'uov_method': uov_method,
            'uov_input_image': uov_input_image,
            'outpaint_selections': outpaint_selections,
            'outpaint_input_image': outpaint_input_image,
            'outpaint_mask_image': outpaint_mask_image,
            'inpaint_input_image': inpaint_input_image,
            'inpaint_context_mask_image': inpaint_context_mask_image,
            'inpaint_additional_prompt': inpaint_additional_prompt,
            'inpaint_mask_image': inpaint_mask_image,
            'inpaint_bb_image': inpaint_bb_image,
            'disable_preview': disable_preview,
            'disable_intermediate_results': disable_intermediate_results,
            'disable_seed_increment': disable_seed_increment,
            'adm_scaler_positive': adm_scaler_positive,
            'adm_scaler_negative': adm_scaler_negative,
            'adm_scaler_end': adm_scaler_end,
            'adaptive_cfg': adaptive_cfg,
            'clip_skip': clip_skip,
            'sampler_name': sampler_name,
            'scheduler_name': scheduler_name,
            'overwrite_step': overwrite_step,
            'overwrite_width': overwrite_width,
            'overwrite_height': overwrite_height,
            'overwrite_vary_strength': overwrite_vary_strength,
            'overwrite_upscale_strength': overwrite_upscale_strength,
            'mixing_image_prompt_and_vary_upscale': mixing_image_prompt_and_vary_upscale,
            'mixing_image_prompt_and_inpaint': mixing_image_prompt_and_inpaint,
            'debugging_cn_preprocessor': debugging_cn_preprocessor,
            'skipping_cn_preprocessor': skipping_cn_preprocessor,
            'canny_low_threshold': canny_low_threshold,
            'canny_high_threshold': canny_high_threshold,
            'controlnet_softness': controlnet_softness,
            
            # inpaint_ctrls
            'debugging_inpaint_preprocessor': debugging_inpaint_preprocessor,
            'inpaint_disable_initial_latent': inpaint_disable_initial_latent,
            'inpaint_engine': inpaint_engine,
            'inpaint_strength': inpaint_strength,
            'inpaint_advanced_masking_checkbox': inpaint_advanced_masking_checkbox,
            'invert_mask_checkbox': invert_mask_checkbox,
            'inpaint_erode_or_dilate': inpaint_erode_or_dilate,
            'inpaint_step2_checkbox': inpaint_step2_checkbox,

            # outpaint_ctrls
            'outpaint_engine': outpaint_engine,
            'outpaint_strength': outpaint_strength,
            'inpaint_outpaint_expansion_size': inpaint_outpaint_expansion_size,
            'outpaint_step2_checkbox': outpaint_step2_checkbox,
        })

        if not args_manager.args.disable_metadata:
            ctrls_dict['save_metadata_to_images'] = save_metadata_to_images
            ctrls_dict['metadata_scheme'] = metadata_scheme

        for i in range(modules.config.default_controlnet_image_count):
            ctrls_dict[f'cn_{i}_image'] = ip_ctrls[i * 4]
            ctrls_dict[f'cn_{i}_stop'] = ip_ctrls[i * 4 + 1]
            ctrls_dict[f'cn_{i}_weight'] = ip_ctrls[i * 4 + 2]
            ctrls_dict[f'cn_{i}_type'] = ip_ctrls[i * 4 + 3]

        import modules.parameter_registry as parameter_registry
        parameter_registry.validate_ctrls(ctrls_dict)

        global ctrls_keys
        ctrls_keys = ['_currentTask'] + list(ctrls_dict.keys())
        ctrls = [currentTask] + list(ctrls_dict.values())

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

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(metadata_ui.load_parameter_button_click, inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('Could not find metadata in the image!')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return metadata_ui.trigger_metadata_import(file, state_is_generating)

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

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[
        modules.config.path_outputs,
        os.path.abspath('javascript'),
        os.path.abspath('css'),
        os.path.abspath('sdxl_styles/samples')
    ],
    blocked_paths=[constants.AUTH_FILENAME]
)


