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
import modules.ui_components.staging_panel as staging_panel
import args_manager
import copy
from modules.setup_utils import download_models

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import javascript_html, css_html
from modules.auth import auth_enabled, check_auth
from modules.util import is_json


import modules.ui_logic as ui_logic
from modules.staging_api import staging_router




# reload_javascript() removed; handled via gr.Blocks(head=...)

title = f'Fooocus {fooocus_version.version}'

def make_nex_image_slot(slot_id, bridge_id, label, extra_attrs=''):
    attrs = f' {extra_attrs}' if extra_attrs else ''
    return f'<nex-image-slot id="{slot_id}" data-bridge-id="{bridge_id}" data-label="{label}"{attrs}></nex-image-slot>'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title, head=javascript_html() + css_html()).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    outpaint_engine_state = gr.State('empty')
    remove_mask_state = gr.State(None)
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



            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='Input Image', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id):
                    with gr.Tab(label='Upscale/Superupscale', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('uov_input_slot', 'uov_input_image_bridge', 'Image', 'data-upload-mode="api" data-path-field-id="uov_input_image_path" data-workspace-field-id="uov_input_workspace_id"'))
                                uov_input_image = gr.Image(label='Image', sources='upload', type='filepath', show_label=False, elem_id='uov_input_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                uov_input_image_path = gr.Textbox(value='', visible=True, elem_id='uov_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                uov_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='uov_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='Method:', choices=['Upscale', 'Super-Upscale'], value='Upscale')
                                upscale_model = gr.Dropdown(label='Upscale Model', choices=['None'], value='None')
                                upscale_scale_info = gr.HTML(value="<b>Scale:</b> Auto-detecting...", elem_id='upscale_scale_info')
                                upscale_scale_override = gr.Slider(label='Scale Override', minimum=0.0, maximum=8.0, step=0.1, value=0.0, info='Set to 0.0 to use model default scale.')
                                
                                with gr.Group(visible=False) as upscale_refinement_container:
                                    upscale_refinement_denoise = gr.Slider(label='Refinement Denoise', minimum=0.0, maximum=1.0, step=0.001, value=0.382)
                                    upscale_refinement_tile_overlap = gr.Slider(label='Refinement Tile Overlap', minimum=0, maximum=256, step=1, value=128)
                                
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Documentation</a>')

                    with gr.Tab(label='Remove', id='remove_tab') as remove_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('remove_base_image_slot', 'remove_base_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="remove_base_image_path" data-workspace-field-id="remove_base_workspace_id"'))
                                remove_base_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, show_label=False, elem_id='remove_base_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                remove_base_image_path = gr.Textbox(value='', visible=True, elem_id='remove_base_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                remove_base_workspace_id = gr.Textbox(value='', visible=True, elem_id='remove_base_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                with gr.Row():
                                    remove_bg_enabled = gr.Checkbox(label='Remove Background', value=False, elem_id='remove_bg_enabled')
                                    remove_obj_enabled = gr.Checkbox(label='Remove Object', value=False, elem_id='remove_obj_enabled')
                                
                                gr.HTML('* <b>Remove Background</b> uses InSpireNet to extract the character.<br>'
                                        '* <b>Remove Object</b> uses MAT to clean the background defined by the mask.')

                            with gr.Column():
                                gr.HTML(make_nex_image_slot('remove_mask_image_slot', 'remove_mask_image_bridge', 'Mask'))
                                remove_mask_image = gr.Image(label='Mask', sources='upload', type='filepath', height=500, elem_id='remove_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                bgr_threshold = gr.Slider(label='BGR Threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.5, info='Higher = tighter cutout; Lower = keep softer edges.')
                                bgr_jit = gr.Checkbox(label='Use JIT (Optimized)', value=True)
                                objr_mask_dilate = gr.Slider(label='Mask Dilate', minimum=0, maximum=128, step=1, value=0, info='Expands the mask for Object Removal.')
                                objr_model = gr.Dropdown(label='OBJR Model', choices=['Places_512_FullData_G.pth'], value='Places_512_FullData_G.pth')
                    with gr.Tab(label='Controlnet', id='ip_tab') as ip_tab:
                        ip_images = []
                        cn_image_paths = []
                        ip_types = []
                        ip_stops = []
                        ip_weights = []
                        ip_ad_cols = []

                        guidance_choices_by_channel = {
                            flags.cn_structural: flags.cn_structural_types,
                            flags.cn_contextual: flags.cn_contextual_types,
                        }

                        def resolve_channel_default(image_count):
                            default_type = flags.resolve_cn_type(modules.config.default_ip_types[image_count])
                            default_channel = flags.get_cn_channel(default_type)
                            if default_channel in guidance_choices_by_channel:
                                return default_channel
                            return flags.cn_contextual

                        def resolve_type_default(image_count, channel):
                            choices = guidance_choices_by_channel.get(channel, flags.cn_contextual_types)
                            default_type = flags.resolve_cn_type(modules.config.default_ip_types[image_count])
                            if default_type in choices:
                                return default_type
                            return choices[0]

                        def update_guidance_type_choices(channel):
                            choices = guidance_choices_by_channel.get(channel, flags.cn_contextual_types)
                            return gr.update(choices=choices, value=choices[0])

                        def create_ip_slot(image_count):
                            default_channel = resolve_channel_default(image_count)
                            default_type = resolve_type_default(image_count, default_channel)
                            with gr.Column():
                                gr.HTML(make_nex_image_slot(
                                    f'ip_image_slot_{image_count}',
                                    f'ip_image_bridge_{image_count}',
                                    f'Guidance Image {image_count}',
                                    f'data-upload-mode="api" data-path-field-id="cn_{image_count - 1}_image_path" data-workspace-field-id="cn_{image_count - 1}_workspace_id"'
                                ))
                                ip_image = gr.Image(
                                    label='Image',
                                    sources='upload',
                                    type='filepath',
                                    show_label=False,
                                    height=300,
                                    value=modules.config.default_ip_images[image_count],
                                    elem_id=f'ip_image_bridge_{image_count}',
                                    elem_classes=['nex-image-slot-bridge']
                                )
                                cn_image_path = gr.Textbox(
                                    value=modules.config.default_ip_images[image_count],
                                    visible=True,
                                    elem_id=f'cn_{image_count - 1}_image_path',
                                    elem_classes=['inpaint-hidden-mask-field'],
                                    show_label=False,
                                    container=False
                                )
                                cn_workspace_id = gr.Textbox(
                                    value='',
                                    visible=True,
                                    elem_id=f'cn_{image_count - 1}_workspace_id',
                                    elem_classes=['inpaint-hidden-mask-field'],
                                    show_label=False,
                                    container=False
                                )
                                ip_images.append(ip_image)
                                cn_image_paths.append(cn_image_path)
                                with gr.Column(visible=True) as ad_col:
                                    with gr.Row():
                                        ip_channel = gr.Radio(
                                            label='Guidance Channel',
                                            choices=[flags.cn_structural, flags.cn_contextual],
                                            value=default_channel,
                                            container=False,
                                            scale=1
                                        )
                                        ip_type = gr.Dropdown(
                                            label='Method',
                                            choices=guidance_choices_by_channel[default_channel],
                                            value=default_type,
                                            container=False,
                                            scale=1
                                        )
                                        ip_channel.change(
                                            fn=update_guidance_type_choices,
                                            inputs=ip_channel,
                                            outputs=ip_type,
                                            queue=False,
                                            show_progress=False
                                        )
                                        ip_types.append(ip_type)

                                    with gr.Row():
                                        ip_stop = gr.Slider(
                                            label='Stop At',
                                            minimum=0.0,
                                            maximum=1.0,
                                            step=0.001,
                                            value=modules.config.default_ip_stop_ats[image_count]
                                        )
                                        ip_stops.append(ip_stop)

                                        ip_weight = gr.Slider(
                                            label='Weight',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.001,
                                            value=modules.config.default_ip_weights[image_count]
                                        )
                                        ip_weights.append(ip_weight)

                                ip_ad_cols.append(ad_col)

                        with gr.Row():
                            with gr.Column(scale=1):
                                for image_count in range(1, modules.config.default_controlnet_image_count + 1, 2):
                                    create_ip_slot(image_count)

                            with gr.Column(scale=1):
                                for image_count in range(2, modules.config.default_controlnet_image_count + 1, 2):
                                    create_ip_slot(image_count)

                        with gr.Group():
                            gr.HTML('<div style="margin-top:20px; border-top:1px solid rgba(128,128,128,0.2); padding-top:15px; font-weight:bold;">Advanced Control</div>')
                            control_panel_result = control_panel.build_control_tab()
                            debugging_cn_preprocessor = control_panel_result['debugging_cn_preprocessor']
                            skipping_cn_preprocessor = control_panel_result['skipping_cn_preprocessor']
                            mixing_image_prompt_and_inpaint = control_panel_result['mixing_image_prompt_and_inpaint']
                            controlnet_softness = control_panel_result['controlnet_softness']
                            canny_low_threshold = control_panel_result['canny_low_threshold']
                            canny_high_threshold = control_panel_result['canny_high_threshold']

                        gr.HTML('* "Controlnet" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">Documentation</a>')



                    with gr.Tab(label='Outpaint', id='outpaint_tab') as outpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('outpaint_input_slot', 'outpaint_input_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="outpaint_input_image_path" data-workspace-field-id="outpaint_input_workspace_id"'))
                                outpaint_input_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, show_label=False, elem_id='outpaint_input_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                outpaint_input_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=['Left'], label='Outpaint Direction')
                                with gr.Column(elem_classes=["step2-toolbox"]):
                                    outpaint_prepare_button = gr.Button(value='Prepare Outpaint', variant='primary', elem_id='outpaint_prepare_button')
                                    outpaint_step2_checkbox = gr.Checkbox(label='2nd Step generation', value=False, visible=False, elem_id='outpaint_step2_checkbox', elem_classes=['step2-status-btn'], container=False)
                                    outpaint_prepare_notice = gr.Markdown(value='')
                                    gr.HTML('<p class="step2-desc">Using base image, BB image, and BB mask to expand the image.</p>')

                                outpaint_panel_result = outpaint_panel.build_outpaint_tab()
                                outpaint_engine = outpaint_panel_result['outpaint_engine']
                                outpaint_strength = outpaint_panel_result['outpaint_strength']
                                inpaint_outpaint_expansion_size = outpaint_panel_result['inpaint_outpaint_expansion_size']

                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Documentation</a>')

                            with gr.Column(visible=True) as outpaint_mask_generation_col:
                                gr.HTML(make_nex_image_slot('outpaint_bb_canvas', 'outpaint_bb_image_bridge', 'BB Image', 'data-upload-mode="api" data-path-field-id="outpaint_bb_image_path" data-workspace-field-id="outpaint_bb_workspace_id" data-tool-group="outpaint"'))
                                outpaint_bb_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_bb_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_bb_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_bb_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML("""
<div id="outpaint-mask-tools" class="mask-workflow-toolbar" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">OUTPAINT MASK</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="outpaint-mask-mode-bb" title="Enable BB Mask">BB Mask</button>
      <button type="button" class="mask-tool-btn active" id="outpaint-mask-mode-disable" title="Disable Masking">Disable</button>
    </div>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="outpaint-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <span id="outpaint-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                outpaint_bb_mask_data = gr.Textbox(value="", visible=True, elem_id="outpaint_bb_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('outpaint_mask_canvas', 'outpaint_mask_image_bridge', 'BB Mask', 'data-upload-mode="api" data-path-field-id="outpaint_mask_image_path" data-workspace-field-id="outpaint_mask_workspace_id"'))
                                outpaint_mask_image_path = gr.Textbox(value='', visible=True, elem_id='outpaint_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='outpaint_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                outpaint_mask_expansion_button = gr.Button(value='Expand Mask (32 pixels)', visible=False)

                    with gr.Tab(label='Inpaint', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(make_nex_image_slot('inpaint_canvas', 'inpaint_input_image_bridge', 'Base Image', 'data-upload-mode="api" data-path-field-id="inpaint_input_image_path" data-workspace-field-id="inpaint_input_workspace_id" data-tool-group="inpaint-base"'))
                                inpaint_input_image = gr.Image(label='Base Image', sources='upload', type='filepath', height=500, elem_id='inpaint_input_image_bridge', show_label=False, elem_classes=['nex-image-slot-bridge'])
                                inpaint_input_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_input_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_input_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_input_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_context_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_context_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                inpaint_replace_bb_nonce = gr.Textbox(value='', visible=True, elem_id='inpaint_replace_bb_nonce', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML("""
<div id="inpaint-mask-tools" style="display:flex; flex-direction:column; gap:14px; margin:8px 0 16px; padding:14px; border:1px solid rgba(128,128,128,0.2); border-radius:12px; background:rgba(128,128,128,0.03);">
  <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center;">
    <span style="font-size:0.9rem; font-weight:700; color:var(--body-text-color); margin-right:4px;">Inpaint Mask</span>
    <div style="display:flex; gap:8px; padding:2px; background:rgba(0,0,0,0.1); border-radius:8px;">
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-context" title="Step 1: Paint Context">Context Mask</button>
      <button type="button" class="mask-tool-btn" id="inpaint-mask-mode-bb" title="Step 2: Paint BB Patch">BB Mask</button>
      <button type="button" class="mask-tool-btn active" id="inpaint-mask-mode-disable" title="Disable Masking">Disable</button>
    </div>
  </div>
  <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding-top:4px; border-top:1px solid rgba(128,128,128,0.1);">
    <label style="display:flex; align-items:center; gap:12px; font-size:0.9rem; font-weight:500; flex-grow:1; min-width:200px;">
      <span style="white-space:nowrap; opacity:0.8;">Brush Size</span>
      <input id="inpaint-mask-size" type="range" min="8" max="160" step="1" value="36" style="flex-grow:1; accent-color:var(--button-primary-background-fill);">
    </label>
    <button type="button" class="mask-tool-btn" id="inpaint-mask-refresh-bb" title="Rebuild BB Image from the current Base Image and Context Mask">Replace BB Image</button>
    <span id="inpaint-mask-status" style="font-size:0.85rem; opacity:0.6; font-style:italic; min-width:120px; text-align:right;">Ready</span>
  </div>
</div>
""")
                                inpaint_toggle_toolbar = gr.Button("Toggle Canvas Toolbar", size="sm", visible=False)
                                inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=True)
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='Additional Prompt Quick List',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=True)
                                with gr.Column(elem_classes=["step2-toolbox"]):
                                    inpaint_step2_checkbox = gr.Checkbox(label='2nd Step generation', value=False, visible=False, elem_id='inpaint_step2_checkbox', elem_classes=['step2-status-btn'], container=False)
                                    gr.HTML('<p class="step2-desc step2-desc--inpaint"><span class="step2-desc__title">Prepare Inpaint</span><span class="step2-desc__body">By adding or drawing masks to fill the Inpaint input images and press Generate to complete the process.</span></p>')

                                inpaint_panel_result = inpaint_panel.build_inpaint_tab()
                                debugging_inpaint_preprocessor = inpaint_panel_result['debugging_inpaint_preprocessor']
                                inpaint_disable_initial_latent = inpaint_panel_result['inpaint_disable_initial_latent']
                                inpaint_engine = inpaint_panel_result['inpaint_engine']
                                inpaint_strength = inpaint_panel_result['inpaint_strength']
                                inpaint_erode_or_dilate = inpaint_panel_result['inpaint_erode_or_dilate']

                                gr.HTML('* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">Documentation</a>')

                            with gr.Column(visible=True) as inpaint_mask_generation_col:
                                gr.HTML(make_nex_image_slot('inpaint_context_mask_canvas', 'inpaint_context_mask_image_bridge', 'Context Mask', 'data-upload-mode="api" data-path-field-id="inpaint_context_mask_image_path" data-workspace-field-id="inpaint_context_mask_workspace_id"'))
                                inpaint_context_mask_image = gr.Image(label='Context Mask', sources='upload', type='filepath', height=500, elem_id='inpaint_context_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_context_mask_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_context_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_context_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_context_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('inpaint_bb_canvas', 'inpaint_bb_image_bridge', 'BB Image', 'data-upload-mode="api" data-path-field-id="inpaint_bb_image_path" data-workspace-field-id="inpaint_bb_workspace_id" data-tool-group="inpaint-bb"'))
                                inpaint_bb_image = gr.Image(label='BB Image', sources='upload', type='filepath', height=500, elem_id='inpaint_bb_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_bb_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_bb_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_bb_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_bb_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_bb_mask_data = gr.Textbox(value="", visible=True, elem_id="inpaint_bb_mask_data", elem_classes=["inpaint-hidden-mask-field"], show_label=False, container=False)
                                gr.HTML(make_nex_image_slot('inpaint_mask_canvas', 'inpaint_mask_image_bridge', 'BB Mask', 'data-upload-mode="api" data-path-field-id="inpaint_mask_image_path" data-workspace-field-id="inpaint_mask_workspace_id"'))
                                inpaint_mask_image = gr.Image(label='BB Mask', sources='upload', type='filepath', height=500, elem_id='inpaint_mask_image_bridge', elem_classes=['nex-image-slot-bridge'])
                                inpaint_mask_image_path = gr.Textbox(value='', visible=True, elem_id='inpaint_mask_image_path', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                inpaint_mask_workspace_id = gr.Textbox(value='', visible=True, elem_id='inpaint_mask_workspace_id', elem_classes=['inpaint-hidden-mask-field'], show_label=False, container=False)
                                




                    with gr.Tab(label='Metadata', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = gr.Image(label='For images created by Fooocus', sources='upload', type='filepath')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')


            current_tab = gr.Textbox(value='uov', visible=False)

            # Phase 3 UI Bindings




        with gr.Column(scale=1, visible=True) as advanced_column:
            with gr.Row():
                gr.HTML('<button id="staging-panel-launcher" class="lg secondary gradio-button" style="width:100%; margin-bottom:12px; font-weight:bold;">\U0001F5C2\uFE0F Open Staging Palette</button>')
                gr.HTML('<button id="monitor-panel-launcher" class="lg secondary gradio-button" style="width:100%; margin-bottom:12px; font-weight:bold;">\U0001F4CA Monitor Dashboard</button>')
            
            with gr.Tab(label='Settings'):
                settings_panel_result = settings_panel.build_settings_tab()
                if not args_manager.args.disable_preset_selection:
                    preset_selection = settings_panel_result['preset_selection']
                aspect_ratios_selection = settings_panel_result['aspect_ratios_selection']
                image_number = settings_panel_result['image_number']
                steps = settings_panel_result['steps']
                sampler_name = settings_panel_result['sampler_name']
                scheduler_name = settings_panel_result['scheduler_name']
                guidance_scale = settings_panel_result['guidance_scale']
                clip_skip = settings_panel_result['clip_skip']
                # output_format moved to debug_panel_result
                negative_prompt = settings_panel_result['negative_prompt']
                seed_random = settings_panel_result['seed_random']
                image_seed = settings_panel_result['image_seed']
                history_link = settings_panel_result['history_link']





            with gr.Tab(label='Models'):
                models_panel_result = models_panel.build_models_tab()
                base_model = models_panel_result['base_model']
                vae_model = models_panel_result['vae_model']
                clip_model = models_panel_result['clip_model']
                
                style_search_bar = models_panel_result['style_search_bar']
                style_selections = models_panel_result['style_selections']
                style_selections_accordion = models_panel_result['style_selections_accordion']

                lora_ctrls = models_panel_result['lora_ctrls']
                refresh_files = models_panel_result['refresh_files']

            with gr.Tab(label='Advanced'):
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
                overwrite_upscale_strength = debug_panel_result['overwrite_upscale_strength']
                disable_preview = debug_panel_result['disable_preview']
                disable_intermediate_results = debug_panel_result['disable_intermediate_results']
                disable_seed_increment = debug_panel_result['disable_seed_increment']
                read_wildcards_in_order = debug_panel_result['read_wildcards_in_order']
                if not args_manager.args.disable_metadata:
                    save_metadata_to_images = debug_panel_result['save_metadata_to_images']
                    metadata_scheme = debug_panel_result['metadata_scheme']

                # Control settings moved to Image Prompt tab
                # (Removed outpaint advanced tab)
                # (Removed inpaint advanced tab)
                
                outpaint_ctrls = [outpaint_engine, outpaint_strength,
                                  inpaint_outpaint_expansion_size, outpaint_step2_checkbox]
                inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                 inpaint_strength, inpaint_erode_or_dilate, inpaint_step2_checkbox]





        state_is_generating = gr.State(False)

        load_data_outputs = [image_number, prompt, negative_prompt, style_selections,
                             steps, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip,
                             base_model, vae_model, clip_model, sampler_name, scheduler_name, 
                             seed_random, image_seed, outpaint_engine_state, inpaint_engine_state,
                             generate_button,
                             load_parameter_button] + lora_ctrls

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
            'uov_input_image': uov_input_image_path,
            'upscale_model': upscale_model,
            'upscale_scale_override': upscale_scale_override,
            'upscale_refinement_denoise': upscale_refinement_denoise,
            'upscale_refinement_tile_overlap': upscale_refinement_tile_overlap,
            'outpaint_selections': outpaint_selections,
            'outpaint_input_image': outpaint_input_image_path,
            'outpaint_mask_image': outpaint_mask_image_path,
            'inpaint_input_image': inpaint_input_image_path,
            'inpaint_context_mask_image': inpaint_context_mask_image_path,
            'inpaint_additional_prompt': inpaint_additional_prompt,
            'inpaint_mask_image': inpaint_mask_image_path,
            'inpaint_bb_image': inpaint_bb_image_path,
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
            'overwrite_width': overwrite_width,
            'overwrite_height': overwrite_height,
            'overwrite_upscale_strength': overwrite_upscale_strength,
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
            'inpaint_erode_or_dilate': inpaint_erode_or_dilate,
            'inpaint_step2_checkbox': inpaint_step2_checkbox,
            'steps': steps,

            # outpaint_ctrls
            'outpaint_engine': outpaint_engine,
            'outpaint_strength': outpaint_strength,
            'inpaint_outpaint_expansion_size': inpaint_outpaint_expansion_size,
            'outpaint_step2_checkbox': outpaint_step2_checkbox,
            'outpaint_bb_image': outpaint_bb_image_path,
            'outpaint_bb_mask_data': outpaint_bb_mask_data,
            'remove_base_image': remove_base_image_path,
            'remove_mask_image': remove_mask_image,
            'remove_bg_enabled': remove_bg_enabled,
            'remove_obj_enabled': remove_obj_enabled,
            'objr_mask_dilate': objr_mask_dilate,
            'bgr_threshold': bgr_threshold,
            'bgr_jit': bgr_jit,
            'objr_model': objr_model,
        })

        if not args_manager.args.disable_metadata:
            ctrls_dict['save_metadata_to_images'] = save_metadata_to_images
            ctrls_dict['metadata_scheme'] = metadata_scheme

        for i in range(modules.config.default_controlnet_image_count):
            ctrls_dict[f'cn_{i}_image'] = cn_image_paths[i]
            ctrls_dict[f'cn_{i}_stop'] = ip_stops[i]
            ctrls_dict[f'cn_{i}_weight'] = ip_weights[i]
            ctrls_dict[f'cn_{i}_type'] = ip_types[i]

        import modules.parameter_registry as parameter_registry
        parameter_registry.validate_ctrls(ctrls_dict)


        ui_elements = {
            'image_input_panel': image_input_panel,
            'uov_tab': uov_tab,
            'inpaint_tab': inpaint_tab,
            'outpaint_tab': outpaint_tab,
            'ip_tab': ip_tab,
            'metadata_tab': metadata_tab,
            'history_link': history_link,
            'style_selections_accordion': style_selections_accordion,
            'state_is_generating': state_is_generating,
            'reset_button': reset_button,
            'stop_button': stop_button,
            'skip_button': skip_button,
            'progress_html': progress_html,
            'progress_window': progress_window,
            'preview_column': preview_column,
            'gallery_column': gallery_column,
            'inpaint_toggle_toolbar': inpaint_toggle_toolbar,
            'outpaint_mask_expansion_button': outpaint_mask_expansion_button,
            'example_inpaint_prompts': example_inpaint_prompts,
            'metadata_import_button': metadata_import_button,
            'load_data_outputs': load_data_outputs,
            'inpaint_mask_generation_col': inpaint_mask_generation_col,
            'outpaint_mask_generation_col': outpaint_mask_generation_col,
            'inpaint_bb_image': inpaint_bb_image,
            'ip_ad_cols': ip_ad_cols,
            'ip_types': ip_types,
            'ip_stops': ip_stops,
            'ip_weights': ip_weights,
            'lora_ctrls': lora_ctrls,
            'style_search_bar': style_search_bar,
            'refresh_files': refresh_files,
            'inpaint_engine_state': inpaint_engine_state,
            'outpaint_engine_state': outpaint_engine_state,
            'generate_button': generate_button,
            'load_parameter_button': load_parameter_button,
            'metadata_input_image': metadata_input_image,
            'metadata_json': metadata_json,
            'inpaint_context_mask_data': inpaint_context_mask_data,
            'inpaint_replace_bb_nonce': inpaint_replace_bb_nonce,
            'inpaint_bb_mask_data': inpaint_bb_mask_data,
            'inpaint_input_image_path': inpaint_input_image_path,
            'inpaint_input_workspace_id': inpaint_input_workspace_id,
            'inpaint_context_mask_image_path': inpaint_context_mask_image_path,
            'inpaint_context_mask_workspace_id': inpaint_context_mask_workspace_id,
            'inpaint_bb_image_path': inpaint_bb_image_path,
            'inpaint_bb_workspace_id': inpaint_bb_workspace_id,
            'inpaint_mask_image_path': inpaint_mask_image_path,
            'inpaint_mask_workspace_id': inpaint_mask_workspace_id,
            'outpaint_bb_mask_data': outpaint_bb_mask_data,
            'outpaint_input_workspace_id': outpaint_input_workspace_id,
            'outpaint_mask_image_path': outpaint_mask_image_path,
            'outpaint_mask_workspace_id': outpaint_mask_workspace_id,
            'outpaint_bb_image_path': outpaint_bb_image_path,
            'outpaint_bb_workspace_id': outpaint_bb_workspace_id,
            'outpaint_prepare_button': outpaint_prepare_button,
            'outpaint_prepare_notice': outpaint_prepare_notice,
            'upscale_refinement_container': upscale_refinement_container,
            'upscale_scale_info': upscale_scale_info,
            'gallery': gallery,
            'seed_random': seed_random,
            'inpaint_tab': inpaint_tab,
            'outpaint_tab': outpaint_tab,
            'remove_tab': remove_tab,
            'remove_bg_enabled': remove_bg_enabled,
            'remove_obj_enabled': remove_obj_enabled,
            'remove_mask_state': remove_mask_state
        }

        if not args_manager.args.disable_preset_selection:
            ui_elements['preset_selection'] = preset_selection

        ui_logic.register_all_events(ctrls_dict, currentTask, ui_elements)


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

# Hijack Gradio's app creation to mount our staging router
import gradio.routes
old_create_app = gradio.routes.App.create_app

@staticmethod
def patched_create_app(*args, **kwargs):
    app = old_create_app(*args, **kwargs)
    from modules.staging_api import staging_router
    from modules.monitor_api import monitor_router
    from modules.image_api import image_router
    app.include_router(staging_router)
    app.include_router(monitor_router)
    app.include_router(image_router)
    return app

gradio.routes.App.create_app = patched_create_app

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



