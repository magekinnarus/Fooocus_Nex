# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn).replace('\\', '/')

    if os.path.exists(fn):
        return f'file={web_path}?{os.path.getmtime(fn)}'
    return f'file={web_path}'


def read_asset(fn):
    fn = fn.replace('/', os.sep)
    full_path = os.path.normpath(os.path.join(script_path, fn))
    if not os.path.exists(full_path):
        print(f'[UI] Asset not found: {full_path}')
        return ""
    # print(f'[UI] Loading asset: {full_path}')
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


def javascript_html():
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    
    # Inline all JS files to avoid 404 routing issues in Gradio 5
    js_files = [
        'javascript/script.js',
        'javascript/contextMenus.js',
        'javascript/localization.js',
        'javascript/zoom.js',
        'javascript/edit-attention.js',
        'javascript/viewer.js',
        'javascript/imageviewer.js'
    ]
    
    for js_file in js_files:
        content = read_asset(js_file)
        if content:
            head += f'<script type="text/javascript">{content}</script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'
    if args_manager.args.theme:
        head += f'<script type="text/javascript">set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head


def css_html():
    content = read_asset('css/style.css')
    if content:
        return f'<style>{content}</style>'
    return ""


def reload_javascript():
    # Deprecated in Gradio 5.x. Head injection handled via gr.Blocks(head=...)
    pass
