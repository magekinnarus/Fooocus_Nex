import gradio as gr
import copy
import modules.config
import modules.style_sorter as style_sorter
from modules.sdxl_styles import legal_style_names

def build_styles_tab():
    """
    Builds the Styles tab: search bar, style checkboxes, and receiver.
    
    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    style_sorter.try_load_sorted_styles(
        style_names=legal_style_names,
        default_selected=modules.config.default_styles)
    default_selected = [x for x in modules.config.default_styles if x in style_sorter.all_styles]

    results['style_search_bar'] = gr.Textbox(
        show_label=False, container=False,
        placeholder="\U0001F50E Type here to search presets ...",
        value="",
        label='Search Prompt Presets'
    )
    
    results['style_selections'] = gr.CheckboxGroup(
        show_label=False, container=False,
        choices=copy.deepcopy(style_sorter.all_styles),
        value=copy.deepcopy(default_selected),
        label='Selected Prompt Presets',
        elem_classes=['style_selections']
    )

    return results
