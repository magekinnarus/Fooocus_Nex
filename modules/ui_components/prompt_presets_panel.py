import gradio as gr
import copy
import modules.config
import modules.prompt_preset_sorter as prompt_preset_sorter
from modules.prompt_presets import legal_prompt_preset_names

def build_prompt_presets_tab():
    """
    Builds the Prompt Presets tab: search bar, preset checkboxes, and receiver.

    Returns:
        dict: Gradio components mapping name to instance.
    """
    results = {}

    prompt_preset_sorter.try_load_sorted_prompt_presets(
        prompt_preset_names=legal_prompt_preset_names,
        default_selected=modules.config.default_prompt_presets)
    default_selected = [x for x in modules.config.default_prompt_presets if x in prompt_preset_sorter.all_prompt_presets]

    results['prompt_preset_search_bar'] = gr.Textbox(
        show_label=False, container=False,
        placeholder="\U0001F50E Type here to search presets ...",
        value="",
        label='Search Prompt Presets'
    )

    results['prompt_preset_selections'] = gr.CheckboxGroup(
        show_label=False, container=False,
        choices=copy.deepcopy(prompt_preset_sorter.all_prompt_presets),
        value=copy.deepcopy(default_selected),
        label='Selected Prompt Presets',
        elem_classes=['prompt_preset_selections']
    )

    return results
