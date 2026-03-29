import json
import gradio as gr
from pathlib import Path
import modules.config
import modules.flags
from modules.flags import SAMPLERS
from modules.util import unquote, get_file_from_folder_list
import modules.meta_parser

def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None

def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)

        if key == 'styles':
            h = [s for s in h if s != 'Fooocus V2']

        results.append(h)
    except:
        results.append(gr.update())

def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except:
        results.append(gr.update())

def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        h = min(h, modules.config.default_max_image_number)
        results.append(h)
    except:
        results.append(1)

def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(h)
    except:
        results.append(-1)

def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None, valid_labels=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        width, height = eval(h)
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if valid_labels is None:
            valid_labels = modules.config.available_aspect_ratios_labels
        if formatted in valid_labels:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())

def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if h == 'empty':
            h = modules.config.default_inpaint_engine_version
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except:
        results.append('empty')
        return None

def get_outpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        if h == 'empty':
            h = modules.config.default_outpaint_engine_version
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        results.append(h)
        return h
    except:
        results.append('empty')
        return None

def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_options
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None

def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

def get_lora(key: str, fallback: str | None, source_dict: dict, results: list):
    try:
        split_data = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]

        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]

        # name validation could be added here if needed

        weight = float(weight)
        results.append(enabled)
        results.append(name)
        results.append(weight)
    except:
        results.append(True)
        results.append('None')
        results.append(1)

def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool):
    inpaint_mode = modules.flags.inpaint_option_default

    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    results = []
    base_model_name = loaded_parameter_dict.get('base_model', loaded_parameter_dict.get('Base Model', modules.config.default_base_model_name))
    resolution_labels = modules.config.get_aspect_ratio_labels_for_model(base_model_name)

    get_image_number('image_number', 'Image Number', loaded_parameter_dict, results)
    get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    get_list('styles', 'Styles', loaded_parameter_dict, results)
    get_steps('steps', 'Steps', loaded_parameter_dict, results)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results, valid_labels=resolution_labels)
    get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_number('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, cast_type=int)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    get_str('vae', 'VAE', loaded_parameter_dict, results)
    get_str('clip_model', 'Force CLIP', loaded_parameter_dict, results)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    get_seed('seed', 'Seed', loaded_parameter_dict, results)
    get_outpaint_engine_version('outpaint_engine_version', 'Outpaint Engine Version', loaded_parameter_dict, results)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results)

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    for i in range(modules.config.default_max_lora_number):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results)

    return results

def parse_meta_from_preset(preset_content):
    assert isinstance(preset_content, dict)
    preset_prepared = {}
    items = preset_content

    for settings_key, meta_key in modules.config.possible_preset_keys.items():
        if settings_key == "default_loras":
            loras = getattr(modules.config, settings_key)
            if settings_key in items:
                loras = items[settings_key]
            for index, lora in enumerate(loras[:modules.config.default_max_lora_number]):
                preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
        elif settings_key == "default_aspect_ratio":
            if settings_key in items and items[settings_key] is not None:
                default_aspect_ratio = items[settings_key]
                width, height = default_aspect_ratio.split('*')
            else:
                default_aspect_ratio = getattr(modules.config, settings_key)
                width, height = default_aspect_ratio.split('횞')
                height = height[:height.index(" ")]
            preset_prepared[meta_key] = (width, height)
        else:
            preset_prepared[meta_key] = items[settings_key] if settings_key in items and items[settings_key] is not None else getattr(modules.config, settings_key)

        if settings_key == "default_styles" or settings_key == "default_aspect_ratio":
            preset_prepared[meta_key] = str(preset_prepared[meta_key])

    return preset_prepared

def trigger_metadata_import(file, state_is_generating):

    parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
    if parameters is None:
        print('Could not find metadata in the image!')
        parsed_parameters = {}
    else:
        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
        parsed_parameters = metadata_parser.to_json(parameters)

    return load_parameter_button_click(parsed_parameters, state_is_generating)



