import os
import gradio as gr
import json


all_prompt_presets = []


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def try_load_sorted_prompt_presets(prompt_preset_names, default_selected):
    global all_prompt_presets

    all_prompt_presets = _dedupe_keep_order(prompt_preset_names or [])

    try:
        if os.path.exists('sorted_prompt_presets.json'):
            with open('sorted_prompt_presets.json', 'rt', encoding='utf-8') as fp:
                sorted_prompt_presets = []
                for x in json.load(fp):
                    if x in all_prompt_presets:
                        sorted_prompt_presets.append(x)
                for x in all_prompt_presets:
                    if x not in sorted_prompt_presets:
                        sorted_prompt_presets.append(x)
                all_prompt_presets = _dedupe_keep_order(sorted_prompt_presets)
    except Exception as e:
        print('Load prompt preset sorting failed.')
        print(e)

    default_selected = _dedupe_keep_order(default_selected or [])
    default_selected = [x for x in default_selected if x in all_prompt_presets]
    unselected = [y for y in all_prompt_presets if y not in default_selected]
    all_prompt_presets = _dedupe_keep_order(default_selected + unselected)

    return


def sort_prompt_presets(selected):
    global all_prompt_presets
    selected = _dedupe_keep_order(selected or [])
    selected = [x for x in selected if x in all_prompt_presets]
    unselected = [y for y in all_prompt_presets if y not in selected]
    sorted_prompt_presets = _dedupe_keep_order(selected + unselected)
    try:
        with open('sorted_prompt_presets.json', 'wt', encoding='utf-8') as fp:
            json.dump(sorted_prompt_presets, fp, indent=4)
    except Exception as e:
        print('Write prompt preset sorting failed.')
        print(e)
    all_prompt_presets = sorted_prompt_presets
    return gr.update(choices=sorted_prompt_presets, value=selected)


def search_prompt_presets(selected, query):
    selected = _dedupe_keep_order(selected or [])
    selected = [x for x in selected if x in all_prompt_presets]
    query = (query or "").strip()
    unselected = [y for y in all_prompt_presets if y not in selected]
    matched = [y for y in unselected if query.lower() in y.lower()] if len(query) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_prompt_presets = _dedupe_keep_order(matched + selected + unmatched)
    return gr.update(choices=sorted_prompt_presets, value=selected)
