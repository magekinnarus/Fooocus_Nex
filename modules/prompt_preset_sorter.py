import os
import gradio as gr
import modules.localization as localization
import json


all_styles = []


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


def try_load_sorted_styles(style_names, default_selected):
    global all_styles

    all_styles = _dedupe_keep_order(style_names or [])

    try:
        if os.path.exists('sorted_styles.json'):
            with open('sorted_styles.json', 'rt', encoding='utf-8') as fp:
                sorted_styles = []
                for x in json.load(fp):
                    if x in all_styles:
                        sorted_styles.append(x)
                for x in all_styles:
                    if x not in sorted_styles:
                        sorted_styles.append(x)
                all_styles = _dedupe_keep_order(sorted_styles)
    except Exception as e:
        print('Load style sorting failed.')
        print(e)

    default_selected = _dedupe_keep_order(default_selected or [])
    default_selected = [x for x in default_selected if x in all_styles]
    unselected = [y for y in all_styles if y not in default_selected]
    all_styles = _dedupe_keep_order(default_selected + unselected)

    return


def sort_styles(selected):
    global all_styles
    selected = _dedupe_keep_order(selected or [])
    selected = [x for x in selected if x in all_styles]
    unselected = [y for y in all_styles if y not in selected]
    sorted_styles = _dedupe_keep_order(selected + unselected)
    try:
        with open('sorted_styles.json', 'wt', encoding='utf-8') as fp:
            json.dump(sorted_styles, fp, indent=4)
    except Exception as e:
        print('Write style sorting failed.')
        print(e)
    all_styles = sorted_styles
    return gr.update(choices=sorted_styles, value=selected)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_styles(selected, query):
    selected = _dedupe_keep_order(selected or [])
    selected = [x for x in selected if x in all_styles]
    query = (query or "").strip()
    unselected = [y for y in all_styles if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_styles = _dedupe_keep_order(matched + selected + unmatched)
    return gr.update(choices=sorted_styles, value=selected)
