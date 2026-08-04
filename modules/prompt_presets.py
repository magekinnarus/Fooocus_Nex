import os
import re
import json
import math

from modules.extra_utils import get_files_from_folder
from random import Random

# cannot use modules.config - validators causing circular imports
prompt_presets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompt_presets/'))


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


prompt_presets = {}
prompt_preset_files = get_files_from_folder(prompt_presets_path, ['.json'])

for x in ['prompt_presets_fooocus.json',
          'prompt_presets_sai.json',
          'prompt_presets_mre.json',
          'prompt_presets_twri.json',
          'prompt_presets_diva.json',
          'prompt_presets_marc_k3nt3l.json']:
    if x in prompt_preset_files:
        prompt_preset_files.remove(x)
        prompt_preset_files.append(x)

for prompt_preset_file in prompt_preset_files:
    try:
        with open(os.path.join(prompt_presets_path, prompt_preset_file), encoding='utf-8') as f:
            for entry in json.load(f):
                name = normalize_key(entry['name'])
                prompt = entry['prompt'] if 'prompt' in entry else ''
                negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                prompt_presets[name] = (prompt, negative_prompt)
    except Exception as e:
        print(str(e))
        print(f'Failed to load prompt preset file {prompt_preset_file}')

prompt_preset_keys = list(prompt_presets.keys())
random_prompt_preset_name = 'Random Preset'
legal_prompt_preset_names = [random_prompt_preset_name] + prompt_preset_keys


def get_random_prompt_preset(rng: Random) -> str:
    return rng.choice(list(prompt_presets.items()))[0]


def apply_prompt_preset(prompt_preset, positive):
    p, n = prompt_presets[prompt_preset]
    return p.replace('{prompt}', positive).splitlines(), n.splitlines(), '{prompt}' in p


def get_words(arrays, total_mult, index):
    if len(arrays) == 1:
        return [arrays[0].split(',')[index]]
    else:
        words = arrays[0].split(',')
        word = words[index % len(words)]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(total_mult / len(words)), index)


def apply_arrays(text, index):
    arrays = re.findall(r'\[\[(.*?)\]\]', text)
    if len(arrays) == 0:
        return text

    print(f'[Arrays] processing: {text}')
    mult = 1
    for arr in arrays:
        words = arr.split(',')
        mult *= len(words)
    
    index %= mult
    chosen_words = get_words(arrays, mult, index)
    
    i = 0
    for arr in arrays:
        text = text.replace(f'[[{arr}]]', chosen_words[i], 1)   
        i = i+1
    
    return text

