import os
import math
import cv2
import numpy as np
import fooocus_version
import modules.config as config
import modules.meta_parser as meta_parser
from modules.private_logger import log
from modules.util import HWC3, resize_image


def yield_result(task_state, imgs, progressbar_index, do_not_show_finished_images=False):
    """
    Updates the task results and yields them for the UI.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    task_state.results.extend(imgs)

    if do_not_show_finished_images:
        return

    task_state.yields.append(['results', task_state.results])


def build_image_wall(task_state):
    """
    Creates a grid (image wall) of all generated images in the current task.
    """
    results = []

    if len(task_state.results) < 2:
        return

    for img in task_state.results:
        if isinstance(img, str) and os.path.exists(img):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not isinstance(img, np.ndarray):
            return
        if img.ndim != 3:
            return
        results.append(img)

    H, W, C = results[0].shape

    for img in results:
        Hn, Wn, Cn = img.shape
        if H != Hn or W != Wn or C != Cn:
            return

    cols = int(math.ceil(float(len(results)) ** 0.5))
    rows = int(math.ceil(float(len(results)) / float(cols)))

    wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            idx = y * cols + x
            if idx < len(results):
                img = results[idx]
                wall[y * H:y * H + H, x * W:x * W + W, :] = img

    task_state.results.append(wall)


def save_and_log(task_state, height, width, images, task_dict, use_expansion, loras, persist_image=True):
    """
    Saves the generated images to disk and logs the generation parameters.
    """
    img_paths = []
    for x in images:
        d = [
            ('Prompt', 'prompt', task_dict['log_positive_prompt']),
            ('Negative Prompt', 'negative_prompt', task_dict['log_negative_prompt']),
            ('Styles', 'styles', str(task_dict['styles'])),
            ('Steps', 'steps', task_state.steps),
            ('Resolution', 'resolution', str((width, height))),
            ('Guidance Scale', 'guidance_scale', task_state.cfg_scale),
            ('Sharpness', 'sharpness', task_state.sharpness),
            ('ADM Guidance', 'adm_guidance', str((
                task_state.adm_scaler_positive,
                task_state.adm_scaler_negative,
                task_state.adm_scaler_end))),
            ('Base Model', 'base_model', task_state.base_model_name),
            ('Force CLIP', 'clip_model', task_state.clip_model_name)
        ]

        if task_state.adaptive_cfg != config.default_cfg_tsnr:
            d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', task_state.adaptive_cfg))

        if task_state.clip_skip > 1:
            d.append(('CLIP Skip', 'clip_skip', task_state.clip_skip))
        
        d.append(('Sampler', 'sampler', task_state.sampler_name))
        d.append(('Scheduler', 'scheduler', task_state.scheduler_name))
        d.append(('VAE', 'vae', task_state.vae_name))
        d.append(('Seed', 'seed', str(task_dict['task_seed'])))

        for li, (n, w) in enumerate(loras):
            if n != 'None':
                d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

        metadata_parser_instance = None
        if task_state.save_metadata_to_images:
            metadata_parser_instance = meta_parser.get_metadata_parser(task_state.metadata_scheme)
            metadata_parser_instance.set_data(
                task_dict['log_positive_prompt'], task_dict['positive'],
                task_dict['log_negative_prompt'], task_dict['negative'],
                task_state.steps, task_state.base_model_name,
                loras, task_state.vae_name, task_state.clip_model_name
            )
        
        d.append(('Metadata Scheme', 'metadata_scheme',
                  task_state.metadata_scheme.value if task_state.save_metadata_to_images else task_state.save_metadata_to_images))
        d.append(('Version', 'version', f'{fooocus_version.app_name} {fooocus_version.version}'))
        
        img_paths.append(log(x, d, metadata_parser_instance, task_state.output_format, task_dict, persist_image))

    return img_paths

