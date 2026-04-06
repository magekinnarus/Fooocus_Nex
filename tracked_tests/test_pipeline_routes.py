import os
import sys

import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.flags as flags
from modules.pipeline.routes import build_generation_route, describe_route
from modules.task_state import TaskState


def test_build_generation_route_maps_default_txt2img_path():
    task_state = TaskState()

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == ['prompt_encode', 'diffusion_batch']


def test_build_generation_route_maps_inpaint_family():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='inpaint',
        inpaint_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'inpaint'
    assert describe_route(route) == [
        'image_input_prepare',
        'controlnet_support_load',
        'inpaint_prepare',
        'prompt_encode',
        'diffusion_batch',
    ]


def test_build_generation_route_maps_controlnet_extensions_explicitly():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='ip',
    )
    task_state.add_cn_task(flags.cn_canny, [np.zeros((8, 8, 3), dtype=np.uint8), 1.0, 1.0])

    route = build_generation_route(task_state)

    assert route.route_id == 'txt2img'
    assert describe_route(route) == [
        'image_input_prepare',
        'controlnet_support_load',
        'prompt_encode',
        'structural_controlnet',
        'contextual_controlnet',
        'diffusion_batch',
    ]


def test_build_generation_route_maps_upscale_family():
    task_state = TaskState(
        input_image_checkbox=True,
        current_tab='uov',
        uov_method='super-upscale',
        uov_input_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    route = build_generation_route(task_state)

    assert route.route_id == 'super_upscale'
    assert describe_route(route) == ['image_input_prepare', 'prompt_encode', 'upscale']
