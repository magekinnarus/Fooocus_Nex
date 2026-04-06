import os

import cv2
import numpy as np
import torch

from backend import resources
from backend import utils as backend_utils
from .mlsd.models import MobileV2_MLSD_Large
from .mlsd.utils import pred_lines
from .teed import TED

DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_MODEL_CACHE = {
    "Depth": {"path": None, "model": None},
    "MistoLine": {"path": None, "model": None},
    "MLSD": {"path": None, "model": None},
}


def _offload_model(model):
    if model is None:
        return
    try:
        model.to(resources.unet_offload_device())
    except Exception:
        pass


def offload_cached_preprocessors():
    for entry in _MODEL_CACHE.values():
        _offload_model(entry["model"])
    resources.soft_empty_cache()


def apply_residency_policy(mode='offload'):
    actions = {'mode': mode, 'count': len(_MODEL_CACHE)}
    for entry in _MODEL_CACHE.values():
        _offload_model(entry['model'])
        if mode == 'destroy':
            entry['model'] = None
            entry['path'] = None
    if mode in ('offload', 'destroy'):
        resources.soft_empty_cache(force=(mode == 'destroy'))
    return actions


def _prepare_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def _get_depth_config(model_path):
    name = os.path.basename(model_path).lower()
    for key, cfg in DEPTH_MODEL_CONFIGS.items():
        if key in name:
            return cfg
    return DEPTH_MODEL_CONFIGS["vitl"]


def _get_cached_model(method, model_path, loader):
    entry = _MODEL_CACHE[method]
    if entry["model"] is None or entry["path"] != model_path:
        _offload_model(entry["model"])
        entry["model"] = loader(model_path)
        entry["path"] = model_path
    return entry["model"]


def _load_depth_model(model_path):
    from .depth_anything_v2 import DepthAnythingV2

    model = DepthAnythingV2(**_get_depth_config(model_path))
    state_dict = _prepare_state_dict(backend_utils.load_torch_file(model_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_teed_model(model_path):
    model = TED()
    state_dict = _prepare_state_dict(backend_utils.load_torch_file(model_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_mlsd_model(model_path):
    model = MobileV2_MLSD_Large()
    state_dict = _prepare_state_dict(backend_utils.load_torch_file(model_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _normalize_depth_input(image):
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def preprocess_depth(image, model_path, input_size=518, max_depth=20.0):
    model = _get_cached_model("Depth", model_path, _load_depth_model)
    device = resources.get_torch_device()
    model = model.to(device)

    with torch.no_grad():
        depth_np = model.infer_image(image, input_size=input_size, max_depth=max_depth)

    depth_np = depth_np.astype(np.float32)
    depth_min = float(depth_np.min())
    depth_max = float(depth_np.max())
    if depth_max > depth_min:
        depth_np = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_np = np.zeros_like(depth_np, dtype=np.float32)

    _offload_model(model)
    result = np.repeat((depth_np.clip(0, 1) * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    return result


def _safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def preprocess_teed(image, model_path):
    model = _get_cached_model("MistoLine", model_path, _load_teed_model)
    device = resources.get_torch_device()
    height, width = image.shape[:2]

    image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad():
        edges = model(image_tensor)

    edges = [edge.detach().cpu().numpy().astype(np.float32)[0, 0] for edge in edges]
    edges = [cv2.resize(edge, (width, height), interpolation=cv2.INTER_LINEAR) for edge in edges]
    edge = 1.0 / (1.0 + np.exp(-np.mean(np.stack(edges, axis=2), axis=2).astype(np.float64)))
    edge = _safe_step(edge, step=2)
    _offload_model(model)
    edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    return np.repeat(edge[..., None], 3, axis=2)


def preprocess_mlsd(image, model_path, score_threshold=0.1, dist_threshold=0.1):
    model = _get_cached_model("MLSD", model_path, _load_mlsd_model)
    device = resources.get_torch_device()
    model = model.to(device)

    result = np.zeros_like(image, dtype=np.uint8)
    with torch.no_grad():
        lines = pred_lines(image, model, [512, 512], score_thr=score_threshold, dist_thr=dist_threshold)
    for line in lines:
        x_start, y_start, x_end, y_end = [int(value) for value in line]
        cv2.line(result, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)

    _offload_model(model)
    return result


def run_structural_preprocessor(method, image, model_path=None):
    if method == "Depth":
        if not model_path:
            raise FileNotFoundError("Depth preprocessor path is missing")
        return preprocess_depth(image, model_path)
    if method == "MistoLine":
        if not model_path:
            raise FileNotFoundError("MistoLine preprocessor path is missing")
        return preprocess_teed(image, model_path)
    if method == "MLSD":
        if not model_path:
            raise FileNotFoundError("MLSD preprocessor path is missing")
        return preprocess_mlsd(image, model_path)
    raise KeyError(f"Unsupported structural preprocessor method: {method}")
