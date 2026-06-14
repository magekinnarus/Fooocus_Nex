import logging
import torch

_LOADED_CONTROLNETS = {}

def _offload_controlnet(model):
    if model is None:
        return
    patcher = getattr(model, 'control_model_wrapped', None)
    if patcher is not None:
        try:
            patcher.detach()
        except Exception:
            pass

def _destroy_controlnet(model):
    if model is None:
        return
    _offload_controlnet(model)
    try:
        model.cleanup()
    except Exception:
        pass

def apply_controlnet_residency(mode='offload'):
    global _LOADED_CONTROLNETS

    actions = {'mode': mode, 'count': len(_LOADED_CONTROLNETS)}
    if mode == 'destroy':
        stale = list(_LOADED_CONTROLNETS.values())
        _LOADED_CONTROLNETS = {}
        for model in stale:
            _destroy_controlnet(model)
    else:
        for model in _LOADED_CONTROLNETS.values():
            _offload_controlnet(model)
    return actions

@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global _LOADED_CONTROLNETS
    import modules.core as core
    cache = {}
    requested_paths = {p for p in model_paths if p is not None}
    stale_paths = [p for p in _LOADED_CONTROLNETS.keys() if p not in requested_paths]

    for stale_path in stale_paths:
        _destroy_controlnet(_LOADED_CONTROLNETS.pop(stale_path, None))

    for p in model_paths:
        if p is not None:
            if p in _LOADED_CONTROLNETS:
                cache[p] = _LOADED_CONTROLNETS[p]
            else:
                cache[p] = core.load_controlnet(p)
    _LOADED_CONTROLNETS = cache
    return
