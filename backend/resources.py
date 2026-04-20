"""
    This file is part of Fooocus_Nex.
    Derived from ComfyUI model_management.py.
"""

import psutil
import logging
from enum import Enum
import torch
import sys
import platform
import weakref
import gc
import ctypes
import time
import os
import backend.memory_governor as memory_governor
from backend import environment_profile as environment_profiles

class VRAMState(Enum):
    DISABLED = 0    # No vram present: no need to move models to vram
    NO_VRAM = 1     # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

class ResourcesConfig:
    def __init__(self):
        self.deterministic = False
        self.directml = None
        self.cpu = False
        self.lowvram = False
        self.novram = False
        self.highvram = False
        self.gpu_only = False
        self.reserve_vram = None
        self.disable_smart_memory = False
        self.disable_xformers = False
        self.use_pytorch_cross_attention = False
        self.use_split_cross_attention = False
        self.use_quad_cross_attention = False
        self.supports_fp8_compute = False
        self.fp32_unet = False
        self.fp64_unet = False
        self.bf16_unet = False
        self.fp16_unet = False
        self.fp8_e4m3fn_unet = False
        self.fp8_e5m2_unet = False
        self.fp8_e8m0fnu_unet = False
        self.fp8_e4m3fn_text_enc = False
        self.fp8_e5m2_text_enc = False
        self.fp16_text_enc = False
        self.bf16_text_enc = False
        self.fp32_text_enc = False
        self.cpu_vae = False
        self.fp16_vae = False
        self.bf16_vae = False
        self.fp32_vae = False
        self.force_upcast_attention = False
        self.async_offload = False
        self.force_channels_last = False
        self.use_sage_attention = False
        self.use_flash_attention = False
        self.force_fp16 = False
        self.force_fp32 = False
        self.fast = []
        self.disable_ipex_optimize = False

config = ResourcesConfig()

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0


def _memory_mb(value):
    return float(value) / (1024 ** 2)


def _classify_model_role(model_patcher):
    patcher_name = type(model_patcher).__name__
    model_obj = getattr(model_patcher, "model", None)
    model_name = type(model_obj).__name__ if model_obj is not None else "UnknownModel"

    role = "model"
    if model_obj is not None:
        if hasattr(model_obj, "diffusion_model"):
            role = "unet"
        elif 'controlnet' in model_name.lower():
            role = "controlnet"
        elif model_name == "CLIP" or hasattr(model_obj, "tokenizer"):
            role = "clip"
        elif model_name == "VAE" or hasattr(model_obj, "first_stage_model"):
            role = "vae"

    gguf_suffix = "[gguf]" if "GGUF" in patcher_name else ""
    return role, patcher_name, model_name, gguf_suffix


def _describe_model_for_logs(model_patcher):
    role, patcher_name, model_name, gguf_suffix = _classify_model_role(model_patcher)
    return f"{role}:{patcher_name}/{model_name}{gguf_suffix}"


def describe_model_patcher(model_patcher):
    return _describe_model_for_logs(model_patcher)

def _residency_profile_name():
    profile = memory_governor.environment_profile()
    return getattr(profile, 'name', environment_profiles.PROFILE_CUSTOM)


def _warn_legacy_vram_mode_if_needed(current_vram_state):
    if current_vram_state not in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        return

    warned = getattr(_warn_legacy_vram_mode_if_needed, '_warned', set())
    if current_vram_state.name in warned:
        return

    logging.warning(
        '[Nex-Memory] Legacy VRAM flag behavior (%s) is deprecated; residency policy now follows profile=%s phase=%s with compatibility fallback.',
        current_vram_state.name,
        _residency_profile_name(),
        memory_governor.current_phase(),
    )
    warned = set(warned)
    warned.add(current_vram_state.name)
    _warn_legacy_vram_mode_if_needed._warned = warned


def _residency_plan_for_phase(target_phase=None, task=None):
    phase_name = normalize_memory_phase(target_phase) if target_phase is not None else current_memory_phase()
    return memory_governor.plan_for_task(task=task, phase=phase_name)


def _emit_residency_log(prefix, *, plan, notes=None, role=None, item=None, action=None):
    payload = {
        'profile': plan.notes.get('profile'),
        'phase': plan.notes.get('phase'),
        'pinned': ','.join(plan.pinned) or '-',
        'warm': ','.join(plan.warm) or '-',
        'evictable': ','.join(plan.evictable) or '-',
    }
    if role is not None:
        payload['role'] = role
    if item is not None:
        payload['item'] = item
    if action is not None:
        payload['action'] = action
    if notes:
        payload.update(notes)

    extras = ' '.join(f"{key}={value}" for key, value in payload.items())
    message = f"[Nex-Residency] {prefix} {extras}"
    print(message)
    logging.info(message)


def _eviction_mode_for_resource(plan, resource_id, *, aggressive=False):
    residency_mode = plan.mode_for(resource_id)
    if residency_mode is None or residency_mode == 'pinned':
        return None
    if aggressive:
        return 'destroy'
    if residency_mode == 'warm':
        return None
    profile_name = plan.notes.get('profile')
    if profile_name in (environment_profiles.PROFILE_COLAB_FREE, environment_profiles.PROFILE_LOCAL_LOW_VRAM):
        return 'destroy'
    return 'offload'


def _has_residency_effect(summary):
    if not isinstance(summary, dict):
        return True
    count_keys = (
        'count',
        'contextual_models',
        'clip_vision_models',
        'insightface_apps',
        'eva_clip_models',
        'face_parsers',
    )
    for key in count_keys:
        try:
            if int(summary.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return True
    return not any(key in summary for key in count_keys)

def _apply_support_residency(plan, *, aggressive=False, notes=None):
    actions = {}

    controlnet_action = _eviction_mode_for_resource(plan, 'controlnet', aggressive=aggressive)
    if controlnet_action is not None:
        try:
            import modules.default_pipeline as default_pipeline
            action = default_pipeline.apply_controlnet_residency(controlnet_action)
            if _has_residency_effect(action):
                actions['controlnet'] = action
        except Exception:
            logging.debug('ControlNet residency cleanup failed.', exc_info=True)

    preprocessor_action = _eviction_mode_for_resource(plan, 'structural_preprocessors', aggressive=aggressive)
    if preprocessor_action is not None:
        try:
            from backend.preprocessors import runtime as preprocessor_runtime
            action = preprocessor_runtime.apply_residency_policy(preprocessor_action)
            if _has_residency_effect(action):
                actions['structural_preprocessors'] = action
        except Exception:
            logging.debug('Structural preprocessor residency cleanup failed.', exc_info=True)

    contextual_action = _eviction_mode_for_resource(plan, 'contextual_adapters', aggressive=aggressive)
    clip_vision_action = _eviction_mode_for_resource(plan, 'clip_vision', aggressive=aggressive)
    insightface_action = _eviction_mode_for_resource(plan, 'insightface', aggressive=aggressive)
    if any(action is not None for action in (contextual_action, clip_vision_action, insightface_action)):
        try:
            import backend.ip_adapter as ip_adapter
            action = ip_adapter.apply_contextual_residency(
                contextual_action or 'offload',
                clip_vision_action=clip_vision_action,
                insightface_action=insightface_action,
            )
            if _has_residency_effect(action):
                actions['contextual_adapters'] = action
        except Exception:
            logging.debug('Contextual adapter residency cleanup failed.', exc_info=True)

    pulid_action = _eviction_mode_for_resource(plan, 'pulid_support', aggressive=aggressive)
    if pulid_action is not None:
        try:
            import backend.pulid_runtime as pulid_runtime
            action = pulid_runtime.apply_contextual_residency(pulid_action)
            if _has_residency_effect(action):
                actions['pulid_support'] = action
        except Exception:
            logging.debug('PuLID residency cleanup failed.', exc_info=True)

    if actions:
        action_notes = dict(notes or {})
        action_notes['support_actions'] = actions
        _emit_residency_log('cleanup', plan=plan, notes=action_notes)

    return actions


def get_supported_float8_types():
    float8_types = []
    try:
        float8_types.append(torch.float8_e4m3fn)
    except:
        pass
    try:
        float8_types.append(torch.float8_e4m3fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e8m0fnu)
    except:
        pass
    return float8_types

FLOAT8_TYPES = get_supported_float8_types()

xpu_available = False
torch_version = ""
try:
    torch_version = torch.version.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
    xpu_available = (torch_version_numeric[0] < 2 or (torch_version_numeric[0] == 2 and torch_version_numeric[1] <= 4)) and torch.xpu.is_available()
except:
    pass

lowvram_available = True

# We'll set this later after config is potentially updated
def apply_config():
    global vram_state, set_vram_to, cpu_state, lowvram_available, directml_enabled, directml_device
    
    if config.deterministic:
        logging.info("Using deterministic algorithms for pytorch")
        torch.use_deterministic_algorithms(True, warn_only=True)

    directml_enabled = False
    if config.directml is not None:
        import torch_directml
        directml_enabled = True
        device_index = config.directml
        if device_index < 0:
            directml_device = torch_directml.device()
        else:
            directml_device = torch_directml.device(device_index)
        logging.info("Using directml with device: {}".format(torch_directml.device_name(device_index)))
        lowvram_available = False 

    if config.cpu:
        cpu_state = CPUState.CPU

    if config.lowvram:
        set_vram_to = VRAMState.LOW_VRAM
        lowvram_available = True
        logging.warning('[Nex-Memory] --lowvram compatibility mode is active; prefer memory_environment_profile for stage-aware residency.')
    elif config.novram:
        set_vram_to = VRAMState.NO_VRAM
        logging.warning('[Nex-Memory] --novram compatibility mode is active; prefer memory_environment_profile for stage-aware residency.')
    elif config.highvram or config.gpu_only:
        vram_state = VRAMState.HIGH_VRAM

    if lowvram_available:
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to

    if cpu_state != CPUState.GPU:
        vram_state = VRAMState.DISABLED

    if cpu_state == CPUState.MPS:
        vram_state = VRAMState.SHARED

apply_config()

try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
    _ = torch.xpu.device_count()
    xpu_available = xpu_available or torch.xpu.is_available()
except:
    xpu_available = xpu_available or (hasattr(torch, "xpu") and torch.xpu.is_available())

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

try:
    import torch_npu  # noqa: F401
    _ = torch.npu.device_count()
    npu_available = torch.npu.is_available()
except:
    npu_available = False

try:
    import torch_mlu  # noqa: F401
    _ = torch.mlu.device_count()
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False

try:
    ixuca_available = hasattr(torch, "corex")
except:
    ixuca_available = False

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False

def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False

def is_ixuca():
    global ixuca_available
    if ixuca_available:
        return True
    return False

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024 #TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_npu = torch.npu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_npu
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_mlu = torch.mlu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_mlu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None

total_vram = float(get_total_memory(get_torch_device())) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if config.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False

def is_amd():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.hip:
            return True
    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if config.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

try:
    if is_nvidia():
        if torch_version_numeric[0] >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and config.use_split_cross_attention == False and config.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
    if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca():
        if config.use_split_cross_attention == False and config.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        elif device.type == "xpu":
            return "{} {}".format(device, torch.xpu.get_device_name(device))
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    elif is_ascend_npu():
        return "{} {}".format(device, torch.npu.get_device_name(device))
    elif is_mlu():
        return "{} {}".format(device, torch.mlu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

current_loaded_models = []
_last_soft_empty_cache = 0.0

class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
        real_model = self.model.model

        if is_intel_xpu() and not config.disable_ipex_optimize and 'ipex' in globals() and real_model is not None:
            with torch.no_grad():
                real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        if self.model_finalizer:
            self.model_finalizer.detach()
            self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model and self.real_model() is not None and self.model is None

def extra_reserved_memory():
    res = 400 * 1024 * 1024
    if any(platform.win32_ver()):
        res = 600 * 1024 * 1024
        # OPTIMIZATION: Reduce reserve on low-VRAM cards to maximize resident space
        if total_vram < 4096:
            res = 250 * 1024 * 1024
        elif total_vram > (15 * 1024):
            res += 100 * 1024 * 1024
    if config.reserve_vram is not None:
        res = config.reserve_vram * 1024 * 1024 * 1024
    return res

def minimum_inference_memory():
    # OPTIMIZATION: More aggressive for 3GB cards
    if total_vram < 4096:
        return (1024 * 1024 * 1024) * 0.4 + extra_reserved_memory()
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    free_mem = get_free_memory(device)
    logging.info(f"[Nex-Memory] Requesting {memory_required / (1024**2):.1f} MB. Free: {free_mem / (1024**2):.1f} MB")

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not config.disable_smart_memory:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        
        if current_loaded_models[i].model_unload(memory_to_free):
            m_size = current_loaded_models[i].model_memory()
            logging.info(f"[Nex-Memory] Offloading model to CPU to free {m_size / (1024**2):.1f} MB")
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False, force_high_vram=False, target_phase=None):
    cleanup_models_gc()
    global vram_state
    
    current_vram_state = VRAMState.HIGH_VRAM if force_high_vram else vram_state
    _warn_legacy_vram_mode_if_needed(current_vram_state)
    residency_plan = _residency_plan_for_phase(target_phase=target_phase)

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models = set(models)
    logging.info(f"[Nex-Memory] load_models_gpu: {len(models)} models requested")
    models_to_load = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models.pop(loaded_model_index)
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            models_to_load.append(loaded_model)
    
    load_device = get_torch_device()

    def _needs_patching(m_wrapper):
        patcher = m_wrapper.model
        model_obj = getattr(patcher, "model", None)
        if model_obj is None:
            return False
        current_uuid = getattr(model_obj, "current_weight_patches_uuid", None)
        return patcher.patches_uuid != current_uuid

    models_to_load = [m for m in models_to_load if m.model.current_loaded_device() != load_device or force_patch_weights or _needs_patching(m)]
    
    if len(models_to_load) == 0:
        return

    start_time = time.time()
    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            old_p = current_loaded_models.pop(i).model
            # We want to unpatch if the model is currently using this patcher's weights
            model_obj = getattr(old_p, "model", None)
            is_current = model_obj is not None and old_p.patches_uuid == getattr(model_obj, "current_weight_patches_uuid", None)
            old_p.detach(unpatch_all=is_current)

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            if current_vram_state != VRAMState.HIGH_VRAM:
                free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = current_vram_state

        model_role, _, _, _ = _classify_model_role(model)
        residency_mode = residency_plan.mode_for(model_role) or 'evictable'
        profile_name = residency_plan.notes.get('profile')
        pinned_full_load = (
            residency_mode == 'pinned'
            and not force_high_vram
            and vram_set_state in (VRAMState.NORMAL_VRAM, VRAMState.HIGH_VRAM)
            and profile_name not in (environment_profiles.PROFILE_COLAB_FREE, environment_profiles.PROFILE_LOCAL_LOW_VRAM)
        )
        effective_force_full_load = force_full_load or pinned_full_load

        lowvram_model_memory = 0
        loaded_memory = loaded_model.model_loaded_memory()
        current_free_mem = None
        if vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not effective_force_full_load:
            current_free_mem = get_free_memory(torch_dev) + loaded_memory
            lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            if total_vram < 4096:
                # Grant more headroom for quantized UNet
                lowvram_model_memory = max(lowvram_model_memory, 256 * 1024 * 1024)
            lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        model_label = _describe_model_for_logs(model)
        target_memory = loaded_model.model_memory_required(torch_dev)
        load_mode = 'partial' if lowvram_model_memory > 0 and lowvram_model_memory < target_memory else 'full'
        _emit_residency_log(
            'load_plan',
            plan=residency_plan,
            role=model_role,
            item=model_label,
            action=load_mode,
            notes={'full_load': effective_force_full_load, 'legacy_vram': vram_set_state.name},
        )
        free_mem_text = 'n/a' if current_free_mem is None else f"{_memory_mb(current_free_mem):.1f}"
        perf_message = (
            f"[Nex-Perf] load_models_gpu item={model_label} device={torch_dev} mode={load_mode} "
            f"vram_state={vram_set_state.name} target={_memory_mb(target_memory):.1f}MB "
            f"loaded={_memory_mb(loaded_memory):.1f}MB budget={_memory_mb(lowvram_model_memory):.1f}MB "
            f"free={_memory_mb(extra_mem):.1f}MB min_req={_memory_mb(minimum_memory_required):.1f}MB "
            f"current_free={free_mem_text}MB"
        )
        print(perf_message)
        logging.info(perf_message)

        model_load_start = time.perf_counter()
        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        model_load_duration = time.perf_counter() - model_load_start
        current_loaded_models.insert(0, loaded_model)

        lowvram_patch_counter = model.lowvram_patch_counter() if hasattr(model, 'lowvram_patch_counter') else 0
        perf_message = (
            f"[Nex-Perf] load_models_gpu complete item={model_label} "
            f"loaded_now={_memory_mb(loaded_model.model_loaded_memory()):.1f}MB "
            f"total={_memory_mb(loaded_model.model_memory()):.1f}MB "
            f"lowvram_patches={lowvram_patch_counter} duration={model_load_duration:.3f}s"
        )
        print(perf_message)
        logging.info(perf_message)

    load_time = time.time() - start_time
    logging.info(f"Nex Model Loading Time: {load_time:.2f} seconds")
    return load_time

def cleanup_models_gc():
    do_gc = False
    for cur in current_loaded_models:
        if cur.is_dead():
            do_gc = True
            break
    if do_gc:
        gc.collect()
        soft_empty_cache()

def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete
    for i in to_delete:
        del current_loaded_models[i]


def loaded_model_state():
    cleanup_models_gc()
    state = []

    for loaded_model in current_loaded_models:
        model_patcher = loaded_model.model
        if model_patcher is None:
            continue

        model_obj = getattr(model_patcher, "model", None)
        role, patcher_name, model_name, gguf_suffix = _classify_model_role(model_patcher)

        try:
            current_device = model_patcher.current_loaded_device()
        except Exception:
            current_device = getattr(model_obj, "device", None)

        try:
            loaded_memory_mb = _memory_mb(loaded_model.model_loaded_memory())
        except Exception:
            loaded_memory_mb = None

        try:
            total_memory_mb = _memory_mb(loaded_model.model_memory())
        except Exception:
            total_memory_mb = None

        lowvram_patch_counter = None
        if hasattr(model_patcher, "lowvram_patch_counter"):
            try:
                lowvram_patch_counter = model_patcher.lowvram_patch_counter()
            except Exception:
                lowvram_patch_counter = None

        state.append({
            "label": _describe_model_for_logs(model_patcher),
            "role": role,
            "patcher_class": patcher_name,
            "model_class": model_name,
            "gguf": bool(gguf_suffix),
            "load_device": str(getattr(model_patcher, "load_device", None)),
            "offload_device": str(getattr(model_patcher, "offload_device", None)),
            "current_loaded_device": str(current_device),
            "currently_used": bool(getattr(loaded_model, "currently_used", False)),
            "loaded_memory_mb": loaded_memory_mb,
            "total_memory_mb": total_memory_mb,
            "model_lowvram": bool(getattr(model_obj, "model_lowvram", False)),
            "lowvram_patch_counter": lowvram_patch_counter,
            "current_weight_patches_uuid": str(getattr(model_obj, "current_weight_patches_uuid", None)),
        })

    return state

def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def soft_empty_cache(force=False):
    if not memory_governor.should_flush_cache(force=force):
        return

    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    memory_governor.note_cache_flush()


def _try_malloc_trim():
    if platform.system() != 'Linux':
        return False

    for library_name in ('libc.so.6', 'libc.so'):
        try:
            libc = ctypes.CDLL(library_name)
        except OSError:
            continue

        trim = getattr(libc, 'malloc_trim', None)
        if trim is None:
            continue

        try:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            return bool(trim(0))
        except Exception:
            logging.debug('malloc_trim call failed.', exc_info=True)
            return False

    return False


def cleanup_memory(reason, *, unload_models=False, force_cache=False, gc_collect=True, trim_host=None, notes=None, target_phase=None, task=None):
    cleanup_notes = dict(notes or {})
    cleanup_notes['reason'] = reason
    cleanup_phase = normalize_memory_phase(target_phase) if target_phase is not None else current_memory_phase()
    cleanup_notes['target_phase'] = cleanup_phase
    before = capture_memory_snapshot(notes={**cleanup_notes, 'stage': 'before_cleanup'})
    residency_plan = _residency_plan_for_phase(target_phase=cleanup_phase, task=task)

    if unload_models:
        unload_all_models()

    support_actions = _apply_support_residency(
        residency_plan,
        aggressive=bool(unload_models or force_cache),
        notes={'reason': reason, 'target_phase': cleanup_phase},
    )

    if gc_collect:
        gc.collect()

    soft_empty_cache(force=force_cache or unload_models or bool(support_actions))

    if trim_host is None:
        trim_host = memory_governor.should_trim_host_memory(snapshot=before, aggressive=bool(unload_models and force_cache))
    trimmed = _try_malloc_trim() if trim_host else False

    after = capture_memory_snapshot(notes={
        **cleanup_notes,
        'stage': 'after_cleanup',
        'trimmed': trimmed,
        'unload_models': bool(unload_models),
        'support_actions': support_actions,
    })

    logging.info(
        '[Nex-Memory] cleanup reason=%s unload_models=%s force_cache=%s target_phase=%s trimmed=%s '
        'ram_before=%sMB ram_after=%sMB vram_before=%sMB vram_after=%sMB',
        reason,
        unload_models,
        force_cache,
        cleanup_phase,
        trimmed,
        'n/a' if before.free_ram_mb is None else f'{before.free_ram_mb:.1f}',
        'n/a' if after.free_ram_mb is None else f'{after.free_ram_mb:.1f}',
        'n/a' if before.free_vram_mb is None else f'{before.free_vram_mb:.1f}',
        'n/a' if after.free_vram_mb is None else f'{after.free_vram_mb:.1f}',
    )
    return after


def prepare_for_checkpoint_switch(*, current_model=None, next_model=None, release_callback=None, notes=None):
    affordance = memory_governor.can_afford(
        minimum_free_ram_mb=memory_governor.governor.policy.checkpoint_switch_ram_headroom_mb,
        phase=memory_governor.MemoryPhase.MODEL_REFRESH,
        notes=notes,
    )
    aggressive = memory_governor.governor.policy.aggressive_checkpoint_switch_reclaim or not affordance.allowed

    logging.info(
        '[Nex-Memory] checkpoint_switch current=%s next=%s allowed=%s aggressive=%s detail=%s',
        current_model,
        next_model,
        affordance.allowed,
        aggressive,
        affordance.reason,
    )

    if release_callback is not None:
        release_callback()

    return cleanup_memory(
        'checkpoint_switch',
        unload_models=True,
        force_cache=True,
        gc_collect=True,
        trim_host=aggressive,
        target_phase=MemoryPhase.MODEL_REFRESH,
        notes={
            'current_model': current_model,
            'next_model': next_model,
            'affordance_allowed': affordance.allowed,
            'affordance_reason': affordance.reason,
            **(notes or {}),
        },
    )


MemoryPhase = memory_governor.MemoryPhase

def normalize_memory_phase(phase):
    return memory_governor.normalize_phase(phase)

def memory_phase_scope(phase, task=None, notes=None, end_notes=None):
    return memory_governor.phase_scope(phase, task=task, notes=notes, end_notes=end_notes)

def begin_memory_phase(phase, task=None, notes=None):
    return memory_governor.begin_phase(phase, task=task, notes=notes)


def end_memory_phase(phase=None, notes=None):
    return memory_governor.end_phase(phase, notes=notes)


def capture_memory_snapshot(notes=None, task=None):
    return memory_governor.capture_snapshot(notes=notes, task=task)


def current_memory_phase():
    return memory_governor.current_phase()

def active_memory_environment_profile():
    return memory_governor.environment_profile()

def memory_policy_summary():
    return memory_governor.policy_summary()

def memory_can_afford(**kwargs):
    return memory_governor.can_afford(**kwargs)

def is_device_type(device, type):
    if hasattr(device, 'type'):
        return device.type == type
    return False

def is_device_cpu(device): return is_device_type(device, 'cpu')
def is_device_mps(device): return is_device_type(device, 'mps')
def is_device_xpu(device): return is_device_type(device, 'xpu')
def is_device_cuda(device): return is_device_type(device, 'cuda')

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device): return False
    if config.force_fp16: return True
    if config.force_fp32: return False
    if cpu_state == CPUState.CPU: return False
    
    # Simplified logic for SDXL focus
    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device): return False
    if config.force_fp32: return False
    return torch.cuda.is_bf16_supported()

def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_device():
    if config.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    if config.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unload_all_models():
    free_memory(1e30, get_torch_device())

def eject_model(model_patcher):
    model_patcher.detach()
    soft_empty_cache()

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

def maximum_vram_for_weights(device=None):
    return (float(get_total_memory(device)) * 0.88 - minimum_inference_memory())

def supports_fp8_compute(device=None):
    if getattr(config, 'supports_fp8_compute', False):
        return True

    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if torch_version_numeric < (2, 3):
        return False

    if any(platform.win32_ver()):
        if torch_version_numeric < (2, 4):
            return False

    return True

def unet_dtype(device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if model_params < 0:
        model_params = 1000000000000000000000
    
    if config.bf16_unet:
        return torch.bfloat16
    if config.fp16_unet:
        return torch.float16
    if config.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if config.fp8_e5m2_unet:
        return torch.float8_e5m2
    
    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype

    if fp8_dtype is not None:
        if supports_fp8_compute(device):
            return fp8_dtype

        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
        return torch.float16
    if torch.bfloat16 in supported_dtypes and should_use_bf16(device, model_params=model_params):
        return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32

def is_directml_enabled():
    global directml_enabled
    if 'directml_enabled' in globals() and directml_enabled:
        return True
    return False

def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return True
    if config.deterministic:
        return False
    if is_directml_enabled():
        return False
    return True

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            with stream:
                return weight.to(dtype=dtype, copy=copy)
        return weight.to(dtype=dtype, copy=copy)

    if stream is not None:
        with stream:
            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()
interrupt_processing = False

def interrupt_current_processing(value=True):
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
