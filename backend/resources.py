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
    elif config.novram:
        set_vram_to = VRAMState.NO_VRAM
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

total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
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
        if total_vram > (15 * 1024):
            res += 100 * 1024 * 1024
    if config.reserve_vram is not None:
        res = config.reserve_vram * 1024 * 1024 * 1024
    return res

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    free_mem = get_free_memory(device)
    print(f"[Nex-Memory] Requesting {memory_required / (1024**2):.1f} MB. Free: {free_mem / (1024**2):.1f} MB")

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
            print(f"[Nex-Memory] Offloading model to CPU to free {m_size / (1024**2):.1f} MB")
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False, force_high_vram=False):
    cleanup_models_gc()
    global vram_state
    
    current_vram_state = VRAMState.HIGH_VRAM if force_high_vram else vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models = set(models)
    print(f"[Nex-Memory] load_models_gpu: {len(models)} models requested")
    models_to_load = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            current_loaded_models.pop(i).model.detach(unpatch_all=False)

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
        
        lowvram_model_memory = 0
        if vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory
            lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)

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

def soft_empty_cache():
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
