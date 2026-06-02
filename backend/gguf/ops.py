# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import logging
import time
from typing import Optional

from ldm_patched.modules import ops as comfy_ops
from backend import resources as comfy_model_management
from .dequant import dequantize_tensor, is_quantized, dequantize_functions

# --- Workspace Management ---

class GGUFWorkspace:
    """
    Manages reusable VRAM buffers to prevent memory amplification during streaming.
    """
    _instance = None
    
    def __init__(self):
        self.q_buffer: Optional[torch.Tensor] = None # Quantized workspace
        self.max_q_size = 0
        self.device: Optional[torch.device] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_buffers(self, q_size_bytes: int, device: torch.device | str):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(f"GGUF workspace requires a CUDA target device, got {device}.")

        # Reallocate if the workspace needs to grow or follow a different CUDA device.
        if self.q_buffer is None or q_size_bytes > self.max_q_size or self.device != device:
            logging.info(f"[GGUF] Allocating quantized workspace: {q_size_bytes / (1024**2):.2f} MB on {device}")
            self.q_buffer = torch.empty(q_size_bytes, dtype=torch.uint8, device=device)
            self.max_q_size = q_size_bytes
            self.device = device

WORKSPACE = GGUFWorkspace.get_instance()


def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A backward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
            return torch.compiler.disable
        return dummy_decorator

torch_compiler_disable = get_torch_compiler_disable_decorator()

_gguf_trace_stats = {
    "calls": 0,
    "quantized_calls": 0,
    "patch_calls": 0,
    "dequant_seconds": 0.0,
    "patch_seconds": 0.0,
    "total_seconds": 0.0,
    "cpu_calls": 0,
    "cuda_calls": 0,
    "other_device_calls": 0,
    "quantized_cpu_calls": 0,
    "quantized_cuda_calls": 0,
    "quantized_other_device_calls": 0,
    "cpu_dequant_seconds": 0.0,
    "cuda_dequant_seconds": 0.0,
    "other_device_dequant_seconds": 0.0,
    "source_cpu_calls": 0,
    "source_cuda_calls": 0,
    "source_other_calls": 0,
    "source_quantized_cpu_calls": 0,
    "source_quantized_cuda_calls": 0,
    "source_quantized_other_calls": 0,
    "source_pinned_cpu_calls": 0,
    "dequant_cpu_process_seconds": 0.0,
    "by_qtype": {},
    "forward_seconds": 0.0,
    "forward_cpu_process_seconds": 0.0,
    "by_forward_op": {},
    "workspace_pinned_copy_calls": 0,
    "workspace_unpinned_copy_calls": 0,
}


def reset_trace_stats():
    for key in _gguf_trace_stats:
        if key in ("by_qtype", "by_forward_op"):
            _gguf_trace_stats[key] = {}
        else:
            _gguf_trace_stats[key] = 0.0 if key.endswith("seconds") else 0


def consume_trace_stats():
    snapshot = dict(_gguf_trace_stats)
    snapshot["by_qtype"] = dict(_gguf_trace_stats.get("by_qtype", {}))
    snapshot["by_forward_op"] = dict(_gguf_trace_stats.get("by_forward_op", {}))
    reset_trace_stats()
    return snapshot


def _qtype_name(tensor):
    qtype = getattr(tensor, "tensor_type", None)
    return getattr(qtype, "name", str(qtype))


def _record_qtype_stats(qtype_name, *, source_device_type, quantized, wall_seconds, cpu_process_seconds, tensor):
    by_qtype = _gguf_trace_stats.setdefault("by_qtype", {})
    entry = by_qtype.setdefault(qtype_name, {
        "calls": 0,
        "quantized_calls": 0,
        "wall_seconds": 0.0,
        "cpu_process_seconds": 0.0,
        "source_cpu_calls": 0,
        "source_cuda_calls": 0,
        "source_other_calls": 0,
        "elements": 0,
        "data_elements": 0,
    })
    entry["calls"] += 1
    if quantized:
        entry["quantized_calls"] += 1
    entry["wall_seconds"] += wall_seconds
    entry["cpu_process_seconds"] += cpu_process_seconds
    if source_device_type == "cpu":
        entry["source_cpu_calls"] += 1
    elif source_device_type == "cuda":
        entry["source_cuda_calls"] += 1
    else:
        entry["source_other_calls"] += 1
    shape = getattr(tensor, "tensor_shape", getattr(tensor, "shape", ()))
    try:
        element_count = 1
        for dim in shape:
            element_count *= int(dim)
        entry["elements"] += element_count
    except Exception:
        pass
    data = getattr(tensor, "data", None)
    try:
        entry["data_elements"] += int(data.numel())
    except Exception:
        pass


def _record_forward_stats(op_name, *, wall_seconds, cpu_process_seconds):
    _gguf_trace_stats["forward_seconds"] += wall_seconds
    _gguf_trace_stats["forward_cpu_process_seconds"] += cpu_process_seconds
    by_op = _gguf_trace_stats.setdefault("by_forward_op", {})
    entry = by_op.setdefault(op_name, {
        "calls": 0,
        "wall_seconds": 0.0,
        "cpu_process_seconds": 0.0,
    })
    entry["calls"] += 1
    entry["wall_seconds"] += wall_seconds
    entry["cpu_process_seconds"] += cpu_process_seconds

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.gguf_dense_delta = getattr(self, "gguf_dense_delta", None)

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.gguf_dense_delta = getattr(self, "gguf_dense_delta", None)
        new.gguf_pinned_host = getattr(self, "gguf_pinned_host", False)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        wrapped = GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy(),
        )
        wrapped.gguf_dense_delta = getattr(self, "gguf_dense_delta", None)
        wrapped.gguf_pinned_host = getattr(self, "gguf_pinned_host", False)
        return wrapped

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                if isinstance(self, torch.nn.Linear):
                    if v.shape == (self.in_features, self.out_features) and self.in_features != self.out_features:
                        if hasattr(v, "tensor_shape"):
                            v.tensor_shape = torch.Size([self.out_features, self.in_features])
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            destination[prefix + "temp.weight"] = torch.zeros(1, device=torch.device("meta")) # Dummy for ComfyUI



        return

    def get_weight(self, tensor, dtype, device=None):
        if tensor is None:
            return
            
        trace_start = time.perf_counter()
        quantized = is_quantized(tensor)
        qtype_name = _qtype_name(tensor)
        source_device_type = tensor.device.type
        source_pinned_host = bool(getattr(tensor, "gguf_pinned_host", False))
        target_device = torch.device(device) if device is not None else tensor.device

        # Workspace streaming is only valid when the current execution target is CUDA.
        use_workspace = quantized and source_device_type == "cpu" and target_device.type == "cuda"
        
        dequant_start = time.perf_counter()
        dequant_cpu_start = time.process_time()

        if use_workspace:
            # 1. Copy quantized weight to GPU workspace (1x copy)
            q_data = tensor.data.view(torch.uint8)
            flat_q_data = q_data.reshape(-1)
            q_size = q_data.numel()
            WORKSPACE.ensure_buffers(q_size, target_device)
            if source_pinned_host:
                _gguf_trace_stats["source_pinned_cpu_calls"] += 1
                _gguf_trace_stats["workspace_pinned_copy_calls"] += 1
            else:
                _gguf_trace_stats["workspace_unpinned_copy_calls"] += 1
            
            # Zero-copy DMA transfer (if pinned)
            target_q_buffer = WORKSPACE.q_buffer[:q_size]
            target_q_buffer.copy_(flat_q_data, non_blocking=True)
            target_q_view = target_q_buffer.view_as(q_data)
            
            # 2. Dequantize inside GPU (In-place dequant not yet supported by all funcs, so we use dequantize_tensor on the GPU view)
            # We create a temporary GGMLTensor view of the workspace buffer
            q_view = GGMLTensor(
                target_q_view,
                tensor_type=tensor.tensor_type,
                tensor_shape=getattr(tensor, "tensor_shape", tensor.shape)
            )
            weight = dequantize_tensor(q_view, dtype, self.dequant_dtype)
        else:
            # Fallback to standard dequant (naive)
            weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        dequant_cpu_duration = time.process_time() - dequant_cpu_start
        dequant_duration = time.perf_counter() - dequant_start

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        dense_delta = getattr(tensor, "gguf_dense_delta", None)
        if dense_delta is not None:
            # LoRA Delta Streaming
            weight = weight + dense_delta.to(device=weight.device, dtype=weight.dtype, non_blocking=True)

        _gguf_trace_stats["calls"] += 1
        _gguf_trace_stats["dequant_seconds"] += dequant_duration
        _gguf_trace_stats["dequant_cpu_process_seconds"] += dequant_cpu_duration
        _record_qtype_stats(qtype_name, source_device_type=source_device_type, quantized=quantized, wall_seconds=dequant_duration, cpu_process_seconds=dequant_cpu_duration, tensor=tensor)
        _gguf_trace_stats["total_seconds"] += time.perf_counter() - trace_start
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy_model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias, dtype, device=device)
            bias = comfy_ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight, dtype, device=device)
        weight = comfy_ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy_ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy_ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            forward_start = time.perf_counter()
            forward_cpu_start = time.process_time()
            out = torch.nn.functional.linear(input, weight, bias)
            _record_forward_stats(
                "Linear",
                wall_seconds=time.perf_counter() - forward_start,
                cpu_process_seconds=time.process_time() - forward_cpu_start,
            )
            return out

    class Conv2d(GGMLLayer, comfy_ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            forward_start = time.perf_counter()
            forward_cpu_start = time.process_time()
            out = self._conv_forward(input, weight, bias)
            _record_forward_stats(
                "Conv2d",
                wall_seconds=time.perf_counter() - forward_start,
                cpu_process_seconds=time.process_time() - forward_cpu_start,
            )
            return out

    class Embedding(GGMLLayer, comfy_ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            forward_start = time.perf_counter()
            forward_cpu_start = time.process_time()
            out = torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)
            _record_forward_stats(
                "Embedding",
                wall_seconds=time.perf_counter() - forward_start,
                cpu_process_seconds=time.process_time() - forward_cpu_start,
            )
            return out

    class LayerNorm(GGMLLayer, comfy_ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            forward_start = time.perf_counter()
            forward_cpu_start = time.process_time()
            out = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            _record_forward_stats(
                "LayerNorm",
                wall_seconds=time.perf_counter() - forward_start,
                cpu_process_seconds=time.process_time() - forward_cpu_start,
            )
            return out

    class GroupNorm(GGMLLayer, comfy_ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            forward_start = time.perf_counter()
            forward_cpu_start = time.process_time()
            out = torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
            _record_forward_stats(
                "GroupNorm",
                wall_seconds=time.perf_counter() - forward_start,
                cpu_process_seconds=time.process_time() - forward_cpu_start,
            )
            return out

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item



