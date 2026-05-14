# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import collections
# import ldm_patched.modules.sd as comfy_sd
import ldm_patched.modules.weight_adapter as weight_adapter
from ldm_patched.modules import utils as comfy_utils
from .. import patching
from ldm_patched.modules import model_management as comfy_model_management

from .ops import GGMLOps
from .loader import gguf_sd_loader
from .dequant import is_quantized, is_torch_compatible

class GGUFModelPatcher(patching.NexModelPatcher):
    patch_on_device = False

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        loaded = super().add_patches(patches, strength_patch=strength_patch, strength_model=strength_model)
        self.gguf_dense_delta_cache = {}
        return loaded

    def _pin_dense_delta(self, delta: torch.Tensor) -> torch.Tensor:
        cpu_delta = delta.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
        try:
            if not cpu_delta.is_pinned():
                cpu_delta = cpu_delta.pin_memory()
        except Exception:
            pass
        return cpu_delta

    def _validate_dense_delta_value(self, key, value):
        if isinstance(value, list):
            raise RuntimeError(f"GGUF dense delta path does not support nested patch payloads for {key}.")

        if isinstance(value, weight_adapter.WeightAdapterBase):
            if isinstance(value, weight_adapter.LoRAAdapter):
                dora_scale = value.weights[4]
                reshape = value.weights[5]
                if dora_scale is not None:
                    raise RuntimeError(f"GGUF dense delta path does not support DoRA LoRA payloads for {key}.")
                if reshape is not None:
                    raise RuntimeError(f"GGUF dense delta path does not support reshape LoRA payloads for {key}.")
                return
            if isinstance(value, weight_adapter.LoHaAdapter):
                if value.weights[7] is not None:
                    raise RuntimeError(f"GGUF dense delta path does not support DoRA LoHa payloads for {key}.")
                return
            if isinstance(value, weight_adapter.LoKrAdapter):
                if value.weights[8] is not None:
                    raise RuntimeError(f"GGUF dense delta path does not support DoRA LoKr payloads for {key}.")
                return
            raise RuntimeError(
                f"GGUF dense delta path does not support adapter family {type(value).__name__} for {key}."
            )

        if isinstance(value, tuple):
            if len(value) == 2 and isinstance(value[0], str):
                patch_type = value[0]
                if patch_type == "diff":
                    return
                raise RuntimeError(
                    f"GGUF dense delta path does not support legacy patch type {patch_type!r} for {key}."
                )
            if len(value) == 1 and isinstance(value[0], torch.Tensor):
                return

        raise RuntimeError(f"GGUF dense delta path cannot normalize payload type {type(value).__name__} for {key}.")

    def _validate_dense_delta_patch(self, key, patch):
        _alpha, value, strength_model, offset, function = patch
        if strength_model != 1.0:
            raise RuntimeError(f"GGUF dense delta path does not support strength_model != 1.0 for {key}.")
        if offset is not None:
            raise RuntimeError(f"GGUF dense delta path does not support offset patches for {key}.")
        if function is not None:
            raise RuntimeError(f"GGUF dense delta path does not support patch functions for {key}.")
        self._validate_dense_delta_value(key, value)

    def _build_dense_delta(self, key, weight):
        cache = getattr(self, "gguf_dense_delta_cache", None)
        if cache is None:
            cache = {}
            self.gguf_dense_delta_cache = cache
        if key in cache:
            return cache[key]

        patches = self.patches.get(key, [])
        for patch in patches:
            self._validate_dense_delta_patch(key, patch)

        try:
            from backend.weight_ops import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight

        dense_delta = torch.zeros(getattr(weight, "tensor_shape", weight.shape), dtype=torch.float32)
        dense_delta = calculate_weight(patches, dense_delta, key, intermediate_dtype=torch.float32)
        dense_delta = self._pin_dense_delta(dense_delta)
        cache[key] = dense_delta
        return dense_delta

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy_utils.get_attr(self.model, key)
        dense_delta = self._build_dense_delta(key, weight)

        if is_quantized(weight):
            out_weight = weight.to(device_to)
            out_weight.patches = []
            out_weight.gguf_dense_delta = dense_delta
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                # OPTIMIZATION: Avoid backup if not doing inplace update and already on offload device
                if inplace_update or weight.device != self.offload_device:
                    self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                        weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                    )

            if device_to is not None:
                temp_weight = comfy_model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = temp_weight + dense_delta.to(device=temp_weight.device, dtype=temp_weight.dtype, non_blocking=True)
            # In Fooocus Nex, float.stochastic_rounding might not be in the same place as ComfyUI
            # We'll use a safer check or skip if missing, as regular rounding is often fine for FLOAT32 -> FLOAT16
            try:
                import ldm_patched.modules.float
                out_weight = ldm_patched.modules.float.stochastic_rounding(out_weight, weight.dtype)
            except (ImportError, AttributeError):
                out_weight = out_weight.to(weight.dtype)

        if inplace_update:
            comfy_utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy_utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
                if hasattr(p, "gguf_dense_delta"):
                    p.gguf_dense_delta = None
            self.gguf_dense_delta_cache = {}
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.gguf_dense_delta_cache = dict(getattr(self, "gguf_dense_delta_cache", {}))
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n
