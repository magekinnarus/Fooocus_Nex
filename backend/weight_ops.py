"""
    Weight Operations for Model Patches
"""

import logging
import torch
from . import resources
from . import float_ops
from . import utils

def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []

def move_weight_functions(m, device):
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory

def calculate_weight(patches, weight, key, intermediate_dtype=None, original_weights=None):
    if intermediate_dtype is None:
        intermediate_dtype = weight.dtype
    for p in patches:
        alpha = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (calculate_weight(v[1:], v[0][1](resources.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True), key, intermediate_dtype=intermediate_dtype), )

        # Standard ComfyUI WeightAdapter support
        import ldm_patched.modules.weight_adapter as weight_adapter
        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, alpha, strength_model, offset, function, intermediate_dtype, original_weights)
            if output is None:
                logging.warning("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            w1 = v[0]
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * resources.cast_to_device(w1, weight.device, weight.dtype)
        elif patch_type == "lora": # Fooocus specific LoRA handling
            mat1 = resources.cast_to_device(v[0], weight.device, intermediate_dtype)
            mat2 = resources.cast_to_device(v[1], weight.device, intermediate_dtype)
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            if v[3] is not None:
                mat3 = resources.cast_to_device(v[3], weight.device, intermediate_dtype)
                final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
            try:
                weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(
                    weight.shape).type(weight.dtype)
            except Exception as e:
                logging.error(f"ERROR {key} {e}")
        elif patch_type == "fooocus":
            w1 = resources.cast_to_device(v[0], weight.device, intermediate_dtype)
            w_min = resources.cast_to_device(v[1], weight.device, intermediate_dtype)
            w_max = resources.cast_to_device(v[2], weight.device, intermediate_dtype)
            w1 = (w1 / 255.0) * (w_max - w_min) + w_min
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} FOOOCUS WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * resources.cast_to_device(w1, weight.device, weight.dtype)
        elif patch_type == "lokr":
            w1 = v[0]
            w2 = v[1]
            w1_a = v[3]
            w1_b = v[4]
            w2_a = v[5]
            w2_b = v[6]
            t2 = v[7]
            dim = None

            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(resources.cast_to_device(w1_a, weight.device, intermediate_dtype),
                              resources.cast_to_device(w1_b, weight.device, intermediate_dtype))
            else:
                w1 = resources.cast_to_device(w1, weight.device, intermediate_dtype)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(resources.cast_to_device(w2_a, weight.device, intermediate_dtype),
                                  resources.cast_to_device(w2_b, weight.device, intermediate_dtype))
                else:
                    w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      resources.cast_to_device(t2, weight.device, intermediate_dtype),
                                      resources.cast_to_device(w2_b, weight.device, intermediate_dtype),
                                      resources.cast_to_device(w2_a, weight.device, intermediate_dtype))
            else:
                w2 = resources.cast_to_device(w2, weight.device, intermediate_dtype)

            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            if v[2] is not None and dim is not None:
                alpha *= v[2] / dim

            try:
                weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                logging.error(f"ERROR {key} {e}")
        elif patch_type == "loha":
            w1a = v[0]
            w1b = v[1]
            if v[2] is not None:
                alpha *= v[2] / w1b.shape[0]
            w2a = v[3]
            w2b = v[4]
            if v[5] is not None:  # cp decomposition
                t1 = v[5]
                t2 = v[6]
                m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  resources.cast_to_device(t1, weight.device, intermediate_dtype),
                                  resources.cast_to_device(w1b, weight.device, intermediate_dtype),
                                  resources.cast_to_device(w1a, weight.device, intermediate_dtype))

                m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  resources.cast_to_device(t2, weight.device, intermediate_dtype),
                                  resources.cast_to_device(w2b, weight.device, intermediate_dtype),
                                  resources.cast_to_device(w2a, weight.device, intermediate_dtype))
            else:
                m1 = torch.mm(resources.cast_to_device(w1a, weight.device, intermediate_dtype),
                              resources.cast_to_device(w1b, weight.device, intermediate_dtype))
                m2 = torch.mm(resources.cast_to_device(w2a, weight.device, intermediate_dtype),
                              resources.cast_to_device(w2b, weight.device, intermediate_dtype))

            try:
                weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                logging.error(f"ERROR {key} {e}")
        elif patch_type == "glora":
            if v[4] is not None:
                alpha *= v[4] / v[0].shape[0]

            a1 = resources.cast_to_device(v[0].flatten(start_dim=1), weight.device, intermediate_dtype)
            a2 = resources.cast_to_device(v[1].flatten(start_dim=1), weight.device, intermediate_dtype)
            b1 = resources.cast_to_device(v[2].flatten(start_dim=1), weight.device, intermediate_dtype)
            b2 = resources.cast_to_device(v[3].flatten(start_dim=1), weight.device, intermediate_dtype)

            weight += ((torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)) * alpha).reshape(weight.shape).type(weight.dtype)
        elif patch_type == "set": # Standard Comfy 'set'
            weight.copy_(v[0])
        elif patch_type == "model_as_lora": # Standard Comfy 'model_as_lora'
             target_weight: torch.Tensor = v[0]
             diff_weight = resources.cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                           resources.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)
             weight += function(alpha * resources.cast_to_device(diff_weight, weight.device, weight.dtype))
        else:
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight

class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches
    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]: #intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return float_ops.stochastic_rounding(calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), intermediate_dtype, seed=string_to_seed(self.key))

        return calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)

def get_key_weight(model, key):
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = utils.get_attr(model, key)
    else:
        op = utils.get_attr(model, op_keys[0])
        try:
            set_func = getattr(op, "set_{}".format(op_keys[1]))
        except AttributeError:
            pass

        try:
            convert_func = getattr(op, "convert_{}".format(op_keys[1]))
        except AttributeError:
            pass

        weight = getattr(op, op_keys[1])
        if convert_func is not None:
            weight = utils.get_attr(model, key)

    return weight, set_func, convert_func
