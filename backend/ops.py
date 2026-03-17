import torch
import contextlib

from ldm_patched.modules.ops import cast_bias_weight, disable_weight_init, manual_cast


@contextlib.contextmanager
def use_patched_ops(operations):
    op_names = ['Linear', 'Conv2d', 'Conv3d', 'GroupNorm', 'LayerNorm']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return

