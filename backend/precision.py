import torch
import logging
import contextlib

def autocast_context(device, enabled=True):
    """
    Returns a consistent autocast context for the given device.
    Uses 'cuda' for GPU and suppresses for CPU if needed.
    """
    if not enabled:
         return contextlib.nullcontext()
         
    # SDXL generally benefits from autocast (fp16) on GPU.
    # We use GPU device type for autocast.
    try:
        if device.type == 'cuda':
            return torch.autocast("cuda", enabled=True)
    except Exception as e:
        logging.warning(f"[Nex] Autocast setup failed: {e}")
        
    return contextlib.nullcontext()

def cast_unet_inputs(x, timesteps, context=None, y=None, control=None, weight_dtype=None):
    """
    Casts UNet inputs to the specified weight_dtype or detects it from x.
    This prevents per-layer upcasting slowness and dtype mismatch errors.
    Modifies the `control` object IN-PLACE to preserve list exhaustion behavior
    required by `openaimodel.py`.
    """
    if weight_dtype is None:
        if hasattr(x, "dtype"):
            weight_dtype = x.dtype
        else:
            return x, timesteps, context, y, control

    if hasattr(x, "to"):
        x = x.to(weight_dtype)
        
    if timesteps is not None and hasattr(timesteps, "to"):
        timesteps = timesteps.to(weight_dtype)
        
    if context is not None and hasattr(context, "to"):
        context = context.to(weight_dtype)
        
    if y is not None and hasattr(y, "to"):
        y = y.to(weight_dtype)

    if control is not None and isinstance(control, dict):
        # CRITICAL: Modify lists IN-PLACE. The UNet (`openaimodel.py`) consumes
        # these tensors using `.pop()`. Recreating the lists here would leave the
        # original lists fully populated, causing ControlNet to re-apply exponentially
        # over diffusion steps and bloating VRAM.
        for k in control:
            if isinstance(control[k], list):
                for i in range(len(control[k])):
                    c = control[k][i]
                    if hasattr(c, "to"):
                        control[k][i] = c.to(weight_dtype)

    return x, timesteps, context, y, control
