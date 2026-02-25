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
    """
    if weight_dtype is None:
        # If not provided, we don't cast (or we could try to detect from a sample weight, 
        # but usually it's passed from the loader/patcher context)
        return x, timesteps, context, y, control

    # Cast all inputs to model precision
    if hasattr(x, "to"):
        x = x.to(weight_dtype)
    
    if timesteps is not None and hasattr(timesteps, "to"):
        timesteps = timesteps.to(weight_dtype)
        
    if context is not None and hasattr(context, "to"):
        context = context.to(weight_dtype)
        
    if y is not None and hasattr(y, "to"):
        y = y.to(weight_dtype)
        
    if control is not None:
        for k in control:
            control[k] = [c.to(weight_dtype) if hasattr(c, "to") else c for c in control[k]]

    return x, timesteps, context, y, control
