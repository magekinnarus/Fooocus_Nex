"""
gr.Image() component hijack for Gradio 5.
This file is a bridge to allow Fooocus_Nex to launch on Gradio 5.
"""

from __future__ import annotations
import numpy as np
import gradio as gr
from typing import Any, Literal
from PIL import Image as _Image
_Image.init()

class Image(gr.Image):
    """
    Gradio 5 compatible shim for the Fooocus Image component.
    Legacy parameters are stripped to prevent gr.Image initialization errors.
    """
    def __init__(
        self,
        value: Any = None,
        *,
        source: Literal["upload", "webcam", "canvas"] = "upload",
        tool: Literal["editor", "select", "sketch", "color-sketch"] | None = None,
        brush_radius: float | None = None,
        brush_color: str = "#000000",
        mask_opacity: float = 0.7,
        **kwargs,
    ):
        # Store legacy properties for Fooocus logic reference
        self.source = source
        self.tool = tool
        self.brush_radius = brush_radius
        self.brush_color = brush_color
        self.mask_opacity = mask_opacity
        
        # Strip parameters that Gradio 5 gr.Image doesn't accept
        legacy_params = [
            'shape', 'invert_colors', 'source', 'tool', 
            'brush_radius', 'brush_color', 'mask_opacity', 
            'streaming', 'mirror_webcam', 'image_mode'
        ]
        for p in legacy_params:
            kwargs.pop(p, None)
            
        super().__init__(value=value, **kwargs)

    def style(self, **kwargs):
        # Legacy method from Gradio 3
        return self

all_components = []

# Hook into Gradio Component initialization to track all created components for localization.
# In Gradio 5, we hook into gr.components.base.Component.
if not hasattr(gr.components.base.Component, 'original_init'):
    gr.components.base.Component.original_init = gr.components.base.Component.__init__
    
    def patched_init(self, *args, **kwargs):
        all_components.append(self)
        return self.original_init(*args, **kwargs)
        
    gr.components.base.Component.__init__ = patched_init


import gradio.routes
import importlib
gradio.routes.asyncio = importlib.reload(gradio.routes.asyncio)

if not hasattr(gradio.routes.asyncio, 'original_wait_for'):
    gradio.routes.asyncio.original_wait_for = gradio.routes.asyncio.wait_for


def patched_wait_for(fut, timeout):
    del timeout
    return gradio.routes.asyncio.original_wait_for(fut, timeout=65535)


gradio.routes.asyncio.wait_for = patched_wait_for

