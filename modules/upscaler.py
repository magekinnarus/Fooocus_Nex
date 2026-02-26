from collections import OrderedDict

import modules.core as core
import torch
import ldm_patched.modules.model_management
import ldm_patched.modules.utils
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model

# Inlined from ldm_patched.contrib.external_upscale_model
class ImageUpscaleWithModel:
    def upscale(self, upscale_model, image):
        device = ldm_patched.modules.model_management.get_torch_device()

        # upscale_model in Fooocus upscaler is a raw torch model (ESRGAN)
        # We need to handle it slightly differently than the ComfyUI node which expects a wrapper
        
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                # in_img shape is [B, C, H, W]
                # ESRGAN expects [1, C, H, W] if not batched, but here it's likely [1, 3, H, W]
                
                # Note: ESRGAN model doesn't have .scale attribute usually in this context, 
                # but Fooocus ESRGAN is 4x.
                scale = 4 
                
                s = ldm_patched.modules.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=scale)
                oom = False
            except ldm_patched.modules.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

opImageUpscaleWithModel = ImageUpscaleWithModel()

model = None


def perform_upscale(img):
    global model

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        model_filename = downloading_upscale_model()
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img
