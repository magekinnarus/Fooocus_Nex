import os
import torch
import time
import math
import ldm_patched.modules.model_base
import ldm_patched.modules.model_management
import modules.anisotropic as anisotropic
import ldm_patched.ldm.modules.attention
import ldm_patched.k_diffusion.sampling
import modules.inpaint_worker as inpaint_worker
import ldm_patched.ldm.modules.diffusionmodules.model
import ldm_patched.modules.sd
import ldm_patched.modules.model_patcher
import ldm_patched.modules.samplers
import ldm_patched.modules.args_parser
import warnings
import safetensors.torch
import ldm_patched.modules.lora
import modules.constants as constants

from ldm_patched.modules.samplers import calc_cond_uncond_batch
from ldm_patched.k_diffusion.sampling import BatchedBrownianTree


# The global patches have been replaced by native backend calls.
# Quality settings are now passed via model_options["quality"] in core.py and sampling.py.

def round_to_64(x):
    h = float(x)
    h = h / 64.0
    h = round(h)
    h = int(h)
    h = h * 64
    return h


def patched_KSamplerX0Inpaint_forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
    if inpaint_worker.current_task is not None:
        latent_processor = self.inner_model.inner_model.process_latent_in
        inpaint_latent = latent_processor(inpaint_worker.current_task.latent).to(x)
        inpaint_mask = inpaint_worker.current_task.latent_mask.to(x)

        if getattr(self, 'energy_generator', None) is None:
            # avoid bad results by using different seeds.
            self.energy_generator = torch.Generator(device='cpu').manual_seed((seed + 1) % constants.MAX_SEED)

        energy_sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(x.shape) - 1))
        current_energy = torch.randn(
            x.size(), dtype=x.dtype, generator=self.energy_generator, device="cpu").to(x) * energy_sigma
        x = x * inpaint_mask + (inpaint_latent + current_energy) * (1.0 - inpaint_mask)

        out = self.inner_model(x, sigma,
                               cond=cond,
                               uncond=uncond,
                               cond_scale=cond_scale,
                               model_options=model_options,
                               seed=seed)

        out = out * inpaint_mask + inpaint_latent * (1.0 - inpaint_mask)
    else:
        out = self.inner_model(x, sigma,
                               cond=cond,
                               uncond=uncond,
                               cond_scale=cond_scale,
                               model_options=model_options,
                               seed=seed)
    return out


def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
    # This patch is largely redundant now as backend/loader.py:patch_controlnet_for_quality
    # handles softness and Timed ADM. We keep it as a placeholder or remove it if unused.
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import timestep_embedding
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)

    guided_hint = self.input_hint_block(hint, emb, context)

    outs = []

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        if guided_hint is not None:
            h = module(h, emb, context)
            h += guided_hint
            guided_hint = None
        else:
            h = module(h, emb, context)
        outs.append(zero_conv(h, emb, context))

    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h, emb, context))

    return outs


def build_loaded(module, loader_name):
    original_loader_name = loader_name + '_origin'

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    def loader(*args, **kwargs):
        result = None
        try:
            result = original_loader(*args, **kwargs)
        except Exception as e:
            result = None
            exp = str(e) + '\n'
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str):
                    if os.path.exists(path):
                        exp += f'File corrupted: {path} \n'
                        corrupted_backup_file = path + '.corrupted'
                        if os.path.exists(corrupted_backup_file):
                            os.remove(corrupted_backup_file)
                        os.replace(path, corrupted_backup_file)
                        if os.path.exists(path):
                            os.remove(path)
                        exp += f'Fooocus has tried to move the corrupted file to {corrupted_backup_file} \n'
                        exp += f'You may try again now and Fooocus will download models again. \n'
            raise ValueError(exp)
        return result

    setattr(module, loader_name, loader)
    return


def patch_all():
    from modules.patch_precision import patch_all_precision
    from modules.patch_clip import patch_all_clip
    import ldm_patched.controlnet.cldm
    import ldm_patched.ldm.modules.diffusionmodules.openaimodel

    if ldm_patched.modules.model_management.directml_enabled:
        ldm_patched.modules.model_management.lowvram_available = True
        ldm_patched.modules.model_management.OOM_EXCEPTION = Exception

    patch_all_precision()
    patch_all_clip()

    ldm_patched.modules.samplers.KSamplerX0Inpaint.forward = patched_KSamplerX0Inpaint_forward

    warnings.filterwarnings(action='ignore', module='torchsde')

    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')

    return
