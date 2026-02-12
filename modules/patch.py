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


class PatchSettings:
    def __init__(self,
                 sharpness=2.0,
                 adm_scaler_end=0.3,
                 positive_adm_scale=1.5,
                 negative_adm_scale=0.8,
                 controlnet_softness=0.25,
                 adaptive_cfg=7.0):
        self.sharpness = sharpness
        self.adm_scaler_end = adm_scaler_end
        self.positive_adm_scale = positive_adm_scale
        self.negative_adm_scale = negative_adm_scale
        self.controlnet_softness = controlnet_softness
        self.adaptive_cfg = adaptive_cfg
        self.global_diffusion_progress = 0
        self.eps_record = None


patch_settings = {}



class BrownianTreeNoiseSamplerPatched:
    transform = None
    tree = None

    @staticmethod
    def global_init(x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        if ldm_patched.modules.model_management.directml_enabled:
            cpu = True

        t0, t1 = transform(torch.as_tensor(sigma_min)), transform(torch.as_tensor(sigma_max))

        BrownianTreeNoiseSamplerPatched.transform = transform
        BrownianTreeNoiseSamplerPatched.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(sigma, sigma_next):
        transform = BrownianTreeNoiseSamplerPatched.transform
        tree = BrownianTreeNoiseSamplerPatched.tree

        t0, t1 = transform(torch.as_tensor(sigma)), transform(torch.as_tensor(sigma_next))
        return tree(t0, t1) / (t1 - t0).abs().sqrt()


def compute_cfg(uncond, cond, cfg_scale, t):
    pid = os.getpid()
    mimic_cfg = float(patch_settings[pid].adaptive_cfg)
    real_cfg = float(cfg_scale)

    real_eps = uncond + real_cfg * (cond - uncond)

    if cfg_scale > patch_settings[pid].adaptive_cfg:
        mimicked_eps = uncond + mimic_cfg * (cond - uncond)
        return real_eps * t + mimicked_eps * (1 - t)
    else:
        return real_eps


def patched_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
    pid = os.getpid()

    if math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        final_x0 = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)[0]

        if patch_settings[pid].eps_record is not None:
            patch_settings[pid].eps_record = ((x - final_x0) / timestep).cpu()

        return final_x0

    positive_x0, negative_x0 = calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)

    positive_eps = x - positive_x0
    negative_eps = x - negative_x0

    alpha = 0.001 * patch_settings[pid].sharpness * patch_settings[pid].global_diffusion_progress

    if alpha < 0.01:
        positive_eps_degraded_weighted = positive_eps
    else:
        positive_eps_degraded = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=positive_x0)
        positive_eps_degraded_weighted = positive_eps_degraded * alpha + positive_eps * (1.0 - alpha)

    final_eps = compute_cfg(uncond=negative_eps, cond=positive_eps_degraded_weighted,
                            cfg_scale=cond_scale, t=patch_settings[pid].global_diffusion_progress)

    if patch_settings[pid].eps_record is not None:
        patch_settings[pid].eps_record = (final_eps / timestep).cpu()

    return x - final_eps


def round_to_64(x):
    h = float(x)
    h = h / 64.0
    h = round(h)
    h = int(h)
    h = h * 64
    return h


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = ldm_patched.modules.model_base.sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)
    target_width = width
    target_height = height
    pid = os.getpid()

    if kwargs.get("prompt_type", "") == "negative":
        width = float(width) * patch_settings[pid].negative_adm_scale
        height = float(height) * patch_settings[pid].negative_adm_scale
    elif kwargs.get("prompt_type", "") == "positive":
        width = float(width) * patch_settings[pid].positive_adm_scale
        height = float(height) * patch_settings[pid].positive_adm_scale

    def embedder(number_list):
        h = self.embedder(torch.tensor(number_list, dtype=torch.float32))
        h = torch.flatten(h).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return h

    width, height = int(width), int(height)
    target_width, target_height = round_to_64(target_width), round_to_64(target_height)

    adm_emphasized = embedder([height, width, 0, 0, target_height, target_width])
    adm_consistent = embedder([target_height, target_width, 0, 0, target_height, target_width])

    clip_pooled = clip_pooled.to(adm_emphasized)
    final_adm = torch.cat((clip_pooled, adm_emphasized, clip_pooled, adm_consistent), dim=1)

    return final_adm


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


def timed_adm(y, timesteps):
    if isinstance(y, torch.Tensor) and int(y.dim()) == 2 and int(y.shape[1]) == 5632:
        y_mask = (timesteps > 999.0 * (1.0 - float(patch_settings[os.getpid()].adm_scaler_end))).to(y)[..., None]
        y_with_adm = y[..., :2816].clone()
        y_without_adm = y[..., 2816:].clone()
        return y_with_adm * y_mask + y_without_adm * (1.0 - y_mask)
    return y


def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import timestep_embedding
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)
    pid = os.getpid()

    guided_hint = self.input_hint_block(hint, emb, context)

    y = timed_adm(y, timesteps)

    outs = []

    hs = []
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

    if patch_settings[pid].controlnet_softness > 0:
        for i in range(10):
            k = 1.0 - float(i) / 9.0
            outs[i] = outs[i] * (1.0 - patch_settings[pid].controlnet_softness * k)

    return outs


def patched_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    # Detect the target precision from the first weight
    weight_dtype = self.input_blocks[0][0].weight.dtype
    
    # Cast all inputs to model precision to prevent per-layer upcasting slowness and dtype mismatch errors
    x = x.to(weight_dtype)
    if context is not None:
        context = context.to(weight_dtype)
    if y is not None:
        y = y.to(weight_dtype)
    if control is not None:
        for k in control:
            control[k] = [c.to(weight_dtype) if hasattr(c, "to") else c for c in control[k]]

    self.current_step = 1.0 - timesteps.to(x) / 999.0
    patch_settings[os.getpid()].global_diffusion_progress = float(self.current_step.detach().cpu().numpy().tolist()[0])

    y = timed_adm(y, timesteps)

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
    image_only_indicator = kwargs.get("image_only_indicator", self.default_image_only_indicator)
    time_context = kwargs.get("time_context", None)

    assert (y is not None) == (
            self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    from ldm_patched.ldm.modules.diffusionmodules.openaimodel import timestep_embedding, forward_timestep_embed, apply_control
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'input')
        if "input_block_patch" in transformer_patches:
            patch = transformer_patches["input_block_patch"]
            for p in patch:
                h = p(h, transformer_options)

        hs.append(h)
        if "input_block_patch_after_skip" in transformer_patches:
            patch = transformer_patches["input_block_patch_after_skip"]
            for p in patch:
                h = p(h, transformer_options)

    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
    h = apply_control(h, control, 'middle')

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        hsp = apply_control(hsp, control, 'output')

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)


def patched_load_models_gpu(*args, **kwargs):
    execution_start_time = time.perf_counter()
    y = ldm_patched.modules.model_management.load_models_gpu_origin(*args, **kwargs)
    moving_time = time.perf_counter() - execution_start_time
    if moving_time > 0.1:
        print(f'[Fooocus Model Management] Moving model(s) has taken {moving_time:.2f} seconds')
    return y


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

    if not hasattr(ldm_patched.modules.model_management, 'load_models_gpu_origin'):
        ldm_patched.modules.model_management.load_models_gpu_origin = ldm_patched.modules.model_management.load_models_gpu

    ldm_patched.modules.model_management.load_models_gpu = patched_load_models_gpu
    ldm_patched.controlnet.cldm.ControlNet.forward = patched_cldm_forward
    ldm_patched.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = patched_unet_forward
    ldm_patched.modules.model_base.SDXL.encode_adm = sdxl_encode_adm_patched
    ldm_patched.modules.samplers.KSamplerX0Inpaint.forward = patched_KSamplerX0Inpaint_forward
    ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSamplerPatched
    ldm_patched.modules.samplers.sampling_function = patched_sampling_function

    warnings.filterwarnings(action='ignore', module='torchsde')

    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')

    return
