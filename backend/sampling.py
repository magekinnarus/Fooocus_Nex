import torch
import math
import logging
import time
from functools import partial
from typing import Any, Callable, List, Dict, Optional, Union, Tuple

# Local imports
from . import schedulers
from . import k_diffusion
from . import precision

# Re-export key constants for registration
SCHEDULER_NAMES = schedulers.SCHEDULER_NAMES

from . import anisotropic

# Note: We avoid importing from ldm_patched directly to keep the backend clean.
# ModelPatcher and BaseModel are expected to be passed as generic objects or Any.

from .cond_utils import (
    add_area_dims, get_area_and_mult, cond_equal_size, can_concat_cond,
    cond_cat, calc_cond_batch, resolve_areas_and_cond_masks_multidim,
    calculate_start_end_timesteps, encode_model_conds, process_conds
)

class KSamplerX0Inpaint:
    def __init__(self, model: Any, sigmas: torch.Tensor):
        self.inner_model = model
        self.sigmas = sigmas
        self.noise = None
        self.latent_image = None
        
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, denoise_mask: Optional[torch.Tensor] = None, model_options: Dict[str, Any] = {}, seed: Optional[int] = None) -> torch.Tensor:
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma, self.noise, self.latent_image) * latent_mask
        
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

class Sampler:
    def sample(self, model_wrap: Any, sigmas: torch.Tensor, extra_args: Dict[str, Any], callback: Optional[Callable], noise: torch.Tensor, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, disable_pbar: bool = False) -> torch.Tensor:
        pass

    def max_denoise(self, model_wrap: Any, sigmas: torch.Tensor) -> bool:
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

class KSAMPLER(Sampler):
    def __init__(self, sampler_function: Callable, extra_options: Dict[str, Any] = {}, inpaint_options: Dict[str, Any] = {}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap: Any, sigmas: torch.Tensor, extra_args: Dict[str, Any], callback: Optional[Callable], noise: torch.Tensor, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, disable_pbar: bool = False) -> torch.Tensor:
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        
        if self.inpaint_options.get("random", False):
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps, x.get("denoised", None))

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples

# Sampler Registry
KSAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"
]

SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

def sampler_names() -> List[str]:
    return SAMPLER_NAMES

def ksampler(sampler_name: str, extra_options: Dict[str, Any] = {}, inpaint_options: Dict[str, Any] = {}) -> KSAMPLER:
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1: return noise
            sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
            return k_diffusion.sample_dpm_fast(model, noise, sigma_min, sigmas[0], len(sigmas) - 1, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1: return noise
            sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
            return k_diffusion.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    else:
        func_name = f"sample_{sampler_name.replace('_cfg_pp', '')}"
        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {sampler_name} not implemented in k_diffusion as {func_name}")
            
    return KSAMPLER(sampler_function, extra_options, inpaint_options)

def sample_sdxl(
    model: Any,
    noise: torch.Tensor,
    positive: Any,
    negative: Any,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    denoise: float = 1.0,
    seed: int = None,
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    callback: Callable = None,
    disable_pbar: bool = False,
    model_options: Dict[str, Any] = {}
) -> torch.Tensor:
    """Main entry point for SDXL sampling."""
    device = noise.device
    ksampler_inst = KSampler(model, steps, device, sampler_name, scheduler, denoise, model_options=model_options)
    return ksampler_inst.sample(
        noise, positive, negative, cfg,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )

def sampler_priority() -> List[str]:
    return ["euler", "euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde_gpu", "lcm"]

def scheduler_names() -> List[str]:
    return schedulers.scheduler_names()

class KSampler:
    SCHEDULERS = scheduler_names()
    SAMPLERS = sampler_names()
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model: Any, steps: int, device: torch.device, sampler: str = None, scheduler: str = None, denoise: float = None, model_options: Dict[str, Any] = {}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS: scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS: sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.denoise = denoise
        self.model_options = model_options
        self.quality = model_options.get("quality", {})
        
        # Apply quality patches to UNet (Timed ADM, precision casting)
        from . import loader
        loader.patch_unet_for_quality(self.model, self.quality)

        self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True
        
        model_sampling = self.model.get_model_object("model_sampling")
        sigmas = schedulers.calculate_sigmas(model_sampling, self.scheduler, steps, model=self.model.model)
        
        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps: int, denoise: float = None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None: sigmas = self.sigmas
        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise: sigmas[-1] = 0
        if start_step is not None:
            if start_step < (len(sigmas) - 1): sigmas = sigmas[start_step:]
            else: return latent_image if latent_image is not None else torch.zeros_like(noise)

        sampler_inst = ksampler(self.sampler)
        cfg_pp = "_cfg_pp" in self.sampler
        cfg_guider = CFGGuider(self.model)
        cfg_guider.set_conds(positive, negative)
        cfg_guider.set_cfg(cfg, cfg_pp=cfg_pp)
        cfg_guider.set_quality(self.quality)
        
        with precision.autocast_context(self.device):
            return cfg_guider.sample(noise, latent_image, sampler_inst, sigmas, denoise_mask, callback, disable_pbar, seed)

def cfg_function(model: Any, cond_pred: torch.Tensor, uncond_pred: torch.Tensor, cond_scale: float, x: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any] = {}, cfg_pp: bool = False, adaptive_cfg: float = 0.0, diffusion_progress: float = 0.0) -> torch.Tensor:
    if "sampler_cfg_function" in model_options:
        args = {
            "cond_denoised": cond_pred, 
            "uncond_denoised": uncond_pred, 
            "cond_scale": cond_scale, 
            "timestep": timestep, 
            "input": x, 
            "model": model, 
            "model_options": model_options
        }
        return x - model_options["sampler_cfg_function"](args)
    
    # Fooocus Adaptive CFG
    if adaptive_cfg > 0.0 and cond_scale > adaptive_cfg:
        # Scale terms to EPS for Fooocus logic similarity
        cond_eps = x - cond_pred
        uncond_eps = x - uncond_pred
        
        real_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
        mimic_eps = uncond_eps + adaptive_cfg * (cond_eps - uncond_eps)
        
        # Blend by progress: real_eps * progress + mimic_eps * (1 - progress)
        final_eps = real_eps * diffusion_progress + mimic_eps * (1.0 - diffusion_progress)
        return x - final_eps

    if cfg_pp:
        # CFG++: cond + (scale - 1) * (cond - uncond)
        cfg_result = cond_pred + (cond_scale - 1.0) * (cond_pred - uncond_pred)
    else:
        # Standard CFG: uncond + scale * (cond - uncond)
        # ComfyUI often uses: uncond + scale * (cond - uncond)
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    
    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result, 
            "cond_denoised": cond_pred, 
            "uncond_denoised": uncond_pred, 
            "cond_scale": cond_scale, 
            "model": model, 
            "sigma": timestep, 
            "model_options": model_options, 
            "input": x
        }
        cfg_result = fn(args)
        
    return cfg_result

def sampling_function(model: Any, x: torch.Tensor, timestep: torch.Tensor, uncond: Any, cond: Any, cond_scale: float, model_options: Dict[str, Any] = {}, seed: Optional[int] = None, cfg_pp: bool = False, sharpness: float = 0.0, adaptive_cfg: float = 0.0) -> torch.Tensor:
    # Calculate diffusion progress (0.0 -> 1.0)
    # sigma is passed as timestep
    model_sampling = model.model_sampling
    t = model_sampling.timestep(timestep)
    diffusion_progress = max(0.0, min(1.0, 1.0 - t.item() / 999.0))

    if math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)

    # out[0] is positive_x0, out[1] is negative_x0
    cond_pred = out[0]
    uncond_pred = out[1]

    # Fooocus Sharpness (Anisotropic Filtering)
    if sharpness > 0.0:
        alpha = 0.001 * sharpness * diffusion_progress
        if alpha >= 0.01:
            positive_eps = x - cond_pred
            # adaptive_anisotropic_filter(x=eps, g=x0)
            degraded_eps = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=cond_pred)
            # Blend: degraded * alpha + original * (1 - alpha)
            positive_eps_weighted = degraded_eps * alpha + positive_eps * (1.0 - alpha)
            # Update cond_pred (x0) back from weighted eps
            cond_pred = x - positive_eps_weighted

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds": conds, "conds_out": [cond_pred, uncond_pred], "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)
        cond_pred = out[0]
        uncond_pred = out[1]

    return cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options=model_options, cfg_pp=cfg_pp, adaptive_cfg=adaptive_cfg, diffusion_progress=diffusion_progress)

class CFGGuider:
    def __init__(self, model_patcher: Any):
        self.model_patcher = model_patcher
        self.model_options = getattr(model_patcher, "model_options", {})
        self.original_conds = {}
        self.cfg = 1.0
        self.conds = {}
        self.inner_model = None
        self.cfg_pp = False
        self.quality = {}

    def set_conds(self, positive: Any, negative: Any):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg: float, cfg_pp: bool = False):
        self.cfg = cfg
        self.cfg_pp = cfg_pp

    def set_quality(self, quality: Dict[str, Any]):
        self.quality = quality

    def inner_set_conds(self, conds: Dict[str, Any]):
        for k, v in conds.items():
            self.original_conds[k] = self.convert_cond(v)

    def convert_cond(self, cond: Any) -> List[Dict[str, Any]]:
        import uuid
        out = []
        if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
            for c in cond:
                temp = c.copy()
                temp["uuid"] = temp.get("uuid", uuid.uuid4())
                out.append(temp)
            return out
            
        for c in cond:
            temp = c[1].copy()
            model_conds = temp.get("model_conds", {})
            if c[0] is not None:
                temp["cross_attn"] = c[0]
            temp["model_conds"] = model_conds
            temp["uuid"] = uuid.uuid4()
            out.append(temp)
        return out

    def predict_noise(self, x: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any] = {}, seed: Optional[int] = None) -> torch.Tensor:
        return sampling_function(
            self.inner_model, 
            x, 
            timestep, 
            self.conds.get("negative"), 
            self.conds.get("positive"), 
            self.cfg, 
            model_options=model_options, 
            seed=seed,
            cfg_pp=self.cfg_pp,
            sharpness=self.quality.get("sharpness", 0.0),
            adaptive_cfg=self.quality.get("adaptive_cfg", 0.0)
        )

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def sample(self, noise: torch.Tensor, latent_image: torch.Tensor, sampler: Any, sigmas: torch.Tensor, denoise_mask: Optional[torch.Tensor] = None, callback: Optional[Callable] = None, disable_pbar: bool = False, seed: Optional[int] = None) -> torch.Tensor:
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = [c.copy() for c in self.original_conds[k]]

        # Load model to GPU before inference
        from . import resources

        gguf_ops = None
        try:
            from .gguf import ops as gguf_ops
            gguf_ops.reset_trace_stats()
        except Exception:
            gguf_ops = None

        sample_total_start = time.perf_counter()
        load_start = time.perf_counter()
        resources.load_models_gpu([self.model_patcher])
        model_load_duration = time.perf_counter() - load_start

        if self.inner_model is None:
            self.inner_model = self.model_patcher.model

        cond_start = time.perf_counter()
        self.conds = process_conds(self.inner_model, noise, self.conds, self.inner_model.get_dtype(), latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        cond_duration = time.perf_counter() - cond_start

        denoise_start = time.perf_counter()
        try:
            return sampler.sample(self, sigmas, {}, callback, noise, latent_image, denoise_mask, disable_pbar)
        finally:
            denoise_duration = time.perf_counter() - denoise_start
            total_duration = time.perf_counter() - sample_total_start
            perf_message = (
                f"[Nex-Perf] sampler timings model_load={model_load_duration:.3f}s "
                f"cond_prep={cond_duration:.3f}s denoise={denoise_duration:.3f}s total={total_duration:.3f}s"
            )
            print(perf_message)
            logging.info(perf_message)

            if gguf_ops is not None:
                stats = gguf_ops.consume_trace_stats()
                if stats.get("calls", 0) > 0:
                    avg_ms = (stats["total_seconds"] / stats["calls"]) * 1000.0
                    perf_message = (
                        f"[Nex-Perf] gguf get_weight calls={stats['calls']} quantized={stats['quantized_calls']} "
                        f"patch_calls={stats['patch_calls']} dequant={stats['dequant_seconds']:.3f}s "
                        f"patch={stats['patch_seconds']:.3f}s total={stats['total_seconds']:.3f}s avg={avg_ms:.3f}ms "
                        f"src_cpu={stats['source_cpu_calls']} src_cuda={stats['source_cuda_calls']} src_other={stats['source_other_calls']} "
                        f"src_quant_cpu={stats['source_quantized_cpu_calls']} src_quant_cuda={stats['source_quantized_cuda_calls']} src_quant_other={stats['source_quantized_other_calls']} "
                        f"cpu_calls={stats['cpu_calls']} cuda_calls={stats['cuda_calls']} other_calls={stats['other_device_calls']} "
                        f"quantized_cpu={stats['quantized_cpu_calls']} quantized_cuda={stats['quantized_cuda_calls']} quantized_other={stats['quantized_other_device_calls']} "
                        f"cpu_dequant={stats['cpu_dequant_seconds']:.3f}s cuda_dequant={stats['cuda_dequant_seconds']:.3f}s other_dequant={stats['other_device_dequant_seconds']:.3f}s"
                    )
                    print(perf_message)
                    logging.info(perf_message)
