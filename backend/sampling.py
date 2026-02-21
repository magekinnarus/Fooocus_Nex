import torch
import torch.nn as nn
import math
import collections
import logging
from functools import partial
from typing import Any, Callable, List, Dict, Optional, Union, Tuple, NamedTuple

# Local imports
from . import schedulers
from . import k_diffusion

# Re-export key constants for registration
SCHEDULER_NAMES = schedulers.SCHEDULER_NAMES

# Note: We avoid importing from ldm_patched directly to keep the backend clean.
# ModelPatcher and BaseModel are expected to be passed as generic objects or Any.

def add_area_dims(area: List[int], num_dims: int) -> List[int]:
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area

def get_area_and_mult(conds: Dict[str, Any], x_in: torch.Tensor, timestep_in: torch.Tensor) -> Optional[collections.namedtuple]:
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    
    if 'area' in conds:
        area = list(conds['area'])
        area = add_area_dims(area, len(dims))
        if (len(area) // 2) > len(dims):
            area = area[:len(dims)] + area[len(area) // 2:(len(area) // 2) + len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    if 'mask' in conds:
        mask_strength = conds.get("mask_strength", 1.0)
        mask = conds['mask']
        assert (mask.shape[1:] == x_in.shape[2:])

        mask = mask[:input_x.shape[0]]
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    
    mult = mask * strength

    # Area fuzzing for smooth transitions
    if 'mask' not in conds and area is not None:
        fuzz = 8
        for i in range(len(dims)):
            rr = min(fuzz, mult.shape[2 + i] // 4)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0 / rr) * (t + 1))
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    if 'cross_attn' in conds:
        conditioning['c_crossattn'] = conds['cross_attn'].to(device=x_in.device, dtype=x_in.dtype)
    if 'concat' in conds:
        conditioning['c_concat'] = conds['concat'].to(device=x_in.device, dtype=x_in.dtype)

    model_conds = conds.get("model_conds", {})
    for c in model_conds:
        if hasattr(model_conds[c], "process_cond"):
            conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)
        else:
            conditioning[c] = model_conds[c].to(device=x_in.device, dtype=x_in.dtype)

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'uuid'])
    return cond_obj(input_x, mult, conditioning, area, conds.get('uuid'))

def cond_equal_size(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    if c1 is c2: return True
    if c1.keys() != c2.keys(): return False
    for k in c1:
        if not hasattr(c1[k], 'can_concat') or not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1: Any, c2: Any) -> bool:
    if c1.input_x.shape != c2.input_x.shape:
        return False
    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        if hasattr(conds[0], 'concat'):
            out[k] = conds[0].concat(conds[1:])
        else:
            out[k] = torch.cat(conds, dim=0)
    return out

def calc_cond_batch(model: Any, conds: List[List[Dict[str, Any]]], x_in: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any]) -> List[torch.Tensor]:
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                to_run.append((p, i))

    while len(to_run) > 0:
        first = to_run[0]
        to_batch = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch.append(x)
        
        items_to_pop = [to_run[i] for i in to_batch]
        for i in sorted(to_batch, reverse=True):
            to_run.pop(i)
            
        batch_input_x = []
        batch_mult = []
        batch_c = []
        batch_cond_indices = []
        batch_areas = []
        
        for p, cond_index in items_to_pop:
            batch_input_x.append(p.input_x)
            batch_mult.append(p.mult)
            batch_c.append(p.conditioning)
            batch_areas.append(p.area)
            batch_cond_indices.append(cond_index)

        batch_chunks = len(batch_cond_indices)
        input_x = torch.cat(batch_input_x)
        c = cond_cat(batch_c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        # Debug keys
        # print("Calc Cond Batch Keys:", c.keys(), "Batch Chunks:", batch_chunks)
        DEBUG_SAMPLING = False # Set to true to see CFG condition batch standard deviations
        if DEBUG_SAMPLING:
            if "c_crossattn" in c:
                cc = c["c_crossattn"]
                if cc.shape[0] >= 2:
                    cc0_std = cc[0].std().item()
                    cc1_std = cc[1].std().item()
                    cc_diff = (cc[0] - cc[1]).abs().mean().item()
                    print(f"Batch Cond Stats: C0 Std={cc0_std:.4f}, C1 Std={cc1_std:.4f}, L1 Diff={cc_diff:.4f}")
                else:
                    print(f"Single Cond Batch element: Std={cc.std().item():.4f}")
            else:
                print("WARNING: c_crossattn MISSING in Batch Conditioning!")

        output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_idx = batch_cond_indices[o]
            a = batch_areas[o]
            if a is None:
                out_conds[cond_idx] += output[o] * batch_mult[o]
                out_counts[cond_idx] += batch_mult[o]
            else:
                out_c = out_conds[cond_idx]
                out_cts = out_counts[cond_idx]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * batch_mult[o]
                out_cts += batch_mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds

def resolve_areas_and_cond_masks_multidim(conditions: List[Dict[str, Any]], dims: Tuple[int, ...], device: torch.device):
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                new_area = []
                for d in range(len(dims)):
                    new_area.append(max(1, round(a[d] * dims[d])))
                for d in range(len(dims)):
                    new_area.append(round(a[d + a_len] * dims[d]))
                modified['area'] = tuple(new_area)
                conditions[i] = modified
                c = modified

        if 'mask' in c:
            mask = c['mask'].to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode='bilinear', align_corners=False).squeeze(1)
            
            modified['mask'] = mask
            conditions[i] = modified

def calculate_start_end_timesteps(model: Any, conds: List[Dict[str, Any]]):
    s = getattr(model, "model_sampling", None)
    if s is None: return

    for t in range(len(conds)):
        x = conds[t]
        timestep_start = None
        timestep_end = None
        
        if 'start_percent' in x:
            timestep_start = s.percent_to_sigma(x['start_percent'])
        if 'end_percent' in x:
            timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if timestep_start is not None: n['timestep_start'] = timestep_start
            if timestep_end is not None: n['timestep_end'] = timestep_end
            conds[t] = n

def encode_model_conds(model_function: Callable, conds: List[Dict[str, Any]], noise: torch.Tensor, device: torch.device, prompt_type: str, **kwargs) -> List[Dict[str, Any]]:
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        
        if len(noise.shape) >= 4:
            params["width"] = params.get("width", noise.shape[3] * 8)
            params["height"] = params.get("height", noise.shape[2] * 8)
        
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k, v in kwargs.items():
            if k not in params:
                params[k] = v

        out = model_function(**params)
        x = x.copy()
        model_conds = x.get('model_conds', {}).copy()
        for k, v in out.items():
            model_conds[k] = v
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

def process_conds(model: Any, noise: torch.Tensor, conds: Dict[str, Any], device: torch.device, latent_image: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    return conds

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
            x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image) * latent_mask
        
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
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

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
    disable_pbar: bool = False
) -> torch.Tensor:
    """Main entry point for SDXL sampling."""
    device = noise.device
    ksampler_inst = KSampler(model, steps, device, sampler_name, scheduler, denoise)
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
        self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True
        
        model_sampling = self.model.get_model_object("model_sampling")
        sigmas = schedulers.calculate_sigmas(model_sampling, self.scheduler, steps)
        
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
        
        return cfg_guider.sample(noise, latent_image, sampler_inst, sigmas, denoise_mask, callback, disable_pbar, seed)

def cfg_function(model: Any, cond_pred: torch.Tensor, uncond_pred: torch.Tensor, cond_scale: float, x: torch.Tensor, timestep: torch.Tensor, model_options: Dict[str, Any] = {}, cfg_pp: bool = False) -> torch.Tensor:
    # Debug Prediction Stats
    if (getattr(cfg_function, "_step_count", 0) % 5 == 0):
        with torch.no_grad():
            c_std = cond_pred.std().item()
            u_std = uncond_pred.std().item()
            diff = (cond_pred - uncond_pred).abs().mean().item()
            print(f"Step {getattr(cfg_function, '_step_count', 0)} CFG: Cond Std={c_std:.4f}, Uncond Std={u_std:.4f}, L1 Diff={diff:.4f}, Scale={cond_scale}")
    
    cfg_function._step_count = getattr(cfg_function, "_step_count", 0) + 1

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

def sampling_function(model: Any, x: torch.Tensor, timestep: torch.Tensor, uncond: Any, cond: Any, cond_scale: float, model_options: Dict[str, Any] = {}, seed: Optional[int] = None, cfg_pp: bool = False) -> torch.Tensor:

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

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds": conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cfg_pp=cfg_pp)

class CFGGuider:
    def __init__(self, model_patcher: Any):
        self.model_patcher = model_patcher
        self.model_options = getattr(model_patcher, "model_options", {})
        self.original_conds = {}
        self.cfg = 1.0
        self.conds = {}
        self.inner_model = None
        self.cfg_pp = False

    def set_conds(self, positive: Any, negative: Any):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg: float, cfg_pp: bool = False):
        self.cfg = cfg
        self.cfg_pp = cfg_pp

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
            cfg_pp=self.cfg_pp
        )

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def sample(self, noise: torch.Tensor, latent_image: torch.Tensor, sampler: Any, sigmas: torch.Tensor, denoise_mask: Optional[torch.Tensor] = None, callback: Optional[Callable] = None, disable_pbar: bool = False, seed: Optional[int] = None) -> torch.Tensor:
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = [c.copy() for c in self.original_conds[k]]

        if self.inner_model is None:
            self.inner_model = self.model_patcher.model
            
        return sampler.sample(self, sigmas, {}, callback, noise, latent_image, denoise_mask, disable_pbar)
