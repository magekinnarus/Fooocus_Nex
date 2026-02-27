import torch
import math
import numpy
import scipy.stats
from typing import Any, Callable, List, Optional, Union, NamedTuple
from functools import partial

def loglinear_interp(t_steps, num_steps):
    xs = numpy.linspace(0, 1, len(t_steps))
    ys = numpy.log(t_steps[::-1])
    new_xs = numpy.linspace(0, 1, num_steps)
    new_ys = numpy.interp(new_xs, xs, ys)
    interped_ys = numpy.exp(new_ys)[::-1].copy()
    return interped_ys

NOISE_LEVELS = {
    "SD1": [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
    "SDXL": [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
}

def simple_scheduler(model_sampling: Any, steps: int, **kwargs) -> torch.Tensor:
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model_sampling: Any, steps: int, **kwargs) -> torch.Tensor:
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)

def normal_scheduler(model_sampling: Any, steps: int, sgm: bool = False, floor: bool = False, **kwargs) -> torch.Tensor:
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero_bool = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero_bool = False
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(float(s.sigma(ts)))

    if append_zero_bool:
        sigs.append(0.0)

    return torch.FloatTensor(sigs)

def beta_scheduler(model_sampling: Any, steps: int, alpha: float = 0.6, beta: float = 0.6, **kwargs) -> torch.Tensor:
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs.append(float(model_sampling.sigmas[int(t)]))
        last_t = t
    sigs.append(0.0)
    return torch.FloatTensor(sigs)

def linear_quadratic_schedule(model_sampling: Any, steps: int, threshold_noise: float = 0.025, linear_steps: Optional[int] = None, **kwargs) -> torch.Tensor:
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * model_sampling.sigma_max.cpu()

def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas

def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_exponential(n: int, sigma_min: float, sigma_max: float, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)

def turbo_scheduler(model_sampling: Any, steps: int, **kwargs) -> torch.Tensor:
    start_step = 0
    timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
    sigmas = model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return sigmas

def align_your_steps_scheduler(model_sampling: Any, steps: int, **kwargs) -> torch.Tensor:
    model = kwargs.get("model", None)
    model_type = "SDXL"
    if model is not None and hasattr(model, "latent_format"):
        model_type = 'SDXL' if model.latent_format.__class__.__name__ == 'SDXL' else 'SD1'
        
    sigmas = NOISE_LEVELS[model_type][:]
    if (steps + 1) != len(sigmas):
        sigmas = loglinear_interp(sigmas, steps + 1)
        
    sigmas = sigmas[-(steps + 1):]
    sigmas[-1] = 0
    return torch.FloatTensor(sigmas)

class SchedulerHandler(NamedTuple):
    handler: Callable[..., torch.Tensor]
    use_ms: bool = True

SCHEDULER_HANDLERS = {
    "simple": SchedulerHandler(simple_scheduler),
    "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
    "karras": SchedulerHandler(get_sigmas_karras, use_ms=False),
    "exponential": SchedulerHandler(get_sigmas_exponential, use_ms=False),
    "ddim_uniform": SchedulerHandler(ddim_scheduler),
    "beta": SchedulerHandler(beta_scheduler),
    "normal": SchedulerHandler(normal_scheduler),
    "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
    "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
    "turbo": SchedulerHandler(turbo_scheduler),
    "align_your_steps": SchedulerHandler(align_your_steps_scheduler),
}

SCHEDULER_NAMES = list(SCHEDULER_HANDLERS.keys())

def scheduler_names() -> List[str]:
    return list(SCHEDULER_HANDLERS.keys())

def calculate_sigmas(model_sampling: Any, scheduler_name: str, steps: int, model: Any = None) -> torch.Tensor:
    if scheduler_name not in SCHEDULER_HANDLERS:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    handler = SCHEDULER_HANDLERS[scheduler_name]
    if handler.use_ms:
        return handler.handler(model_sampling, steps, model=model)
    else:
        return handler.handler(steps, float(model_sampling.sigma_min), float(model_sampling.sigma_max))
