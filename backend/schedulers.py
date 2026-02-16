import torch
import math
import numpy
import scipy.stats
from typing import Any, Callable, List, Optional, Union, NamedTuple
from functools import partial

def simple_scheduler(model_sampling: Any, steps: int) -> torch.Tensor:
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model_sampling: Any, steps: int) -> torch.Tensor:
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

def normal_scheduler(model_sampling: Any, steps: int, sgm: bool = False, floor: bool = False) -> torch.Tensor:
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

def beta_scheduler(model_sampling: Any, steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
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

def linear_quadratic_schedule(model_sampling: Any, steps: int, threshold_noise: float = 0.025, linear_steps: Optional[int] = None) -> torch.Tensor:
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
}

SCHEDULER_NAMES = list(SCHEDULER_HANDLERS.keys())

def scheduler_names() -> List[str]:
    return list(SCHEDULER_HANDLERS.keys())

def calculate_sigmas(model_sampling: Any, scheduler_name: str, steps: int) -> torch.Tensor:
    if scheduler_name not in SCHEDULER_HANDLERS:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    handler = SCHEDULER_HANDLERS[scheduler_name]
    if handler.use_ms:
        return handler.handler(model_sampling, steps)
    else:
        return handler.handler(steps, float(model_sampling.sigma_min), float(model_sampling.sigma_max))
