import torch
from typing import Callable, Protocol, TypedDict, Optional, List

class UnetApplyFunction(Protocol):
    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

class UnetApplyConds(TypedDict):
    c_concat: Optional[torch.Tensor]
    c_crossattn: Optional[torch.Tensor]
    control: Optional[torch.Tensor]
    transformer_options: Optional[dict]

class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: UnetApplyConds
    cond_or_uncond: List[int]

UnetWrapperFunction = Callable[[UnetApplyFunction, UnetParams], torch.Tensor]

class CallbacksMP:
    ON_CLONE = "on_clone"
    ON_LOAD = "on_load_after"
    ON_DETACH = "on_detach_after"
    ON_CLEANUP = "on_cleanup"
    ON_PRE_RUN = "on_pre_run"
    ON_PREPARE_STATE = "on_prepare_state"
    ON_APPLY_HOOKS = "on_apply_hooks"
    ON_REGISTER_ALL_HOOK_PATCHES = "on_register_all_hook_patches"
    ON_INJECT_MODEL = "on_inject_model"
    ON_EJECT_MODEL = "on_eject_model"

    @classmethod
    def init_callbacks(cls) -> dict[str, dict[str, list[Callable]]]:
        return {}

class WrappersMP:
    OUTER_SAMPLE = "outer_sample"
    PREPARE_SAMPLING = "prepare_sampling"
    SAMPLER_SAMPLE = "sampler_sample"
    CALC_COND_BATCH = "calc_cond_batch"
    APPLY_MODEL = "apply_model"
    DIFFUSION_MODEL = "diffusion_model"

    @classmethod
    def init_wrappers(cls) -> dict[str, dict[str, list[Callable]]]:
        return {}

class PatcherInjection:
    def __init__(self, inject: Callable, eject: Callable):
        self.inject = inject
        self.eject = eject

class WeightAdapterBase:
    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
        raise NotImplementedError
