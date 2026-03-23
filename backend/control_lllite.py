import torch

from backend import patching, resources, utils
from backend.controlnet import ControlBase, broadcast_image_to


def _extra_options_to_module_prefix(extra_options):
    block = extra_options["block"]
    block_index = extra_options["block_index"]
    if block[0] == "input":
        return f"lllite_unet_input_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    if block[0] == "middle":
        return f"lllite_unet_middle_block_1_transformer_blocks_{block_index}"
    if block[0] == "output":
        return f"lllite_unet_output_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    raise ValueError(f"Unsupported ControlLLLite block type: {block[0]}")


class LLLitePatch:
    def __init__(self, modules, control=None):
        self.modules = modules
        self.control = control

    def set_control(self, control):
        self.control = control
        return self

    def clone_with_control(self, control):
        return LLLitePatch(self.modules, control)

    def to(self, device):
        for name, module in self.modules.items():
            self.modules[name] = module.to(device)
        return self

    def cleanup(self):
        for module in self.modules.values():
            module.cleanup()

    def __call__(self, q, k, v, extra_options):
        control = self.control
        if control is None:
            return q, k, v
        if control.timestep_range is not None and control.t is not None:
            if control.t > control.timestep_range[0] or control.t < control.timestep_range[1]:
                return q, k, v

        module_prefix = _extra_options_to_module_prefix(extra_options)
        module_prefix += "_attn1" if q.shape[-1] == k.shape[-1] else "_attn2"

        q_name = module_prefix + "_to_q"
        k_name = module_prefix + "_to_k"
        v_name = module_prefix + "_to_v"

        if q_name in self.modules:
            q = q + self.modules[q_name](q, control)
        if k_name in self.modules:
            k = k + self.modules[k_name](k, control)
        if v_name in self.modules:
            v = v + self.modules[v_name](v, control)
        return q, k, v


class LLLiteModule(torch.nn.Module):
    def __init__(self, name, is_conv2d, in_dim, depth, cond_emb_dim, mlp_dim):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d

        modules = [torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0)]
        if depth == 1:
            modules += [torch.nn.ReLU(inplace=True), torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0)]
        elif depth == 2:
            modules += [torch.nn.ReLU(inplace=True), torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0)]
        else:
            modules += [
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0),
            ]
        self.conditioning1 = torch.nn.Sequential(*modules)

        if is_conv2d:
            self.down = torch.nn.Sequential(torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0), torch.nn.ReLU(inplace=True))
            self.mid = torch.nn.Sequential(torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0), torch.nn.ReLU(inplace=True))
            self.up = torch.nn.Sequential(torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0))
        else:
            self.down = torch.nn.Sequential(torch.nn.Linear(in_dim, mlp_dim), torch.nn.ReLU(inplace=True))
            self.mid = torch.nn.Sequential(torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim), torch.nn.ReLU(inplace=True))
            self.up = torch.nn.Sequential(torch.nn.Linear(mlp_dim, in_dim))

        self.cond_emb = None
        self.prev_batch = 0
        self.prev_shape = None

    def cleanup(self):
        self.cond_emb = None
        self.prev_batch = 0
        self.prev_shape = None

    def _build_cond_emb(self, control, x):
        cond_hint = control.cond_hint.to(x.device, dtype=x.dtype)
        cx = self.conditioning1(cond_hint)
        if not self.is_conv2d:
            n, c, h, w = cx.shape
            cx = cx.view(n, c, h * w).permute(0, 2, 1)
        self.cond_emb = cx
        self.prev_batch = x.shape[0]
        self.prev_shape = tuple(x.shape)

    def forward(self, x, control):
        if self.cond_emb is None or self.prev_batch != x.shape[0] or self.prev_shape != tuple(x.shape):
            self._build_cond_emb(control, x)
        cx = self.cond_emb
        if x.shape[0] != cx.shape[0]:
            repeat_factor = x.shape[0] // cx.shape[0]
            cx = cx.repeat(repeat_factor, 1, 1, 1) if self.is_conv2d else cx.repeat(repeat_factor, 1, 1)

        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        return cx * control.strength


class ControlLLLiteModules(torch.nn.Module):
    def __init__(self, patch_attn1, patch_attn2):
        super().__init__()
        self.patch_attn1_modules = torch.nn.ModuleList(list(patch_attn1.modules.values()))
        self.patch_attn2_modules = torch.nn.ModuleList(list(patch_attn2.modules.values()))


class ControlLLLite(ControlBase):
    def __init__(self, patch_attn1, patch_attn2, device=None):
        super().__init__(device=device)
        self.patch_attn1 = patch_attn1.clone_with_control(self)
        self.patch_attn2 = patch_attn2.clone_with_control(self)
        self.control_model = ControlLLLiteModules(self.patch_attn1, self.patch_attn2)
        self.control_model_wrapped = patching.NexModelPatcher(
            self.control_model,
            load_device=self.device,
            offload_device=resources.unet_offload_device(),
        )
        self.t = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0)):
        super().set_cond_hint(cond_hint * 2.0 - 1.0, strength, timestep_percent_range)
        return self

    def copy(self):
        control = ControlLLLite(self.patch_attn1, self.patch_attn2, device=self.device)
        self.copy_to(control)
        return control

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        model.set_model_attn1_patch(self.patch_attn1.set_control(self))
        model.set_model_attn2_patch(self.patch_attn2.set_control(self))

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None and (t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]):
            return control_prev

        self.t = t[0]
        target_width = x_noisy.shape[3] * 8
        target_height = x_noisy.shape[2] * 8
        if self.cond_hint is None or target_height != self.cond_hint.shape[2] or target_width != self.cond_hint.shape[3]:
            self.cond_hint = utils.common_upscale(
                self.cond_hint_original,
                target_width,
                target_height,
                "nearest-exact",
                "center",
            ).to(x_noisy.dtype).to(self.device)

        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        return control_prev

    def cleanup(self):
        self.t = None
        self.patch_attn1.cleanup()
        self.patch_attn2.cleanup()
        super().cleanup()


def load_controllllite(ckpt_path, controlnet_data=None):
    if controlnet_data is None:
        controlnet_data = utils.load_torch_file(ckpt_path)

    module_weights = {}
    for key, value in controlnet_data.items():
        module_name, _, weight_name = key.partition('.')
        module_weights.setdefault(module_name, {})[weight_name] = value

    modules = {}
    load_device = resources.get_torch_device()
    first_dtype = next(iter(controlnet_data.values())).dtype
    target_dtype = torch.float32 if resources.is_device_cpu(load_device) else first_dtype

    for module_name, weights in module_weights.items():
        if 'conditioning1.4.weight' in weights:
            depth = 3
        elif weights['conditioning1.2.weight'].shape[-1] == 4:
            depth = 2
        else:
            depth = 1

        module = LLLiteModule(
            name=module_name,
            is_conv2d=weights['down.0.weight'].ndim == 4,
            in_dim=weights['down.0.weight'].shape[1],
            depth=depth,
            cond_emb_dim=weights['conditioning1.0.weight'].shape[0] * 2,
            mlp_dim=weights['down.0.weight'].shape[0],
        )
        module.load_state_dict(weights, strict=True)
        modules[module_name] = module.to(dtype=target_dtype)

    patch_attn1 = LLLitePatch(modules=modules)
    patch_attn2 = LLLitePatch(modules=modules)
    return ControlLLLite(patch_attn1=patch_attn1, patch_attn2=patch_attn2, device=load_device)
