import math
import os

import torch

from backend import controlnet_compat, lora, ops, patching, resources, utils

import ldm_patched.controlnet.cldm
import ldm_patched.t2ia.adapter


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        repeats = per_batch // tensor.shape[0]
        remainder = per_batch % tensor.shape[0]
        tensor = torch.cat([tensor] * repeats + [tensor[:remainder]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    return torch.cat([tensor] * batched_number, dim=0)


class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.global_average_pooling = False
        self.timestep_range = None
        self.device = device or resources.get_torch_device()
        self.previous_controlnet = None

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0)):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (
            percent_to_timestep_function(self.timestep_percent_range[0]),
            percent_to_timestep_function(self.timestep_percent_range[1]),
        )
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, control):
        control.cond_hint_original = self.cond_hint_original
        control.strength = self.strength
        control.timestep_percent_range = self.timestep_percent_range
        control.global_average_pooling = self.global_average_pooling

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        out = {"input": [], "middle": [], "output": []}

        if control_input is not None:
            for control_tensor in control_input:
                x = control_tensor
                if x is not None:
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out["input"].insert(0, x)

        if control_output is not None:
            for i, control_tensor in enumerate(control_output):
                key = "middle" if i == (len(control_output) - 1) else "output"
                x = control_tensor
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].append(x)

        if control_prev is not None:
            for key in ["input", "middle", "output"]:
                merged = out[key]
                for i, prev_val in enumerate(control_prev[key]):
                    if i >= len(merged):
                        merged.append(prev_val)
                    elif prev_val is not None:
                        if merged[i] is None:
                            merged[i] = prev_val
                        elif merged[i].shape[0] < prev_val.shape[0]:
                            merged[i] = prev_val + merged[i]
                        else:
                            merged[i] += prev_val

        return out


class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(device)
        self.control_model = control_model
        self.load_device = load_device or resources.get_torch_device()
        self.control_model_wrapped = patching.NexModelPatcher(
            self.control_model,
            load_device=self.load_device,
            offload_device=resources.unet_offload_device(),
        )
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype
        self._nex_pending_quality = None

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        target_width = x_noisy.shape[3] * 8
        target_height = x_noisy.shape[2] * 8
        if self.cond_hint is None or target_height != self.cond_hint.shape[2] or target_width != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = utils.common_upscale(
                self.cond_hint_original,
                target_width,
                target_height,
                "nearest-exact",
                "center",
            ).to(dtype).to(self.device)

        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond["c_crossattn"]
        y = cond.get("y", None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(
            x=x_noisy.to(dtype),
            hint=self.cond_hint,
            timesteps=timestep.float(),
            context=context.to(dtype),
            y=y,
        )
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        control = ControlNet(
            self.control_model,
            global_average_pooling=self.global_average_pooling,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )
        self.copy_to(control)
        return control

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling
        if self._nex_pending_quality:
            from backend import loader
            loader.patch_controlnet_for_quality(self, self._nex_pending_quality)

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()


class ControlLoraOps:
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None
            self.weight_function = []
            self.bias_function = []
            self.comfy_cast_weights = True

        def forward(self, input_tensor):
            weight, bias = ops.cast_bias_weight(self, input_tensor)
            if self.up is not None:
                delta = torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))
                delta = delta.reshape(self.weight.shape).type(input_tensor.dtype)
                return torch.nn.functional.linear(input_tensor, weight + delta, bias)
            return torch.nn.functional.linear(input_tensor, weight, bias)

    class Conv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode
            self.weight = None
            self.bias = None
            self.up = None
            self.down = None
            self.weight_function = []
            self.bias_function = []
            self.comfy_cast_weights = True

        def forward(self, input_tensor):
            weight, bias = ops.cast_bias_weight(self, input_tensor)
            if self.up is not None:
                delta = torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))
                delta = delta.reshape(self.weight.shape).type(input_tensor.dtype)
                return torch.nn.functional.conv2d(
                    input_tensor,
                    weight + delta,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            return torch.nn.functional.conv2d(
                input_tensor,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, device=None):
        ControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling
        self.control_model = None
        self.manual_cast_dtype = None

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()

        if self.manual_cast_dtype is None:
            class control_lora_ops(ControlLoraOps, ops.disable_weight_init):
                pass
        else:
            class control_lora_ops(ControlLoraOps, ops.manual_cast):
                pass
            dtype = self.manual_cast_dtype

        controlnet_config["operations"] = control_lora_ops
        controlnet_config["dtype"] = dtype
        self.control_model = ldm_patched.controlnet.cldm.ControlNet(**controlnet_config)
        self.control_model.to(resources.get_torch_device())

        for key, weight in model.diffusion_model.state_dict().items():
            try:
                utils.set_attr(self.control_model, key, weight)
            except Exception:
                pass

        target_device = resources.get_torch_device()
        for key, value in self.control_weights.items():
            if key != "lora_controlnet":
                utils.set_attr(self.control_model, key, value.to(dtype).to(target_device))

        if self._nex_pending_quality:
            from backend import loader
            loader.patch_controlnet_for_quality(self, self._nex_pending_quality)

    def copy(self):
        control = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(control)
        return control

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        return ControlBase.get_models(self)

    def inference_memory_requirements(self, dtype):
        return utils.calculate_parameters(self.control_weights) * resources.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)


def load_controlnet(ckpt_path, model=None):
    controlnet_data = utils.load_torch_file(ckpt_path)
    if "lora_controlnet" in controlnet_data:
        return ControlLora(controlnet_data)

    controlnet_config = None
    unet_dtype = resources.unet_dtype()
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:
        controlnet_config = controlnet_compat.unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
        diffusers_keys = lora.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            for suffix in [".weight", ".bias"]:
                key_in = f"controlnet_down_blocks.{count}{suffix}"
                key_out = f"zero_convs.{count}.0{suffix}"
                if key_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[key_in] = key_out
            count += 1

        count = 0
        loop = True
        while loop:
            for suffix in [".weight", ".bias"]:
                if count == 0:
                    key_in = f"controlnet_cond_embedding.conv_in{suffix}"
                else:
                    key_in = f"controlnet_cond_embedding.blocks.{count - 1}{suffix}"
                key_out = f"input_hint_block.{count * 2}{suffix}"
                if key_in not in controlnet_data:
                    key_in = f"controlnet_cond_embedding.conv_out{suffix}"
                    loop = False
                diffusers_keys[key_in] = key_out
            count += 1

        new_state_dict = {}
        for key, mapped_key in diffusers_keys.items():
            if key in controlnet_data:
                new_state_dict[mapped_key] = controlnet_data.pop(key)

        if len(controlnet_data.keys()) > 0:
            print("leftover keys:", controlnet_data.keys())
        controlnet_data = new_state_dict

    pth_key = "control_model.zero_convs.0.0.weight"
    state_key = "zero_convs.0.0.weight"
    pth = False
    if pth_key in controlnet_data:
        pth = True
        prefix = "control_model."
    elif state_key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data)
        if net is None:
            print("error checkpoint does not contain controlnet or t2i adapter data", ckpt_path)
        return net

    if controlnet_config is None:
        controlnet_config = controlnet_compat.model_config_from_unet(controlnet_data, prefix, True).unet_config

    load_device = resources.get_torch_device()
    manual_cast_dtype = controlnet_compat.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = ops.manual_cast
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data[f"{prefix}input_hint_block.0.weight"].shape[1]
    control_model = ldm_patched.controlnet.cldm.ControlNet(**controlnet_config)

    if pth:
        if "difference" in controlnet_data:
            if model is not None:
                resources.load_models_gpu([model])
                model_state_dict = model.model_state_dict()
                for key in controlnet_data:
                    control_prefix = "control_model."
                    if key.startswith(control_prefix):
                        sd_key = f"diffusion_model.{key[len(control_prefix):]}"
                        if sd_key in model_state_dict:
                            control_delta = controlnet_data[key]
                            control_delta += model_state_dict[sd_key].type(control_delta.dtype).to(control_delta.device)
            else:
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass

        weights_loader = WeightsLoader()
        weights_loader.control_model = control_model
        missing, unexpected = weights_loader.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    print(missing, unexpected)

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
        global_average_pooling = True

    return ControlNet(
        control_model,
        global_average_pooling=global_average_pooling,
        load_device=load_device,
        manual_cast_dtype=manual_cast_dtype,
    )


class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev

        target_width, target_height = self.scale_image_to(x_noisy.shape[3] * 8, x_noisy.shape[2] * 8)
        if self.cond_hint is None or target_height != self.cond_hint.shape[2] or target_width != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = utils.common_upscale(
                self.cond_hint_original,
                target_width,
                target_height,
                "nearest-exact",
                "center",
            ).float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)

        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = [None if item is None else item.clone() for item in self.control_input]
        mid = None
        if self.t2i_model.xl is True:
            mid = control_input[-1:]
            control_input = control_input[:-1]
        return self.control_merge(control_input, mid, control_prev, x_noisy.dtype)

    def copy(self):
        control = T2IAdapter(self.t2i_model, self.channels_in)
        self.copy_to(control)
        return control


def load_t2i_adapter(t2i_data):
    if "adapter" in t2i_data:
        t2i_data = t2i_data["adapter"]
    if "adapter.body.0.resnets.0.block1.weight" in t2i_data:
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace[f"adapter.body.{i}.resnets.{j}."] = f"body.{i * 2 + j}."
            prefix_replace[f"adapter.body.{i}."] = f"body.{i * 2}."
        prefix_replace["adapter."] = ""
        t2i_data = utils.state_dict_prefix_replace(t2i_data, prefix_replace)

    keys = t2i_data.keys()
    if "body.0.in_conv.weight" in keys:
        channels_in = t2i_data["body.0.in_conv.weight"].shape[1]
        model_ad = ldm_patched.t2ia.adapter.Adapter_light(
            cin=channels_in,
            channels=[320, 640, 1280, 1280],
            nums_rb=4,
        )
    elif "conv_in.weight" in keys:
        channels_in = t2i_data["conv_in.weight"].shape[1]
        channel = t2i_data["conv_in.weight"].shape[0]
        kernel_size = t2i_data["body.0.block2.weight"].shape[2]
        use_conv = len([key for key in keys if key.endswith("down_opt.op.weight")]) > 0
        xl = channels_in in (256, 768)
        model_ad = ldm_patched.t2ia.adapter.Adapter(
            cin=channels_in,
            channels=[channel, channel * 2, channel * 4, channel * 4][:4],
            nums_rb=2,
            ksize=kernel_size,
            sk=True,
            use_conv=use_conv,
            xl=xl,
        )
    else:
        return None

    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        print("t2i missing", missing)
    if len(unexpected) > 0:
        print("t2i unexpected", unexpected)

    return T2IAdapter(model_ad, model_ad.input_channels)


