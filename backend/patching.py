from __future__ import annotations

import collections
import copy
import inspect
import logging
import math
import uuid
from typing import Callable, Optional

import torch

from . import float_ops
from . import resources
from . import utils
UnetWrapperFunction = Callable

from backend.weight_ops import string_to_seed, wipe_lowvram_weight, move_weight_functions, calculate_weight, LowVramPatch, get_key_weight




class NexModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False, current_device=None, **kwargs):
        self.size = size
        self.model = model
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.current_device = current_device
        self.weight_inplace_update = weight_inplace_update
        self.force_cast_weights = False
        self.patches_uuid = uuid.uuid4()
        self.parent = None

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

        if not hasattr(self.model, 'current_weight_patches_uuid'):
            self.model.current_weight_patches_uuid = None
            
    
    class _DummyContext:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    def use_ejected(self, skip_and_inject_on_exit_only=False):
        return self._DummyContext()

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = resources.module_size(self.model)
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def lowvram_patch_counter(self):
        return self.model.lowvram_patch_counter

    def clone(self):
        n = self.__class__(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.parent = self

        n.force_cast_weights = self.force_cast_weights
        return n
        
    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone):
        if not self.is_clone(clone):
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True
        return False
        
    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, sampler_calc_cond_batch_function):
        self.model_options["sampler_calc_cond_batch_function"] = sampler_calc_cond_batch_function

    def set_model_unet_function_wrapper(self, unet_wrapper_function: UnetWrapperFunction):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def set_model_emb_patch(self, patch):
        self.set_model_patch(patch, "emb_patch")

    def set_model_forward_timestep_embed_patch(self, patch):
        self.set_model_patch(patch, "forward_timestep_embed_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_compute_dtype(self, dtype):
        self.add_object_patch("manual_cast_dtype", dtype)
        if dtype is not None:
            self.force_cast_weights = True
        self.patches_uuid = uuid.uuid4() #TODO: optimize by preventing a full model reload for this

    def add_weight_wrapper(self, name, function):
        self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
        self.patches_uuid = uuid.uuid4()

    def get_model_object(self, name: str) -> torch.nn.Module:
        """Retrieves a nested attribute from an object using dot notation considering
        object patches.

        Args:
            name (str): The attribute path using dot notation (e.g. "model.layer.weight")

        Returns:
            The value of the requested attribute

        Example:
            patcher = ModelPatcher()
            weight = patcher.get_model_object("layer1.conv.weight")
        """
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches = self.patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    self.patches[key] = current_patches

            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            bk = self.backup.get(k, None)
            hbk = self.hook_backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if hbk is not None:
                weight = hbk[0]
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        with self.use_ejected():
            sd = self.model.state_dict()
            keys = list(sd.keys())
            if filter_prefix is not None:
                for k in keys:
                    if not k.startswith(filter_prefix):
                        sd.pop(k)
            return sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        inplace_update = self.weight_inplace_update or inplace_update
        weight, set_func, convert_func = get_key_weight(self.model, key)
        target_dtype = resources.unet_dtype(device=device_to, weight_dtype=weight.dtype)

        if key not in self.patches:
            if device_to is not None or target_dtype != weight.dtype:
                temp_weight = resources.cast_to_device(weight, device_to, target_dtype, copy=True)
                if convert_func is not None:
                    temp_weight = convert_func(temp_weight, inplace=True)
                
                if set_func is None:
                    if inplace_update:
                        utils.copy_to_param(self.model, key, temp_weight)
                    else:
                        utils.set_attr_param(self.model, key, temp_weight)
                else:
                    set_func(temp_weight, inplace_update=inplace_update, seed=string_to_seed(key))
            return

        if key not in self.backup:
            # OPTIMIZATION: Only backup if we are doing an inplace update or if the weight is not already on the offload device
            if inplace_update or weight.device != self.offload_device:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = resources.cast_to_device(weight, device_to, target_dtype, copy=True)
        else:
            temp_weight = weight.to(target_dtype, copy=True)

        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = calculate_weight(self.patches[key], temp_weight, key, intermediate_dtype=target_dtype, original_weights=self.backup)
        if set_func is None:
            out_weight = float_ops.stochastic_rounding(out_weight, target_dtype, seed=string_to_seed(key))
            if inplace_update:
                utils.copy_to_param(self.model, key, out_weight)
            else:
                utils.set_attr_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def _load_list(self):
        loading = []
        for n, m in self.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                loading.append((resources.module_size(m), n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            m.weight_function = [LowVramPatch(weight_key, self.patches)]
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            m.bias_function = [LowVramPatch(bias_key, self.patches)]
                            patch_counter += 1

                    cast_weight = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        with self.use_ejected():
            for k in self.object_patches:
                old = utils.set_attr(self.model, k, self.object_patches[k])
                if k not in self.object_patches_backup:
                    self.object_patches_backup[k] = old

            if lowvram_model_memory == 0:
                full_load = True
            else:
                full_load = False

            if load_weights:
                self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    utils.copy_to_param(self.model, k, bk.weight)
                else:
                    utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0):
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()
            for unload in unload_list:
                if memory_to_free < memory_freed:
                    break
                module_mem = unload[0]
                n = unload[1]
                m = unload[2]
                params = unload[3]

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                    move_weight = True
                    for param in params:
                        key = "{}.{}".format(n, param)
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                hooks_unpatched = True

                            if bk.inplace_update:
                                utils.copy_to_param(self.model, key, bk.weight)
                            else:
                                utils.set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                m.weight_function.append(LowVramPatch(weight_key, self.patches))
                                patch_counter += 1
                            if bias_key in self.patches:
                                m.bias_function.append(LowVramPatch(bias_key, self.patches))
                                patch_counter += 1
                            cast_weight = True

                        if cast_weight:
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        logging.debug("freed {}".format(n))

            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
            # TODO: force_patch_weights should not unload + reload full model
            used = self.model.model_loaded_weight_memory
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
            if unpatch_weights:
                extra_memory += (used - self.model.model_loaded_weight_memory)

            self.patch_model(load_weights=False)
            full_load = False
            if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
                return 0
            if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
                full_load = True
            current_used = self.model.model_loaded_weight_memory
            try:
                self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
            except Exception as e:
                self.detach()
                raise e

            return self.model.model_loaded_weight_memory - current_used

    def detach(self, unpatch_all=True):
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def current_loaded_device(self):
        return self.model.device
