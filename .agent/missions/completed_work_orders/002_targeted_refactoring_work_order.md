# Mission 002 Work Orders

These are self-contained instructions for an external agent (Gemini 3 Flash) to execute.

## Job 1: 1.5.A — Patch Registry
**Goal:** Create a documentation file listing all monkey-patches.
**Files:** `modules/patch.py`, `modules/patch_clip.py`, `modules/patch_precision.py`.
**Instructions:**
1.  Create a new file `modules/PATCH_MANIFEST.md`.
2.  Read the three source files listed above.
3.  Identify every function or method that is being assigned to an external module (e.g., `ldm_patched.modules.lora.calculate_weight = calculate_weight_patched`).
4.  Create a markdown table with columns: `Target Function`, `Replacement Function`, `Source File`, `Purpose (Brief)`.
5.  Populate the table with all active patches.

---

## Job 2: 1.5.B — Absorb `calculate_weight`
**Goal:** Move `calculate_weight` logic into `model_patcher.py` and update calls.
**Critical:** This function must support both Fooocus types AND standard Comfy types.
**Instructions:**

### Part 1: Modify `ldm_patched/modules/model_patcher.py`
1.  Add the following imports if missing:
    ```python
    import ldm_patched.modules.weight_adapter as weight_adapter
    ```
2.  Insert this *merged* `calculate_weight` function before the `LowVramPatch` class (around line 125):
    ```python
    def calculate_weight(patches, weight, key, intermediate_dtype=None, original_weights=None):
        if intermediate_dtype is None:
            intermediate_dtype = weight.dtype
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]
            offset = p[3]
            function = p[4]
            if function is None:
                function = lambda a: a
    
            old_weight = None
            if offset is not None:
                old_weight = weight
                weight = weight.narrow(offset[0], offset[1], offset[2])
    
            if strength_model != 1.0:
                weight *= strength_model
    
            if isinstance(v, list):
                v = (calculate_weight(v[1:], v[0][1](<ldm_patched.modules.model_management.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True>), inplace=True), key, intermediate_dtype=intermediate_dtype), )
    
            # Standard ComfyUI WeightAdapter support
            if isinstance(v, weight_adapter.WeightAdapterBase):
                output = v.calculate_weight(weight, key, alpha, strength_model, offset, function, intermediate_dtype, original_weights)
                if output is None:
                    logging.warning("Calculate Weight Failed: {} {}".format(v.name, key))
                else:
                    weight = output
                    if old_weight is not None:
                        weight = old_weight
                continue
    
            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]
    
            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += alpha * ldm_patched.modules.model_management.cast_to_device(w1, weight.device, weight.dtype)
            elif patch_type == "lora": # Fooocus specific LoRA handling
                mat1 = ldm_patched.modules.model_management.cast_to_device(v[0], weight.device, intermediate_dtype)
                mat2 = ldm_patched.modules.model_management.cast_to_device(v[1], weight.device, intermediate_dtype)
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    mat3 = ldm_patched.modules.model_management.cast_to_device(v[3], weight.device, intermediate_dtype)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1),
                                    mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(
                        weight.shape).type(weight.dtype)
                except Exception as e:
                    logging.error(f"ERROR {key} {e}")
            elif patch_type == "fooocus":
                w1 = ldm_patched.modules.model_management.cast_to_device(v[0], weight.device, intermediate_dtype)
                w_min = ldm_patched.modules.model_management.cast_to_device(v[1], weight.device, intermediate_dtype)
                w_max = ldm_patched.modules.model_management.cast_to_device(v[2], weight.device, intermediate_dtype)
                w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning("WARNING SHAPE MISMATCH {} FOOOCUS WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += alpha * ldm_patched.modules.model_management.cast_to_device(w1, weight.device, weight.dtype)
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dim = None
    
                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1_a, weight.device, intermediate_dtype),
                                  ldm_patched.modules.model_management.cast_to_device(w1_b, weight.device, intermediate_dtype))
                else:
                    w1 = ldm_patched.modules.model_management.cast_to_device(w1, weight.device, intermediate_dtype)
    
                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype),
                                      ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          ldm_patched.modules.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                          ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype),
                                          ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype))
                else:
                    w2 = ldm_patched.modules.model_management.cast_to_device(w2, weight.device, intermediate_dtype)
    
                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim
    
                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    logging.error(f"ERROR {key} {e}")
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t1, weight.device, intermediate_dtype),
                                      ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
                                      ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, intermediate_dtype))
    
                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                      ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
                                      ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, intermediate_dtype))
                else:
                    m1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, intermediate_dtype),
                                  ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, intermediate_dtype))
                    m2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, intermediate_dtype),
                                  ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, intermediate_dtype))
    
                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    logging.error(f"ERROR {key} {e}")
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]
    
                a1 = ldm_patched.modules.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, intermediate_dtype)
                a2 = ldm_patched.modules.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, intermediate_dtype)
                b1 = ldm_patched.modules.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, intermediate_dtype)
                b2 = ldm_patched.modules.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, intermediate_dtype)
    
                weight += ((torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)) * alpha).reshape(weight.shape).type(weight.dtype)
            elif patch_type == "set": # Standard Comfy 'set'
                weight.copy_(v[0])
            elif patch_type == "model_as_lora": # Standard Comfy 'model_as_lora'
                 target_weight: torch.Tensor = v[0]
                 diff_weight = ldm_patched.modules.model_management.cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                               ldm_patched.modules.model_management.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)
                 weight += function(alpha * ldm_patched.modules.model_management.cast_to_device(diff_weight, weight.device, weight.dtype))
            else:
                logging.warning("patch type not recognized {} {}".format(patch_type, key))
    
            if old_weight is not None:
                weight = old_weight
    
        return weight
    ```
3.  Update logic in `LowVramPatch.__call__` (inside `model_patcher.py`) to call `calculate_weight(...)` instead of `ldm_patched.modules.lora.calculate_weight(...)`.
4.  Update logic in `ModelPatcher.patch_weight_to_device` (inside `model_patcher.py`) to call `calculate_weight(...)` instead of `ldm_patched.modules.lora.calculate_weight(...)`.

### Part 2: Clean `modules/patch.py`
1.  Remove the `calculate_weight_patched` function completely.
2.  In `patch_all()`, REMOVE the line: `ldm_patched.modules.lora.calculate_weight = calculate_weight_patched`.

---

## Job 3: 1.5.C — `nex_loader.py` Adapter
**Goal:** Create a new module that delegates checkpoint loading.
**Instructions:**
1.  Create `modules/nex_loader.py`.
2.  Content:
    ```python
    import modules.config
    from ldm_patched.modules.sd import load_checkpoint_guess_config
    from modules.config import path_embeddings
    
    def load_checkpoint(ckpt_filename, vae_filename=None):
        """
        Loads a checkpoint using the standard pipeline.
        Future expansion: will support GGUF loading here.
        """
        # Delegate to existing ComfyUI/Fooocus loader
        unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(
            ckpt_filename, 
            embedding_directory=path_embeddings,
            vae_filename_param=vae_filename
        )
        return unet, clip, vae, vae_filename, clip_vision
    ```
3.  Verify that it imports cleanly.