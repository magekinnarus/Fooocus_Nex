import math
import numpy as np
from typing import List, NamedTuple
import modules.core as core
import modules.flags as flags
import modules.blending as blending
from backend import resources

class TileInfo(NamedTuple):
    crop: tuple  # (x1, y1, x2, y2)
    tile_image: np.ndarray
    x: int
    y: int
    w: int
    h: int

def select_tile_resolution(full_w, full_h, min_overlap=128):
    """
    Smart Auto-Tiling 2.0: Iterates through all buckets and selects the one 
    that minimizes the total tile count (nx * ny).
    """
    buckets = []
    for s in flags.sdxl_aspect_ratios:
        w, h = map(int, s.split('*'))
        buckets.append((w, h))

    best_config = None
    min_total_tiles = float('inf')
    min_waste = float('inf')

    for bw, bh in buckets:
        # Calculate nx: how many tiles of width bw to cover full_w with min_overlap
        if full_w <= bw:
            nx = 1
            overlap_w = 0
        else:
            nx = math.ceil((full_w - min_overlap) / (bw - min_overlap))
            overlap_w = (nx * bw - full_w) / (nx - 1)
        
        # Calculate ny: how many tiles of height bh to cover full_h with min_overlap
        if full_h <= bh:
            ny = 1
            overlap_h = 0
        else:
            ny = math.ceil((full_h - min_overlap) / (bh - min_overlap))
            overlap_h = (ny * bh - full_h) / (ny - 1)

        total_tiles = nx * ny
        # Total waste is a combination of overlaps and aspect ratio mismatch
        waste = (overlap_w if nx > 1 else (bw - full_w)) + (overlap_h if ny > 1 else (bh - full_h))

        if total_tiles < min_total_tiles or (total_tiles == min_total_tiles and waste < min_waste):
            min_total_tiles = total_tiles
            min_waste = waste
            best_config = ((bw, bh), nx, ny, int(overlap_w), int(overlap_h))

    bucket, nx, ny, overlap_w, overlap_h = best_config
    print(f'[Smart Tiling] Optimized Layout: {nx}x{ny} grid using {bucket[0]}x{bucket[1]} bucket.')
    print(f'[Smart Tiling] Actual Overlap: {overlap_w}px (Horiz), {overlap_h}px (Vert)')
    
    return bucket, nx, ny, overlap_w, overlap_h

def split_into_tiles(image: np.ndarray, bucket_w: int, bucket_h: int, nx: int, ny: int, overlap_w: int, overlap_h: int) -> List[TileInfo]:
    H, W, C = image.shape
    tiles = []
    
    stride_w = bucket_w - overlap_w
    stride_h = bucket_h - overlap_h
    
    for i in range(ny):
        for j in range(nx):
            x1 = j * stride_w
            y1 = i * stride_h
            
            # Boundary correction
            if x1 + bucket_w > W: x1 = W - bucket_w
            if y1 + bucket_h > H: y1 = H - bucket_h
            
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = x1 + bucket_w, y1 + bucket_h
            
            tile_image = image[y1:y2, x1:x2]
            
            # Final check: Ensure we ARE at bucket size (in case of very small images)
            if tile_image.shape[0] != bucket_h or tile_image.shape[1] != bucket_w:
                import cv2
                tile_image = cv2.resize(tile_image, (bucket_w, bucket_h), interpolation=cv2.INTER_LANCZOS4)

            tiles.append(TileInfo(
                crop=(x1, y1, x2, y2),
                tile_image=tile_image.copy(),
                x=x1, y=y1, w=bucket_w, h=bucket_h
            ))
            
    return tiles


def stitch_tiles(tiles: List[TileInfo], full_size: tuple, bucket_w: int, bucket_h: int) -> np.ndarray:
    H, W, C = full_size
    output = np.zeros((H, W, C), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    base_weight_map = blending.sin_blend_2d(bucket_w, bucket_h).cpu().numpy()
    
    for t in tiles:
        x1, y1, x2, y2 = t.crop
        output[y1:y2, x1:x2] += t.tile_image.astype(np.float32) * base_weight_map[:, :, None]
        weights[y1:y2, x1:x2] += base_weight_map
        
    output /= np.maximum(weights[:, :, None], 1e-5)
    return np.clip(output, 0, 255).astype(np.uint8)


def _require_unified_tiled_runtime_owner(task_state):
    runtime_owner = str(getattr(task_state, 'sdxl_runtime_owner', '') or '').strip().lower()
    if runtime_owner == 'unified':
        return

    policy = getattr(task_state, 'sdxl_execution_policy', None)
    if policy is None or not bool(getattr(policy, 'enabled', False)):
        reason = 'the SDXL execution policy is disabled'
    else:
        base_model_name = str(getattr(task_state, 'base_model_name', '') or '').strip().lower()
        if base_model_name.endswith('.gguf'):
            reason = 'GGUF SDXL execution is owned by a different runtime'
        else:
            reason = f"runtime owner resolved to {runtime_owner or 'legacy/shared'}"
    raise RuntimeError(
        'Super-upscale tiled refinement requires the unified SDXL runtime; '
        f'{reason}. No legacy tiled fallback is available.'
    )


def _resolve_tiled_prompt_blueprint(task_state, prompt_task=None):
    prompt_task = prompt_task or {}
    prompt = str(prompt_task.get('task_prompt', task_state.prompt) or '')
    negative_prompt = str(prompt_task.get('task_negative_prompt', task_state.negative_prompt) or '')

    positive_texts = tuple(str(item) for item in (prompt_task.get('positive') or [prompt]))
    negative_texts = tuple(str(item) for item in (prompt_task.get('negative') or [negative_prompt]))

    positive_top_k = int(prompt_task.get('positive_top_k', len(positive_texts)) or max(1, len(positive_texts)))
    negative_top_k = int(prompt_task.get('negative_top_k', len(negative_texts)) or max(1, len(negative_texts)))
    seed = int(prompt_task.get('task_seed', task_state.seed))

    return {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'positive_texts': positive_texts,
        'negative_texts': negative_texts,
        'positive_top_k': positive_top_k,
        'negative_top_k': negative_top_k,
        'seed': seed,
    }


def apply_tiled_diffusion_refinement(task_state, upscaled_image: np.ndarray, progressbar_callback=None, prompt_task=None):
    from backend import resources
    from backend.sdxl_unified_runtime import UnifiedSDXLRuntime, UnifiedSDXLRuntimeConfig
    from modules.pipeline.inference import _resolve_unified_checkpoint_path, _resolve_unified_vae_path
    import modules.pipeline.preprocessing as preprocessing

    _require_unified_tiled_runtime_owner(task_state)

    H, W, C = upscaled_image.shape

    with resources.memory_phase_scope(
        resources.MemoryPhase.TILED_REFINE,
        task=task_state,
        notes={'image_size': [W, H]},
        end_notes={'completed': True},
    ):
        # Pre-flight cleanup: Clear everything to maximize tile headroom.
        resources.cleanup_memory('tiled_refine_preflight', unload_models=True, force_cache=True, trim_host=True, target_phase=resources.MemoryPhase.TILED_REFINE)
        
        min_overlap = getattr(task_state, 'upscale_refinement_tile_overlap', 128)
        bucket, nx, ny, overlap_w, overlap_h = select_tile_resolution(W, H, min_overlap)
        bucket_w, bucket_h = bucket
        
        denoise = getattr(task_state, 'upscale_refinement_denoise', 0.382)
        tiles = split_into_tiles(upscaled_image, bucket_w, bucket_h, nx, ny, overlap_w, overlap_h)

        print(f'[Tiled Refinement] Processing {len(tiles)} tiles...')

        final_scheduler_name = preprocessing.patch_samplers(task_state)
        prompt_blueprint = _resolve_tiled_prompt_blueprint(task_state, prompt_task=prompt_task)

        # Merge active LoRAs
        resolved_loras = list(getattr(task_state, 'loras_processed', None) or getattr(task_state, 'loras', []) or [])
        merged_loras = tuple((str(path), float(weight)) for path, weight in resolved_loras)

        quality = {
            "sharpness": float(getattr(task_state, 'sharpness', 2.0)),
            "adaptive_cfg": float(getattr(task_state, 'adaptive_cfg', 7.0)),
            "adm_scaler_positive": float(getattr(task_state, 'adm_scaler_positive', 1.5)),
            "adm_scaler_negative": float(getattr(task_state, 'adm_scaler_negative', 0.8)),
            "adm_scaler_end": float(getattr(task_state, 'adm_scaler_end', 0.3)),
            "controlnet_softness": float(getattr(task_state, 'controlnet_softness', 0.25)),
        }

        policy = getattr(task_state, 'sdxl_execution_policy', None)
        stream_budget = float(getattr(policy, 'stream_budget_mb', 256.0))

        config_kwargs = dict(
            model_variant='sdxl',
            execution_class=str(
                getattr(task_state, 'sdxl_execution_family', None)
                or getattr(policy, 'execution_family', None)
                or 'standard_sdxl'
            ),
            streamlike_budget_mb=stream_budget,
            quality=quality,
            checkpoint_path=_resolve_unified_checkpoint_path(task_state),
            vae_path=_resolve_unified_vae_path(task_state),
            prompt=prompt_blueprint['prompt'],
            negative_prompt=prompt_blueprint['negative_prompt'],
            positive_texts=prompt_blueprint['positive_texts'],
            negative_texts=prompt_blueprint['negative_texts'],
            positive_top_k=prompt_blueprint['positive_top_k'],
            negative_top_k=prompt_blueprint['negative_top_k'],
            width=int(bucket_w),
            height=int(bucket_h),
            steps=int(task_state.steps),
            cfg=float(task_state.cfg_scale),
            sampler=str(task_state.sampler_name),
            scheduler=str(final_scheduler_name),
            seed=prompt_blueprint['seed'],
            clip_layer=-abs(int(getattr(task_state, 'clip_skip', 1) or 1)),
            batch_size=1,
            lora_specs=merged_loras,
            denoise_strength=denoise,
        )

        runtime = UnifiedSDXLRuntime(UnifiedSDXLRuntimeConfig(**config_kwargs))
        try:
            prepared_inputs, _ = runtime.prepare_inputs()
            
            refined_tiles = []
            for i, t in enumerate(tiles):
                if progressbar_callback:
                    progressbar_callback(task_state, int(task_state.current_progress + (i/len(tiles))*10), f'Refining tile {i+1}/{len(tiles)} ...')
                
                # VAE encode tile in VAE_ENCODE memory phase scope
                pixels = core.numpy_to_pytorch(t.tile_image)
                with resources.memory_phase_scope(
                    resources.MemoryPhase.VAE_ENCODE,
                    task=task_state,
                    notes={'route': 'tiled_refine', 'tile_size': [t.w, t.h], 'denoise': float(denoise)},
                    end_notes={'completed': True},
                ):
                    resources.load_models_gpu([runtime.vae.patcher])
                    latent_dict = core.encode_vae(vae=runtime.vae, pixels=pixels)
                
                # Update prepared inputs with current tile's latent samples
                prepared_inputs.payload["initial_latent"] = latent_dict["samples"]
                
                # Run denoise using unified runtime
                denoise_result = runtime.denoise_prepared_inputs(
                    prepared_inputs,
                    disable_pbar=True,
                )
                
                # Decode refined latent (non-tiled)
                decoded_images, _, _ = runtime.decode_latent(denoise_result.samples, tiled=False)
                
                refined_img = core.pytorch_to_numpy(decoded_images)[0]
                refined_tiles.append(t._replace(tile_image=refined_img))
                
                # Post-tile cleanup
                resources.cleanup_memory('tiled_refine_tile_complete', notes={'tile_index': i}, trim_host=False, target_phase=resources.MemoryPhase.TILED_REFINE)
                
        finally:
            runtime.close()
        
        if progressbar_callback:
            progressbar_callback(task_state, task_state.current_progress + 10, 'Stitching tiles ...')
            
        result = stitch_tiles(refined_tiles, (H, W, C), bucket_w, bucket_h)
        
        # Final sweep: Leave the GPU clean
        resources.cleanup_memory('tiled_refine_finalize', unload_models=True, force_cache=True, target_phase=resources.MemoryPhase.FINALIZE)
        
        return result
