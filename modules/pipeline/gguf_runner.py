from __future__ import annotations

import gc
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from backend import (
    anisotropic,
    cond_utils,
    conditioning,
    k_diffusion,
    loader,
    precision,
    resources,
    sampling,
)
from ldm_patched.modules import latent_formats
from modules import util
import modules.core as core
from modules.core import pytorch_to_numpy
from modules.model_manager import default_model_manager
from modules.pipeline.inference import get_sampling_callback
from modules.pipeline.output import save_and_log
from modules.pipeline.stage_runtime import PipelineRoute, PipelineRouteContext
from modules.sdxl_styles import apply_arrays, apply_style, get_random_style, random_style_name
from modules.util import parse_lora_references_from_prompt, remove_empty_str, safe_str
import modules.config
import modules.constants as constants

logger = logging.getLogger(__name__)


@dataclass
class DiffusionReadyArtifacts:
    """Phase-owned artifacts consumed by the dedicated GGUF denoise path."""

    positive_conds: list[dict[str, Any]]
    negative_conds: list[dict[str, Any]]
    initial_latent: torch.Tensor
    noise: torch.Tensor
    denoise_mask: Optional[torch.Tensor]
    sigmas: torch.Tensor
    conditioning_fingerprint: str
    delta_fingerprint: str
    engine_fingerprint: str
    input_fingerprint: Optional[str]
    controlnet_fingerprint: Optional[str]


@dataclass
class DiffusionReadyArtifactMetrics:
    clip_encode: float = 0.0
    adm_build: float = 0.0
    phase0_prep: float = 0.0
    cond_prepare: float = 0.0


class GGUFPipelineRunner:
    """
    Dedicated production runner for the low-VRAM SDXL GGUF txt2img path.
    """

    def run(self, route: PipelineRoute, context: PipelineRouteContext) -> PipelineRouteContext:
        task_state = context.task_state
        device = resources.get_torch_device()
        total_start = time.perf_counter()
        quality = self._build_quality(task_state)

        prompt_tasks, processed_loras, use_expansion = self._prepare_prompt_tasks(task_state)
        task_state.loras_processed = processed_loras
        task_state.use_expansion = use_expansion
        context.prompt_tasks = list(prompt_tasks)
        context.all_steps = max(int(task_state.steps) * max(len(prompt_tasks), 1), 1)

        if context.progressbar_callback is not None:
            task_state.current_progress += 1
            context.progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')

        unet = None
        clip = None
        vae = None
        using_shared_pipeline_components = False
        task_metrics: list[dict[str, float]] = []

        try:
            with resources.memory_phase_scope(resources.MemoryPhase.MODEL_REFRESH, task=task_state):
                shared_components = self._try_reuse_loaded_pipeline_components(context, processed_loras)
                if shared_components is not None:
                    using_shared_pipeline_components = True
                    unet, clip, vae = shared_components
                    clip.clip_layer(-abs(int(task_state.clip_skip or 1)))
                    loader.patch_unet_for_quality(unet, quality)
                else:
                    unet_path, clip_l_path, clip_g_path, vae_path = self._resolve_components(task_state)
                    unet = loader.load_sdxl_unet(unet_path, dtype=torch.float16)
                    loader.patch_unet_for_quality(unet, quality)

                    clip = loader.load_sdxl_clip(clip_l_path, clip_g_path, dtype=torch.float16)
                    clip.clip_layer(-abs(int(task_state.clip_skip or 1)))

                    vae = loader.load_vae(
                        vae_path,
                        dtype=torch.float32,
                        latent_format=latent_formats.SDXL(),
                    )

                    if processed_loras:
                        self._apply_loras_to_gguf(unet, clip, processed_loras)

            if context.progressbar_callback is not None:
                task_state.current_progress += 1
                context.progressbar_callback(task_state, task_state.current_progress, 'Processing prompts ...')

            if len(getattr(task_state, 'goals', [])) > 0:
                task_state.current_progress += 1
                if context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, task_state.current_progress, 'Image processing ...')

            context.preparation_steps = task_state.current_progress
            task_state.yields.append(['preview', (task_state.current_progress, 'Moving model to GPU ...', None)])
            processing_start_time = time.perf_counter()

            for i, task_dict in enumerate(prompt_tasks):
                if task_state.last_stop is not False:
                    resources.interrupt_current_processing()

                if context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, task_state.current_progress, f'Preparing task {i + 1}/{len(prompt_tasks)} ...')

                current_task_id = i
                total_count = len(prompt_tasks)
                sampling_callback = get_sampling_callback(
                    task_state,
                    context.progressbar_callback,
                    current_task_id,
                    total_count,
                    context.preparation_steps,
                    context.all_steps,
                )

                with resources.memory_phase_scope(
                    resources.MemoryPhase.DIFFUSION,
                    task=task_state,
                    notes={'task_index': i},
                    end_notes={'completed': True},
                ):
                    diffusion_ready, artifact_metrics = self._build_diffusion_ready_artifacts(
                        unet,
                        clip,
                        task_state,
                        task_dict,
                        context.sdxl_policy,
                        device,
                    )

                with resources.memory_phase_scope(
                    resources.MemoryPhase.DIFFUSION,
                    task=task_state,
                    notes={'task_index': i},
                    end_notes={'completed': True},
                ):
                    denoise_result = self._denoise(
                        unet,
                        diffusion_ready,
                        task_state,
                        quality,
                        task_dict['task_seed'],
                        sampling_callback,
                        device,
                    )

                with resources.memory_phase_scope(
                    resources.MemoryPhase.DECODE,
                    task=task_state,
                    notes={'task_index': i},
                    end_notes={'completed': True},
                ):
                    images, vae_attach_duration, vae_decode_duration = self._decode_latent(
                        vae,
                        denoise_result.samples,
                        device,
                    )

                numpy_images = pytorch_to_numpy(images)
                current_progress = int(
                    context.preparation_steps
                    + ((100 - context.preparation_steps) / float(context.all_steps)) * ((i + 1) * int(task_state.steps))
                )

                if context.progressbar_callback is not None:
                    context.progressbar_callback(
                        task_state,
                        current_progress,
                        f'Saving image {i + 1}/{len(prompt_tasks)} to system ...',
                    )

                img_paths = save_and_log(
                    task_state,
                    task_state.height,
                    task_state.width,
                    numpy_images,
                    task_dict,
                    task_state.use_expansion,
                    processed_loras,
                )

                if context.yield_result_callback is not None:
                    show_results = not bool(getattr(task_state, 'disable_intermediate_results', False))
                    context.yield_result_callback(
                        task_state,
                        img_paths,
                        current_progress,
                        do_not_show_finished_images=not show_results,
                    )

                task_metrics.append(
                    {
                        'clip_encode': artifact_metrics.clip_encode,
                        'adm_build': artifact_metrics.adm_build,
                        'latent_noise_prep': artifact_metrics.phase0_prep,
                        'cond_prepare_explicit': artifact_metrics.cond_prepare,
                        'sampler_model_attach': denoise_result.sampler_model_attach,
                        'denoise_wall': denoise_result.denoise_wall,
                        'gguf_dequant': float(denoise_result.gguf_trace_stats.get('dequant_seconds', 0.0)),
                        'vae_attach': vae_attach_duration,
                        'vae_decode': vae_decode_duration,
                    }
                )
                resources.cleanup_memory(
                    'gguf_task_image_complete',
                    gc_collect=False,
                    notes={'task_index': i},
                    target_phase=resources.MemoryPhase.DECODE,
                    task=task_state,
                )

            total_wall = time.perf_counter() - total_start
            processing_wall = time.perf_counter() - processing_start_time
            telemetry = self._build_telemetry(task_metrics, total_wall, processing_wall, len(prompt_tasks), task_state.steps)
            context.complete_route(telemetry=telemetry, completed=True, tasks_processed=len(prompt_tasks))
            return context
        finally:
            if unet is not None:
                unet.detach()
            if vae is not None:
                vae.patcher.detach()
            if clip is not None:
                clip.patcher.detach()
            if not using_shared_pipeline_components:
                gc.collect()
            resources.soft_empty_cache(force=True)

    def _build_quality(self, task_state: Any) -> Dict[str, Any]:
        quality = {
            "sharpness": task_state.sharpness,
            "adm_scaler_end": task_state.adm_scaler_end,
            "adm_scale_positive": task_state.adm_scaler_positive,
            "adm_scale_negative": task_state.adm_scaler_negative,
        }
        if task_state.adaptive_cfg > 0:
            quality["adaptive_cfg"] = task_state.adaptive_cfg
        return quality

    def _build_telemetry(
        self,
        task_metrics: list[dict[str, float]],
        total_wall: float,
        processing_wall: float,
        tasks_processed: int,
        steps: int,
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {
            "route_label": "gguf_lowvram_runner",
            "tasks_processed": tasks_processed,
            "total_wall": total_wall,
            "processing_wall": processing_wall,
        }
        if not task_metrics:
            return telemetry

        keys = (
            'clip_encode',
            'adm_build',
            'latent_noise_prep',
            'cond_prepare_explicit',
            'sampler_model_attach',
            'denoise_wall',
            'gguf_dequant',
            'vae_attach',
            'vae_decode',
        )
        for key in keys:
            telemetry[key] = sum(metric[key] for metric in task_metrics) / len(task_metrics)
        telemetry['denoise_s_per_it'] = telemetry['denoise_wall'] / max(1, int(steps))
        return telemetry

    def _prepare_prompt_tasks(self, task_state: Any) -> tuple[list[dict[str, Any]], list[tuple[str, float]], bool]:
        prompt = task_state.prompt
        negative_prompt = task_state.negative_prompt
        image_number = int(getattr(task_state, 'image_number', 1) or 1)
        disable_seed_increment = bool(getattr(task_state, 'disable_seed_increment', False))
        use_expansion = bool(getattr(task_state, 'use_expansion', False))
        use_style = bool(getattr(task_state, 'use_style', True))

        prompts = remove_empty_str([safe_str(p) for p in str(prompt).splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in str(negative_prompt).splitlines()], default='')
        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        edit_additional_prompt = ''
        if 'inpaint' in getattr(task_state, 'goals', []):
            edit_additional_prompt = str(getattr(task_state, 'inpaint_additional_prompt', '') or '')
        elif 'outpaint' in getattr(task_state, 'goals', []):
            edit_additional_prompt = str(getattr(task_state, 'outpaint_additional_prompt', '') or '')

        if edit_additional_prompt:
            prompt = edit_additional_prompt if prompt == '' else edit_additional_prompt + '\n' + prompt

        if prompt == '':
            use_expansion = False

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

        processed_loras, prompt = parse_lora_references_from_prompt(
            prompt,
            list(getattr(task_state, 'loras', [])),
            modules.config.default_max_lora_number,
        )

        tasks: list[dict[str, Any]] = []
        task_rng = random.Random(task_state.seed)
        base_styles = list(getattr(task_state, 'style_selections', []) or [])

        for i in range(image_number):
            if disable_seed_increment:
                task_seed = task_state.seed % (constants.MAX_SEED + 1)
            else:
                task_seed = (task_state.seed + i) % (constants.MAX_SEED + 1)

            task_prompt = apply_arrays(prompt, i)
            task_negative_prompt = negative_prompt
            positive_basic_workloads: list[str] = []
            negative_basic_workloads: list[str] = []

            task_styles = list(base_styles)
            if use_style:
                for j, style_name in enumerate(task_styles):
                    selected_style = style_name
                    if selected_style == random_style_name:
                        selected_style = get_random_style(task_rng)
                        task_styles[j] = selected_style
                    p, n, _ = apply_style(selected_style, positive=task_prompt)
                    positive_basic_workloads += p
                    negative_basic_workloads += n

                positive_basic_workloads = [task_prompt] + positive_basic_workloads
                negative_basic_workloads = [task_negative_prompt] + negative_basic_workloads
            else:
                positive_basic_workloads.append(task_prompt)
                negative_basic_workloads.append(task_negative_prompt)

            positive_basic_workloads += extra_positive_prompts
            negative_basic_workloads += extra_negative_prompts
            positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

            tasks.append(
                dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + extra_negative_prompts),
                    styles=task_styles,
                )
            )

        return tasks, processed_loras, use_expansion

    def _resolve_components(self, task_state: Any) -> Tuple[str, str, str, str]:
        unet_path = self._resolve_model_path(
            task_state.base_model_name,
            root_key='unet',
            folder_paths=modules.config.path_unet,
            required_label='GGUF UNet',
        )

        clip_entry = default_model_manager.resolve_companion_clip(unet_path, installed_only=True)
        if clip_entry is None:
            raise RuntimeError(f"Could not resolve companion CLIP for GGUF model: {unet_path}")

        clip_path = self._resolve_model_path(
            clip_entry,
            root_key='clip',
            folder_paths=modules.config.paths_clips,
            required_label='companion CLIP',
        )
        vae_name = task_state.vae_name
        if not vae_name or vae_name == 'Default (Same as model)':
            vae_path = self._find_existing_file('sdxl_vae.safetensors', modules.config.path_vae)
            if vae_path is None:
                raise RuntimeError("Could not find default SDXL VAE (sdxl_vae.safetensors)")
        else:
            vae_path = self._find_existing_file(vae_name, modules.config.path_vae)
            if vae_path is None:
                raise RuntimeError(f"Could not find selected VAE: {vae_name}")

        return unet_path, clip_path, clip_path, vae_path

    def _resolve_model_path(
        self,
        selector_or_entry: Any,
        *,
        root_key: str,
        folder_paths: Any,
        required_label: str,
    ) -> str:
        direct_path = getattr(selector_or_entry, 'absolute_path', None)
        if isinstance(direct_path, str) and os.path.isfile(direct_path):
            return direct_path

        if isinstance(selector_or_entry, str) and os.path.isfile(selector_or_entry):
            return selector_or_entry

        inventory_path = None
        if hasattr(selector_or_entry, 'id') and hasattr(selector_or_entry, 'root_key'):
            inventory = default_model_manager.inventory_record(selector_or_entry)
            inventory_path = getattr(inventory, 'installed_path', None)
            if isinstance(inventory_path, str) and os.path.isfile(inventory_path):
                return inventory_path

        for candidate in (
            getattr(selector_or_entry, 'relative_path', None),
            getattr(selector_or_entry, 'name', None),
            selector_or_entry if isinstance(selector_or_entry, str) else None,
        ):
            resolved_path = self._find_existing_file(candidate, folder_paths)
            if resolved_path is not None:
                return resolved_path

        catalog_entry = modules.config.resolve_model_catalog_entry(
            selector_or_entry if isinstance(selector_or_entry, str) else getattr(selector_or_entry, 'id', None),
            root_keys=(root_key,),
            folder_paths=folder_paths,
        )
        if catalog_entry is not None and catalog_entry is not selector_or_entry:
            inventory = default_model_manager.inventory_record(catalog_entry)
            inventory_path = getattr(inventory, 'installed_path', None)
            if isinstance(inventory_path, str) and os.path.isfile(inventory_path):
                return inventory_path
            resolved_path = self._find_existing_file(getattr(catalog_entry, 'relative_path', None), folder_paths)
            if resolved_path is not None:
                return resolved_path

        attempted_selector = (
            direct_path
            or inventory_path
            or getattr(selector_or_entry, 'relative_path', None)
            or getattr(selector_or_entry, 'name', None)
            or selector_or_entry
        )
        raise RuntimeError(f"Could not resolve installed path for {required_label}: {attempted_selector}")

    def _find_existing_file(self, name: Any, folders: Any) -> str | None:
        if not name:
            return None

        if isinstance(name, str) and os.path.isfile(name):
            return name

        candidate = util.get_file_from_folder_list(name, folders)
        if isinstance(candidate, str) and os.path.isfile(candidate):
            return candidate
        return None

    def _try_reuse_loaded_pipeline_components(
        self,
        context: PipelineRouteContext,
        processed_loras: list[tuple[str, float]],
    ) -> tuple[Any, Any, Any] | None:
        try:
            import modules.default_pipeline as default_pipeline
        except Exception:
            return None

        refresh_state = getattr(default_pipeline, 'refresh_state', None)
        final_unet = getattr(default_pipeline, 'final_unet', None)
        final_clip = getattr(default_pipeline, 'final_clip', None)
        final_vae = getattr(default_pipeline, 'final_vae', None)
        if not isinstance(refresh_state, dict) or final_unet is None or final_clip is None or final_vae is None:
            return None

        task_state = context.task_state
        normalize_selector = lambda value: str(value or '').strip()
        expected_state = {
            'base_model_name': normalize_selector(task_state.base_model_name),
            'loras': sorted(processed_loras),
            'base_model_additional_loras': sorted(context.base_model_additional_loras or []),
            'vae_name': normalize_selector(task_state.vae_name),
            'clip_name': normalize_selector(task_state.clip_model_name),
        }
        actual_state = {
            'base_model_name': normalize_selector(refresh_state.get('base_model_name')),
            'loras': sorted(refresh_state.get('loras') or []),
            'base_model_additional_loras': sorted(refresh_state.get('base_model_additional_loras') or []),
            'vae_name': normalize_selector(refresh_state.get('vae_name')),
            'clip_name': normalize_selector(refresh_state.get('clip_name')),
        }
        if expected_state != actual_state:
            return None

        print('[Nex-Pipeline] Reusing already-loaded UI SDXL components for dedicated GGUF runner.')
        return final_unet, final_clip, final_vae

    def _apply_loras_to_gguf(self, unet: Any, clip: Any, loras: List[Tuple[str, float]]) -> None:
        from backend import lora as lora_backend
        from backend.lora import model_lora_keys_clip, model_lora_keys_unet

        unet_keys = model_lora_keys_unet(unet.model, {})
        unet_keys.update({key: key for key in unet.model.state_dict().keys()})

        clip_keys = model_lora_keys_clip(clip.cond_stage_model, {})
        clip_keys.update({key: key for key in clip.cond_stage_model.state_dict().keys()})

        for filename, weight in loras:
            if filename == 'None':
                continue

            lora_path = util.get_file_from_folder_list(filename, modules.config.paths_lora_lookup)
            if not lora_path:
                logger.warning("LoRA file not found: %s", filename)
                continue

            lora_sd = loader.utils.load_torch_file(lora_path)
            try:
                unet_patches = lora_backend.load_lora(lora_sd, unet_keys, log_missing=False)
                if unet_patches:
                    unet.add_patches(unet_patches, weight)

                clip_patches = lora_backend.load_lora(lora_sd, clip_keys, log_missing=False)
                if clip_patches:
                    clip.add_patches(clip_patches, weight)
            finally:
                del lora_sd
                resources.soft_empty_cache(force=False)

    def _encode_prompt_pair(
        self,
        clip: Any,
        positive_texts: list[str],
        negative_texts: list[str],
        positive_top_k: int,
        negative_top_k: int,
        policy: Any,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        residency_mode = getattr(policy, 'clip_residency_mode', 'gpu_then_offload')
        encode_start = time.perf_counter()

        if residency_mode == 'cpu_only':
            with torch.inference_mode():
                positive_cond, positive_pooled = self._encode_text_workloads(
                    clip,
                    positive_texts,
                    positive_top_k,
                    use_explicit_residency=False,
                )
                negative_cond, negative_pooled = self._encode_text_workloads(
                    clip,
                    negative_texts,
                    negative_top_k,
                    use_explicit_residency=False,
                )
            return {
                "positive": {"cond": positive_cond, "pooled": positive_pooled},
                "negative": {"cond": negative_cond, "pooled": negative_pooled},
            }, {"clip_encode": time.perf_counter() - encode_start}

        resources.prepare_models_for_stage(
            [clip.patcher],
            stage_name="text_encode_reconcile",
            target_phase=resources.MemoryPhase.PROMPT_ENCODE,
        )
        clip.patcher.patch_model(device_to=device, lowvram_model_memory=0)

        try:
            with torch.inference_mode(), precision.autocast_context(device, enabled=True):
                positive_cond, positive_pooled = self._encode_text_workloads(
                    clip,
                    positive_texts,
                    positive_top_k,
                    use_explicit_residency=True,
                )
                negative_cond, negative_pooled = self._encode_text_workloads(
                    clip,
                    negative_texts,
                    negative_top_k,
                    use_explicit_residency=True,
                )
        finally:
            clip.patcher.detach()

        return {
            "positive": {"cond": positive_cond, "pooled": positive_pooled},
            "negative": {"cond": negative_cond, "pooled": negative_pooled},
        }, {"clip_encode": time.perf_counter() - encode_start}

    def _encode_text_workloads(
        self,
        clip: Any,
        texts: list[str],
        pool_top_k: int,
        *,
        use_explicit_residency: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_list: list[torch.Tensor] = []
        pooled_acc: Optional[torch.Tensor] = None

        for i, text in enumerate(texts):
            cond, pooled = conditioning.encode_text_sdxl(
                clip,
                text,
                use_explicit_residency=use_explicit_residency,
            )
            cond_list.append(cond)
            if i < pool_top_k:
                pooled_acc = pooled if pooled_acc is None else pooled_acc + pooled

        if not cond_list or pooled_acc is None:
            raise RuntimeError("GGUF prompt encoding produced no conditioning payload.")

        return torch.cat(cond_list, dim=1), pooled_acc

    def _build_diffusion_ready_artifacts(
        self,
        unet: Any,
        clip: Any,
        task_state: Any,
        task_dict: dict[str, Any],
        policy: Any,
        device: torch.device,
    ) -> tuple[DiffusionReadyArtifacts, DiffusionReadyArtifactMetrics]:
        metrics = DiffusionReadyArtifactMetrics()

        with resources.memory_phase_scope(
            resources.MemoryPhase.PROMPT_ENCODE,
            task=task_state,
            notes={'task_seed': task_dict['task_seed']},
            end_notes={'completed': True},
        ):
            encoded_prompt_pair, encode_metrics = self._encode_prompt_pair(
                clip,
                task_dict['positive'],
                task_dict['negative'],
                task_dict['positive_top_k'],
                task_dict['negative_top_k'],
                policy,
                device,
            )
        metrics.clip_encode = encode_metrics['clip_encode']

        adm_start = time.perf_counter()
        adm_pair = conditioning.build_sdxl_adm_pair(
            encoded_prompt_pair,
            task_state.width,
            task_state.height,
            target_width=task_state.width,
            target_height=task_state.height,
            adm_scale_positive=task_state.adm_scaler_positive,
            adm_scale_negative=task_state.adm_scaler_negative,
        )
        metrics.adm_build = time.perf_counter() - adm_start

        positive = [[
            encoded_prompt_pair["positive"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["positive"]["pooled"],
                "model_conds": {"y": adm_pair["positive"]},
            },
        ]]
        negative = [[
            encoded_prompt_pair["negative"]["cond"],
            {
                "pooled_output": encoded_prompt_pair["negative"]["pooled"],
                "model_conds": {"y": adm_pair["negative"]},
            },
        ]]

        phase0_start = time.perf_counter()
        initial_latent, noise, denoise_mask, input_fingerprint = self._prepare_phase0_inputs(
            unet,
            task_state,
            task_dict['task_seed'],
            device,
        )
        metrics.phase0_prep = time.perf_counter() - phase0_start

        processed_conds, metrics.cond_prepare = self._prepare_direct_conds(
            unet,
            positive,
            negative,
            task_dict['task_seed'],
            initial_latent,
            noise,
            denoise_mask,
            device,
        )
        sigmas = self._calculate_sigmas(
            unet,
            task_state,
            device,
            self._build_quality(task_state),
        )

        return DiffusionReadyArtifacts(
            positive_conds=processed_conds.get("positive") or [],
            negative_conds=processed_conds.get("negative") or [],
            initial_latent=initial_latent,
            noise=noise,
            denoise_mask=denoise_mask,
            sigmas=sigmas,
            conditioning_fingerprint="",
            delta_fingerprint="",
            engine_fingerprint=str(getattr(task_state, 'base_model_name', '') or ''),
            input_fingerprint=input_fingerprint,
            controlnet_fingerprint=None,
        ), metrics

    def _prepare_phase0_inputs(
        self,
        unet: Any,
        task_state: Any,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[str]]:
        prepared = getattr(task_state, 'initial_latent', None)
        if isinstance(prepared, dict) and isinstance(prepared.get("samples"), torch.Tensor):
            dtype = unet.model.get_dtype()
            latent = prepared["samples"].to(device=device, dtype=dtype)
            denoise_mask = prepared.get("noise_mask")
            if isinstance(denoise_mask, torch.Tensor):
                denoise_mask = denoise_mask.to(device=device, dtype=dtype)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            input_fingerprint = (
                f"prepared:{tuple(int(dim) for dim in latent.shape)}:"
                f"{tuple(int(dim) for dim in denoise_mask.shape) if isinstance(denoise_mask, torch.Tensor) else 'none'}"
            )
            return latent, noise, denoise_mask, input_fingerprint

        latent, noise = self._create_latent_and_noise(
            unet,
            task_state.width,
            task_state.height,
            seed,
            device,
        )
        return latent, noise, None, None

    def _prepare_direct_conds(
        self,
        unet: Any,
        positive: Any,
        negative: Any,
        seed: int,
        latent_image: torch.Tensor,
        noise: torch.Tensor,
        denoise_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[Dict[str, Any], float]:
        def convert_sampler_cond(cond: Any) -> list[Dict[str, Any]]:
            out = []
            for cross_attn, payload in cond:
                converted = payload.copy()
                if cross_attn is not None:
                    converted["cross_attn"] = cross_attn
                converted["model_conds"] = converted.get("model_conds", {})
                converted["uuid"] = uuid.uuid4()
                out.append(converted)
            return out

        conds = {
            "positive": convert_sampler_cond(positive),
            "negative": convert_sampler_cond(negative),
        }

        cond_start = time.perf_counter()
        processed = cond_utils.process_conds(
            unet.model,
            noise,
            conds,
            device,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            seed=seed,
        )
        return processed, time.perf_counter() - cond_start

    def _calculate_sigmas(
        self,
        unet: Any,
        task_state: Any,
        device: torch.device,
        quality: Dict[str, Any],
    ) -> torch.Tensor:
        sampler_instance = sampling.KSampler(
            unet,
            task_state.steps,
            device,
            task_state.sampler_name,
            task_state.scheduler_name,
            float(getattr(task_state, 'denoising_strength', 1.0) or 1.0),
            model_options={"quality": quality},
        )
        return sampler_instance.sigmas

    def _create_latent_and_noise(
        self,
        unet: Any,
        width: int,
        height: int,
        seed: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_h = height // 8
        latent_w = width // 8
        dtype = unet.model.get_dtype()

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        noise = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latent = torch.zeros(
            (1, 4, latent_h, latent_w),
            device=device,
            dtype=dtype,
        )
        return latent, noise

    def _denoise(
        self,
        unet: Any,
        diffusion_ready: DiffusionReadyArtifacts,
        task_state: Any,
        quality: Dict[str, Any],
        seed: int,
        sampling_callback: Optional[Callable[..., None]],
        device: torch.device,
    ) -> Any:
        sampler_instance = sampling.KSampler(
            unet,
            task_state.steps,
            device,
            task_state.sampler_name,
            task_state.scheduler_name,
            float(getattr(task_state, 'denoising_strength', 1.0) or 1.0),
            model_options={"quality": quality},
        )
        sigmas = diffusion_ready.sigmas if diffusion_ready.sigmas is not None else sampler_instance.sigmas
        if sigmas.shape[-1] == 0:
            from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFDenoiseResult

            return DirectSDXLGGUFDenoiseResult(
                samples=diffusion_ready.initial_latent,
                cond_prepare_duration=0.0,
                sampler_model_attach=0.0,
                denoise_wall=0.0,
                denoise_cpu_proc=0.0,
                gguf_trace_stats={},
            )

        sampler_function = self._resolve_sampler_function(task_state.sampler_name)

        attach_start = time.perf_counter()
        budget_bytes = int(resources.maximum_vram_for_weights(device))
        model_size = int(unet.model_size())
        lowvram_model_memory = 0 if budget_bytes >= model_size else budget_bytes
        unet.patch_model(device_to=device, lowvram_model_memory=lowvram_model_memory)
        sampler_model_attach_duration = time.perf_counter() - attach_start

        from backend.gguf import ops as gguf_ops

        gguf_ops.reset_trace_stats()
        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        previewer = self._resolve_previewer(unet, task_state)

        try:
            with torch.inference_mode(), precision.autocast_context(device):
                model_sampling = unet.model.model_sampling
                max_sigma = float(model_sampling.sigma_max)
                sigma = float(sigmas[0])
                max_denoise = math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma
                scaled_noise = model_sampling.noise_scaling(
                    sigmas[0],
                    diffusion_ready.noise,
                    diffusion_ready.initial_latent,
                    max_denoise,
                )

                total_steps = len(sigmas) - 1

                def k_callback(payload):
                    if sampling_callback is not None:
                        denoised = payload.get("denoised")
                        preview_image = None
                        if previewer is not None and denoised is not None:
                            try:
                                preview_image = previewer(denoised, payload["i"], total_steps)
                            except Exception:
                                preview_image = None
                        sampling_callback(
                            payload["i"],
                            denoised,
                            payload.get("x"),
                            total_steps,
                            preview_image,
                        )

                samples = sampler_function(
                    self._build_model_callable(unet, diffusion_ready, task_state, quality),
                    scaled_noise,
                    sigmas,
                    extra_args={"denoise_mask": diffusion_ready.denoise_mask, "seed": seed},
                    callback=k_callback if sampling_callback is not None else None,
                    disable=True,
                )
                samples = model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        finally:
            denoise_wall = time.perf_counter() - denoise_start
            denoise_cpu_proc = time.process_time() - denoise_cpu_start
            gguf_trace_stats = dict(gguf_ops.consume_trace_stats())
            unet.detach()

        from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFDenoiseResult

        return DirectSDXLGGUFDenoiseResult(
            samples=samples,
            cond_prepare_duration=0.0,
            sampler_model_attach=sampler_model_attach_duration,
            denoise_wall=denoise_wall,
            denoise_cpu_proc=denoise_cpu_proc,
            gguf_trace_stats=gguf_trace_stats,
        )

    def _resolve_previewer(self, unet: Any, task_state: Any) -> Optional[Callable[..., Any]]:
        if bool(getattr(task_state, 'disable_preview', False)):
            return None
        try:
            return core.get_previewer(unet)
        except Exception:
            return None

    def _resolve_sampler_function(self, sampler_name: str) -> Callable:
        if sampler_name == "dpm_fast":
            def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_fast(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    len(sigmas) - 1,
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_fast_function

        if sampler_name == "dpm_adaptive":
            def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
                if len(sigmas) <= 1:
                    return noise
                sigma_min = sigmas[-1] if sigmas[-1] > 0 else sigmas[-2]
                return k_diffusion.sample_dpm_adaptive(
                    model,
                    noise,
                    sigma_min,
                    sigmas[0],
                    extra_args=extra_args,
                    callback=callback,
                    disable=disable,
                )

            return dpm_adaptive_function

        func_name = f"sample_{sampler_name.replace('_cfg_pp', '')}"
        sampler_function = getattr(k_diffusion, func_name, None)
        if sampler_function is None:
            raise ValueError(f"Sampler {sampler_name} not implemented in k_diffusion")
        return sampler_function

    def _build_model_callable(
        self,
        unet: Any,
        diffusion_ready: DiffusionReadyArtifacts,
        task_state: Any,
        quality: Dict[str, Any],
    ) -> Callable:
        def model_fn(x: torch.Tensor, sigma: torch.Tensor, **_: Any) -> torch.Tensor:
            negative_conds = diffusion_ready.negative_conds
            if math.isclose(task_state.cfg_scale, 1.0):
                negative_conds = None

            cond_pred, uncond_pred = self._calc_fullframe_cond_batch(
                unet,
                [diffusion_ready.positive_conds, negative_conds],
                x,
                sigma,
            )
            return self._apply_cfg(unet, task_state, quality, x, sigma, cond_pred, uncond_pred)

        return model_fn

    def _calc_fullframe_cond_batch(
        self,
        unet: Any,
        conds: list[Optional[list[Dict[str, Any]]]],
        x_in: torch.Tensor,
        timestep: torch.Tensor,
    ) -> list[torch.Tensor]:
        out_conds = [torch.zeros_like(x_in) for _ in conds]
        out_counts = [torch.ones_like(x_in) * 1e-37 for _ in conds]
        to_run = []

        for cond_index, cond in enumerate(conds):
            if cond is None:
                continue
            for cond_entry in cond:
                prepared = cond_utils.get_area_and_mult(cond_entry, x_in, timestep)
                if prepared is None:
                    continue
                if prepared.area is not None or prepared.input_x.shape != x_in.shape:
                    raise ValueError("GGUF txt2img runner only supports full-frame conditions.")
                to_run.append((prepared, cond_index))

        while len(to_run) > 0:
            first = to_run[0]
            to_batch = []
            for index in range(len(to_run)):
                if cond_utils.can_concat_cond(to_run[index][0], first[0]):
                    to_batch.append(index)

            batch_items = [to_run[index] for index in to_batch]
            for index in sorted(to_batch, reverse=True):
                to_run.pop(index)

            batch_input_x = [prepared.input_x for prepared, _ in batch_items]
            batch_mult = [prepared.mult for prepared, _ in batch_items]
            batch_conditioning = [prepared.conditioning for prepared, _ in batch_items]
            batch_cond_indices = [cond_index for _, cond_index in batch_items]

            input_x = torch.cat(batch_input_x)
            conditioning_batch = cond_utils.cond_cat(batch_conditioning)
            timestep_batch = torch.cat([timestep] * len(batch_cond_indices))
            outputs = unet.model.apply_model(input_x, timestep_batch, **conditioning_batch).chunk(len(batch_cond_indices))

            for output, cond_index, mult in zip(outputs, batch_cond_indices, batch_mult):
                out_conds[cond_index] += output * mult
                out_counts[cond_index] += mult

        for index in range(len(out_conds)):
            out_conds[index] /= out_counts[index]
        return out_conds

    def _apply_cfg(
        self,
        unet: Any,
        task_state: Any,
        quality: Dict[str, Any],
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
    ) -> torch.Tensor:
        model_sampling = unet.model.model_sampling
        t = model_sampling.timestep(timestep).float()
        diffusion_progress = max(0.0, min(1.0, 1.0 - float(t.reshape(-1)[0].item()) / 999.0))

        sharpness = float(quality.get("sharpness", 0.0))
        if sharpness > 0.0:
            alpha = 0.001 * sharpness * diffusion_progress
            if alpha >= 0.01:
                positive_eps = x - cond_pred
                degraded_eps = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=cond_pred)
                positive_eps_weighted = degraded_eps * alpha + positive_eps * (1.0 - alpha)
                cond_pred = x - positive_eps_weighted

        adaptive_cfg = float(quality.get("adaptive_cfg", 0.0))
        if adaptive_cfg > 0.0 and task_state.cfg_scale > adaptive_cfg:
            cond_eps = x - cond_pred
            uncond_eps = x - uncond_pred
            real_eps = uncond_eps + task_state.cfg_scale * (cond_eps - uncond_eps)
            mimic_eps = uncond_eps + adaptive_cfg * (cond_eps - uncond_eps)
            final_eps = real_eps * diffusion_progress + mimic_eps * (1.0 - diffusion_progress)
            return x - final_eps

        if "_cfg_pp" in task_state.sampler_name:
            return cond_pred + (task_state.cfg_scale - 1.0) * (cond_pred - uncond_pred)
        return uncond_pred + (cond_pred - uncond_pred) * task_state.cfg_scale

    def _decode_latent(self, vae: Any, latent: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float, float]:
        attach_start = time.perf_counter()
        vae.patcher.patch_model(device_to=device, lowvram_model_memory=0)
        vae.first_stage_model.to(device=device, dtype=torch.float32)
        vae_attach_duration = time.perf_counter() - attach_start

        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                scaled_latent = vae.latent_format.process_out(latent)

                try:
                    direct_latent = scaled_latent.to(device=device, dtype=torch.float32)
                    pixels = vae.first_stage_model.decode(direct_latent).float()
                    images = torch.clamp((pixels + 1.0) / 2.0, min=0.0, max=1.0).movedim(1, -1).cpu()
                except (resources.OOM_EXCEPTION, torch.OutOfMemoryError):
                    resources.soft_empty_cache(force=True)
                    decode_dtype = next(vae.first_stage_model.parameters()).dtype
                    decode_fn = lambda latent_tensor: vae.first_stage_model.decode(
                        latent_tensor.to(device=device, dtype=decode_dtype)
                    ).float()

                    from backend import utils as backend_utils

                    p3 = backend_utils.tiled_scale(
                        scaled_latent,
                        decode_fn,
                        64,
                        64,
                        16,
                        upscale_amount=8,
                        output_device='cpu',
                    )
                    p1 = backend_utils.tiled_scale(
                        scaled_latent,
                        decode_fn,
                        32,
                        128,
                        16,
                        upscale_amount=8,
                        output_device='cpu',
                    )
                    p2 = backend_utils.tiled_scale(
                        scaled_latent,
                        decode_fn,
                        128,
                        32,
                        16,
                        upscale_amount=8,
                        output_device='cpu',
                    )

                    pixels = (p1 + p2 + p3) / 3.0
                    images = torch.clamp((pixels + 1.0) / 2.0, min=0.0, max=1.0).movedim(1, -1)
        finally:
            vae.patcher.detach()

        vae_decode_duration = time.perf_counter() - decode_start
        return images, vae_attach_duration, vae_decode_duration
