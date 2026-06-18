from __future__ import annotations
"""
Nex Compatibility-Only / Tooling-Only / Legacy-Origin Module.
This module supports legacy GGUF running and is not used in modern/authoritative production generation routes.
"""


import gc
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from threading import RLock
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
from backend import lora_artifacts
from ldm_patched.modules import latent_formats
from modules import util
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


@dataclass(frozen=True)
class ResolvedGGUFLoraArtifact:
    requested_name: str
    resolved_path: str
    artifact: lora_artifacts.AdapterArtifact
    target_families: tuple[str, ...]
    clip_target_keys: tuple[str, ...]
    unet_target_keys: tuple[str, ...]

    @property
    def is_clip_side(self) -> bool:
        return len(self.clip_target_keys) > 0

    @property
    def is_unet_side(self) -> bool:
        return len(self.unet_target_keys) > 0


@dataclass
class GGUFWarmArtifactState:
    """
    Dedicated GGUF warm state that survives across requests.

    W06g Step 1 establishes the persistence authority only. Selective
    fingerprint-driven reuse is wired in later steps.
    """

    warm_unet: Any = None
    engine_fingerprint: Optional[str] = None
    conditioning_fingerprint: Optional[str] = None
    delta_fingerprint: Optional[str] = None
    input_fingerprint: Optional[str] = None
    controlnet_fingerprint: Optional[str] = None
    conditioning_artifacts: Optional[dict[str, Any]] = None
    delta_artifacts: Optional[dict[str, Any]] = None
    input_artifacts: Optional[dict[str, Any]] = None
    controlnet_hints: Any = None

    def clear_cached_artifacts(self) -> None:
        self.conditioning_fingerprint = None
        self.delta_fingerprint = None
        self.input_fingerprint = None
        self.controlnet_fingerprint = None
        self.conditioning_artifacts = None
        self.delta_artifacts = None
        self.input_artifacts = None
        self.controlnet_hints = None

    def reset(self, *, release_engine: bool = False) -> None:
        if release_engine:
            _detach_gguf_warm_unet(self.warm_unet)
            self.warm_unet = None
            self.engine_fingerprint = None
        self.clear_cached_artifacts()


_GGUF_WARM_STATE_LOCK = RLock()
_GGUF_WARM_STATE = GGUFWarmArtifactState()


def _detach_gguf_warm_unet(unet: Any) -> None:
    if unet is None:
        return
    try:
        unet.detach()
    except Exception:
        pass


def get_gguf_warm_state() -> GGUFWarmArtifactState:
    with _GGUF_WARM_STATE_LOCK:
        return _GGUF_WARM_STATE


def reset_gguf_warm_state(*, release_engine: bool = False) -> None:
    with _GGUF_WARM_STATE_LOCK:
        _GGUF_WARM_STATE.reset(release_engine=release_engine)


class GGUFPipelineRunner:
    """
    Retained legacy GGUF runner used by regression tests and tooling.
    Modern authoritative production routes do not dispatch through this runner.
    """

    def run(self, route: PipelineRoute, context: PipelineRouteContext) -> PipelineRouteContext:
        task_state = context.task_state
        device = resources.get_torch_device()
        total_start = time.perf_counter()
        quality = self._build_quality(task_state)

        prompt_tasks, processed_loras, use_expansion = self._prepare_prompt_tasks(task_state)
        task_state.loras_processed = processed_loras
        task_state.gguf_lora_artifacts = ()
        task_state.use_expansion = use_expansion
        context.prompt_tasks = list(prompt_tasks)
        context.all_steps = max(int(task_state.steps) * max(len(prompt_tasks), 1), 1)

        if context.progressbar_callback is not None:
            task_state.current_progress += 1
            context.progressbar_callback(task_state, task_state.current_progress, 'Loading models ...')

        unet = None
        task_metrics: list[dict[str, float]] = []

        try:
            with resources.memory_phase_scope(resources.MemoryPhase.MODEL_REFRESH, task=task_state):
                unet_path, clip_l_path, clip_g_path, vae_path = self._resolve_components(task_state)
                task_state.gguf_engine_path = unet_path
                engine_fingerprint = self._build_engine_fingerprint(task_state)
                task_state.gguf_engine_fingerprint = engine_fingerprint
                task_state.gguf_clip_l_path = clip_l_path
                task_state.gguf_clip_g_path = clip_g_path
                task_state.gguf_vae_path = vae_path

                unet = self._acquire_warm_gguf_unet(unet_path, quality, engine_fingerprint)
                clip = self._load_transient_clip(
                    clip_l_path,
                    clip_g_path,
                    clip_skip=task_state.clip_skip,
                )
                try:
                    resolved_lora_artifacts = self._resolve_gguf_lora_artifacts(unet, clip, processed_loras)
                    task_state.gguf_lora_artifacts = tuple(resolved_lora_artifacts)
                    self._prepare_warm_gguf_delta(unet, resolved_lora_artifacts)
                finally:
                    self._dispose_transient_clip(clip)

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
                        task_state.gguf_vae_path,
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
                self._release_gguf_unet_for_warm_state(unet)
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
            from backend.sdxl_unified_runtime import _resolve_shared_sdxl_vae_path
            vae_path = _resolve_shared_sdxl_vae_path()
            if vae_path is None:
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

        normalized_name = str(name).replace('\\', '/').replace('/', os.sep)
        candidate = util.get_file_from_folder_list(normalized_name, folders)
        if isinstance(candidate, str) and os.path.isfile(candidate):
            return candidate
        return None

    def _try_reuse_loaded_pipeline_components(
        self,
        context: PipelineRouteContext,
        processed_loras: list[tuple[str, float]],
    ) -> tuple[Any, Any, Any] | None:
        return None

    def _acquire_warm_gguf_unet(
        self,
        unet_path: str,
        quality: Dict[str, Any],
        engine_fingerprint: str,
    ) -> Any:
        with _GGUF_WARM_STATE_LOCK:
            warm_state = _GGUF_WARM_STATE
            if warm_state.warm_unet is not None and warm_state.engine_fingerprint == engine_fingerprint:
                loader.patch_unet_for_quality(warm_state.warm_unet, quality)
                return warm_state.warm_unet

            old_unet = warm_state.warm_unet
            warm_state.reset(release_engine=False)
            warm_state.engine_fingerprint = engine_fingerprint
            warm_state.warm_unet = None

        if old_unet is not None:
            _detach_gguf_warm_unet(old_unet)

        unet = loader.load_sdxl_unet(unet_path, dtype=torch.float16)
        loader.patch_unet_for_quality(unet, quality)

        with _GGUF_WARM_STATE_LOCK:
            _GGUF_WARM_STATE.warm_unet = unet
            _GGUF_WARM_STATE.engine_fingerprint = engine_fingerprint
        return unet

    def _release_gguf_unet_for_warm_state(self, unet: Any) -> None:
        with _GGUF_WARM_STATE_LOCK:
            if _GGUF_WARM_STATE.warm_unet is unet:
                _detach_gguf_warm_unet(unet)
                return
        _detach_gguf_warm_unet(unet)

    def _resolve_lora_filename(self, filename: str) -> str | None:
        if filename == 'None':
            return None
        if os.path.exists(filename):
            return filename
        resolved = util.get_file_from_folder_list(filename, modules.config.paths_lora_lookup)
        if resolved and os.path.exists(resolved):
            return resolved
        logger.warning("LoRA file not found: %s", filename)
        return None

    def _load_transient_clip(
        self,
        clip_l_path: str,
        clip_g_path: str,
        *,
        clip_skip: Any,
    ) -> Any:
        clip = loader.load_sdxl_clip(clip_l_path, clip_g_path, dtype=torch.float16)
        clip.clip_layer(-abs(int(clip_skip or 1)))
        return clip

    def _dispose_transient_clip(self, clip: Any) -> None:
        if clip is None:
            return
        try:
            clip.patcher.detach()
        except Exception:
            pass

    def _resolve_gguf_lora_artifacts(
        self,
        unet: Any,
        clip: Any,
        loras: List[Tuple[str, float]],
    ) -> tuple[ResolvedGGUFLoraArtifact, ...]:
        from backend import lora as lora_backend
        from backend.lora import model_lora_keys_clip, model_lora_keys_unet

        unet_keys = model_lora_keys_unet(unet.model, {})
        unet_keys.update({key: key for key in unet.model.state_dict().keys()})

        clip_keys = model_lora_keys_clip(clip.cond_stage_model, {})
        clip_keys.update({key: key for key in clip.cond_stage_model.state_dict().keys()})

        resolved_artifacts: list[ResolvedGGUFLoraArtifact] = []
        component_artifacts: list[lora_artifacts.AdapterArtifact] = []
        component_labels: list[str] = []
        for filename, weight in loras:
            lora_path = self._resolve_lora_filename(filename)
            if lora_path is None:
                continue

            lora_sd = loader.utils.load_torch_file(lora_path)
            try:
                loaded_patches: dict[str, Any] = {}
                unet_patches = lora_backend.load_lora(lora_sd, unet_keys, log_missing=False)
                clip_patches = lora_backend.load_lora(lora_sd, clip_keys, log_missing=False)
                if unet_patches:
                    loaded_patches.update(unet_patches)
                if clip_patches:
                    loaded_patches.update(clip_patches)
                artifact = lora_artifacts.normalize_loaded_lora_artifact(
                    source_path=lora_path,
                    default_scale=weight,
                    loaded_patches=loaded_patches,
                )
                component_artifacts.append(artifact)
                component_labels.append(lora_path)
            finally:
                del lora_sd
                resources.soft_empty_cache(force=False)

        if not component_artifacts:
            return ()

        merged_artifact = lora_artifacts.merge_loaded_lora_artifacts(
            component_artifacts,
            source_path=" || ".join(component_labels),
        )
        classification = lora_artifacts.classify_artifact_targets(merged_artifact)
        requested_name = " + ".join(os.path.basename(label) for label in component_labels)
        resolved_artifacts.append(
            ResolvedGGUFLoraArtifact(
                requested_name=requested_name,
                resolved_path=merged_artifact.source_path,
                artifact=merged_artifact,
                target_families=classification.target_families,
                clip_target_keys=classification.clip_target_keys,
                unet_target_keys=classification.unet_target_keys,
            )
        )
        return tuple(resolved_artifacts)

    def _clear_gguf_unet_lora_state(self, unet: Any) -> None:
        unet.detach()
        unet.patches = {}
        unet.backup.clear()
        unet.object_patches_backup.clear()
        unet.gguf_dense_delta_cache = {}
        unet.patches_uuid = uuid.uuid4()

    def _prepare_warm_gguf_delta(
        self,
        unet: Any,
        resolved_artifacts: tuple[ResolvedGGUFLoraArtifact, ...],
    ) -> None:
        delta_fingerprint = self._build_delta_fingerprint(resolved_artifacts)

        with _GGUF_WARM_STATE_LOCK:
            warm_state = _GGUF_WARM_STATE
            if warm_state.warm_unet is unet and warm_state.delta_fingerprint == delta_fingerprint:
                return

        self._clear_gguf_unet_lora_state(unet)
        self._apply_lora_artifacts_to_unet(unet, resolved_artifacts)

        with _GGUF_WARM_STATE_LOCK:
            if _GGUF_WARM_STATE.warm_unet is unet:
                _GGUF_WARM_STATE.delta_fingerprint = delta_fingerprint
                _GGUF_WARM_STATE.delta_artifacts = {
                    "artifact_signature": tuple(
                        resolved.artifact.artifact_id
                        for resolved in resolved_artifacts
                        if resolved.is_unet_side
                    ),
                }

    def _apply_lora_artifacts_to_unet(
        self,
        unet: Any,
        resolved_artifacts: tuple[ResolvedGGUFLoraArtifact, ...],
    ) -> None:
        from backend.lora import model_lora_keys_unet

        unet_keys = model_lora_keys_unet(unet.model, {})
        unet_keys.update({key: key for key in unet.model.state_dict().keys()})

        for resolved in resolved_artifacts:
            if resolved.is_unet_side:
                lora_artifacts.apply_artifact_to_patcher(
                    unet,
                    resolved.artifact,
                    unet_keys,
                    target_family="unet",
                )

    def _apply_lora_artifacts_to_clip(
        self,
        clip: Any,
        resolved_artifacts: tuple[ResolvedGGUFLoraArtifact, ...],
    ) -> None:
        from backend.lora import model_lora_keys_clip

        clip_keys = model_lora_keys_clip(clip.cond_stage_model, {})
        clip_keys.update({key: key for key in clip.cond_stage_model.state_dict().keys()})

        for resolved in resolved_artifacts:
            if resolved.is_clip_side:
                lora_artifacts.apply_artifact_to_patcher(
                    clip,
                    resolved.artifact,
                    clip_keys,
                    target_family="clip",
                )

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
        task_state: Any,
        task_dict: dict[str, Any],
        policy: Any,
        device: torch.device,
    ) -> tuple[DiffusionReadyArtifacts, DiffusionReadyArtifactMetrics]:
        metrics = DiffusionReadyArtifactMetrics()
        conditioning_fingerprint = self._build_conditioning_fingerprint(
            task_state,
            task_dict,
            tuple(getattr(task_state, 'gguf_lora_artifacts', ()) or ()),
        )

        cached_conditioning = self._load_cached_conditioning_artifacts(conditioning_fingerprint)
        if cached_conditioning is None:
            clip = self._load_transient_clip(
                task_state.gguf_clip_l_path,
                task_state.gguf_clip_g_path,
                clip_skip=task_state.clip_skip,
            )
            with resources.memory_phase_scope(
                resources.MemoryPhase.PROMPT_ENCODE,
                task=task_state,
                notes={'task_seed': task_dict['task_seed']},
                end_notes={'completed': True},
            ):
                try:
                    self._apply_lora_artifacts_to_clip(
                        clip,
                        tuple(getattr(task_state, 'gguf_lora_artifacts', ()) or ()),
                    )
                    encoded_prompt_pair, encode_metrics = self._encode_prompt_pair(
                        clip,
                        task_dict['positive'],
                        task_dict['negative'],
                        task_dict['positive_top_k'],
                        task_dict['negative_top_k'],
                        policy,
                        device,
                    )
                finally:
                    self._dispose_transient_clip(clip)
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
            self._store_conditioning_artifacts(
                conditioning_fingerprint,
                encoded_prompt_pair,
                adm_pair,
            )
        else:
            encoded_prompt_pair = cached_conditioning["encoded_prompt_pair"]
            adm_pair = cached_conditioning["adm_pair"]

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
        fingerprints = self._build_diffusion_fingerprints(
            task_state,
            task_dict,
            input_fingerprint=input_fingerprint,
        )

        return DiffusionReadyArtifacts(
            positive_conds=processed_conds.get("positive") or [],
            negative_conds=processed_conds.get("negative") or [],
            initial_latent=initial_latent,
            noise=noise,
            denoise_mask=denoise_mask,
            sigmas=sigmas,
            conditioning_fingerprint=conditioning_fingerprint,
            delta_fingerprint=fingerprints["delta_fingerprint"],
            engine_fingerprint=fingerprints["engine_fingerprint"],
            input_fingerprint=fingerprints["input_fingerprint"],
            controlnet_fingerprint=fingerprints["controlnet_fingerprint"],
        ), metrics

    def _load_cached_conditioning_artifacts(
        self,
        conditioning_fingerprint: str,
    ) -> Optional[dict[str, Any]]:
        with _GGUF_WARM_STATE_LOCK:
            artifacts = _GGUF_WARM_STATE.conditioning_artifacts or {}
            cached = artifacts.get(conditioning_fingerprint)
            if cached is None:
                return None
            _GGUF_WARM_STATE.conditioning_fingerprint = conditioning_fingerprint
            return self._clone_cached_value(cached)

    def _store_conditioning_artifacts(
        self,
        conditioning_fingerprint: str,
        encoded_prompt_pair: dict[str, Any],
        adm_pair: dict[str, Any],
    ) -> None:
        cached = {
            "encoded_prompt_pair": self._cache_to_cpu(encoded_prompt_pair),
            "adm_pair": self._cache_to_cpu(adm_pair),
        }
        with _GGUF_WARM_STATE_LOCK:
            artifacts = dict(_GGUF_WARM_STATE.conditioning_artifacts or {})
            artifacts[conditioning_fingerprint] = cached
            _GGUF_WARM_STATE.conditioning_artifacts = artifacts
            _GGUF_WARM_STATE.conditioning_fingerprint = conditioning_fingerprint

    def _build_diffusion_fingerprints(
        self,
        task_state: Any,
        task_dict: dict[str, Any],
        *,
        input_fingerprint: Optional[str],
    ) -> dict[str, Optional[str]]:
        resolved_lora_artifacts = tuple(getattr(task_state, 'gguf_lora_artifacts', ()) or ())
        return {
            "conditioning_fingerprint": self._build_conditioning_fingerprint(
                task_state,
                task_dict,
                resolved_lora_artifacts,
            ),
            "delta_fingerprint": self._build_delta_fingerprint(resolved_lora_artifacts),
            "engine_fingerprint": self._build_engine_fingerprint(task_state),
            "input_fingerprint": self._build_input_fingerprint(task_state, input_fingerprint),
            "controlnet_fingerprint": self._build_controlnet_fingerprint(task_state),
        }

    def _build_conditioning_fingerprint(
        self,
        task_state: Any,
        task_dict: dict[str, Any],
        resolved_lora_artifacts: tuple[ResolvedGGUFLoraArtifact, ...],
    ) -> str:
        clip_loras = []
        for resolved in resolved_lora_artifacts:
            if not resolved.is_clip_side:
                continue
            clip_loras.append(
                {
                    "artifact_id": resolved.artifact.artifact_id,
                    "side_hash": lora_artifacts.fingerprint_artifact_entries(
                        resolved.artifact,
                        target_family="clip",
                    ),
                    "strength": float(resolved.artifact.default_scale),
                    "target_keys": list(resolved.clip_target_keys),
                }
            )

        payload = {
            "positive_workloads": list(task_dict.get("positive") or []),
            "negative_workloads": list(task_dict.get("negative") or []),
            "styles": [str(style) for style in (task_dict.get("styles") or [])],
            "clip_skip": int(getattr(task_state, "clip_skip", 1) or 1),
            "positive_top_k": int(task_dict.get("positive_top_k") or 0),
            "negative_top_k": int(task_dict.get("negative_top_k") or 0),
            "clip_side_loras": clip_loras,
        }
        return self._fingerprint_json_payload(payload)

    def _build_delta_fingerprint(
        self,
        resolved_lora_artifacts: tuple[ResolvedGGUFLoraArtifact, ...],
    ) -> str:
        unet_loras = []
        for resolved in resolved_lora_artifacts:
            if not resolved.is_unet_side:
                continue
            unet_loras.append(
                {
                    "artifact_id": resolved.artifact.artifact_id,
                    "side_hash": lora_artifacts.fingerprint_artifact_entries(
                        resolved.artifact,
                        target_family="unet",
                    ),
                    "strength": float(resolved.artifact.default_scale),
                    "target_keys": list(resolved.unet_target_keys),
                }
            )
        return self._fingerprint_json_payload({"unet_side_loras": unet_loras})

    def _build_engine_fingerprint(self, task_state: Any) -> str:
        engine_path = str(getattr(task_state, "gguf_engine_path", "") or "")
        engine_payload = {
            "selector": str(getattr(task_state, "base_model_name", "") or ""),
            "resolved_path": engine_path,
        }
        if engine_path and os.path.isfile(engine_path):
            try:
                stat = os.stat(engine_path)
                engine_payload["size"] = int(stat.st_size)
                engine_payload["mtime_ns"] = int(stat.st_mtime_ns)
            except OSError:
                pass
        return self._fingerprint_json_payload(engine_payload)

    def _build_input_fingerprint(
        self,
        task_state: Any,
        prepared_input_fingerprint: Optional[str],
    ) -> Optional[str]:
        if prepared_input_fingerprint:
            return prepared_input_fingerprint

        raw_inputs = {}
        for attr_name in (
            "input_image",
            "uov_input_image",
            "inpaint_image",
            "input_mask",
            "inpaint_mask",
        ):
            value = getattr(task_state, attr_name, None)
            if value is not None:
                raw_inputs[attr_name] = self._fingerprint_value(value)

        if not raw_inputs:
            return None
        return self._fingerprint_json_payload(raw_inputs)

    def _build_controlnet_fingerprint(self, task_state: Any) -> str:
        has_controlnet = bool(getattr(task_state, "controlnet_tasks", None))
        payload = {
            "reserved": True,
            "implemented": False,
            "requested": has_controlnet,
        }
        return self._fingerprint_json_payload(payload)

    def _fingerprint_json_payload(self, payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _fingerprint_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            digest = hashlib.sha256()
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
            return {"tensor_sha256": digest.hexdigest()}
        if isinstance(value, dict):
            return {str(key): self._fingerprint_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [self._fingerprint_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        if hasattr(value, "tobytes") and hasattr(value, "shape"):
            try:
                digest = hashlib.sha256()
                digest.update(str(tuple(int(dim) for dim in value.shape)).encode("utf-8"))
                digest.update(value.tobytes())
                return {"array_sha256": digest.hexdigest()}
            except Exception:
                pass
        return repr(value)

    def _cache_to_cpu(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=torch.device("cpu")).contiguous()
        if isinstance(value, dict):
            return {key: self._cache_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cache_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cache_to_cpu(item) for item in value)
        return value

    def _clone_cached_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: self._clone_cached_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_cached_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_cached_value(item) for item in value)
        return value

    def _prepare_phase0_inputs(
        self,
        unet: Any,
        task_state: Any,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[str]]:
        prepared = getattr(task_state, 'initial_latent', None)
        if isinstance(prepared, dict) and isinstance(prepared.get("samples"), torch.Tensor):
            input_fingerprint = self._fingerprint_json_payload(
                {
                    "prepared_latent": self._fingerprint_value(prepared["samples"]),
                    "noise_mask": self._fingerprint_value(prepared.get("noise_mask")),
                }
            )
            cached_phase0 = self._load_cached_phase0_input(input_fingerprint)
            if cached_phase0 is None:
                self._store_phase0_input(input_fingerprint, prepared)
                cached_phase0 = self._load_cached_phase0_input(input_fingerprint)
            if cached_phase0 is not None:
                return self._materialize_cached_phase0_input(unet, cached_phase0, seed, device, input_fingerprint)

        latent, noise = self._create_latent_and_noise(
            unet,
            task_state.width,
            task_state.height,
            seed,
            device,
        )
        return latent, noise, None, None

    def _load_cached_phase0_input(self, input_fingerprint: str) -> Optional[dict[str, Any]]:
        with _GGUF_WARM_STATE_LOCK:
            artifacts = _GGUF_WARM_STATE.input_artifacts or {}
            cached = artifacts.get(input_fingerprint)
            if cached is None:
                return None
            _GGUF_WARM_STATE.input_fingerprint = input_fingerprint
            return self._clone_cached_value(cached)

    def _store_phase0_input(self, input_fingerprint: str, prepared: dict[str, Any]) -> None:
        cached = {
            "samples": self._cache_to_cpu(prepared.get("samples")),
            "noise_mask": self._cache_to_cpu(prepared.get("noise_mask")),
        }
        with _GGUF_WARM_STATE_LOCK:
            artifacts = dict(_GGUF_WARM_STATE.input_artifacts or {})
            artifacts[input_fingerprint] = cached
            _GGUF_WARM_STATE.input_artifacts = artifacts
            _GGUF_WARM_STATE.input_fingerprint = input_fingerprint

    def _materialize_cached_phase0_input(
        self,
        unet: Any,
        cached_phase0: dict[str, Any],
        seed: int,
        device: torch.device,
        input_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[str]]:
        dtype = unet.model.get_dtype()
        latent = cached_phase0["samples"].to(device=device, dtype=dtype)
        denoise_mask = cached_phase0.get("noise_mask")
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
        return latent, noise, denoise_mask, input_fingerprint

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
                    self._log_gguf_sampling_progress(payload, total_steps)
                    if sampling_callback is not None:
                        sampling_callback(
                            payload["i"],
                            payload.get("denoised"),
                            payload.get("x"),
                            total_steps,
                            None,
                        )
                    self._log_gguf_step_trace(task_state, payload, total_steps)

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

    def _should_trace_gguf_steps(self, task_state: Any) -> bool:
        if bool(getattr(task_state, 'gguf_debug_step_trace', False)):
            return True
        env_value = str(os.getenv('NEX_GGUF_STEP_TRACE', '') or '').strip().lower()
        return env_value in {'1', 'true', 'yes', 'on'}

    def _log_gguf_sampling_progress(self, payload: dict[str, Any], total_steps: int) -> None:
        step_index = int(payload.get('i', -1))
        if step_index < 0 or total_steps <= 0:
            return

        sigma_value = payload.get('sigma')
        try:
            if isinstance(sigma_value, torch.Tensor):
                sigma_value = float(sigma_value.reshape(-1)[0].item())
            elif sigma_value is not None:
                sigma_value = float(sigma_value)
        except Exception:
            sigma_value = None

        percent = ((step_index + 1) / float(total_steps)) * 100.0
        message = (
            f"[Nex-GGUF-Progress] step={step_index + 1}/{total_steps} "
            f"percent={percent:.1f}"
        )
        if sigma_value is not None:
            message += f" sigma={sigma_value:.6f}"
        print(message)
        logging.info(message)

    @staticmethod
    def _tensor_trace_stats(tensor: Any) -> Optional[dict[str, Any]]:
        if not isinstance(tensor, torch.Tensor):
            return None
        try:
            stats_tensor = tensor.detach().float()
            return {
                'shape': list(stats_tensor.shape),
                'mean': float(stats_tensor.mean().item()),
                'std': float(stats_tensor.std(unbiased=False).item()) if stats_tensor.numel() > 1 else 0.0,
                'min': float(stats_tensor.min().item()),
                'max': float(stats_tensor.max().item()),
            }
        except Exception:
            return None

    def _log_gguf_step_trace(self, task_state: Any, payload: dict[str, Any], total_steps: int) -> None:
        if not self._should_trace_gguf_steps(task_state):
            return

        step_index = int(payload.get('i', -1))
        if step_index < 0:
            return

        should_log = step_index < 4 or step_index >= max(total_steps - 1, 0)
        if not should_log:
            return

        sigma_value = payload.get('sigma')
        try:
            if isinstance(sigma_value, torch.Tensor):
                sigma_value = float(sigma_value.reshape(-1)[0].item())
            elif sigma_value is not None:
                sigma_value = float(sigma_value)
        except Exception:
            sigma_value = None

        trace_payload = {
            'step': step_index + 1,
            'total_steps': total_steps,
            'sigma': sigma_value,
            'denoised': self._tensor_trace_stats(payload.get('denoised')),
            'latent_x': self._tensor_trace_stats(payload.get('x')),
            'engine_fingerprint': str(getattr(task_state, 'gguf_engine_fingerprint', '') or ''),
        }
        message = f"[Nex-GGUF-StepTrace] {json.dumps(trace_payload, sort_keys=True)}"
        print(message)
        logging.info(message)

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

    def _decode_latent(self, vae_path: str, latent: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float, float]:
        vae = loader.load_vae(
            vae_path,
            dtype=torch.float32,
            latent_format=latent_formats.SDXL(),
        )
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
