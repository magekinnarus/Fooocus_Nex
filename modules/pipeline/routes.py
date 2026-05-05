from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from backend import environment_profile as environment_profiles
import modules.flags as flags
from modules.pipeline.stage_runtime import (
    PipelineResourceRequirement,
    PipelineRoute,
    PipelineRouteContext,
    PipelineStage,
    PipelineStageResult,
    StageMemoryEstimate,
)


def _describe_route_resources(*requirements: PipelineResourceRequirement) -> Sequence[PipelineResourceRequirement]:
    return requirements


def _estimated_megapixels(task_state) -> float:
    width = max(int(getattr(task_state, 'width', 0) or 0), 1)
    height = max(int(getattr(task_state, 'height', 0) or 0), 1)
    return float(width * height) / 1_000_000.0


def _expects_controlnet_extension(task_state) -> bool:
    if not getattr(task_state, 'input_image_checkbox', False):
        return False
    if task_state.current_tab == 'ip':
        return True
    if getattr(task_state, 'mixing_image_prompt_and_inpaint', False):
        return True
    if getattr(task_state, 'mixing_image_prompt_and_outpaint', False):
        return True
    return any(len(tasks) > 0 for tasks in task_state.cn_tasks.values())


def _has_outpaint_request(task_state) -> bool:
    if not getattr(task_state, 'input_image_checkbox', False):
        return False
    mixed_cn_outpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_outpaint', False)
    has_mixed_outpaint_request = mixed_cn_outpaint_workflow and task_state.outpaint_input_image is not None and (
        getattr(task_state, 'outpaint_step2_checkbox', False)
        or bool(getattr(task_state, 'outpaint_selections', []))
        or getattr(task_state, 'outpaint_mask_image', None) is not None
    )
    return (task_state.current_tab == 'outpaint' or has_mixed_outpaint_request) and task_state.outpaint_input_image is not None


def _has_inpaint_request(task_state) -> bool:
    if not getattr(task_state, 'input_image_checkbox', False):
        return False
    mixed_cn_inpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_inpaint', False)
    mixed_cn_outpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_outpaint', False)
    has_mixed_outpaint_request = mixed_cn_outpaint_workflow and task_state.outpaint_input_image is not None and (
        getattr(task_state, 'outpaint_step2_checkbox', False)
        or bool(getattr(task_state, 'outpaint_selections', []))
        or getattr(task_state, 'outpaint_mask_image', None) is not None
    )
    has_mixed_inpaint_request = mixed_cn_inpaint_workflow and task_state.inpaint_input_image is not None
    return (
        task_state.current_tab == 'inpaint'
        or (has_mixed_inpaint_request and not has_mixed_outpaint_request)
    ) and task_state.inpaint_input_image is not None


def _is_flux_fill_inpaint_request(task_state) -> bool:
    if task_state.current_tab != 'inpaint' or task_state.inpaint_input_image is None:
        return False

    try:
        import modules.objr_engine as objr_engine

        return objr_engine.normalize_flux_fill_inpaint_route(getattr(task_state, 'inpaint_route', None)) == objr_engine.FLUX_FILL_INPAINT_ROUTE_FLUX
    except Exception:
        return False


def _is_upscale_request(task_state) -> bool:
    if not getattr(task_state, 'input_image_checkbox', False):
        return False
    if task_state.current_tab != 'uov':
        return False
    if task_state.uov_input_image is None:
        return False
    method = str(getattr(task_state, 'uov_method', '') or '').lower()
    if method == flags.disabled.casefold():
        return False
    return 'upscale' in method


def describe_route(route: PipelineRoute) -> list[str]:
    return [stage.stage_id for stage in route.stages]


def _save_step1_result(context: PipelineRouteContext, payload, description: str) -> None:
    from modules.pipeline.output import save_and_log

    if payload is None:
        return

    images_to_save = payload if isinstance(payload, (list, tuple)) else [payload]
    normalized_images = []
    for image in images_to_save:
        if isinstance(image, np.ndarray) and image.ndim == 2:
            normalized_images.append(np.stack([image] * 3, axis=-1))
        else:
            normalized_images.append(image)

    task_state = context.task_state
    if context.progressbar_callback is not None:
        context.progressbar_callback(task_state, 100, f'Saving {description} ...')

    img_paths = save_and_log(
        task_state,
        task_state.height,
        task_state.width,
        normalized_images,
        {
            'log_positive_prompt': task_state.prompt,
            'log_negative_prompt': task_state.negative_prompt,
            'positive': [],
            'negative': [],
            'styles': task_state.style_selections,
            'task_seed': task_state.seed,
            'description': description,
        },
        False,
        task_state.loras,
    )
    if context.yield_result_callback is not None:
        context.yield_result_callback(task_state, img_paths, 100, do_not_show_finished_images=True)


def _resolve_inpaint_prompt(task_state) -> str:
    prompt = str(getattr(task_state, 'prompt', '') or '').strip()
    additional_prompt = str(getattr(task_state, 'inpaint_additional_prompt', '') or '').strip()
    if additional_prompt == '':
        return prompt
    if prompt == '':
        return additional_prompt
    return additional_prompt + '\n' + prompt


def _should_force_flux_host_cleanup() -> bool:
    try:
        from backend import resources

        profile = resources.active_memory_environment_profile()
        profile_name = getattr(profile, "name", None)
        return profile_name in (
            environment_profiles.PROFILE_COLAB_FREE,
            environment_profiles.PROFILE_LOCAL_LOW_VRAM,
        )
    except Exception:
        return False


def _build_flux_preview_transform(active_session):
    previewer_holder = {"previewer": None, "latent_format": None, "resolved": False}

    def decode_preview(preview_payload):
        try:
            import torch
            from PIL import Image
            from ldm_patched.taesd.taesd import TAESD
            from ldm_patched.utils.latent_visualization import Latent2RGBPreviewer, TAESDPreviewerImpl
            import ldm_patched.utils.path_utils as path_utils
        except Exception:
            return None

        if not isinstance(preview_payload, torch.Tensor):
            return preview_payload if isinstance(preview_payload, np.ndarray) else None

        previewer = previewer_holder["previewer"]
        latent_format = previewer_holder["latent_format"]
        if not previewer_holder["resolved"]:
            previewer_holder["resolved"] = True
            unet_patcher = getattr(active_session, "unet_patcher", None)
            vae = getattr(active_session, "vae", None)
            load_device = getattr(unet_patcher, "load_device", None)
            patcher_model = getattr(unet_patcher, "model", None)
            latent_format = getattr(patcher_model, "latent_format", None)
            if latent_format is None:
                latent_format = getattr(getattr(patcher_model, "model", None), "latent_format", None)
            if latent_format is None and vae is not None:
                latent_format = getattr(vae, "latent_format", None)
            if load_device is None and vae is not None:
                load_device = getattr(getattr(vae, "patcher", None), "load_device", None)

            previewer_holder["latent_format"] = latent_format
            if latent_format is not None:
                taesd_decoder_path = None
                taesd_decoder_name = getattr(latent_format, "taesd_decoder_name", None)
                if taesd_decoder_name:
                    try:
                        taesd_decoder_file = next(
                            (fn for fn in path_utils.get_filename_list("vae_approx") if fn.startswith(taesd_decoder_name)),
                            "",
                        )
                        if taesd_decoder_file:
                            taesd_decoder_path = path_utils.get_full_path("vae_approx", taesd_decoder_file)
                    except Exception:
                        taesd_decoder_path = None
                if taesd_decoder_path and load_device is not None:
                    try:
                        previewer = TAESDPreviewerImpl(TAESD(None, taesd_decoder_path).to(load_device))
                    except Exception:
                        previewer = None
                if previewer is None and getattr(latent_format, "latent_rgb_factors", None) is not None:
                    try:
                        previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors)
                    except Exception:
                        previewer = None
            previewer_holder["previewer"] = previewer
            latent_format = previewer_holder["latent_format"]

        if previewer is None:
            return None

        try:
            preview_latent = preview_payload.detach()
            if latent_format is not None and hasattr(latent_format, "process_out"):
                preview_latent = latent_format.process_out(preview_latent)
            preview_image = previewer.decode_latent_to_preview(preview_latent)
        except Exception:
            return None

        if isinstance(preview_image, Image.Image):
            return np.asarray(preview_image.convert("RGB"))
        try:
            preview_array = np.asarray(preview_image)
        except Exception:
            return None
        return preview_array if preview_array.ndim == 3 else None

    return decode_preview


def sync_flux_fill_route_session(route: PipelineRoute, task_state, *, progress: bool = False):
    import modules.objr_engine as objr_engine

    selected_engine = objr_engine.normalize_objr_engine(getattr(task_state, "objr_engine", None))
    if route.family == "flux_fill":
        selected_engine = objr_engine.OBJR_ENGINE_FLUX_FILL
    try:
        return objr_engine.reconcile_active_flux_fill_session(
            route_family=route.family,
            selected_engine=selected_engine,
            conditioning=getattr(task_state, "flux_fill_conditioning", None),
            progress=progress,
        )
    except Exception:
        objr_engine.end_active_flux_fill_session(reason="flux_session_start_failed")
        return None


class ImageInputPreparationStage(PipelineStage):
    stage_id = 'image_input_prepare'
    phase_name = 'image_input_prepare'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='image_input',
                description='User-provided image inputs, masks, and route assets resolved for the active family.',
                resource_type='input',
                owner='modules.pipeline.image_input',
                tags=('route-entry',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'input-shape-dependent'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.image_input import apply_image_input

        task_state = context.task_state
        with resources.memory_phase_scope(
            resources.MemoryPhase.IMAGE_INPUT_PREPARE,
            task=task_state,
            notes={'current_tab': task_state.current_tab},
            end_notes={'completed': True},
        ):
            payload = apply_image_input(task_state, context.base_model_additional_loras, context.progressbar_callback)
        context.update_image_input_result(payload)
        return PipelineStageResult(notes={'goals': list(task_state.goals)})


class ControlNetSupportLoadStage(PipelineStage):
    stage_id = 'controlnet_support_load'
    phase_name = 'model_refresh'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='controlnet_models',
                description='Structural ControlNet models refreshed for the current route.',
                owner='modules.default_pipeline',
                tags=('controlnet', 'structural'),
            ),
            PipelineResourceRequirement(
                resource_id='contextual_support_models',
                description='Contextual adapter support assets such as CLIP vision and insightface.',
                owner='backend.ip_adapter',
                tags=('controlnet', 'contextual'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'support-model-load'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import load_controlnet_support_models

        task_state = context.task_state
        if not task_state.input_image_checkbox:
            return PipelineStageResult()

        task_state.current_progress = max(task_state.current_progress, 1)
        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, task_state.current_progress, 'Loading ControlNets ...')

        load_controlnet_support_models(context.image_input_result)
        return PipelineStageResult()


class InpaintPreparationStage(PipelineStage):
    stage_id = 'inpaint_prepare'
    phase_name = 'vae_encode'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='inpaint_assets',
                description='Prepared inpaint image, context mask, BB image, and retained inpaint context.',
                resource_type='artifact',
                owner='modules.pipeline.inpaint',
                tags=('inpaint', 'latent'),
            ),
            PipelineResourceRequirement(
                resource_id='candidate_vae',
                description='VAE selected for inpaint latent encoding.',
                owner='modules.default_pipeline',
                tags=('vae',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(ram_mb=round(max(128.0, megapixels * 96.0), 1), notes={'basis': 'image-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import EarlyReturnException, apply_inpaint

        task_state = context.task_state
        try:
            apply_inpaint(
                task_state,
                context.image_input_result.get('inpaint_image'),
                context.image_input_result.get('inpaint_mask'),
                context.progressbar_callback,
                context.yield_result_callback,
            )
        except EarlyReturnException as exc:
            _save_step1_result(context, exc.payload, 'Phase 1 Inpaint BB')
            return PipelineStageResult(route_complete=True, notes={'early_return': True, 'route': 'inpaint'})
        return PipelineStageResult()


class OutpaintPreparationStage(PipelineStage):
    stage_id = 'outpaint_prepare'
    phase_name = 'vae_encode'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='outpaint_assets',
                description='Prepared outpaint canvas, mask, and retained outpaint context.',
                resource_type='artifact',
                owner='modules.pipeline.outpaint',
                tags=('outpaint', 'latent'),
            ),
            PipelineResourceRequirement(
                resource_id='candidate_vae',
                description='VAE selected for outpaint latent encoding.',
                owner='modules.default_pipeline',
                tags=('vae',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(ram_mb=round(max(128.0, megapixels * 96.0), 1), notes={'basis': 'expanded-canvas-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import apply_outpaint_inference_setup

        task_state = context.task_state
        apply_outpaint_inference_setup(
            task_state,
            context.image_input_result.get('outpaint_image'),
            context.image_input_result.get('outpaint_mask'),
            context.progressbar_callback,
            context.yield_result_callback,
        )
        return PipelineStageResult()


class PromptEncodingStage(PipelineStage):
    stage_id = 'prompt_encode'
    phase_name = 'prompt_encode'

    def describe_resources(self, context: PipelineRouteContext):
        task_state = context.task_state
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='base_model',
                description=f'Base model {task_state.base_model_name!r} prepared for prompt encoding.',
                owner='modules.default_pipeline',
                tags=('checkpoint', 'clip', 'vae'),
            ),
            PipelineResourceRequirement(
                resource_id='prompt_conditions',
                description='Positive and negative conditioning retained for downstream stages.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('conditioning',),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'conditioning-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.preprocessing import apply_overrides, process_prompt

        task_state = context.task_state
        apply_overrides(task_state)

        if context.image_input_result.get('skip_prompt_processing', False):
            context.prompt_tasks = []
            return PipelineStageResult(notes={'prompt_processing': 'skipped'})

        context.prompt_tasks = process_prompt(task_state, context.base_model_additional_loras, context.progressbar_callback)
        resources.cleanup_memory('encoding_to_diffusion', notes={'goals': list(task_state.goals)}, target_phase=resources.MemoryPhase.DIFFUSION, task=task_state)
        return PipelineStageResult(notes={'task_count': len(context.prompt_tasks)})


class StructuralControlNetStage(PipelineStage):
    stage_id = 'structural_controlnet'
    phase_name = 'structural_preprocess'

    def finalize(self, context: PipelineRouteContext, *, result=None, error=None):
        from backend import resources

        if error is not None:
            return

        contextual_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_contextual)
        next_phase = resources.MemoryPhase.CONTEXTUAL_PREPROCESS if sum(len(tasks) for tasks in contextual_tasks.values()) > 0 else resources.MemoryPhase.CONTROL_APPLY
        resources.cleanup_memory(
            'structural_preprocess_complete',
            gc_collect=False,
            target_phase=next_phase,
            notes={'route_id': context.route_id},
        )

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='structural_preprocessors',
                description='Structural preprocessors for Canny, CPDS, Depth, MistoLine, and MLSD guidance.',
                owner='backend.preprocessors.runtime',
                tags=('controlnet', 'structural'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'preprocessor-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import preprocess_structural_controlnets

        if not context.has_goal('cn'):
            return PipelineStageResult(notes={'status': 'skipped'})

        structural_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_structural)
        if sum(len(tasks) for tasks in structural_tasks.values()) == 0:
            return PipelineStageResult(notes={'status': 'skipped'})

        preprocess_structural_controlnets(
            context.task_state,
            structural_preprocessor_paths=context.image_input_result.get('structural_preprocessor_paths'),
        )
        return PipelineStageResult()


class ContextualControlNetStage(PipelineStage):
    stage_id = 'contextual_controlnet'
    phase_name = 'contextual_preprocess'

    def finalize(self, context: PipelineRouteContext, *, result=None, error=None):
        from backend import resources

        if error is not None:
            return

        resources.cleanup_memory(
            'contextual_preprocess_complete',
            gc_collect=False,
            target_phase=resources.MemoryPhase.DIFFUSION,
            notes={'route_id': context.route_id},
        )

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='contextual_adapters',
                description='Contextual guidance assets such as IP-Adapter, FaceID, and PuLID support models.',
                owner='backend.ip_adapter',
                tags=('controlnet', 'contextual'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(notes={'strategy': 'adapter-count-dependent'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import preprocess_contextual_controlnets

        if not context.has_goal('cn'):
            return PipelineStageResult(notes={'status': 'skipped'})

        contextual_tasks = context.task_state.get_cn_tasks_for_channel(flags.cn_contextual)
        if sum(len(tasks) for tasks in contextual_tasks.values()) == 0:
            return PipelineStageResult(notes={'status': 'skipped'})

        preprocess_contextual_controlnets(
            context.task_state,
            contextual_assets=context.image_input_result.get('contextual_assets'),
        )
        return PipelineStageResult()


class DiffusionTaskStage(PipelineStage):
    stage_id = 'diffusion_batch'
    phase_name = 'diffusion'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='prompt_tasks',
                description='Per-image prompt task dictionaries with retained conditioning.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('conditioning', 'tasks'),
            ),
            PipelineResourceRequirement(
                resource_id='diffusion_models',
                description='UNet, VAE, and optional ControlNet state used during iterative task execution.',
                owner='modules.default_pipeline',
                tags=('unet', 'vae', 'diffusion'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(512.0, megapixels * 768.0), 1), notes={'basis': 'route-resolution'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules.pipeline.preprocessing import apply_overrides, patch_samplers
        from modules.pipeline.inference import process_task

        task_state = context.task_state
        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, 'Image processing ...')

        steps, _, _ = apply_overrides(task_state)
        context.all_steps = max(steps * task_state.image_number, 1)
        context.preparation_steps = task_state.current_progress
        context.final_scheduler_name = patch_samplers(task_state)

        task_state.yields.append(['preview', (task_state.current_progress, 'Moving model to GPU ...', None)])
        context.processing_start_time = time.perf_counter()

        for i, task_dict in enumerate(context.prompt_tasks):
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, f'Preparing task {i + 1}/{task_state.image_number} ...')
            execution_start_time = time.perf_counter()
            interrupted_action = None

            try:
                process_task(
                    task_state,
                    task_dict,
                    i,
                    task_state.image_number,
                    context.all_steps,
                    context.preparation_steps,
                    task_state.denoising_strength,
                    context.final_scheduler_name,
                    task_state.loras,
                    context.image_input_result.get('controlnet_paths', {}),
                    context.progressbar_callback,
                    context.yield_result_callback,
                )
            except resources.InterruptProcessingException:
                if task_state.last_stop == 'skip':
                    print('User skipped')
                    task_state.last_stop = False
                    interrupted_action = 'skip'
                else:
                    print('User stopped')
                    interrupted_action = 'stop'
            finally:
                if 'c' in task_dict:
                    del task_dict['c']
                if 'uc' in task_dict:
                    del task_dict['uc']
                resources.cleanup_memory('task_image_complete', gc_collect=False, notes={'task_index': i}, target_phase=resources.MemoryPhase.DECODE, task=task_state)

            if interrupted_action == 'skip':
                continue
            if interrupted_action == 'stop':
                break

            print(f'Task {i + 1} time: {time.perf_counter() - execution_start_time:.2f}s')

        print(f'Total processing time: {time.perf_counter() - context.processing_start_time:.2f}s')
        return PipelineStageResult(route_complete=True, notes={'completed': True, 'tasks_processed': len(context.prompt_tasks)})


class FluxFillInpaintStage(PipelineStage):
    stage_id = 'flux_inpaint'
    phase_name = 'diffusion'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='flux_session',
                description='Resident Flux UNet, AE, and prompt-conditioning cache used for Flux Inpaint.',
                owner='modules.objr_engine',
                tags=('flux', 'inpaint'),
            ),
            PipelineResourceRequirement(
                resource_id='inpaint_context',
                description='Prepared Inpaint tab context and blend mask carried into Flux Fill.',
                resource_type='artifact',
                owner='modules.pipeline.image_input',
                tags=('inpaint', 'flux'),
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(512.0, megapixels * 640.0), 1), notes={'basis': 'flux-fill-resolution'})

    def execute(self, context: PipelineRouteContext):
        from backend import resources
        from modules import objr_engine
        from modules.pipeline.image_input import prepare_flux_inpaint_context
        from modules.pipeline.inference import get_sampling_callback
        from modules.pipeline.inpaint import InpaintPipeline
        from modules.pipeline.output import save_and_log

        task_state = context.task_state
        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, 'Preparing Flux Fill Inpaint ...')

        ctx = task_state.inpaint_context
        if ctx is None:
            inpaint_image = context.image_input_result.get('inpaint_image')
            inpaint_mask = context.image_input_result.get('inpaint_mask')
            ctx = prepare_flux_inpaint_context(task_state, inpaint_image, inpaint_mask)

        active_session = objr_engine.ensure_active_flux_fill_session(
            tier=objr_engine.select_flux_fill_tier(),
            conditioning=getattr(task_state, 'flux_fill_conditioning', None),
            progress=False,
        )

        prompt_text = _resolve_inpaint_prompt(task_state)
        stitcher = InpaintPipeline()
        output_images: list[np.ndarray] = []
        img_paths: list[str] = []
        total_count = max(1, int(getattr(task_state, 'image_number', 1) or 1))
        base_seed = int(task_state.seed)
        output_height, output_width = ctx.original_image.shape[:2]
        task_state.width = output_width
        task_state.height = output_height
        all_steps = max(int(task_state.steps) * total_count, 1)
        preparation_steps = task_state.current_progress
        preview_transform = _build_flux_preview_transform(active_session)
        force_host_cleanup = _should_force_flux_host_cleanup()
        for image_index in range(total_count):
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, f'Flux Fill Inpaint {image_index + 1}/{total_count} ...')

            seed = base_seed if getattr(task_state, 'disable_seed_increment', False) else base_seed + image_index
            callback = get_sampling_callback(
                task_state,
                context.progressbar_callback,
                image_index,
                total_count,
                preparation_steps,
                all_steps,
                preview_transform=preview_transform,
            )
            interrupted_action = None
            try:
                resources.throw_exception_if_processing_interrupted()
                result = active_session.generate_inpaint(
                    ctx.bb_image,
                    ctx.bb_mask,
                    prompt=prompt_text,
                    seed=seed,
                    steps=int(task_state.steps),
                    sampler=task_state.sampler_name,
                    scheduler=task_state.scheduler_name,
                    guidance=objr_engine.FLUX_FILL_GUIDANCE_DEFAULT,
                    mode='baseline',
                    callback=callback,
                    disable_pbar=True,
                    progress=False,
                )
            except resources.InterruptProcessingException:
                if task_state.last_stop == 'skip':
                    print('User skipped')
                    task_state.last_stop = False
                    interrupted_action = 'skip'
                else:
                    print('User stopped')
                    interrupted_action = 'stop'
            if interrupted_action == 'skip':
                continue
            if interrupted_action == 'stop':
                break

            stitched_image = stitcher.stitch(ctx, np.asarray(result.output_image))
            output_images.append(stitched_image)

            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, 100, f'Saving Flux Fill Inpaint {image_index + 1}/{total_count} to system ...')

            task_dict = {
                'log_positive_prompt': prompt_text,
                'log_negative_prompt': task_state.negative_prompt,
                'positive': [],
                'negative': [],
                'styles': task_state.style_selections,
                'task_seed': seed,
                'description': 'Flux Fill Inpaint',
            }
            current_img_paths = save_and_log(task_state, output_height, output_width, [stitched_image], task_dict, False, task_state.loras)
            img_paths.extend(current_img_paths)
            if context.yield_result_callback is not None:
                context.yield_result_callback(
                    task_state,
                    current_img_paths,
                    100,
                    do_not_show_finished_images=task_state.disable_intermediate_results,
                )
            resources.cleanup_memory(
                'flux_inpaint_image_complete',
                gc_collect=force_host_cleanup,
                trim_host=force_host_cleanup,
                notes={'task_index': image_index, 'route_id': getattr(context, 'route_id', 'flux_inpaint')},
                target_phase=resources.MemoryPhase.DIFFUSION,
                task=task_state,
            )
        return PipelineStageResult(route_complete=True, notes={'completed': True, 'route': 'flux_inpaint', 'tasks_processed': len(output_images)})


class UpscaleStage(PipelineStage):
    stage_id = 'upscale'
    phase_name = 'upscale'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='upscaler_model',
                description='GAN upscaler model with optional tiled refinement follow-up.',
                owner='modules.upscaler',
                tags=('upscale',),
            ),
            PipelineResourceRequirement(
                resource_id='retained_conditions',
                description='Prompt conditioning retained for super-upscale tiled refinement.',
                resource_type='artifact',
                owner='modules.pipeline.preprocessing',
                tags=('upscale', 'conditioning'),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        megapixels = _estimated_megapixels(context.task_state)
        return StageMemoryEstimate(vram_mb=round(max(256.0, megapixels * 384.0), 1), notes={'basis': 'upscale-resolution'})

    def execute(self, context: PipelineRouteContext):
        from modules.pipeline.image_input import apply_upscale
        from modules.pipeline.output import save_and_log
        from modules.pipeline.tiled_refinement import apply_tiled_diffusion_refinement

        task_state = context.task_state
        if len(task_state.goals) > 0:
            task_state.current_progress += 1
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, task_state.current_progress, 'Image processing ...')

        direct_return = apply_upscale(task_state, context.progressbar_callback)
        if not direct_return:
            task_state.uov_input_image = apply_tiled_diffusion_refinement(task_state, task_state.uov_input_image, context.progressbar_callback)

        if context.progressbar_callback is not None:
            context.progressbar_callback(task_state, 100, 'Saving image to system ...')

        img_paths = save_and_log(
            task_state,
            task_state.height,
            task_state.width,
            [task_state.uov_input_image],
            {
                'log_positive_prompt': task_state.prompt,
                'log_negative_prompt': task_state.negative_prompt,
                'positive': [],
                'negative': [],
                'styles': task_state.style_selections,
                'task_seed': task_state.seed,
            },
            task_state.use_expansion,
            task_state.loras,
        )
        if context.yield_result_callback is not None:
            context.yield_result_callback(task_state, img_paths, 100, do_not_show_finished_images=True)
        return PipelineStageResult(route_complete=True, notes={'completed': True})


class RemovalStage(PipelineStage):
    stage_id = 'removal'
    phase_name = 'removal'

    def describe_resources(self, context: PipelineRouteContext):
        return _describe_route_resources(
            PipelineResourceRequirement(
                resource_id='bgr_engine',
                description='Background removal engine loaded on demand.',
                owner='modules.bgr_engine',
                tags=('removal',),
                optional=True,
            ),
            PipelineResourceRequirement(
                resource_id='objr_engine',
                description='Object removal engine loaded on demand.',
                owner='modules.objr_engine',
                tags=('removal',),
                optional=True,
            ),
        )

    def estimate_memory(self, context: PipelineRouteContext):
        return StageMemoryEstimate(vram_mb=2048.0, notes={'basis': 'engine-load-headroom'})

    def execute(self, context: PipelineRouteContext):
        import modules.bgr_engine as bgr_engine
        import modules.objr_engine as objr_engine
        from backend import resources
        from modules.pipeline.inference import get_sampling_callback

        task_state = context.task_state
        selected_engine = objr_engine.normalize_objr_engine(task_state.objr_engine)
        use_flux_session = selected_engine == objr_engine.OBJR_ENGINE_FLUX_FILL and objr_engine.has_active_flux_fill_session()
        resources.begin_memory_phase('removal', notes={'goals': list(task_state.goals)})
        try:
            if context.progressbar_callback is not None:
                context.progressbar_callback(task_state, 5, 'Clearing VRAM for Removal Models...')
            if not use_flux_session:
                resources.cleanup_memory('removal_preflight', unload_models=True, force_cache=True, trim_host=True, notes={'goals': list(task_state.goals)}, target_phase=resources.MemoryPhase.REMOVAL)

            if flags.remove_bg in task_state.goals:
                if context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, 10, 'Background Removal Starting...')
                char_path, mask_path = bgr_engine.remove_background_from_file(
                    filepath=task_state.remove_base_image,
                    threshold=task_state.bgr_threshold,
                    jit=task_state.bgr_jit,
                )
                bgr_engine.unload_model()
                if context.yield_result_callback is not None:
                    context.yield_result_callback(
                        task_state,
                        [char_path, mask_path],
                        50 if flags.remove_obj in task_state.goals else 100,
                        do_not_show_finished_images=True,
                    )
                if flags.remove_obj in task_state.goals:
                    task_state.remove_mask_image = mask_path

            if flags.remove_obj in task_state.goals:
                if context.progressbar_callback is not None:
                    context.progressbar_callback(task_state, 60 if flags.remove_bg in task_state.goals else 10, 'Object Removal Starting...')
                removal_prep_steps = 60 if flags.remove_bg in task_state.goals else 10
                flux_callback = None
                if selected_engine == objr_engine.OBJR_ENGINE_FLUX_FILL:
                    flux_callback = get_sampling_callback(
                        task_state,
                        context.progressbar_callback,
                        0,
                        1,
                        removal_prep_steps,
                        max(int(task_state.steps), 1),
                        preview_transform=_build_flux_preview_transform(objr_engine.get_active_flux_fill_session()),
                    )
                res_path = objr_engine.remove_object_from_file(
                    image_path=task_state.remove_base_image,
                    mask_path=task_state.remove_mask_image,
                    seed=task_state.seed,
                    mask_dilate=task_state.objr_mask_dilate,
                    engine=task_state.objr_engine,
                    flux_conditioning=task_state.flux_fill_conditioning,
                    flux_prompt=task_state.remove_prompt,
                    flux_prompt_cache=task_state.flux_fill_prompt_cache,
                    flux_mask_blur=task_state.objr_mask_blur,
                    flux_blend_mode=task_state.objr_blend_mode,
                    flux_steps=int(task_state.steps),
                    flux_sampler=task_state.sampler_name,
                    flux_scheduler=task_state.scheduler_name,
                    flux_callback=flux_callback,
                    flux_disable_pbar=True,
                )
                objr_engine.unload_model()
                if context.yield_result_callback is not None:
                    context.yield_result_callback(task_state, [res_path], 100, do_not_show_finished_images=True)

            return PipelineStageResult(route_complete=True, notes={'completed': True})
        finally:
            resources.end_memory_phase('removal', notes={'completed': True})


def build_generation_route(task_state) -> PipelineRoute:
    expects_controlnet = _expects_controlnet_extension(task_state)

    if task_state.input_image_checkbox and (flags.remove_bg in task_state.goals or flags.remove_obj in task_state.goals):
        return PipelineRoute(
            route_id='removal',
            family='removal',
            display_name='Removal',
            stages=[RemovalStage()],
        )

    if _is_upscale_request(task_state):
        route_id = 'super_upscale' if 'super-upscale' in str(task_state.uov_method).lower() else 'upscale'
        return PipelineRoute(
            route_id=route_id,
            family='upscale',
            display_name='Upscale',
            stages=[ImageInputPreparationStage(), PromptEncodingStage(), UpscaleStage()],
        )

    if _has_outpaint_request(task_state):
        stages: list[PipelineStage] = [ImageInputPreparationStage(), ControlNetSupportLoadStage(), OutpaintPreparationStage(), PromptEncodingStage()]
        if expects_controlnet:
            stages.extend([StructuralControlNetStage(), ContextualControlNetStage()])
        stages.append(DiffusionTaskStage())
        return PipelineRoute(
            route_id='outpaint',
            family='image_input',
            display_name='Outpaint',
            stages=stages,
        )

    if _is_flux_fill_inpaint_request(task_state):
        stages = [ImageInputPreparationStage(), FluxFillInpaintStage()]
        return PipelineRoute(
            route_id='flux_inpaint',
            family='flux_fill',
            display_name='Flux Inpaint',
            stages=stages,
        )

    if _has_inpaint_request(task_state):
        stages = [ImageInputPreparationStage(), ControlNetSupportLoadStage(), InpaintPreparationStage(), PromptEncodingStage()]
        if expects_controlnet:
            stages.extend([StructuralControlNetStage(), ContextualControlNetStage()])
        stages.append(DiffusionTaskStage())
        return PipelineRoute(
            route_id='inpaint',
            family='image_input',
            display_name='Inpaint',
            stages=stages,
        )

    stages = []
    if task_state.input_image_checkbox:
        stages.append(ImageInputPreparationStage())
    if task_state.input_image_checkbox:
        stages.append(ControlNetSupportLoadStage())
    stages.append(PromptEncodingStage())
    if expects_controlnet:
        stages.extend([StructuralControlNetStage(), ContextualControlNetStage()])
    stages.append(DiffusionTaskStage())
    return PipelineRoute(
        route_id='txt2img',
        family='txt2img',
        display_name='Txt2Img',
        stages=stages,
    )
