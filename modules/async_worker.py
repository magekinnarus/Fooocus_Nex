import os
import time
import traceback
import threading
import re

import torch

import backend.resources as resources
from backend import process_transition
from backend import sdxl_runtime_policy
import modules.config
import modules.flags as flags
from modules.task_state import TaskState
from modules.pipeline.output import build_image_wall, yield_result
from modules.pipeline.routes import build_generation_route, describe_route, sync_flux_fill_route_session
from modules.pipeline.stage_runtime import PipelineRouteContext, PipelineStageRunner

class AsyncTask:
    callback_steps: float = 0.0

    def __init__(self, args):
        import uuid
        self.task_id = str(uuid.uuid4())[:8]
        self.enqueue_time = time.time()
        self.ui_delivered_result_count = 0

        from modules.flags import MetadataScheme
        from modules.util import get_enabled_loras
        from modules.config import default_max_lora_number
        import args_manager

        self.state = TaskState()
        self.yields = self.state.yields
        self.results = self.state.results # Shared reference
        self.is_valid = len(args) > 0
        
        if not self.is_valid:
            return

        if isinstance(args, list):
            raise TypeError("AsyncTask received a positional args list instead of a named dictionary. Clear your browser cache and restart.")

        s = self.state

        import modules.parameter_registry as registry
        for param in registry.PARAM_REGISTRY:
            if param.task_field is None:
                continue
            
            val = args.get(param.name, param.default)
            if param.transform and val is not None:
                try:
                    val = param.transform(val)
                except (ValueError, TypeError):
                    val = param.default
            setattr(s, param.task_field, val)

        s.original_steps = s.steps

        lora_data = []
        for i in range(default_max_lora_number):
            enabled = bool(args.get(f'lora_{i}_enabled', False))
            name = str(args.get(f'lora_{i}_model', 'None'))
            weight = float(args.get(f'lora_{i}_weight', 1.0))
            lora_data.append((enabled, name, weight))
        s.loras = get_enabled_loras(lora_data)

        if not getattr(args_manager.args, 'disable_metadata', False):
            s.save_metadata_to_images = args.get('save_metadata_to_images', False)
            scheme_val = args.get('metadata_scheme', 'fooocus_nex')
            try:
                s.metadata_scheme = MetadataScheme(scheme_val)
            except ValueError:
                s.metadata_scheme = MetadataScheme.FOOOCUS
        else:
            s.save_metadata_to_images = False
            s.metadata_scheme = MetadataScheme.FOOOCUS

        def has_controlnet_input(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ''
            if isinstance(value, dict):
                for key in ['image', 'mask', 'background']:
                    item = value.get(key)
                    if isinstance(item, str) and item.strip() != '':
                        return True
                    if item is not None and not isinstance(item, str):
                        return True
                return False
            return True

        from modules.config import default_controlnet_image_count
        for i in range(default_controlnet_image_count):
            cn_img = args.get(f'cn_{i}_image')
            cn_stop = args.get(f'cn_{i}_stop', 1.0)
            cn_weight = args.get(f'cn_{i}_weight', 1.0)
            raw_cn_type = args.get(f'cn_{i}_type')
            if not has_controlnet_input(cn_img):
                continue

            cn_type = flags.resolve_cn_type(raw_cn_type, default=None)
            if cn_type is None or not s.add_cn_task(cn_type, [cn_img, cn_stop, cn_weight]):
                print(f'[ControlNet] Skipping unsupported guidance type: {raw_cn_type!r}')

    @property
    def generate_image_grid(self): return self.state.generate_image_grid
    @property
    def last_stop(self): return self.state.last_stop
    @last_stop.setter
    def last_stop(self, value): self.state.last_stop = value
    @property
    def processing(self): return self.state.processing
    @processing.setter
    def processing(self, value): self.state.processing = value


async_tasks = []
_active_task = None
_active_task_mutex = threading.RLock()


def set_active_task(task):
    global _active_task
    with _active_task_mutex:
        _active_task = task


def get_active_task():
    with _active_task_mutex:
        return _active_task

def cancel_task(task_id: str) -> bool:
    global async_tasks
    with _active_task_mutex:
        active = _active_task
        if active and getattr(active, 'task_id', None) == task_id:
            request_interrupt('stop', active)
            return True
        for i, task in enumerate(async_tasks):
            if getattr(task, 'task_id', None) == task_id:
                async_tasks.pop(i)
                task.yields.append(['finish', []])
                return True
    return False


def request_interrupt(action, task=None):
    # Flux stop/skip interrupts are intentionally non-destructive.
    # Route-entry reconciliation decides whether a later route switch should tear residency down.
    target = get_active_task()
    if target is None:
        target = task
    if target is not None:
        target.last_stop = action
    resources.interrupt_current_processing()
    return target if target is not None else task


def progressbar(task_state, number, text):
    resources.throw_exception_if_processing_interrupted()
    task_state.current_progress = int(number)
    task_state.current_status_text = str(text or '')
    print(f'[Fooocus] {text}')
    task_state.yields.append(['preview', (number, text, None)])


@torch.no_grad()
@torch.inference_mode()
def _release_route_runtime_state(task_state):
    task_state.initial_latent = None
    task_state.positive_cond = None
    task_state.negative_cond = None
    task_state.uov_input_image = None
    task_state.inpaint_input_image = None
    task_state.inpaint_mask_image = None
    task_state.inpaint_context = None
    task_state.context_mask = None
    task_state.outpaint_input_image = None
    task_state.outpaint_mask_image = None
    for cn_type in list(task_state.cn_tasks.keys()):
        task_state.cn_tasks[cn_type] = []
    task_state.ensure_cn_task_maps()


def _resolve_preflight_additional_loras(task_state) -> list:
    additional_loras = []

    # 1. Inpaint / Outpaint patch LoRA
    try:
        from modules import flags, config
        from modules.objr_engine import is_flux_fill_inpaint_route

        # Check outpaint
        mixed_cn_outpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_outpaint', False)
        has_mixed_outpaint_request = mixed_cn_outpaint_workflow and getattr(task_state, 'outpaint_input_image', None) is not None and (
            getattr(task_state, 'outpaint_step2_checkbox', False)
            or bool(getattr(task_state, 'outpaint_selections', []))
            or getattr(task_state, 'outpaint_mask_image', None) is not None
        )
        is_outpaint = (task_state.current_tab == 'outpaint' or has_mixed_outpaint_request) and getattr(task_state, 'outpaint_input_image', None) is not None

        # Check inpaint
        mixed_cn_inpaint_workflow = task_state.current_tab == 'ip' and getattr(task_state, 'mixing_image_prompt_and_inpaint', False)
        has_mixed_inpaint_request = mixed_cn_inpaint_workflow and getattr(task_state, 'inpaint_input_image', None) is not None
        is_inpaint = not is_outpaint and (task_state.current_tab == 'inpaint' or has_mixed_inpaint_request) and getattr(task_state, 'inpaint_input_image', None) is not None

        use_flux_fill_inpaint = task_state.current_tab == 'inpaint' and is_flux_fill_inpaint_route(getattr(task_state, 'inpaint_route', None))

        if (is_outpaint or is_inpaint) and not use_flux_fill_inpaint:
            engine = getattr(task_state, 'outpaint_engine', 'None') if is_outpaint else getattr(task_state, 'inpaint_engine', 'None')
            engine = flags.normalize_inpaint_engine_version(engine, default=flags.INPAINT_ENGINE_NONE)
            if engine != flags.INPAINT_ENGINE_NONE:
                inpaint_patch_model_path = config.downloading_inpaint_models(engine)
                additional_loras.append((inpaint_patch_model_path, 1.0))
    except Exception:
        pass

    # 2. FaceID LoRA
    try:
        from modules import flags, model_registry

        contextual_tasks = task_state.get_cn_tasks_for_channel(flags.cn_contextual)
        if len(contextual_tasks.get(flags.cn_faceid, [])) > 0:
            faceid_lora_path = model_registry.ensure_asset('contextual.faceid.lora')
            additional_loras.append((faceid_lora_path, 1.0))
    except Exception:
        pass

    return additional_loras


def _resolve_sdxl_process_key(task_state) -> process_transition.ProcessKey | None:
    from modules.pipeline.inference import resolve_unified_sdxl_process_key

    return resolve_unified_sdxl_process_key(
        task_state,
        loras=getattr(task_state, 'loras', []) or [],
        base_model_additional_loras=getattr(task_state, 'base_model_additional_loras', []) or [],
    )


def _resolve_flux_fill_process_key(
    task_state,
    *,
    route_family: str | None = None,
    selected_engine: str | None = None,
) -> process_transition.ProcessKey | None:
    try:
        import modules.objr_engine as objr_engine

        if selected_engine is None:
            selected_engine = objr_engine.normalize_objr_engine(getattr(task_state, 'objr_engine', None))
        if str(route_family or '').strip().lower() == 'flux_fill':
            selected_engine = objr_engine.OBJR_ENGINE_FLUX_FILL
        if selected_engine != objr_engine.OBJR_ENGINE_FLUX_FILL:
            return None

        asset_paths = objr_engine.resolve_flux_fill_asset_paths(
            conditioning=getattr(task_state, 'flux_fill_conditioning', None),
            progress=False,
        )
        return process_transition.build_process_key(
            family=process_transition.PROCESS_FAMILY_FLUX_FILL,
            process_class=process_transition.PROCESS_CLASS_FLUX_FILL,
            authoritative_identity=tuple(sorted(asset_paths.items())),
            route_family='flux_fill',
        )
    except Exception:
        return None


def _resolve_requested_process_key(task_state, route) -> process_transition.ProcessKey | None:
    import modules.objr_engine as objr_engine

    selected_engine = objr_engine.normalize_objr_engine(getattr(task_state, 'objr_engine', None))
    expects_flux_process = route.family == 'flux_fill' or selected_engine == objr_engine.OBJR_ENGINE_FLUX_FILL
    if expects_flux_process:
        return _resolve_flux_fill_process_key(
            task_state,
            route_family=route.family,
            selected_engine=selected_engine,
        )
    if getattr(task_state.sdxl_execution_policy, 'enabled', False):
        return _resolve_sdxl_process_key(task_state)
    return None


def _release_process_boundary(current_key, requested_key):
    if current_key is None or requested_key is None:
        return None

    if current_key.family == process_transition.PROCESS_FAMILY_SDXL:
        import backend.resources as resources
        import modules.default_pipeline as default_pipeline

        current_model_name = getattr(current_key, 'authoritative_identity', (None,))[0] if getattr(current_key, 'authoritative_identity', None) else None
        next_model_name = getattr(requested_key, 'authoritative_identity', (None,))[0] if getattr(requested_key, 'authoritative_identity', None) else None

        resources.prepare_for_checkpoint_switch(
            current_model=current_model_name,
            next_model=next_model_name,
            release_callback=None,
            notes={
                'reason': 'route_transition',
                'current_process_key': process_transition.describe_process_key(current_key),
                'next_process_key': process_transition.describe_process_key(requested_key),
            },
        )

        return default_pipeline.release_sdxl_runtime_state(
            current_process_key=current_key,
            next_process_key=requested_key,
            current_model_name=current_model_name,
            next_model_name=next_model_name,
            reason='route_transition',
            hard_reset=False,
        )

    if current_key.family == process_transition.PROCESS_FAMILY_FLUX_FILL:
        import modules.objr_engine as objr_engine

        return objr_engine.end_active_flux_fill_session(reason='route_transition')

    return None


def _apply_process_transition_gate(requested_key):
    if requested_key is None:
        return None

    current_key = process_transition.get_active_process_key()
    decision = process_transition.evaluate_process_transition(requested_key)
    if decision.reset_required:
        _release_process_boundary(current_key, requested_key)
        process_transition.clear_active_process_key()
    return decision


def _sync_route_process_activation(route, task_state, requested_process_key):
    if route.family != "flux_fill":
        return None

    sync_result = sync_flux_fill_route_session(route, task_state, progress=False)

    import modules.objr_engine as objr_engine

    if (
        requested_process_key is not None
        and requested_process_key.family == process_transition.PROCESS_FAMILY_FLUX_FILL
        and objr_engine.has_active_flux_fill_session()
    ):
        process_transition.set_active_process_key(requested_process_key)
    else:
        process_transition.clear_active_process_key()

    return sync_result


@torch.no_grad()
@torch.inference_mode()
def handler(async_task: AsyncTask):
    async_task.last_stop = False
    task_state = async_task.state
    task_state.processing = True
    task_state.current_progress = 0
    resources.begin_memory_phase('task', notes={'goals': list(task_state.goals)})

    print(f'[Parameters] Seed = {task_state.seed}')
    dims = re.findall(r'\d+', str(task_state.aspect_ratios_selection))
    if len(dims) < 2:
        raise ValueError(f'Invalid aspect ratio selection: {task_state.aspect_ratios_selection!r}')
    task_state.width, task_state.height = int(dims[0]), int(dims[1])

    # Resolve model taxonomy first
    resolved_taxonomy = modules.config.resolve_model_taxonomy(task_state.base_model_name)
    if sdxl_runtime_policy.is_legacy_sdxl_gguf_selection(
        architecture=getattr(resolved_taxonomy, 'architecture', None),
        base_model_name=task_state.base_model_name,
    ):
        message = (
            'SDXL GGUF base models are deprecated and no longer supported. '
            'Select an SDXL checkpoint base model instead.'
        )
        print(f'[Nex Error] {message}')
        task_state.yields.append(['preview', (0, message, None)])
        raise ValueError(message)

    # Resolve execution policy
    active_profile = resources.active_memory_environment_profile()
    task_state.sdxl_execution_policy = sdxl_runtime_policy.resolve_sdxl_execution_policy(
        architecture=getattr(resolved_taxonomy, 'architecture', None),
        base_model_name=task_state.base_model_name,
        profile=active_profile,
        requested_residency_class=getattr(task_state, 'sdxl_residency_class', None) or None,
    )
    task_state.sdxl_execution_family = str(getattr(task_state.sdxl_execution_policy, 'execution_family', '') or '')
    task_state.sdxl_residency_class = str(getattr(task_state.sdxl_execution_policy, 'residency_class', '') or '')

    with resources.memory_phase_scope(
        resources.MemoryPhase.ROUTE_SELECT,
        task=task_state,
        notes={
            'current_tab': task_state.current_tab,
            'input_image_checkbox': bool(task_state.input_image_checkbox),
        },
        end_notes={'completed': True},
    ):
        route = build_generation_route(task_state)

    task_state.runtime_route_id = route.route_id
    task_state.runtime_route_family = route.family
    task_state.runtime_route_display_name = route.display_name

    print(f"[Route] {route.route_id}: {' -> '.join(describe_route(route))}")

    # Pre-resolve and pre-populate expected additional LoRAs so preflight/postflight keys match
    task_state.base_model_additional_loras = _resolve_preflight_additional_loras(task_state)

    requested_process_key = _resolve_requested_process_key(task_state, route)

    _apply_process_transition_gate(requested_process_key)

    _sync_route_process_activation(route, task_state, requested_process_key)

    route_context = PipelineRouteContext(
        async_task=async_task,
        task_state=task_state,
        route_id=route.route_id,
        route_family=route.family,
        execution_family=getattr(task_state.sdxl_execution_policy, 'execution_family', None),
        residency_class=resources.normalize_sdxl_residency_class(getattr(task_state, 'sdxl_residency_class', None)),
        sdxl_policy=task_state.sdxl_execution_policy,
        progressbar_callback=progressbar,
        yield_result_callback=yield_result,
        base_model_additional_loras=list(task_state.base_model_additional_loras),
    )
    PipelineStageRunner().run(route, route_context)

    task_state.processing = False
    _release_route_runtime_state(task_state)


def worker():
    pid = os.getpid()
    print(f'Started worker with PID {pid}')
    
    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            set_active_task(task)
            try:
                handler(task)
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': bool(task.state.generate_image_grid)},
                    end_notes={'completed': True, 'success': True},
                ):
                    if task.state.generate_image_grid:
                        build_image_wall(task.state)
                    task.yields.append(['finish', task.results])
            except resources.InterruptProcessingException:
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': False},
                    end_notes={'completed': True, 'success': False, 'interrupted': True},
                ):
                    task.yields.append(['finish', task.results])
            except:
                traceback.print_exc()
                with resources.memory_phase_scope(
                    resources.MemoryPhase.FINALIZE,
                    task=task.state,
                    notes={'generate_image_grid': False},
                    end_notes={'completed': True, 'success': False},
                ):
                    task.yields.append(['finish', task.results])
            finally:
                set_active_task(None)
                resources.cleanup_memory('task_finalize', force_cache=True, notes={'completed': True}, target_phase=resources.MemoryPhase.FINALIZE)
                resources.end_memory_phase('task', notes={'completed': True})


threading.Thread(target=worker, daemon=True).start()
