from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Dict, Optional

from . import resources


def _memory_snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    if hasattr(snapshot, "__dataclass_fields__"):
        return asdict(snapshot)
    return dict(snapshot)


def snapshot_inference_state(tag: str, *, notes: Optional[Dict[str, Any]] = None, task=None) -> Dict[str, Any]:
    snapshot_notes = dict(notes or {})
    snapshot_notes["tag"] = tag
    memory = resources.capture_memory_snapshot(notes=snapshot_notes, task=task)
    loaded_models = resources.loaded_model_state()
    return {
        "tag": tag,
        "phase": resources.current_memory_phase(),
        "memory": _memory_snapshot_to_dict(memory),
        "loaded_models": loaded_models,
        "loaded_model_count": len(loaded_models),
    }


def attach_patcher_for_stage(
    model_patcher: Any,
    stage: str,
    *,
    memory_required: int = 0,
    force_patch_weights: bool = False,
    minimum_memory_required: Optional[int] = None,
    force_full_load: bool = False,
    force_high_vram: bool = False,
    target_phase=None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    before = snapshot_inference_state(f"{stage}_before_attach", notes=notes)
    started_at = time.perf_counter()
    resources.load_models_gpu(
        [model_patcher],
        memory_required=memory_required,
        force_patch_weights=force_patch_weights,
        minimum_memory_required=minimum_memory_required,
        force_full_load=force_full_load,
        force_high_vram=force_high_vram,
        target_phase=target_phase,
    )
    duration_s = time.perf_counter() - started_at
    after = snapshot_inference_state(f"{stage}_after_attach", notes=notes)
    return {
        "action": "attach",
        "stage": stage,
        "model": resources.describe_model_patcher(model_patcher),
        "duration_s": duration_s,
        "target_phase": resources.normalize_memory_phase(target_phase) if target_phase is not None else None,
        "before": before,
        "after": after,
    }


def detach_patcher_after_stage(
    model_patcher: Any,
    stage: str,
    *,
    unpatch_all: bool = True,
    flush_cache: bool = False,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    before = snapshot_inference_state(f"{stage}_before_detach", notes=notes)
    started_at = time.perf_counter()
    model_patcher.detach(unpatch_all=unpatch_all)
    if flush_cache:
        resources.soft_empty_cache(force=True)
    duration_s = time.perf_counter() - started_at
    after = snapshot_inference_state(f"{stage}_after_detach", notes=notes)
    return {
        "action": "detach",
        "stage": stage,
        "model": resources.describe_model_patcher(model_patcher),
        "duration_s": duration_s,
        "flush_cache": flush_cache,
        "before": before,
        "after": after,
    }


def reset_inference_run_state(
    reason: str,
    *,
    unload_models: bool = False,
    force_cache: bool = True,
    gc_collect: bool = True,
    trim_host=None,
    target_phase=None,
    notes: Optional[Dict[str, Any]] = None,
    task=None,
) -> Dict[str, Any]:
    before = snapshot_inference_state(f"{reason}_before_reset", notes=notes, task=task)
    started_at = time.perf_counter()
    cleanup_snapshot = resources.cleanup_memory(
        reason,
        unload_models=unload_models,
        force_cache=force_cache,
        gc_collect=gc_collect,
        trim_host=trim_host,
        notes=notes,
        target_phase=target_phase,
        task=task,
    )
    duration_s = time.perf_counter() - started_at
    after = snapshot_inference_state(f"{reason}_after_reset", notes=notes, task=task)
    return {
        "action": "reset",
        "reason": reason,
        "duration_s": duration_s,
        "target_phase": resources.normalize_memory_phase(target_phase) if target_phase is not None else None,
        "cleanup_snapshot": _memory_snapshot_to_dict(cleanup_snapshot),
        "before": before,
        "after": after,
    }
