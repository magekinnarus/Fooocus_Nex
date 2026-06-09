from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import urllib.parse

import modules.async_worker as worker


@dataclass
class CompletedTaskRecord:
    task_id: str
    prompt: str
    model_name: str
    seed: object
    images: list[str]


_state_mutex = threading.RLock()
completed_tasks_history: list[CompletedTaskRecord] = []
_last_seen_active_task = None
_last_active_task_id: str | None = None
_last_progress_state = {"visible": False, "number": 0, "text": ""}
_last_preview_value = None
_last_preview_revision = 0


def build_file_url(path: str) -> str:
    return f"/file={urllib.parse.quote(str(path), safe='')}"


def build_completed_image_url(task_id: str, image_index: int) -> str:
    safe_task_id = urllib.parse.quote(str(task_id or ""), safe="")
    return f"/runtime_surface_api/completed_image/{safe_task_id}/{int(image_index)}"


def build_prompt_preview(prompt: str, *, limit: int = 40) -> str:
    prompt_preview = str(prompt or "")[:limit]
    if len(str(prompt or "")) > limit:
        prompt_preview += "..."
    if not prompt_preview.strip():
        prompt_preview = "Image generation"
    return prompt_preview


def reset_runtime_surface_state():
    global _last_seen_active_task, _last_active_task_id
    global _last_preview_value, _last_preview_revision, _last_progress_state
    with _state_mutex:
        completed_tasks_history.clear()
        _last_seen_active_task = None
        _last_active_task_id = None
        _last_preview_value = None
        _last_preview_revision = 0
        _last_progress_state = {"visible": False, "number": 0, "text": ""}


def set_progress_state(*, visible: bool, number: int = 0, text: str = ""):
    with _state_mutex:
        _last_progress_state["visible"] = bool(visible)
        _last_progress_state["number"] = int(number or 0)
        _last_progress_state["text"] = str(text or "")


def _set_preview_value(value):
    global _last_preview_value, _last_preview_revision
    _last_preview_value = value
    _last_preview_revision += 1


def get_preview_state():
    with _state_mutex:
        return _last_preview_value, _last_preview_revision


def _record_completed_task(task, images):
    if not images:
        return False

    task_id = getattr(task, "task_id", None)
    if task_id is None or any(record.task_id == task_id for record in completed_tasks_history):
        return False

    completed_tasks_history.append(
        CompletedTaskRecord(
            task_id=task_id,
            prompt=getattr(getattr(task, "state", None), "prompt", ""),
            model_name=getattr(getattr(task, "state", None), "base_model_name", ""),
            seed=getattr(getattr(task, "state", None), "seed", ""),
            images=list(images),
        )
    )
    if len(completed_tasks_history) > 50:
        completed_tasks_history.pop(0)
    return True


def _drain_task_events(task):
    latest_preview_value = None
    latest_progress_pct = None
    latest_progress_msg = None
    finished_images = None

    while len(task.yields) > 0:
        flag, product = task.yields.pop(0)
        if flag == "preview":
            pct, msg, img = product
            latest_progress_pct = pct
            latest_progress_msg = msg
            if img is not None:
                latest_preview_value = img
        elif flag == "finish":
            finished_images = list(product) if isinstance(product, list) else [product]
            _record_completed_task(task, finished_images)

    return latest_preview_value, latest_progress_pct, latest_progress_msg, finished_images


def drain_worker_state():
    global _last_seen_active_task, _last_active_task_id
    with _state_mutex:
        active_task = worker.get_active_task()
        active_task_id = getattr(active_task, "task_id", None)

        previous_task = None
        if (
            _last_seen_active_task is not None
            and getattr(_last_seen_active_task, "task_id", None) != active_task_id
        ):
            previous_task = _last_seen_active_task

        if previous_task is not None:
            _, _, _, previous_finished_images = _drain_task_events(previous_task)
            if previous_finished_images:
                _set_preview_value(previous_finished_images[0])

        if active_task is not None and _last_active_task_id != active_task_id:
            set_progress_state(visible=True, number=1, text="Waiting for task to start ...")
            _last_active_task_id = active_task_id

        if active_task is not None:
            latest_preview_value, latest_progress_pct, latest_progress_msg, finished_images = _drain_task_events(active_task)

            if finished_images:
                _set_preview_value(finished_images[0])

            if latest_preview_value is not None:
                _set_preview_value(latest_preview_value)

            if latest_progress_msg is not None:
                set_progress_state(
                    visible=True,
                    number=latest_progress_pct or 0,
                    text=latest_progress_msg,
                )

            _last_seen_active_task = active_task
        else:
            if _last_active_task_id is not None:
                set_progress_state(visible=False, number=0, text="")
            _last_active_task_id = None
            if previous_task is not None:
                _last_seen_active_task = None


def _serialize_task(task):
    if task is None:
        return None

    state = getattr(task, "state", None)
    return {
        "task_id": getattr(task, "task_id", ""),
        "prompt_preview": build_prompt_preview(getattr(state, "prompt", "")),
        "model_name": str(getattr(state, "base_model_name", "") or ""),
        "seed": getattr(state, "seed", ""),
        "progress": max(0, min(int(getattr(state, "current_progress", 0) or 0), 100)),
        "status_text": str(getattr(state, "current_status_text", "") or "").strip(),
    }


def get_completed_image_path(task_id: str, image_index: int) -> str | None:
    if image_index < 0:
        return None

    with _state_mutex:
        for record in completed_tasks_history:
            if record.task_id != task_id:
                continue
            if image_index >= len(record.images):
                return None
            image_path = str(record.images[image_index] or "")
            if not image_path or not os.path.exists(image_path):
                return None
            return image_path
    return None


def remove_completed_task(task_id: str) -> bool:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return False

    with _state_mutex:
        for index, record in enumerate(completed_tasks_history):
            if record.task_id != normalized_task_id:
                continue
            completed_tasks_history.pop(index)
            return True
    return False


def get_runtime_snapshot():
    drain_worker_state()
    with _state_mutex:
        active_task = worker.get_active_task()
        pending_tasks = list(worker.async_tasks)
        active_payload = _serialize_task(active_task)
        if active_payload is not None and not active_payload["status_text"]:
            active_payload["status_text"] = "Waiting for task to start ..."

        return {
            "progress": dict(_last_progress_state),
            "running": active_payload,
            "pending": [_serialize_task(task) for task in pending_tasks],
            "completed": [
                {
                    "task_id": record.task_id,
                    "prompt_preview": build_prompt_preview(record.prompt),
                    "model_name": str(record.model_name or ""),
                    "seed": record.seed,
                    "images": list(record.images),
                    "image_urls": [
                        build_completed_image_url(record.task_id, image_index)
                        for image_index, _ in enumerate(record.images)
                    ],
                }
                for record in reversed(completed_tasks_history)
            ],
            "queue_count": len(pending_tasks) + (1 if active_task is not None else 0),
        }


def request_skip_active():
    active_task = worker.get_active_task()
    if active_task is not None:
        worker.request_interrupt("skip", active_task)


def _clear_progress_if_idle():
    if worker.get_active_task() is None and len(worker.async_tasks) == 0:
        set_progress_state(visible=False, number=0, text="")


def request_cancel_task(task_id: str):
    worker.cancel_task(task_id)
    _clear_progress_if_idle()


def request_delete_completed_task(task_id: str) -> bool:
    return remove_completed_task(task_id)


def request_clear_all():
    active_task = worker.get_active_task()
    if active_task is not None:
        worker.request_interrupt("stop", active_task)
    while len(worker.async_tasks) > 0:
        task = worker.async_tasks.pop(0)
        task.yields.append(["finish", []])
    _clear_progress_if_idle()
