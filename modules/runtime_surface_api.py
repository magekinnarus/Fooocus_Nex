from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from modules import runtime_surface_state


runtime_surface_router = APIRouter()


@runtime_surface_router.get("/runtime_surface_api/state")
async def get_runtime_surface_state():
    return JSONResponse(content={"status": "success", "state": runtime_surface_state.get_runtime_snapshot()})


@runtime_surface_router.get("/runtime_surface_api/completed_image/{task_id}/{image_index}")
async def get_runtime_surface_completed_image(task_id: str, image_index: int):
    image_path = runtime_surface_state.get_completed_image_path(task_id, image_index)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Completed image not found")
    return FileResponse(image_path)


@runtime_surface_router.post("/runtime_surface_api/action")
async def post_runtime_surface_action(payload: dict = Body(...)):
    action = str((payload or {}).get("action", "")).strip().lower()
    task_id = str((payload or {}).get("task_id", "")).strip()

    if action == "skip":
        runtime_surface_state.request_skip_active()
    elif action == "cancel":
        if not task_id:
            raise HTTPException(status_code=400, detail="Missing task_id for cancel action")
        runtime_surface_state.request_cancel_task(task_id)
    elif action == "delete_completed":
        if not task_id:
            raise HTTPException(status_code=400, detail="Missing task_id for delete_completed action")
        runtime_surface_state.request_delete_completed_task(task_id)
    elif action == "clear_all":
        runtime_surface_state.request_clear_all()
    elif action == "refresh":
        pass
    else:
        raise HTTPException(status_code=400, detail="Unsupported runtime surface action")

    return JSONResponse(content={"status": "success", "state": runtime_surface_state.get_runtime_snapshot()})
