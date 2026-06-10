from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

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


@runtime_surface_router.get("/runtime_surface_api/preview_image")
async def get_runtime_surface_preview_image(
    revision: int | None = Query(default=None),
    max_width: int | None = Query(default=None),
    max_height: int | None = Query(default=None),
):
    runtime_surface_state.drain_worker_state()
    _preview_value, preview_revision = runtime_surface_state.get_preview_state()
    preview_headers = {"Cache-Control": "no-store, max-age=0"}
    requested_max_width = max(0, int(max_width or 0))
    requested_max_height = max(0, int(max_height or 0))

    if requested_max_width <= 0 and requested_max_height <= 0:
        preview_path = runtime_surface_state.get_preview_image_path()
        if preview_path is not None:
            preview_headers["X-Nex-Preview-Revision"] = str(int(preview_revision))
            return FileResponse(preview_path, headers=preview_headers)

    preview_bytes, media_type, encoded_revision = runtime_surface_state.get_preview_image_bytes(
        max_width=requested_max_width,
        max_height=requested_max_height,
    )
    if preview_bytes is None or media_type is None:
        preview_path = runtime_surface_state.get_preview_image_path()
        if preview_path is None:
            raise HTTPException(status_code=404, detail="Preview image not found")
        preview_headers["X-Nex-Preview-Revision"] = str(int(preview_revision))
        return FileResponse(preview_path, headers=preview_headers)

    preview_headers["X-Nex-Preview-Revision"] = str(int(encoded_revision))
    return Response(content=preview_bytes, media_type=media_type, headers=preview_headers)


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
