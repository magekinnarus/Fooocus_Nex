import os
import io
import base64
import shutil
import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np

import modules.config
import modules.mask_processing as mask_proc

image_router = APIRouter()

def get_workspaces_root():
    """Returns the root directory for all allocated workspaces."""
    root = os.path.abspath(os.path.join(modules.config.path_outputs, "workspaces"))
    os.makedirs(root, exist_ok=True)
    return root

def get_workspace_dir(workspace_id: str):
    """Returns the directory for a specific workspace ID."""
    if not workspace_id or not all(c.isalnum() or c == '_' for c in workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID")
    
    root = get_workspaces_root()
    workspace_dir = os.path.join(root, workspace_id)
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir

@image_router.post("/image_api/upload")
async def upload_image(
    workspace_id: str = Form(...),
    file: UploadFile = File(...),
    preserve_metadata: bool = Form(False)
):
    """Uploads an image to a specific workspace."""
    workspace_dir = get_workspace_dir(workspace_id)

    try:
        contents = await file.read()

        if preserve_metadata:
            filename = Path(file.filename or "base.png").name or "base.png"
            filepath = os.path.join(workspace_dir, filename)
            with open(filepath, "wb") as f:
                f.write(contents)
        else:
            img = Image.open(io.BytesIO(contents)).convert("RGBA")
            filepath = os.path.join(workspace_dir, "base.png")
            img.save(filepath, format="PNG")
            filename = "base.png"

        return JSONResponse(content={
            "status": "success",
            "workspace_id": workspace_id,
            "filename": filename,
            "path": filepath,
            "url": f"/image_api/image/{workspace_id}/{filename}"
        })
    except Exception as e:
        print(f"[ImageAPI] Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@image_router.get("/image_api/image/{workspace_id}/{filename}")
async def get_image(workspace_id: str, filename: str):
    """Serves an image from a specific workspace."""
    workspace_dir = get_workspace_dir(workspace_id)
    filepath = os.path.join(workspace_dir, filename)
    
    # Security check: ensure the file is inside the workspace dir
    if not os.path.abspath(filepath).startswith(os.path.abspath(workspace_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if os.path.exists(filepath):
        return FileResponse(filepath)
    
    raise HTTPException(status_code=404, detail="Image not found")

@image_router.post("/image_api/compute_context")
async def compute_context(
    workspace_id: str = Form(...),
    mask_base64: str = Form(...)
):
    """
    Computes inpaint context and BB patch based on a base64 mask.
    Saves context_mask.png and bb_patch.png to the workspace.
    """
    workspace_dir = get_workspace_dir(workspace_id)
    base_image_path = os.path.join(workspace_dir, "base.png")
    
    if not os.path.exists(base_image_path):
        raise HTTPException(status_code=404, detail="Base image not found in workspace")
    
    try:
        # Load inputs
        original_image = mask_proc.unpack_gradio_data(base_image_path)
        context_mask = mask_proc.ensure_numpy(mask_base64, mode='L')
        
        if original_image is None or context_mask is None:
            raise HTTPException(status_code=400, detail="Invalid image or mask data")
        
        # Compute core logic
        ctx = mask_proc.core_compute_inpaint_step1_context(original_image, context_mask)
        
        # Save results to workspace
        mask_proc.save_to_png(context_mask, os.path.join(workspace_dir, "context_mask.png"))
        mask_proc.save_to_png(ctx.bb_image, os.path.join(workspace_dir, "bb_patch.png"))
        
        return JSONResponse(content={
            "status": "success",
            "context_mask_url": f"/image_api/image/{workspace_id}/context_mask.png",
            "bb_patch_url": f"/image_api/image/{workspace_id}/bb_patch.png"
        })
    except Exception as e:
        print(f"[ImageAPI] Compute context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@image_router.post("/image_api/compute_bb")
async def compute_bb(
    workspace_id: str = Form(...),
    mask_base64: str = Form(...)
):
    """
    Computes the bounding box mask based on base64 input.
    Saves bb_mask.png to the workspace.
    """
    workspace_dir = get_workspace_dir(workspace_id)
    
    try:
        bb_mask = mask_proc.ensure_numpy(mask_base64, mode='L')
        if bb_mask is None:
            raise HTTPException(status_code=400, detail="Invalid mask data")
            
        mask_proc.save_to_png(bb_mask, os.path.join(workspace_dir, "bb_mask.png"))
        
        return JSONResponse(content={
            "status": "success",
            "bb_mask_url": f"/image_api/image/{workspace_id}/bb_mask.png"
        })
    except Exception as e:
        print(f"[ImageAPI] Compute BB error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@image_router.delete("/image_api/workspace/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Deletes a specific workspace directory."""
    workspace_dir = get_workspace_dir(workspace_id)
    
    try:
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
            return JSONResponse(content={"status": "success"})
        else:
            raise HTTPException(status_code=404, detail="Workspace not found")
    except Exception as e:
        print(f"[ImageAPI] Delete workspace error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_old_workspaces(max_age_hours=24):
    """Utility to clean up workspaces older than a certain age."""
    root = get_workspaces_root()
    now = datetime.datetime.now()
    
    for item in os.listdir(root):
        path = os.path.join(root, item)
        if os.path.isdir(path):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
            if (now - mtime).total_seconds() > (max_age_hours * 3600):
                try:
                    shutil.rmtree(path)
                    print(f"[ImageAPI] Cleaned up old workspace: {item}")
                except Exception as e:
                    print(f"[ImageAPI] Failed to clean up {item}: {e}")
