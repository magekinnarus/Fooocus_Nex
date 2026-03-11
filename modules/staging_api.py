import os
import io
import urllib.request
import urllib.parse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import modules.config
import modules.util
from PIL import Image

staging_router = APIRouter()

def get_staging_dir():
    staging_dir = os.path.join(modules.config.path_outputs, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir

@staging_router.get("/staging_api/images")
async def list_staging_images():
    """Returns a list of image URLs currently in the staging directory."""
    staging_dir = get_staging_dir()
    files = []
    
    # Sort files by modification time (newest first)
    try:
        entries = sorted(
            [e for e in os.scandir(staging_dir) if e.is_file() and e.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))],
            key=lambda e: e.stat().st_mtime,
            reverse=True
        )
        for entry in entries:
            # We return URLs structured for our dedicated API endpoint
            # This is more reliable than Gradio's internal /file= route
            files.append({
                "name": entry.name,
                "url": f"/staging_api/image/{entry.name}"
            })
    except Exception as e:
        print(f"Error reading staging dir: {e}")
        
    return JSONResponse(content={"images": files})

@staging_router.post("/staging_api/upload")
async def upload_staging_image(
    file: UploadFile = File(None),
    url: str = Form(None)
):
    """Accepts either a File upload or an existing URL/Base64 to save to staging."""
    staging_dir = get_staging_dir()
    
    try:
        img = None
        if file is not None:
             # Handle direct file upload
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            
        elif url is not None:
             # Handle drag-and-drop URL copy
             if url.startswith("data:image"):
                 # Handle base64
                 header, encoded = url.split(",", 1)
                 import base64
                 data = base64.b64decode(encoded)
                 img = Image.open(io.BytesIO(data)).convert("RGB")
             elif "/file=" in url:
                 # Handle Gradio Local URL
                 filepath = url.split("/file=", 1)[1].split("?")[0]
                 filepath = urllib.parse.unquote(filepath)
                 if os.path.exists(filepath):
                     img = Image.open(filepath).convert("RGB")
                 else:
                     raise HTTPException(status_code=400, detail="Local file not found")
             elif url.startswith("file://"):
                 # Handle local file URL
                 filepath = url.replace("file:///", "").replace("file://", "")
                 filepath = urllib.parse.unquote(filepath)
                 if os.path.exists(filepath):
                     img = Image.open(filepath).convert("RGB")
                 else:
                     raise HTTPException(status_code=400, detail="Local file not found")
             elif url.startswith("http"):
                 # Download remote URL
                 req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                 with urllib.request.urlopen(req) as response:
                     data = response.read()
                     img = Image.open(io.BytesIO(data)).convert("RGB")
             else:
                 raise HTTPException(status_code=400, detail="Invalid URL format")
        else:
             raise HTTPException(status_code=400, detail="Must provide file or url")
             
        if img:
            # Save image to staging dir
            import datetime
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_%f")
            filename = f"staged_{time_str}.png"
            filepath = os.path.join(staging_dir, filename)
            img.save(filepath, format="PNG")
            return JSONResponse(content={"status": "success", "file": filename})
        
        return JSONResponse(content={"status": "error", "message": "Failed to process image"})
        
    except Exception as e:
        print(f"Staging upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.delete("/staging_api/delete")
async def delete_staging_image(name: str):
    """Deletes a specific image from the staging directory."""
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)
    
    # Security check: ensure the file is actually inside the staging dir
    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return JSONResponse(content={"status": "success"})
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        print(f"Staging delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.post("/staging_api/clear")
async def clear_staging_images():
    """Clears all images from the staging directory."""
    staging_dir = get_staging_dir()
    try:
        import shutil
        # We don't want to delete the dir itself, just the contents
        for filename in os.listdir(staging_dir):
            file_path = os.path.join(staging_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        print(f"Staging clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@staging_router.get("/staging_api/image/{name}")
async def get_staging_image(name: str):
    """Serves a specific image from the staging directory."""
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)
    
    # Security check
    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    if os.path.exists(filepath):
        from fastapi.responses import FileResponse
        return FileResponse(filepath)
    
    raise HTTPException(status_code=404, detail="Image not found")
