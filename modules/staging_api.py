import os
import io
import urllib.request
import urllib.parse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
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
            img = Image.open(io.BytesIO(contents)).convert("RGBA")
            
        elif url is not None:
             # Handle drag-and-drop URL copy
             if url.startswith("data:image"):
                 # Handle base64
                 header, encoded = url.split(",", 1)
                 import base64
                 data = base64.b64decode(encoded)
                 img = Image.open(io.BytesIO(data)).convert("RGBA")
             elif "/file=" in url:
                 # Handle Gradio Local URL
                 filepath = url.split("/file=", 1)[1].split("?")[0]
                 filepath = urllib.parse.unquote(filepath)
                 if os.path.exists(filepath):
                     img = Image.open(filepath).convert("RGBA")
                 else:
                     raise HTTPException(status_code=400, detail="Local file not found")
             elif url.startswith("file://"):
                 # Handle local file URL
                 filepath = url.replace("file:///", "").replace("file://", "")
                 filepath = urllib.parse.unquote(filepath)
                 if os.path.exists(filepath):
                     img = Image.open(filepath).convert("RGBA")
                 else:
                     raise HTTPException(status_code=400, detail="Local file not found")
             elif url.startswith("http"):
                 # Download remote URL
                 req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                 with urllib.request.urlopen(req) as response:
                     data = response.read()
                     img = Image.open(io.BytesIO(data)).convert("RGBA")
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
            return JSONResponse(content={
                "status": "success",
                "file": filename,
                "filepath": filepath,
                "url": f"/staging_api/image/{filename}",
            })
        
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

@staging_router.post("/staging_api/gimp_target")
async def set_gimp_target(name: str):
    """Sets a specific image as the current target for GIMP retrieval."""
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)
    
    # Security check
    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        target_file = os.path.join(staging_dir, ".gimp_target.txt")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(name)
        return JSONResponse(content={"status": "success", "target": name})
    except Exception as e:
        print(f"Staging GIMP target error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@staging_router.get("/staging_api/gimp_target")
async def get_gimp_target():
    """Returns the current GIMP target image file."""
    staging_dir = get_staging_dir()
    target_file = os.path.join(staging_dir, ".gimp_target.txt")
    
    if not os.path.exists(target_file):
        raise HTTPException(status_code=404, detail="No GIMP target set")
        
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            name = f.read().strip()
            
        filepath = os.path.join(staging_dir, name)
        
        # Security check
        if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
            raise HTTPException(status_code=403, detail="Forbidden")
            
        if os.path.exists(filepath):
            from fastapi.responses import FileResponse
            return FileResponse(filepath)
            
        raise HTTPException(status_code=404, detail="Targeted image no longer exists")
    except Exception as e:
        print(f"Staging GIMP retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _safe_staging_path(name: str):
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, name)
    if not os.path.abspath(filepath).startswith(os.path.abspath(staging_dir)):
        raise HTTPException(status_code=403, detail="Forbidden")
    return staging_dir, filepath


def _center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side))


@staging_router.post("/staging_api/compose_face_grid")
async def compose_face_grid(payload: dict = Body(...)):
    names = payload.get('names') if isinstance(payload, dict) else None
    if not isinstance(names, list) or not names:
        raise HTTPException(status_code=400, detail='Please select at least one staged image')

    selected_names = []
    for name in names:
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if clean_name and clean_name not in selected_names:
            selected_names.append(clean_name)

    if not selected_names:
        raise HTTPException(status_code=400, detail='Please select at least one staged image')

    selected_names = selected_names[:9]
    grid_size = 1 if len(selected_names) == 1 else 2 if len(selected_names) <= 4 else 3
    canvas_size = 1024
    cell_size = canvas_size // grid_size

    tiles = []
    for name in selected_names:
        _, filepath = _safe_staging_path(name)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f'Staged image not found: {name}')
        with Image.open(filepath) as source_img:
            tile = _center_crop_square(source_img.convert('RGBA')).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            tiles.append(tile)

    if not tiles:
        raise HTTPException(status_code=400, detail='No staged images were available to compose')

    canvas = Image.new('RGBA', (cell_size * grid_size, cell_size * grid_size), (255, 255, 255, 255))
    total_cells = grid_size * grid_size
    for index in range(total_cells):
        tile = tiles[index % len(tiles)]
        x = (index % grid_size) * cell_size
        y = (index // grid_size) * cell_size
        canvas.paste(tile, (x, y))

    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    filename = f'face_grid_{grid_size}x{grid_size}_{time_str}.png'
    staging_dir = get_staging_dir()
    filepath = os.path.join(staging_dir, filename)
    canvas.save(filepath, format='PNG')

    return JSONResponse(content={
        'status': 'success',
        'file': filename,
        'filepath': filepath,
        'url': f'/staging_api/image/{filename}',
        'selected_count': len(selected_names),
        'grid_size': grid_size,
    })


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
