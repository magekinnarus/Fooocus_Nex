import psutil
import torch
import subprocess
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import ldm_patched.modules.model_management as model_management

monitor_router = APIRouter()

def get_gpu_utilization(device_index=0):
    try:
        # Use nvidia-smi with specific index to get usage percentage
        result = subprocess.check_output(
            ["nvidia-smi", "-i", str(device_index), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return float(result.strip())
    except:
        return 0.0

@monitor_router.get("/nex_api/monitor")
async def get_monitor_data():
    """Returns real-time CPU, RAM, GPU, and VRAM usage."""
    try:
        # 1. CPU & RAM (Global)
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        
        # 2. GPU & VRAM (Current Device)
        gpu_usage = 0.0
        vram_usage = 0.0
        vram_total = 0.0
        vram_free = 0.0
        
        device = model_management.get_torch_device()
        device_index = 0
        if device.type == 'cuda':
            try:
                device_index = device.index if device.index is not None else 0
                gpu_usage = get_gpu_utilization(device_index)
            except:
                gpu_usage = get_gpu_utilization(0)
            try:
                # VRAM
                free, total = torch.cuda.mem_get_info(device)
                vram_total = total / (1024**2)
                vram_free = free / (1024**2)
                vram_used = vram_total - vram_free
                vram_usage = (vram_used / vram_total) * 100 if vram_total > 0 else 0
            except:
                pass

        return JSONResponse(content={
            "cpu": round(cpu_usage, 1),
            "ram": round(ram_usage, 1),
            "gpu": round(gpu_usage, 1),
            "vram": round(vram_usage, 1),
            "vram_details": {
                "free": round(vram_free, 0),
                "total": round(vram_total, 0)
            },
            "device": str(device)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
