# API Contract Document: P4-M01

This document defines the API contract for the new FastAPI-based server that decouples the Fooocus UI from the backend engine.

## Endpoints

| Method | Path | Purpose | Request Body | Response |
|--------|------|---------|--------------|----------|
| `GET` | `/health` | Server health | — | `{"status": "ready", "model": "..."}` |
| `GET` | `/models` | List models, LoRAs, VAEs | — | `{"base_models": [...], "loras": [...], "vae": [...]}` |
| `GET` | `/styles` | List available styles | — | `{"styles": [...]}` |
| `POST` | `/generate` | txt2img / img2img | `GenerateRequest` | `GenerateResponse` |
| `POST` | `/inpaint` | Inpaint | `InpaintRequest` | `GenerateResponse` |
| `POST` | `/outpaint` | Outpaint | `OutpaintRequest` | `GenerateResponse` |
| `POST` | `/interrupt` | Cancel generation | — | `{"status": "interrupted"}` |
| `WS` | `/ws/progress` | Progress streaming | — | `ProgressMessage` stream |

## Core Schemas (Pydantic)

```python
from pydantic import BaseModel

class LoraSpec(BaseModel):
    name: str
    weight: float = 1.0

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    style_selections: list[str] = []
    performance: str = "Speed"      # Speed | Quality | Extreme Speed | Lightning | Hyper-SD
    aspect_ratio: str = "1152×896"  # From flags.sdxl_aspect_ratios
    image_number: int = 1
    seed: int = -1                  # -1 = random
    sharpness: float = 2.0
    cfg_scale: float = 7.0
    sampler_name: str = "dpmpp_2m_sde_gpu"
    scheduler_name: str = "karras"
    base_model: str | None = None   # Override default model
    loras: list[LoraSpec] = []
    # img2img fields (optional)
    input_image: str | None = None  # base64 PNG
    denoise_strength: float = 1.0

class InpaintRequest(GenerateRequest):
    input_image: str               # base64 PNG (required)
    mask: str                      # base64 grayscale PNG
    inpaint_mode: str = "Inpaint"  # Inpaint | Improve Detail
    denoise_strength: float = 0.85

class OutpaintRequest(GenerateRequest):
    input_image: str               # base64 PNG
    mask: str | None = None        # For step 2 (user-provided mask)
    direction: str = "Right"       # Left | Right | Top | Bottom
    expansion: int = 384           # Pixels to expand (multiple of 32)

class GenerateResponse(BaseModel):
    images: list[str]              # base64 PNG or file paths
    seed: int
    metadata: dict = {}

class ProgressMessage(BaseModel):
    type: str                      # "progress" | "preview" | "complete" | "error"
    progress: int = 0              # 0-100
    message: str = ""
    preview: str | None = None     # base64 preview image (optional)
```

## Conventions

- **Errors**: HTTP status codes (400 validation, 409 busy, 500 internal) + JSON `{"error": "message"}`
- **Busy handling**: If a generation is running, new requests return 409 Conflict
- **Image format**: base64-encoded PNG for both upload and download
- **Seeds**: -1 means random; response always includes the actual seed used
