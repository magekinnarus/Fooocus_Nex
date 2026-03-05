# Work Order: P4-M01-W01 — API Contract Design & Server Scaffold
**ID:** P4-M01-W01
**Mission:** P4-M01
**Status:** Ready
**Depends On:** P3-M12-1 (completed)

## Mandatory Reading
- `.agent/missions/active/P4-M01_mission_brief.md`
- `.agent/summaries/04_Inference_Architectural_Guideline.md`
- `Fooocus_Nex/modules/async_worker.py` (current `handler()` + `AsyncTask.__init__()`)
- `Fooocus_Nex/modules/config.py` (model paths, defaults, resolution lists)
- `Fooocus_Nex/modules/flags.py` (performance modes, aspect ratios, generation modes)

## Objective
Define the API contract (endpoints, schemas, conventions) and build the FastAPI server scaffold
that starts up, loads models, and responds to health checks. After this work order, `python
api_server.py` starts a server that loads the SDXL model and responds to `GET /health` and
`GET /models`.

## Scope

### 1. API Contract Document

Create `.agent/reference/api_contract.md` defining:

#### Endpoints
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

#### Core Schemas (Pydantic)

```python
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

class LoraSpec(BaseModel):
    name: str
    weight: float = 1.0

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

#### Conventions
- **Errors**: HTTP status codes (400 validation, 409 busy, 500 internal) + JSON `{"error": "message"}`
- **Busy handling**: If a generation is running, new requests return 409 Conflict
- **Image format**: base64-encoded PNG for both upload and download
- **Seeds**: -1 means random; response always includes the actual seed used

### 2. FastAPI Server Scaffold

Create `Fooocus_Nex/api_server.py`:

```python
# Key components:
app = FastAPI(title="Nex Engine API", version="0.1.0")

# CORS for local React dev server (localhost:5173)
app.add_middleware(CORSMiddleware, ...)

# Static file serving for output images
app.mount("/outputs", StaticFiles(directory=config.path_outputs), name="outputs")

# Startup: load default model
@app.on_event("startup")
async def startup():
    # Call refresh_everything() to load default model
    # This reuses existing default_pipeline.py initialization

# Health check
@app.get("/health")
async def health():
    return {"status": "ready", "model": current_model_name}

# Model listing
@app.get("/models")
async def list_models():
    # Read from config.path_checkpoints, config.path_loras, etc.
```

### 3. Entry Point

The server should be launchable via:
```
python api_server.py                    # Default: localhost:8000
python api_server.py --port 8080       # Custom port
python api_server.py --host 0.0.0.0    # For Colab/remote access
```

Uses `uvicorn` as the ASGI server.

## Implementation Steps

### Step 1: API Contract Document
- [ ] Create `.agent/reference/api_contract.md` with full endpoint/schema specification
- [ ] Review against current `AsyncTask.__init__()` parameter list to ensure coverage

### Step 2: FastAPI Project Setup
- [ ] Create `Fooocus_Nex/api_server.py` with FastAPI app, CORS, static files
- [ ] Add `fastapi`, `uvicorn`, `python-multipart` to requirements
- [ ] Implement `GET /health` endpoint
- [ ] Implement model loading on startup (reuse `default_pipeline.refresh_everything()`)

### Step 3: Model & Style Listing
- [ ] Implement `GET /models` — scan model directories, return lists
- [ ] Implement `GET /styles` — return available style list from `sdxl_styles.py`
- [ ] **Verification**: `curl localhost:8000/health` returns ready status with model name
- [ ] **Verification**: `curl localhost:8000/models` returns model file lists

## Success Criteria
1. `python api_server.py` starts without errors, loads SDXL GGUF model
2. `GET /health` returns `{"status": "ready", "model": "<model_name>"}`
3. `GET /models` returns lists of base models, LoRAs, and VAEs
4. `GET /styles` returns available style names
5. No Gradio imports in `api_server.py`
6. CORS allows requests from `localhost:5173` (Vite dev server)
7. API contract document exists and covers all planned endpoints
