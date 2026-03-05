# Work Order: P4-M01-W02 — txt2img Endpoint & Progress Streaming
**ID:** P4-M01-W02
**Mission:** P4-M01
**Status:** Completed
**Depends On:** P4-M01-W01

## Mandatory Reading
- `.agent/reference/api_contract.md` (produced by W01)
- `.agent/missions/active/P4-M01_mission_brief.md`
- `Fooocus_Nex/modules/async_worker.py` — `handler()` flow
- `Fooocus_Nex/modules/pipeline/inference.py` — `process_task()`
- `Fooocus_Nex/modules/pipeline/preprocessing.py` — `process_prompt()`, `apply_overrides()`
- `Fooocus_Nex/modules/default_pipeline.py` — `refresh_everything()`, `process_diffusion()`

## Objective
Implement the `POST /generate` endpoint for txt2img/img2img and real-time progress streaming.
After this work order, a `curl` command generates an image and a WebSocket client receives
progress updates during generation.

## Scope

### 1. TaskState Construction from JSON

The current `AsyncTask.__init__()` reads 70+ positional arguments from Gradio `gr.State` slots.
The API server must construct an equivalent `TaskState` from the `GenerateRequest` schema.

**Approach**: Create a factory function that maps `GenerateRequest` fields to `TaskState` attributes
with sensible defaults for all fields the frontend doesn't set:

```python
def create_task_state(request: GenerateRequest) -> TaskState:
    """Construct TaskState from API request, filling defaults for UI-only fields."""
    state = TaskState()
    state.prompt = request.prompt
    state.negative_prompt = request.negative_prompt
    state.style_selections = request.style_selections
    state.seed = request.seed if request.seed >= 0 else random.randint(0, 2**31)
    # ... map remaining fields
    # Set defaults for UI-only fields (input_image_checkbox, etc.)
    return state
```

**Key mapping points:**
- `aspect_ratio` string → `width`, `height` integers
- `performance` string → `Performance` enum → step/sampler overrides
- `loras` list → format expected by `process_prompt()`
- `seed = -1` → generate random seed

### 2. Headless Orchestrator (`api_handler`)

Create the API-side equivalent of `async_worker.handler()`. This function:

1. Constructs `TaskState` from the request
2. Calls `apply_overrides(state)` for performance presets
3. Calls `process_prompt(state, ...)` for CLIP encoding
4. Calls `process_task(state, ...)` for sampling + decoding
5. Returns generated images

**Critical difference from `handler()`**: No `async_task` wrapper, no Gradio yield mechanism.
Instead, progress is reported via a shared queue that the WebSocket endpoint reads.

```python
# Simplified flow (txt2img path from handler()):
async def api_handler(request: GenerateRequest, progress_queue: asyncio.Queue):
    state = create_task_state(request)
    
    # Performance overrides
    if state.performance_selection == Performance.EXTREME_SPEED:
        set_lcm_defaults(state, lambda s, n, t: progress_queue.put_nowait(...))
    
    # Prompt processing + model loading
    base_model_additional_loras = []
    tasks = process_prompt(state, base_model_additional_loras, progressbar_fn)
    
    # Generation
    for i, task in enumerate(tasks):
        imgs, img_paths, progress = process_task(
            state, task, i, state.image_number, all_steps,
            preparation_steps, state.denoising_strength,
            final_scheduler_name, state.loras, None, None,
            progressbar_callback=progressbar_fn
        )
    
    return GenerateResponse(images=img_paths, seed=state.seed, metadata={...})
```

### 3. Progress Streaming

**Mechanism**: WebSocket at `/ws/progress`

During generation, the orchestrator pushes progress messages to an `asyncio.Queue`.
The WebSocket endpoint reads from this queue and sends JSON messages to the client:

```json
{"type": "progress", "progress": 45, "message": "Sampling step 9/20, image 1/1 ..."}
{"type": "preview", "progress": 50, "message": "...", "preview": "<base64_png>"}
{"type": "complete", "progress": 100, "message": "Done"}
```

**Bridging the callback**: The existing pipeline uses `task_state.yields.append(...)` for progress.
The API handler either:
- (a) Replaces `yields` with the asyncio queue, or
- (b) Polls `yields` in a loop and forwards to the queue

Option (b) is lower-risk — it doesn't modify the pipeline code.

### 4. Generation Endpoint

```python
@app.post("/generate")
async def generate(request: GenerateRequest):
    if generation_lock.locked():
        raise HTTPException(409, "A generation is already in progress")
    
    async with generation_lock:
        result = await run_in_executor(api_handler, request, progress_queue)
        return result
```

**Threading note**: The pipeline uses PyTorch and must run in a thread (not async). Use
`asyncio.get_event_loop().run_in_executor()` to run `api_handler` in a thread pool while
keeping the FastAPI event loop responsive for WebSocket messages.

## Implementation Steps

### Step 1: TaskState Factory
- [x] Create `create_task_state(request)` mapping function
- [x] Map all `GenerateRequest` fields → `TaskState` attributes
- [x] Set safe defaults for UI-only fields (no image input, no ControlNet)
- [x] **Verification**: Construct a TaskState, verify all required fields are populated

### Step 2: Headless Orchestrator
- [x] Create `api_handler()` function following `handler()` txt2img path
- [x] Wire progress reporting to asyncio queue
- [x] Handle the threading boundary (PyTorch in thread, FastAPI in event loop)
- [x] **Verification**: Call `api_handler()` directly with a test request, verify image generation

### Step 3: Generate Endpoint + WebSocket
- [x] Implement `POST /generate` with generation lock
- [x] Implement `/ws/progress` WebSocket endpoint
- [x] Bridge pipeline progress → WebSocket messages
- [x] **Verification (end-to-end)**:
  - `curl -X POST localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "a photo of a cat"}'` → returns image
  - WebSocket client (wscat or Python script) connects to `/ws/progress` and receives progress messages during generation
  - Generated image matches same-seed output from Gradio UI

## Success Criteria
1. `POST /generate` with a prompt returns at least one generated image
2. WebSocket at `/ws/progress` streams progress updates during generation
3. Same seed produces identical output as current Gradio UI
4. 409 returned if a generation is already running
5. Random seed assignment works correctly when `seed = -1`
6. Performance presets (Speed, Quality, etc.) apply correctly
7. LoRA specifications in request are applied to generation
