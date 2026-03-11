# Fooocus-Nex Masking & Update Mechanism

### 1. Component Definition (Python)
In `webui.py`, masking components are defined with unique `elem_id`s:
- `inpaint_canvas`: The base image upload/editor.
- `inpaint_context_mask_canvas`: The Step 1 mask upload (if using direct upload).
- `inpaint_bb_canvas`: The Step 2 bounding box image.

### 2. Canvas Overlay (Javascript: `inpaint_mask.js`)
The custom drawing logic operates entirely client-side:
- **Polling**: `window.setInterval(refreshAll, 500)` runs every 0.5s.
- **Surface Sync**: `refreshMode` finds the root `elem_id`, then locates the nested `<img>` element.
- **Sizing**: It calculates the displayed size of the image (`getContainSize`) using `img.naturalWidth` and `img.naturalHeight`. 
- **Overlay**: It injects a `<canvas>` as a sibling to the `<img>`, perfectly sized to match the content.
- **Persistence**: A `MutationObserver` watches for image uploads (`src` changes) to trigger immediate re-syncing without waiting for the interval.

### 3. Data Flow (Gradio <-> JS)
- **Hidden Fields**: `webui.py` includes hidden `gr.Textbox` components (e.g., `inpaint_context_mask_data`).
- **Exporting**: When painting ends (`pointerup`), the JS converts the canvas to a Base64 PNG (`toDataURL`).
- **Injection**: The JS writes this Base64 string directly into the hidden textbox's `.value` and dispatches `input` and `change` events.
- **Python Trigger**: The `.change()` event on the hidden field triggers the backend `mask_processing.py` logic.

### 4. The "5-Slot Limit" Breakdown (Hypothesis)
Chrome and most modern browsers limit **concurrent connections to a single origin to 6**.
- **The Culprit**: `gr.Image(type='filepath')` works by fetching files from the Gradio server via `/file=...` URLs.
- **The Cascade**:
    1. If you have 3 Inpaint images + 2 Gallery previews + 2 Staging Slots = 7 requests.
    2. The 7th request (often a critical mask image or the base image) is **queued** by the browser.
    3. `inpaint_mask.js` polls every 500ms. If the image is stuck in the connection queue, `img.naturalWidth` remains `0`.
    4. The script sees `size = null` and sets `canvas.style.display = 'none'`.
- **The Result**: Masking "breaks" because the browser is too busy loading staging thumbnails to fetch the metadata/data for the masking target.
