# Work Order: P3-M03-W02

**Title:** Build `backend/conditioning.py` (CLIP & ADM Logic)
**Mission:** P3-M03
**Status:** Pending (Blocked by W01)
**Priority:** High

## Objective
Extract SDXL text encoding (CLIP) and Adaptive Domain Mixing (ADM) conditioning logic into `backend/conditioning.py`.

## Source Material
-   **Source of Truth:** `ComfyUI_reference/comfy/sd.py` (for CLIP loading) and `ComfyUI_reference/comfy/samplers.py` (for conditioning structure).
-   **Dependency:** `backend/resources.py` (created in W01).

## Tasks

### 1. Create `backend/conditioning.py`
-   [ ] **Extract** `CLIPTextEncode` logic.
-   [ ] **Extract** `timestep_embedding` and ADM formatting logic.
-   [ ] **Refactor**: Ensure all functions use typed arguments (e.g., `def encode(clip: Any, text: str) -> Tensor`).
-   [ ] **Import**: Use `backend.resources` for device management, NOT `ldm_patched`.

### 2. Verify
-   [ ] **Create** `tests/test_backend_conditioning.py`.
-   [ ] **Test:**
    -   Mock a CLIP object.
    -   Call `encode()` and verify tensor shape.
    -   Verify ADM embeddings are generated correctly for SDXL resolutions.

## Deliverables
1.  `Fooocus_Nex/backend/conditioning.py`
2.  `tests/test_backend_conditioning.py`
