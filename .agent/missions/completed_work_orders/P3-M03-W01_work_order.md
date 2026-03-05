# Work Order: P3-M03-W01

**Title:** Build `backend/resources.py` (Device & Memory Manager)
**Mission:** P3-M03
**Status:** Ready
**Priority:** Critical (Blocker for W02)

## Objective
Extract the "Smart Memory Management" logic from `ComfyUI_reference` into a clean, standalone module `backend/resources.py`.

## Source Material
-   **Source of Truth:** `ComfyUI_reference/comfy/model_management.py`
-   **Reference:** `.agent/reference/ldm_patched_analysis.md` (Read "Category B: The Bad Updates")

## Tasks

### 1. Create `backend/resources.py`
-   [ ] **Copy** contents of `ComfyUI_reference/comfy/model_management.py`.
-   [ ] **Clean**:
    -   Remove `args` imports (we will pass config explicitly).
    -   Remove `LoadedModel.legacy_mode` logic (Category B: Hybrid Wrapper).
    -   Remove `directml` specific branches if they are redundant (check `.agent/reference/ldm_patched_analysis.md`).
    -   Remove `sys.argv` checks.

### 2. Implement "High VRAM" Configuration
-   [ ] **Add** a configuration class or function argument (e.g., `load_models_gpu(..., force_high_vram=False)`).
-   [ ] **Logic:** If `force_high_vram` is True, bypass the "Low VRAM" checks and force-load functionality. This replaces the hardcoded `google.colab` hack.

### 3. Verify
-   [ ] **Create** `tests/test_backend_resources.py`.
-   [ ] **Test:**
    -   Import `backend.resources`.
    -   Call `get_torch_device()`.
    -   Mock a generic Model object (with a `model_size()` method).
    -   Call `load_models_gpu([mock_model])`.
    -   Assert that the mock model is "loaded" (state updated).

## Deliverables
1.  `Fooocus_Nex/backend/resources.py`
2.  `tests/test_backend_resources.py`
