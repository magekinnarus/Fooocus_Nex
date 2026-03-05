# Mission Work List: P3-M03

**Mission:** Backend Foundation: Resources & Conditioning
**Status:** Active
**Analyst:** Code Manager (Role 1)

## Strategic Context
This mission builds the foundation of the "Clean Architecture" backend. We are extracting core logic from `ComfyUI_reference` into `Fooocus_Nex/backend/`, explicitly avoiding any dependency on `ldm_patched`.

## Work Orders

| ID | Title | Status | Dependencies |
|---|---|---|---|
| **P3-M03-W01** | **Build `backend/resources.py`** | **Complete** | None |
| **P3-M03-W02** | **Build `backend/conditioning.py`** | **Ready** | P3-M03-W01 |

## Logic & Sequencing
1.  **W01 (Resources)** must be completed first. Models cannot be loaded without a device/memory manager. This module will also implement the "Colab Fix" (Force High VRAM) natively.
2.  **W02 (Conditioning)** follows. It requires `resources.py` to handle CLIP models.

## Verification Strategy
-   Each Work Order MUST produce a standalone test script (`tests/test_backend_*.py`).
-   Verification is successful if the test script runs without errors and independently of the legacy engine.
