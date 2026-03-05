# Work Order: P3-M10-W04 — UI Integration Bugs and Memory Optimization
**ID:** P3-M10-W04
**Mission:** P3-M10
**Status:** Completed
**Depends On:** P3-M10-W03

## Mandatory Reading
- `.agent/temp/refactoring_assessment.md` (Background on technical debt in `async_worker.py` and `webui.py`)
- `P3-M10_integration_bugs.md` (Bug log for P3-M10-W03)

## Objective
Address three critical post-integration bugs found during P3-M10 testing: consecutive generation hangs, memory/RAM leaks, and non-functional Skip/Stop UI buttons. This work order serves as an immediate fix, guided by the strategies in the refactoring assessment, to ensure stable operation before moving to Mission 11.

## Scope

### 1. Fix Consecutive Generation Hang (`async_worker.py`)
**Issue:** The `worker()` thread crashes at the end of the first generation because `task.results` is accessed but `AsyncTask` is missing the `results` attribute and is not populated with images. Because the thread dies, subsequent generations simply hang.
**Tasks:**
- Initialize `self.results = []` in `AsyncTask.__init__`.
- In `handler()`, append `img_paths` to `async_task.results` (or `task.results`) after `process_task` successfully completes.

### 2. Fix RAM / Memory Spikes (`async_worker.py`)
**Issue:** RAM remains at ~63% after generation because `backend.resources.soft_empty_cache()` was not invoked at the end of the generation loop, keeping models and tensors fully resident.
**Tasks:**
- Import `backend.resources` in `async_worker.py`.
- Call `backend.resources.soft_empty_cache()` at the end of the `worker()` loop or within `stop_processing()`.

### 3. Fix Skip/Stop UI Buttons (`webui.py` & `async_worker.py`)
**Issue:** `webui.py` still relies on `ldm_patched.modules.model_management` to set the interrupt flags, while the native backend checks `backend.resources.InterruptProcessingException`.
**Tasks:**
- In `webui.py` (e.g., lines 35, 170, 177), replace `import ldm_patched.modules.model_management` with `import backend.resources`.
- Update the mutex tracking in `generate_clicked` (`backend.resources.interrupt_processing_mutex`) and `backend.resources.interrupt_processing = False`.
- Update the Skip/Stop button event handlers to call `backend.resources.interrupt_current_processing()`.
- Ensure `async_worker.py` catches `backend.resources.InterruptProcessingException` instead of the `ldm_patched` version.

## Verification
- **Consecutive Generations:** Queue two images back-to-back from the UI. Both should generate successfully.
- **Memory Optimization:** Observe process RAM usage. It should return close to idle levels (~43-45%) after generation completes.
- **Skip/Stop Verification:** Pressing "Stop" or "Skip" in the UI should immediately halt the generation and return to the idle state.

## Success Criteria
- The backend `worker` thread survives unlimited consecutive generations via the UI.
- RAM usage decreases after generation.
- Skip and Stop buttons correctly interrupt the new backend pipeline.
- `ldm_patched.modules.model_management.InterruptProcessingException` is fully removed from `async_worker.py` and `webui.py`.
