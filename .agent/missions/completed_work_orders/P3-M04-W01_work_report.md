# Work Report: P3-M04-W01 — Core Sampling Logic

## Summary
Successfully extracted the core sampling engine components from `ComfyUI_reference` into `Fooocus_Nex/backend/sampling.py`. The extraction focused on a clean SDXL-centric implementation, stripping away ComfyUI's complex hook and generic model management systems while preserving the essential sampling algorithms.

## Changes

### [NEW] [sampling.py](file:///d:/AI/Fooocus_revision/Fooocus_Nex/backend/sampling.py)
A new module containing the core sampling logic:
- `CFGGuider`: Manages the sampling flow and CFG application.
- `sampling_function`: Per-step denoiser function.
- `cfg_function`: Core CFG arithmetic.
- `calc_cond_batch`: Orchestrates batching of multi-part conditioning.
- `process_conds`: Main entry point for preparing conditioning data before sampling.
- `resolve_areas_and_cond_masks_multidim`: Handles region-based/masked conditioning.
- `calculate_start_end_timesteps`: Manages conditioning schedule (start/end percent).
- `encode_model_conds`: Wraps SDXL-specific conditioning encoding (extra_conds).

## Verification Results

### Automated Tests
- **Import Test**: Passed. The module can be imported without errors.
- **Instantiation Test**: Passed. `CFGGuider` can be instantiated with a mock `ModelPatcher`.
- **Dependency Check**: Successfully avoided all imports from `ldm_patched`.

## Dependency Inventory

| Dependency | Purpose | Source |
|------------|---------|--------|
| `torch` | Tensor operations | standard |
| `math`, `collections`, `uuid` | Utilities | standard |
| `ModelPatcher` | Model wrapper (runtime) | Passed as arg |
| `BaseModel` | Model base (runtime) | Passed as arg |

## Notes
- Integrated `process_conds` earlier than planned in the work order to ensure `CFGGuider` has a complete functional structure for preparation.
- Simplified batching logic by removing complex memory estimation (`model.memory_required` check).
    - **Risk Assessment:** Low for standard SDXL generation (batch size ~2 for cond/uncond).
    - **Future Work:** May need reintroduction if OOMs occur during heavy multi-ControlNet/ImagePrompt usage.
- Stubbed/Simplified hook-related logic to keep imports clean.
