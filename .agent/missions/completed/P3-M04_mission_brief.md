# Mission Brief: P3-M04 — Sampling Engine
**ID:** P3-M04
**Phase:** 3
**Date Issued:** 2026-02-15
**Status:** Draft
**Depends On:** P3-M03 (Resources & Conditioning)
**Work List:** `.agent/missions/active/P3-M04_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/reference/P3-M01_reference_trace.md` (Stage 3: Sampling — **critical**)
- `.agent/reference/ldm_patched_analysis.md`

## Objective

Build `backend/sampling.py` — a clean SDXL sampling module extracted from `ComfyUI_reference`. This module wraps the **CFGGuider**, **k-diffusion sampler loop**, and **scheduler functions** into a single, callable interface that accepts conditioned inputs (from `backend/conditioning.py`) and produces denoised latents ready for VAE decode.

This is the **highest-complexity extraction** in Phase 3. The reference trace identifies the sampling stage as the deepest call chain with the most entanglement.

## The Strategy: "Clean Extraction" & "Standalone Verification"
> [!IMPORTANT]
> **Source of Truth:** All logic MUST be extracted from `ComfyUI_reference`.
> **Forbidden:** Do NOT import from `ldm_patched` under any circumstances for new logic.
> **Verification:** Since the full engine is not ready, every new module must be verified with a standalone script in `tests/`.

## Scope

### In Scope

1. **CFG Guidance Logic** — Extract `CFGGuider` from `ComfyUI_reference/comfy/samplers.py`. This manages:
   - Setting positive/negative conditions
   - CFG scale application (`cfg_function`: `uncond + (cond - uncond) * scale`)
   - The `sampling_function` per-step denoiser

2. **K-Diffusion Sampler Integration** — Extract the **full `KSAMPLER_NAMES` list** from `ComfyUI_reference` (36 samplers). Key additions over current Fooocus:
   - **CFG++ variants** (critical for Illustrious SDXL-variants): `euler_cfg_pp`, `euler_ancestral_cfg_pp`, `dpmpp_2s_ancestral_cfg_pp`, `dpmpp_2m_cfg_pp`, `res_multistep_cfg_pp`, `res_multistep_ancestral_cfg_pp`, `gradient_estimation_cfg_pp`
   - **New community samplers**: `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `res_multistep_ancestral`, `gradient_estimation`, `er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`
   - `KSAMPLER` class / sampler dispatch
   - Model wrapper that translates between CFGGuider and k-diffusion's expected interface

3. **Scheduler Functions** — Extract the **full `SCHEDULER_HANDLERS` dict** from `ComfyUI_reference/comfy/samplers.py`:
   - **Must include:** `normal`, `karras`, `sgm_uniform`, `exponential`, `simple`, `ddim_uniform`, **`beta`** (critical for Illustrious SDXL-variants), `linear_quadratic`, `kl_optimal`
   - **Exclude:** `turbo`, `align_your_steps`, `tcd`, `edm_playground_v2.5` — these are Fooocus-specific monkey-patches from `contrib/` that are no longer relevant. LoRA-based performance presets (Turbo, Hyper-SD, LCM) do not need dedicated schedulers; they work with standard schedulers.

4. **`process_conds`** — The SDXL-specific condition processing that runs inside the sampling loop:
   - `resolve_areas_and_cond_masks()`
   - `calculate_start_end_timesteps()`
   - `encode_model_conds()` → calls `SDXL.extra_conds()` to produce `c_crossattn` + `y`

5. **Clean API** — A top-level function like:
   ```python
   def sample_sdxl(
       model_patcher: ModelPatcher,
       positive: ConditioningData,
       negative: ConditioningData,
       latent: torch.Tensor,
       steps: int,
       cfg_scale: float,
       sampler_name: str,
       scheduler_name: str,
       seed: int,
       denoise: float = 1.0,
   ) -> torch.Tensor:
   ```

### Out of Scope
- **VAE decode** — Separate module (M05 or later)
- **LoRA/patching** — `ModelPatcher` is used as-is; the patching internals are a future mission
- **ControlNet** — Will be added as an extension later
- **Hooks/wrapper system** (`patcher_extension.py`) — Only import if absolutely required by `CFGGuider`; do not extract the full hooks API
- **Inpainting mask logic** — Excluded from this mission; sampling API should accept but not manage masks
- Modifying any existing `modules/` or `ldm_patched/` code

## Reference Files
- `ComfyUI_reference/comfy/samplers.py` — CFGGuider, KSAMPLER, sampling_function, cfg_function
- `ComfyUI_reference/comfy/sample.py` — Top-level sample() entry point
- `ComfyUI_reference/comfy/sampler_helpers.py` — Condition processing helpers
- `ComfyUI_reference/comfy/k_diffusion/sampling.py` — Sampler implementations
- `ComfyUI_reference/comfy/model_base.py` — `SDXL.extra_conds()` for condition encoding
- `.agent/reference/P3-M01_reference_trace.md` (Stage 3)

## Constraints
- Follow all design principles from DR-001
- New code in `backend/sampling.py` only (+ constants in `backend/defs/sdxl.py` if needed)
- `sampling.py` may import from `backend/resources.py` and `backend/conditioning.py`
- `sampling.py` will need `ModelPatcher` — it may import this from `ComfyUI_reference` (or from the existing `ldm_patched` if `ModelPatcher` is identical). Document this dependency explicitly.
- Include the Dependency Inventory table per DR-001

## Deliverables
- [ ] `backend/sampling.py` — the main module
- [ ] Updated `backend/defs/sdxl.py` — any new constants (sampler names, default sigmas, etc.)
- [ ] `tests/test_backend_sampling.py` — standalone verification script
- [ ] Dependency Inventory in work report

## Success Criteria
1. `sample_sdxl()` can be called with a loaded `ModelPatcher`, conditioning data, and noise — and produces a denoised latent tensor of the expected shape
2. Support for all 36 ComfyUI_reference samplers (including CFG++ variants) and all 9 core schedulers (including `beta`)
3. `sampling.py` has zero imports from `ldm_patched.modules.model_management` or `ldm_patched.modules.conds`
4. CFG guidance produces different outputs for different CFG scales (basic sanity check)
5. Standalone test passes without requiring the full Fooocus pipeline

## Work Orders
Registered in mission work-list (`P3-M04_work_list.md`):
- `P3-M04-W01` — Extract CFGGuider + sampling_function + cfg_function
- `P3-M04-W02` — Extract schedulers + KSAMPLER + integrate into `sample_sdxl()` API
- `P3-M04-W03` — Verification & dependency inventory

## Notes
- **Complexity Warning:** The reference trace (Stage 3) shows that `CFGGuider.sample()` → `outer_sample()` → `inner_sample()` → `KSAMPLER.sample()` is a 5-level deep call chain. The extraction should flatten this where possible without changing the algorithm.
- **ModelPatcher Dependency:** This is the first module that genuinely *needs* `ModelPatcher` at runtime (not just at load time). The dependency must be documented clearly and may need a thin wrapper.
- **`process_conds` Entanglement:** The `resolve_areas_and_cond_masks()` logic handles region-based conditioning (areas, masks). For SDXL simple generation, most of this is a no-op. The extraction should handle the common case cleanly and leave the area/mask logic as an optional code path.
- **CFG++ Variants (Illustrious):** The `*_cfg_pp` samplers use a different CFG formulation that is important for Illustrious SDXL-variant models. These must be included as first-class samplers, not afterthoughts.
- **LoRA-Based Presets (Turbo/Hyper/LCM):** These are LoRA-driven performance modes, not fundamentally different scheduling algorithms. They work fine with standard schedulers (`normal`, `karras`, etc.) and do not need dedicated scheduler entries. Fooocus's `lcm`/`turbo`/`tcd` scheduler monkey-patches are legacy baggage we do not carry forward.
