# Work Order: P3-M04-W02 — Samplers, Schedulers & API

**Mission:** P3-M04
**Work Order ID:** P3-M04-W02
**Status:** Completed
**Depends On:** P3-M04-W01

## Objective
Implement the full suite of samplers and schedulers from `ComfyUI_reference` and expose the top-level `sample_sdxl` API.

## Requirements

1.  **Extract Samplers (`KSAMPLER`):**
    -   Source: `ComfyUI_reference/comfy/k_diffusion/sampling.py` and `ComfyUI_reference/comfy/samplers.py`.
    -   Implement the `KSAMPLER` function/class that dispatches to k-diffusion samplers.
    -   Include **ALL 36 samplers**, including `*_cfg_pp` variants.
    -   Include `sampler_names()` function.

2.  **Extract Schedulers:**
    -   Source: `ComfyUI_reference/comfy/samplers.py`.
    -   Implement `calculate_sigmas_scheduler` or equivalent dispatch logic.
    -   Include **ALL 9 schedulers**: `normal`, `karras`, `sgm_uniform`, `simple`, `ddim_uniform`, `beta`, `linear_quadratic`, `kl_optimal`, `exponential`.
    -   Include `scheduler_names()` function.

3.  **Implement `sample_sdxl`:**
    -   Create the high-level entry point in `Fooocus_Nex/backend/sampling.py`:
        ```python
        def sample_sdxl(
            model_patcher: Any, # Typed as ModelPatcher
            positive: Any,      # Typed as Conditioning
            negative: Any,
            latent_image: torch.Tensor,
            seed: int,
            steps: int,
            cfg: float,
            sampler_name: str,
            scheduler: str,
            denoise: float = 1.0,
            disable_noise: bool = False,
            start_step: int = None,
            last_step: int = None,
            force_full_denoise: bool = False,
        ) -> torch.Tensor:
            ...
        ```
    -   This function should instantiate `CFGGuider`, set conditions, and call the sampler.
    -   It must handle the "noise" generation (using `prepare_noise` if needed or standard torch.randn).

4.  **Conditioning Resolution (`process_conds`):**
    -   Implementing the logic to resolve areas and encode model conditions (SDXL `y` and `crossattn`).
    -   You may need to port `comfy.sampler_helpers.get_additional_models` equivalents if they are used, OR just strictly follow the SDXL path.
    -   Crucial: SDXL expects `adm` (resolution) embeddings. Ensure these are passed from `positive`/`negative` conditioning to the model.

## Definitions
-   Make sure `backend/defs/sdxl.py` is updated with constants if needed (e.g. default sigmas if not dynamically calculated).

## Verification
-   Update `tests/test_backend_sampling.py` to test specific samplers/schedulers.
-   Ensure `beta` scheduler works (critical for Illustrious).
