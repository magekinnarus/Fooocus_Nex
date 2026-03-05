# Work Order: Create Validation Script

**ID:** P3-M06-W01
**Mission:** P3-M06
**Status:** Ready

## Objective
Create a standalone Python script `scripts/validate_p3.py` to verify the end-to-end functionality of the extracted backend modules (loader, conditioning, sampling, decode) without using `ldm_patched`.

## Requirements
1. **Script Location:** `scripts/validate_p3.py`
2. **Dependencies:**
   - MUST NOT import `ldm_patched` directly.
   - MUST use `backend.loader`, `backend.conditioning`, `backend.sampling`, `backend.decode`, `backend.resources`.
3. **Functionality:**
   - Load an SDXL model (checkpoint or GGUF).
   - Encode a simple prompt (e.g., "A beautiful landscape").
   - Perform sampling (e.g., 20 steps, Euler Ancestral).
   - Decode the latent to an image.
   - Save the image as `completed_image.png`.
4. **Resources:**
   - Ensure memory is managed correctly (model loading/offloading) using `backend.resources`.
   - Must run on 4GB VRAM target (sequential offloading if needed).

## Implementation Steps
1. Create `scripts/` directory if it doesn't exist.
2. Create `scripts/validate_p3.py`.
3. Implement model loading using `backend.loader.load_file_from_path` (or similar).
4. Implement text encoding using `backend.conditioning.encode`.
5. Implement sampling using `backend.sampling.sample`.
6. Implement decoding using `backend.decode.decode_variant`.
7. Add main execution block to run the pipeline and save the image.

## Acceptance Criteria
- [ ] Script runs successfully without errors.
- [ ] `completed_image.png` is created and is a valid image.
- [ ] No `ldm_patched` imports in the script (except potentially for typing if absolutely necessary, but preferably not).
