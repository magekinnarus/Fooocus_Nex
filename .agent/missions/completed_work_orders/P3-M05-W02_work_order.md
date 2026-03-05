# Work Order: P3-M05-W02 — Verification

**ID:** P3-M05-W02
**Mission:** P3-M05 — VAE Decode
**Status:** Pending
**Depends On:** P3-M05-W01

## Reference Material
- **Mission Brief:** `.agent/missions/active/P3-M05_mission_brief.md`
- **Implementation:** `backend/decode.py` (to be created in W01)
Verify the correctness of the extracted VAE decode module using a standalone test script.

## Success Criteria
1. `tests/test_backend_decode.py` exists and passes.
2. Test verifies:
   - Output shape is correct `[B, H*8, W*8, 3]`.
   - Output range is `[0, 1]`.
   - Tiled decode produces similar output to standard decode (within small tolerance).
   - No errors on CPU or GPU (if available).
3. Test runs without needing the full app (only `backend` modules).

## Tasks
1. **Create Test Script**:
   - `tests/test_backend_decode.py`
   - Setup: Load a VAE (can use a mock VAE or load a real one via `backend/loader.py` if available and efficient, or just mock the `first_stage_model`).
   - *Mocking is preferred for speed and isolation*, but loading a real VAE ensures integration is correct. A mix is best: verification script can take an optional path to a VAE.
   2. **Test Cases**:
      - `test_decode_shapes`: Pass a random latent `[1, 4, 64, 64]` and check output is `[1, 512, 512, 3]`.
      - `test_tiled_vs_standard`: Run both on same input, assert `torch.allclose` or small MSE.
      - `test_device_movement`: Ensure tensors move to/from GPU correctly if available.

## Technical Notes
- Use `unittest` or `pytest`.
- Use `torch.randn` for input latents.
- If mocking VAE, ensure the `decode` method signature matches expected `first_stage_model` interface.
