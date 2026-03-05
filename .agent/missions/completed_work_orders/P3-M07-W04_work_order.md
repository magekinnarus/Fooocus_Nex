# Work Order: P3-M07-W04 — Performance Profiling & Optimization

**Mission:** P3-M07
**Work Order ID:** P3-M07-W04
**Status:** Pending
**Assignee:** Role 2
**Prerequisites:** P3-M07-W02 (Working app.py)

## Objective
Analyze the performance of `app.py` and optimize it to be within 20% of ComfyUI's generation speed on comparable hardware.

## Context
Goal: Achieve <14.4 sec/it (if ComfyUI is 12 sec/it) on local hardware. Current GGUF path via `llama.cpp` was around 20 sec/it in previous tests. We need to identify bottlenecks in our Python-side orchestration vs the underlying C++ execution.

## Tasks

### 1. Baselining
- [ ] **Measure ComfyUI**: Run a standard generation (e.g., SDXL 1024x1024, 20 steps, dpmpp_2m_sde) on the target hardware. Record total time and s/it.
- [ ] **Measure app.py**: Run the exact same configuration using `app.py`. Record total time and s/it.
- [ ] **Gap Analysis**: Calculate the percentage difference.

### 2. Profiling
- [ ] Use `cProfile` or `py-spy` on `app.py` to see where time is spent.
- [ ] Check for overhead in:
    - VAE decoding (is it running on CPU? float32?).
    - Prompt encoding (re-running unnecessarily?).
    - Sampler step loop (Python overhead vs torch/gguf operation).

### 3. Optimization
- [ ] **Address findings**:
    - If VAE is slow: Ensure it's on GPU and using fp16/bf16 if supported.
    - If Sampler is slow: Check `k_diffusion` integration.
    - If Model Loading is slow: Check for redundant disk reads or memory swaps.
- [ ] **Retest**: Run `app.py` again and verifying the improvement.

## Deliverables
- `profiling_report.md`: Summary of baseline vs `app.py` performance, bottlenecks found, and fixes applied.
- Optimized `app.py` (if code changes were needed).
