from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional benchmark dependency
    psutil = None

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.flux_fill_fp8_benchmark_contract import (
    FLUX_FILL_BENCHMARK_MODES,
    FluxFillBenchmarkArtifactBundle,
    FluxFillBenchmarkRunInput,
    FluxFillBenchmarkRunMetrics,
    normalize_flux_fill_benchmark_mode,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        from backend.staging_manager import ExecutionClass as _ExecutionClass

        if isinstance(value, _ExecutionClass):
            return value.value
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _load_artifact(path: Path) -> Any:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload
    return payload


def _load_tensor_artifact(path: Path) -> torch.Tensor:
    import torch

    payload = _load_artifact(path)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in ("tensor", "latent", "samples", "mask", "denoise_mask"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Artifact at {path} does not contain a tensor payload.")


def _load_conditioning_artifact(path: Path):
    from backend.flux.flux_fill_pipeline import load_flux_empty_conditioning_cache

    return load_flux_empty_conditioning_cache(path, map_location="cpu")


def _mb_to_bytes(value: float | int | None) -> int | None:
    if value is None:
        return None
    return int(float(value) * 1024 * 1024)


def _save_samples_artifact(path: Path, *, samples, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "samples": samples.detach().cpu(),
            "metadata": dict(metadata),
        },
        path,
    )


def _decode_preview_image(
    *,
    samples,
    ae_path: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    from PIL import Image
    from backend.flux.flux_fill_pipeline import decode_flux_fill_latent

    decoded = decode_flux_fill_latent(
        samples.detach().cpu(),
        ae_path,
        stitch=False,
        load_device=device,
        offload_device="cpu",
        cleanup_vae=True,
    )
    image = decoded.stitched_image if decoded.stitched_image is not None else decoded.bb_image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return {
        "status": "ok",
        "image_path": str(output_path),
        "decode_device": str(device),
        "shape": list(image.shape),
        "timings": dict(decoded.timings),
        "runtime": dict(decoded.metadata),
    }


def _load_unet(
    mode: str,
    bundle: FluxFillBenchmarkArtifactBundle,
    execution_class: Any | None,
    *,
    prefetch_depth: int | None,
    max_prefetch_bytes: int | None,
    vram_guard_bytes: int | None,
    vram_guard_margin_bytes: int | None,
    prefetch_scan_ahead: int | None,
    bandwidth_limit_mb_s: float | None,
):
    from backend.flux.flux_fill_pipeline import (
        load_flux_fill_native_unet,
        load_flux_fill_native_unet_streaming,
    )
    unet_path = bundle.unet_path_for_mode(mode)

    if mode == "fp8_resident":
        return load_flux_fill_native_unet(
            unet_path,
            load_device="cuda" if _cuda_available() else "cpu",
            offload_device="cpu",
            execution_class=execution_class,
        )
    if mode == "fp8_stream_async":
        start = time.perf_counter()
        patcher = load_flux_fill_native_unet_streaming(
            unet_path,
            load_device="cpu",
            offload_device="cpu",
            execution_class=execution_class,
            prefetch_depth=prefetch_depth,
            max_prefetch_bytes=max_prefetch_bytes,
            vram_guard_bytes=vram_guard_bytes,
            vram_guard_margin_bytes=vram_guard_margin_bytes,
            prefetch_scan_ahead=prefetch_scan_ahead,
            bandwidth_limit_mb_s=bandwidth_limit_mb_s,
        )
        async_load_wall = time.perf_counter() - start
        return patcher, async_load_wall
    raise ValueError(f"Unsupported benchmark mode: {mode!r}")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _peak_pinned_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        stats = torch.cuda.memory_stats()
        keys = [
            "pinned_memory_allocated.all.current",
            "pinned_memory_allocated",
            "pinned_memory_reserved.all.current",
            "pinned_memory_reserved",
        ]
        total = 0.0
        for key in keys:
            value = stats.get(key)
            if value is not None:
                total = max(total, float(value) / (1024 * 1024))
        return total
    except Exception:
        return 0.0


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if psutil is not None else None
        self.peak_rss_bytes = 0
        self.peak_pinned_mb = 0.0

    def __enter__(self) -> "MemorySampler":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set() and self._process is not None:
            try:
                rss = int(self._process.memory_info().rss)
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            except Exception:
                pass
            try:
                self.peak_pinned_mb = max(self.peak_pinned_mb, _peak_pinned_mb())
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        import torch

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _sum_trace_wall(trace_stats: dict[str, Any] | None) -> float:
    if not isinstance(trace_stats, dict):
        return 0.0
    return float(sum(float(item.get("wall_seconds", 0.0)) for item in trace_stats.values()))


def _sum_trace_cpu(trace_stats: dict[str, Any] | None) -> float:
    if not isinstance(trace_stats, dict):
        return 0.0
    return float(sum(float(item.get("cpu_process_seconds", 0.0)) for item in trace_stats.values()))


def _scheduler_hit_rate(stats: dict[str, Any] | None) -> float:
    if not isinstance(stats, dict):
        return 0.0
    enqueued = int(stats.get("prefetch_enqueued", 0))
    hits = int(stats.get("prefetch_hits", 0))
    if enqueued <= 0:
        return 0.0
    return float(hits) / float(enqueued)


def _mb_per_s(byte_count: float, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return 0.0
    return (float(byte_count) / (1024.0 * 1024.0)) / (float(elapsed_ms) / 1000.0)


def _build_overlap_note(mode: str, scheduler_stats: dict[str, Any] | None, non_blocking_supported: bool) -> str:
    if mode != "fp8_stream_async":
        return "resident_baseline" if mode == "fp8_resident" else ""
    if not non_blocking_supported:
        return "scheduler_present_non_blocking_unsupported"
    if not isinstance(scheduler_stats, dict):
        return "scheduler_missing"
    if int(scheduler_stats.get("prefetch_hits", 0)) > 0:
        return "layer_prefetch_hits_observed"
    if int(scheduler_stats.get("prefetch_enqueued", 0)) > 0:
        return "layer_prefetch_enqueued_no_hits"
    return "scheduler_attached_no_prefetch_activity"


def _build_gpu_busy_note(
    *,
    denoise_wall: float,
    denoise_cpu_proc: float,
    scheduler_stats: dict[str, Any] | None,
) -> str:
    wall_cpu_gap = max(0.0, denoise_wall - denoise_cpu_proc)
    if isinstance(scheduler_stats, dict) and int(scheduler_stats.get("prefetch_hits", 0)) > 0 and wall_cpu_gap > 0.0:
        return "cpu_wall_gap_plus_prefetch_hits"
    if wall_cpu_gap > 0.0:
        return "wall_gt_cpu_expected_for_gpu_overlap"
    return "cpu_bound_or_overlap_unproven"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tools-only Flux Fill fp8 feasibility benchmark.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M13-W08b"))
    parser.add_argument("--bundle", required=True, help="Path to a JSON file describing the precomputed artifact bundle.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30, help="Denoise step count for the benchmark run.")
    parser.add_argument("--modes", nargs="+", default=list(FLUX_FILL_BENCHMARK_MODES), choices=list(FLUX_FILL_BENCHMARK_MODES))
    parser.add_argument("--prefetch-depth", type=int, default=None, help="Optional async scheduler prefetch depth override.")
    parser.add_argument("--prefetch-max-mb", type=float, default=None, help="Optional async scheduler max prefetch bytes override in MB.")
    parser.add_argument("--vram-guard-mb", type=float, default=None, help="Optional async scheduler VRAM guard ceiling in MB.")
    parser.add_argument("--vram-guard-margin-mb", type=float, default=None, help="Optional async scheduler free-VRAM margin in MB.")
    parser.add_argument("--prefetch-scan-ahead", type=int, default=None, help="Optional async scheduler scan-ahead override.")
    parser.add_argument("--bandwidth-limit-mb-s", type=float, default=None, help="Optional async scheduler bandwidth limit in MB/s for streamed host->GPU transfer pacing.")
    parser.add_argument("--save-latents", action="store_true", help="Persist denoised latent artifacts for each run.")
    parser.add_argument("--decode-preview", action="store_true", help="Decode each run's denoised latent after the timed benchmark window.")
    parser.add_argument("--decode-with-unet-resident", action="store_true", help="Keep the UNet resident through decode to test UNet+VAE coexistence.")
    parser.add_argument("--decode-device", default=None, help="Optional device override for preview decode, e.g. cuda or cpu.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def _load_bundle(bundle_path: Path) -> FluxFillBenchmarkArtifactBundle:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    return FluxFillBenchmarkArtifactBundle(
        fp8_unet_path=Path(payload["fp8_unet_path"]),
        ae_path=Path(payload["ae_path"]),
        conditioning_cache_path=Path(payload["conditioning_cache_path"]),
        source_latent_path=Path(payload["source_latent_path"]),
        concat_latent_path=Path(payload["concat_latent_path"]),
        denoise_mask_path=Path(payload["denoise_mask_path"]),
        provenance=dict(payload.get("provenance", {})),
        q4_gguf_unet_path=Path(payload["q4_gguf_unet_path"]) if payload.get("q4_gguf_unet_path") else None,
    )


def _execution_class_for_mode(mode: str):
    from backend.staging_manager import ExecutionClass

    normalized = normalize_flux_fill_benchmark_mode(mode)
    if normalized == "fp8_stream_async":
        return ExecutionClass.FLUX_STREAMING_T3
    if normalized == "fp8_resident":
        return ExecutionClass.FLUX_RESIDENT_T5 if _cuda_available() else ExecutionClass.FLUX_STREAMING_T3
    raise ValueError(f"Unsupported benchmark mode: {mode!r}")


def _run_case(
    *,
    mode: str,
    bundle: FluxFillBenchmarkArtifactBundle,
    run_label: str,
    steps: int,
    output_dir: Path,
    prefetch_depth: int | None,
    max_prefetch_bytes: int | None,
    vram_guard_bytes: int | None,
    vram_guard_margin_bytes: int | None,
    prefetch_scan_ahead: int | None,
    bandwidth_limit_mb_s: float | None,
    save_latents: bool,
    decode_preview: bool,
    decode_with_unet_resident: bool,
    decode_device: str | None,
) -> dict[str, Any]:
    import torch
    from backend.flux.flux_fill_pipeline import (
        FluxFillConfig,
        FluxFillPrecomputedDenoiseInput,
        _cleanup_model_patcher,
        denoise_flux_fill_precomputed_latent,
        load_flux_fill_native_unet,
        load_flux_fill_native_unet_streaming,
        measure_pinned_module_tensors,
    )
    unet_path = bundle.unet_path_for_mode(mode)
    execution_class = _execution_class_for_mode(mode)

    if mode == "fp8_resident":
        unet_patcher = load_flux_fill_native_unet(
            unet_path,
            load_device="cuda" if torch.cuda.is_available() else "cpu",
            offload_device="cpu",
            execution_class=execution_class,
        )
        async_load_wall = 0.0
    else:
        unet_patcher, async_load_wall = _load_unet(
            mode,
            bundle,
            execution_class,
            prefetch_depth=prefetch_depth,
            max_prefetch_bytes=max_prefetch_bytes,
            vram_guard_bytes=vram_guard_bytes,
            vram_guard_margin_bytes=vram_guard_margin_bytes,
            prefetch_scan_ahead=prefetch_scan_ahead,
            bandwidth_limit_mb_s=bandwidth_limit_mb_s,
        )

    source_latent = _load_tensor_artifact(bundle.source_latent_path)
    concat_latent = _load_tensor_artifact(bundle.concat_latent_path)
    denoise_mask = _load_tensor_artifact(bundle.denoise_mask_path)
    empty_conditioning = _load_conditioning_artifact(bundle.conditioning_cache_path)

    config = FluxFillConfig(
        unet_path=unet_path,
        ae_path=bundle.ae_path,
        conditioning_cache_path=bundle.conditioning_cache_path,
        tier="fp8_tools_only",
        seed=882699830973928,
        steps=int(steps),
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
        guidance=15.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        execution_class=execution_class,
    )
    precomputed_input = FluxFillPrecomputedDenoiseInput(
        source_latent=source_latent,
        concat_latent=concat_latent,
        denoise_mask=denoise_mask,
        empty_conditioning=empty_conditioning,
        seed=config.seed,
        guidance=config.guidance,
        steps=config.steps,
        cfg=config.cfg,
        sampler=config.sampler,
        scheduler=config.scheduler,
        denoise=config.denoise,
    )

    cleanup_unet_after_denoise = not (decode_preview and decode_with_unet_resident)
    with MemorySampler() as memory:
        cuda_start = None
        cuda_end = None
        if torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        start = time.perf_counter()
        result = denoise_flux_fill_precomputed_latent(
            config,
            precomputed_input,
            unet_patcher=unet_patcher,
            load_device=config.device,
            cleanup_unet=cleanup_unet_after_denoise,
            disable_pbar=True,
        )
        total_wall = time.perf_counter() - start
        if cuda_start is not None and cuda_end is not None:
            cuda_end.record()
    try:
        import torch

        peak_vram_mb = float(torch.cuda.max_memory_reserved()) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        peak_vram_allocated_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        cuda_elapsed_ms = float(cuda_start.elapsed_time(cuda_end)) if cuda_start is not None and cuda_end is not None else 0.0
    except Exception:
        peak_vram_mb = 0.0
        peak_vram_allocated_mb = 0.0
        cuda_elapsed_ms = 0.0

    denoise_wall = float(result.timings.get("denoise_wall", 0.0))
    denoise_cpu_proc = float(result.timings.get("denoise_cpu_proc", 0.0))
    flux_options = getattr(unet_patcher, "model_options", {}).get("flux_fill", {})
    detected_config = flux_options.get("detected_config", {}) if isinstance(flux_options, dict) else {}
    sampling_perf = getattr(unet_patcher, "model_options", {}).get("_nex_sampling_perf", {})
    scheduler_stats = result.metadata.get("streaming_scheduler", {})
    host_pinned_mb = max(
        float(flux_options.get("host_pinned_bytes", 0)) / (1024 * 1024),
        float(measure_pinned_module_tensors(getattr(unet_patcher, "model", None))) / (1024 * 1024),
        _peak_pinned_mb(),
        float(memory.peak_pinned_mb),
    )
    wall_cpu_gap_s = max(0.0, denoise_wall - denoise_cpu_proc)
    cpu_wall_ratio = (denoise_cpu_proc / denoise_wall) if denoise_wall > 0.0 else 0.0
    overlap_note = _build_overlap_note(mode, scheduler_stats, bool(flux_options.get("non_blocking_supported", False)))
    gpu_busy_note = _build_gpu_busy_note(
        denoise_wall=denoise_wall,
        denoise_cpu_proc=denoise_cpu_proc,
        scheduler_stats=scheduler_stats,
    )
    sampler_trace = sampling_perf.get("sampler_trace", {})
    cond_batch_trace = sampling_perf.get("cond_batch_trace", {})
    apply_model_trace = sampling_perf.get("apply_model_trace", {})
    latent_artifact_path = None
    decode_preview_result = None
    samples_metadata = {
        "mode": mode,
        "run_label": run_label,
        "steps": int(steps),
        "device": str(config.device),
        "sample_shape": list(result.samples.shape),
    }
    total_transfer_bytes = float(scheduler_stats.get("prefetch_bytes", 0) + scheduler_stats.get("direct_copy_bytes", 0)) if isinstance(scheduler_stats, dict) else 0.0
    total_copy_cuda_ms = float(scheduler_stats.get("prefetch_copy_cuda_ms", 0.0) + scheduler_stats.get("direct_copy_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0
    total_throttle_cuda_ms = float(scheduler_stats.get("bandwidth_throttle_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0
    try:
        if save_latents or decode_preview:
            latent_artifact_path = output_dir / f"{run_label}_samples.pt"
            _save_samples_artifact(
                latent_artifact_path,
                samples=result.samples,
                metadata=samples_metadata,
            )
        if decode_preview:
            preview_output_path = output_dir / f"{run_label}_preview.png"
            preview_device = str(decode_device or ("cuda" if torch.cuda.is_available() else "cpu"))
            decode_preview_result = _decode_preview_image(
                samples=result.samples,
                ae_path=bundle.ae_path,
                output_path=preview_output_path,
                device=preview_device,
            )
            decode_preview_result["decode_with_unet_resident"] = bool(decode_with_unet_resident)
    except Exception as exc:
        decode_preview_result = {
            "status": "error",
            "decode_with_unet_resident": bool(decode_with_unet_resident),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
    finally:
        if not cleanup_unet_after_denoise:
            _cleanup_model_patcher(unet_patcher, cleanup=True)
            try:
                from backend import resources

                resources.soft_empty_cache()
            except Exception:
                pass

    return {
        "mode": mode,
        "run_label": run_label,
        "total_wall": total_wall,
        "denoise_wall": denoise_wall,
        "denoise_cpu_proc": denoise_cpu_proc,
        "denoise_s_per_it": denoise_wall / max(1, config.steps),
        "wall_cpu_gap_s": wall_cpu_gap_s,
        "cpu_wall_ratio": cpu_wall_ratio,
        "peak_rss_mb": float(memory.peak_rss_bytes) / (1024 * 1024),
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_allocated_mb": peak_vram_allocated_mb,
        "host_pinned_mb": host_pinned_mb,
        "cuda_elapsed_ms": cuda_elapsed_ms,
        "async_load_wall": async_load_wall,
        "execution_class": execution_class.value,
        "overlap_note": overlap_note,
        "gpu_busy_note": gpu_busy_note,
        "metadata": result.metadata,
        "phase_timings": dict(result.timings),
        "sampling_perf": sampling_perf,
        "scheduler_stats": scheduler_stats,
        "scheduler_hit_rate": _scheduler_hit_rate(scheduler_stats),
        "prefetch_enqueued": int(scheduler_stats.get("prefetch_enqueued", 0)) if isinstance(scheduler_stats, dict) else 0,
        "prefetch_hits": int(scheduler_stats.get("prefetch_hits", 0)) if isinstance(scheduler_stats, dict) else 0,
        "prefetch_misses": int(scheduler_stats.get("prefetch_misses", 0)) if isinstance(scheduler_stats, dict) else 0,
        "prefetch_mb": float(scheduler_stats.get("prefetch_bytes", 0)) / (1024 * 1024) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_copy_wall_s": float(scheduler_stats.get("prefetch_copy_wall_s", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_copy_cuda_ms": float(scheduler_stats.get("prefetch_copy_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_effective_mb_s": _mb_per_s(
            float(scheduler_stats.get("prefetch_bytes", 0)) if isinstance(scheduler_stats, dict) else 0.0,
            float(scheduler_stats.get("prefetch_copy_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        ),
        "prefetch_skipped_size": int(scheduler_stats.get("prefetch_skipped_size", 0)) if isinstance(scheduler_stats, dict) else 0,
        "prefetch_skipped_vram": int(scheduler_stats.get("prefetch_skipped_vram", 0)) if isinstance(scheduler_stats, dict) else 0,
        "prefetch_guard_mb": float(scheduler_stats.get("vram_guard_bytes", 0)) / (1024 * 1024) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_guard_margin_mb": float(scheduler_stats.get("vram_guard_margin_bytes", 0)) / (1024 * 1024) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_max_mb": float(scheduler_stats.get("max_prefetch_bytes", 0)) / (1024 * 1024) if isinstance(scheduler_stats, dict) else 0.0,
        "bandwidth_limit_mb_s": float(scheduler_stats.get("bandwidth_limit_mb_s", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "direct_copy_mb": float(scheduler_stats.get("direct_copy_bytes", 0)) / (1024 * 1024) if isinstance(scheduler_stats, dict) else 0.0,
        "direct_copy_cuda_ms": float(scheduler_stats.get("direct_copy_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "direct_copy_calls": int(scheduler_stats.get("direct_copy_calls", 0)) if isinstance(scheduler_stats, dict) else 0,
        "direct_copy_effective_mb_s": _mb_per_s(
            float(scheduler_stats.get("direct_copy_bytes", 0)) if isinstance(scheduler_stats, dict) else 0.0,
            float(scheduler_stats.get("direct_copy_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        ),
        "prefetch_throttle_cuda_ms": float(scheduler_stats.get("prefetch_throttle_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "prefetch_throttle_events": int(scheduler_stats.get("prefetch_throttle_events", 0)) if isinstance(scheduler_stats, dict) else 0,
        "direct_throttle_cuda_ms": float(scheduler_stats.get("direct_throttle_cuda_ms", 0.0)) if isinstance(scheduler_stats, dict) else 0.0,
        "direct_throttle_events": int(scheduler_stats.get("direct_throttle_events", 0)) if isinstance(scheduler_stats, dict) else 0,
        "bandwidth_throttle_cuda_ms": total_throttle_cuda_ms,
        "bandwidth_throttle_events": int(scheduler_stats.get("bandwidth_throttle_events", 0)) if isinstance(scheduler_stats, dict) else 0,
        "stream_transfer_mb": total_transfer_bytes / (1024 * 1024),
        "stream_copy_cuda_ms": total_copy_cuda_ms,
        "stream_copy_plus_throttle_cuda_ms": total_copy_cuda_ms + total_throttle_cuda_ms,
        "stream_effective_mb_s": _mb_per_s(total_transfer_bytes, total_copy_cuda_ms),
        "stream_effective_with_throttle_mb_s": _mb_per_s(total_transfer_bytes, total_copy_cuda_ms + total_throttle_cuda_ms),
        "stream_waits": int(scheduler_stats.get("stream_waits", 0)) if isinstance(scheduler_stats, dict) else 0,
        "direct_copy_stream_uses": int(scheduler_stats.get("direct_copy_stream_uses", 0)) if isinstance(scheduler_stats, dict) else 0,
        "sampler_trace_wall": _sum_trace_wall(sampler_trace),
        "sampler_trace_cpu": _sum_trace_cpu(sampler_trace),
        "cond_trace_wall": _sum_trace_wall(cond_batch_trace),
        "cond_trace_cpu": _sum_trace_cpu(cond_batch_trace),
        "apply_model_trace_wall": _sum_trace_wall(apply_model_trace),
        "apply_model_trace_cpu": _sum_trace_cpu(apply_model_trace),
        "flux_fill_mode": flux_options.get("mode", ""),
        "detected_weight_dtype": str(detected_config.get("weight_dtype", "")),
        "manual_cast_dtype": str(detected_config.get("manual_cast_dtype", "")),
        "non_blocking_supported": bool(flux_options.get("non_blocking_supported", False)),
        "streaming_scheduler_kind": flux_options.get("streaming_scheduler_kind", ""),
        "streaming_scheduler_policy": flux_options.get("streaming_scheduler_policy", {}),
        "scheduled_module_count": int(flux_options.get("scheduled_module_count", 0)),
        "direct_safetensors_load": bool(flux_options.get("direct_safetensors_load", False)),
        "single_host_artifact": bool(flux_options.get("single_host_artifact", False)),
        "runtime_weight_dtype": str(flux_options.get("runtime_weight_dtype", "")),
        "runtime_weight_mb": float(flux_options.get("runtime_weight_bytes", 0)) / (1024 * 1024),
        "native_unet_load_diagnostics": result.metadata.get("native_unet_load_diagnostics", {}),
        "native_unet_runtime_before_denoise": result.metadata.get("native_unet_runtime_before_denoise", {}),
        "native_unet_runtime_after_denoise": result.metadata.get("native_unet_runtime_after_denoise", {}),
        "sample_shape": list(result.samples.shape),
        "samples_artifact_path": str(latent_artifact_path) if latent_artifact_path is not None else None,
        "decode_preview": decode_preview_result,
    }


def main() -> int:
    args = _parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(Path(args.bundle))
    modes = [normalize_flux_fill_benchmark_mode(mode) for mode in args.modes]
    run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]

    try:
        results: list[dict[str, Any]] = []
        for mode in modes:
            for run_label in run_labels:
                payload = _run_case(
                    mode=mode,
                    bundle=bundle,
                    run_label=run_label,
                    steps=args.steps,
                    output_dir=output_dir,
                    prefetch_depth=args.prefetch_depth,
                    max_prefetch_bytes=_mb_to_bytes(args.prefetch_max_mb),
                    vram_guard_bytes=_mb_to_bytes(args.vram_guard_mb),
                    vram_guard_margin_bytes=_mb_to_bytes(args.vram_guard_margin_mb),
                    prefetch_scan_ahead=args.prefetch_scan_ahead,
                    bandwidth_limit_mb_s=args.bandwidth_limit_mb_s,
                    save_latents=bool(args.save_latents),
                    decode_preview=bool(args.decode_preview),
                    decode_with_unet_resident=bool(args.decode_with_unet_resident),
                    decode_device=args.decode_device,
                )
                results.append(payload)
                _append_jsonl(output_dir / "benchmark_results.jsonl", payload)
                print(json.dumps(payload, default=_json_default))

        summary = {
            "bundle": asdict(bundle),
            "modes": modes,
            "runs": args.runs,
            "results": results,
            "scheduler_overrides": {
                "prefetch_depth": args.prefetch_depth,
                "prefetch_max_mb": args.prefetch_max_mb,
                "vram_guard_mb": args.vram_guard_mb,
                "vram_guard_margin_mb": args.vram_guard_margin_mb,
                "prefetch_scan_ahead": args.prefetch_scan_ahead,
                "bandwidth_limit_mb_s": args.bandwidth_limit_mb_s,
            },
            "artifact_options": {
                "save_latents": bool(args.save_latents),
                "decode_preview": bool(args.decode_preview),
                "decode_with_unet_resident": bool(args.decode_with_unet_resident),
                "decode_device": args.decode_device,
            },
            "notes": args.notes,
            "output_dir": str(output_dir),
            "gpu_busy_contract": {
                "wall_vs_cpu": True,
                "overlap_required": True,
                "utilization_signal": "denoise_wall_and_cpu_proc_plus_scheduler_hits_plus_stream_waits_plus_cuda_memory_peaks",
            },
        }
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps({"summary": str(output_dir / "summary.json"), "output_dir": str(output_dir)}, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "output_dir": str(output_dir),
        }
        if args.traceback:
            import traceback

            error["traceback"] = traceback.format_exc()
        _write_json(output_dir / "error.json", error)
        print(json.dumps(error, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
