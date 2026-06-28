from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _mb(value: float | int) -> float:
    return round(float(value) / (1024 * 1024), 2)


def _current_rss_bytes() -> int:
    if psutil is None:
        return 0
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


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
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(self._process.memory_info().rss))
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


def _parse_bucket(text: str) -> tuple[int, int]:
    raw = str(text or "").strip().lower().replace("x", "*")
    if "*" not in raw:
        raise ValueError(f"Bucket must look like 832x1216 or 832*1216, got {text!r}.")
    width_text, height_text = raw.split("*", 1)
    width = int(width_text)
    height = int(height_text)
    if width < 8 or height < 8:
        raise ValueError(f"Bucket dimensions must be >= 8, got {width}x{height}.")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"Bucket dimensions must be divisible by 8, got {width}x{height}.")
    return width, height


def _resolve_bucket_list(bucket_args: list[str] | None) -> list[tuple[int, int]]:
    if bucket_args:
        return [_parse_bucket(item) for item in bucket_args]

    from modules import flags

    return [_parse_bucket(item) for item in flags.sdxl_aspect_ratios]


def _latent_token_metrics(width: int, height: int) -> dict[str, int]:
    latent_w = width // 8
    latent_h = height // 8
    token_w = (latent_w + 1) // 2
    token_h = (latent_h + 1) // 2
    return {
        "latent_w": latent_w,
        "latent_h": latent_h,
        "token_w": token_w,
        "token_h": token_h,
        "image_tokens": token_w * token_h,
    }


def _effective_flux_attention_backend(device_label: str, requested_backend: str) -> str:
    try:
        from ldm_patched.modules.flux import math as flux_math

        backends = getattr(flux_math, "_FLUX_ATTENTION_RUNTIME_BACKENDS", {})
        return str(backends.get((requested_backend, device_label), requested_backend))
    except Exception:
        return requested_backend


def _resolve_assets(args: argparse.Namespace) -> dict[str, Path]:
    from modules import model_registry
    from backend.flux_fill_v3.activation import (
        FLUX_FILL_AE_ASSET_ID,
        FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID,
        FLUX_FILL_UNET_ASSET_BY_TIER,
    )

    unet_path = Path(args.unet_path) if args.unet_path else Path(
        model_registry.ensure_asset(FLUX_FILL_UNET_ASSET_BY_TIER[args.tier], progress=False)
    )
    ae_path = Path(args.ae_path) if args.ae_path else Path(
        model_registry.ensure_asset(FLUX_FILL_AE_ASSET_ID, progress=False)
    )
    conditioning_cache_path = Path(args.conditioning_cache_path) if args.conditioning_cache_path else Path(
        model_registry.ensure_asset(FLUX_FILL_EMPTY_CONDITIONING_ASSET_ID, progress=False)
    )
    return {
        "unet_path": unet_path,
        "ae_path": ae_path,
        "conditioning_cache_path": conditioning_cache_path,
    }


def _prepare_bucket_tensors(
    *,
    batch_size: int,
    latent_h: int,
    latent_w: int,
):
    import torch

    source = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32, device="cpu")
    concat = torch.zeros_like(source)
    denoise_mask = torch.ones((batch_size, 1, latent_h, latent_w), dtype=torch.float32, device="cpu")
    return source, concat, denoise_mask


def _summarize_bucket(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}

    keys_mean = (
        "wall_seconds",
        "seconds_per_step",
        "peak_rss_mb",
        "peak_rss_delta_mb",
        "cuda_peak_allocated_mb",
        "peak_pinned_mb",
        "image_tokens_per_second",
        "megapixels_per_second",
    )
    summary: dict[str, Any] = {
        "runs": len(runs),
        "sample_mean_abs_last": runs[-1].get("sample_mean_abs"),
    }
    for key in keys_mean:
        values = [float(run.get(key, 0.0)) for run in runs]
        summary[f"avg_{key}"] = round(sum(values) / max(len(values), 1), 4)
        summary[f"min_{key}"] = round(min(values), 4)
        summary[f"max_{key}"] = round(max(values), 4)
    return summary


def _print_ranked_table(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    print("")
    print("Rank  Bucket     Tokens  s/it   rel  tok/s  peakRSS")
    for index, row in enumerate(results, start=1):
        print(
            f"{index:>4}  "
            f"{row['bucket_label']:<10} "
            f"{row['image_tokens']:>6}  "
            f"{row['avg_seconds_per_step']:>5.2f}  "
            f"{row['relative_to_best']:>4.2f}  "
            f"{row['avg_image_tokens_per_second']:>5.1f}  "
            f"{row['avg_peak_rss_delta_mb']:>7.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Flux Fill v3 denoise throughput across exact SDXL bucket shapes."
    )
    parser.add_argument("--tier", default="fp8", choices=("fp8", "q8_0", "q4_k_s"), help="Flux Fill UNet tier asset.")
    parser.add_argument("--unet-path", default=None, help="Optional explicit UNet path.")
    parser.add_argument("--ae-path", default=None, help="Optional explicit AE path.")
    parser.add_argument("--conditioning-cache-path", default=None, help="Optional explicit empty-conditioning cache path.")
    parser.add_argument("--attention-backend", default="auto", choices=("auto", "sdpa", "xformers", "xformers_only"))
    parser.add_argument("--device", default=None, help="Optional torch device override.")
    parser.add_argument("--category", default="inpaint", choices=("inpaint", "removal"))
    parser.add_argument("--steps", type=int, default=8, help="Denoise steps per benchmark run.")
    parser.add_argument("--guidance", type=float, default=15.0, help="Flux guidance value.")
    parser.add_argument("--sampler", default="euler", help="Flux sampler name.")
    parser.add_argument("--scheduler", default="simple", help="Flux scheduler name.")
    parser.add_argument("--prefetch-depth", type=int, default=1, help="Streaming loader prefetch depth.")
    parser.add_argument("--prefetch-chunk-mb", type=int, default=64, help="Streaming loader prefetch chunk size in MB.")
    parser.add_argument("--seed", type=int, default=1415417960480890870, help="Fixed seed used by the denoise path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic latents.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per bucket before measurement.")
    parser.add_argument("--repeats", type=int, default=2, help="Measured runs per bucket.")
    parser.add_argument("--buckets", nargs="*", default=None, help="Optional bucket list such as 832x1216 896x1152.")
    parser.add_argument(
        "--recommend-within-ratio",
        type=float,
        default=1.15,
        help="Mark buckets as recommended when avg s/it is within this ratio of the best bucket.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["NEX_FLUX_ATTENTION_BACKEND"] = str(args.attention_backend)

    import torch
    from backend import resources
    from backend.flux_fill_v3.contracts import FluxFillRequest
    from backend.flux_fill_v3.spine import StreamingUnetSpine
    from backend.flux_fill_v3.t5_worker import DiskPagedTextWorker

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / f"flux_bucket_bench_{_now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        assets = _resolve_assets(args)
        buckets = _resolve_bucket_list(args.buckets)
        request = FluxFillRequest(
            unet_path=assets["unet_path"],
            ae_path=assets["ae_path"],
            conditioning_cache_path=assets["conditioning_cache_path"],
            seed=int(args.seed),
            steps=int(args.steps),
            guidance=float(args.guidance),
            sampler=str(args.sampler),
            scheduler=str(args.scheduler),
            prefetch_depth=int(args.prefetch_depth),
            prefetch_chunk_mb=int(args.prefetch_chunk_mb),
            device=args.device,
            prompt="",
            category=args.category,
        )
        text_worker = DiskPagedTextWorker(request)
        empty_conditioning = text_worker.get_conditioning()
        spine = StreamingUnetSpine(request)

        load_started = time.perf_counter()
        spine.start()
        spine_load_wall_seconds = time.perf_counter() - load_started
        device_label = str(spine.device)
        effective_attention_backend = args.attention_backend

        results: list[dict[str, Any]] = []
        try:
            for width, height in buckets:
                metrics = _latent_token_metrics(width, height)
                source, concat, denoise_mask = _prepare_bucket_tensors(
                    batch_size=int(args.batch_size),
                    latent_h=metrics["latent_h"],
                    latent_w=metrics["latent_w"],
                )

                bucket_label = f"{width}x{height}"
                print(f"[Flux Bucket Bench] {bucket_label} latent={metrics['latent_w']}x{metrics['latent_h']} tokens={metrics['image_tokens']}")

                for _ in range(max(int(args.warmup_runs), 0)):
                    resources.soft_empty_cache(force=True)
                    gc.collect()
                    with torch.no_grad():
                        spine.denoise(source, concat, denoise_mask, empty_conditioning)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    effective_attention_backend = _effective_flux_attention_backend(device_label, args.attention_backend)

                measured_runs: list[dict[str, Any]] = []
                for run_index in range(max(int(args.repeats), 1)):
                    resources.soft_empty_cache(force=True)
                    gc.collect()
                    rss_before = _current_rss_bytes()
                    with MemorySampler() as memory:
                        wall_started = time.perf_counter()
                        with torch.no_grad():
                            samples, _sigmas = spine.denoise(source, concat, denoise_mask, empty_conditioning)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        wall_seconds = time.perf_counter() - wall_started

                    cuda_peak_allocated_mb = 0.0
                    if torch.cuda.is_available():
                        try:
                            cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
                        except Exception:
                            cuda_peak_allocated_mb = 0.0

                    seconds_per_step = wall_seconds / max(int(args.steps), 1)
                    megapixels = float(width * height) / 1_000_000.0
                    image_tokens_per_second = float(metrics["image_tokens"]) / max(seconds_per_step, 1e-9)
                    megapixels_per_second = megapixels / max(seconds_per_step, 1e-9)
                    effective_attention_backend = _effective_flux_attention_backend(device_label, args.attention_backend)
                    run_payload = {
                        "bucket_label": bucket_label,
                        "width": width,
                        "height": height,
                        "latent_w": metrics["latent_w"],
                        "latent_h": metrics["latent_h"],
                        "token_w": metrics["token_w"],
                        "token_h": metrics["token_h"],
                        "image_tokens": metrics["image_tokens"],
                        "batch_size": int(args.batch_size),
                        "steps": int(args.steps),
                        "requested_attention_backend": args.attention_backend,
                        "effective_attention_backend": effective_attention_backend,
                        "device": device_label,
                        "run_index": run_index,
                        "wall_seconds": round(wall_seconds, 4),
                        "seconds_per_step": round(seconds_per_step, 4),
                        "peak_rss_mb": round(_mb(memory.peak_rss_bytes), 2),
                        "peak_rss_delta_mb": round(max(_mb(memory.peak_rss_bytes - rss_before), 0.0), 2),
                        "peak_pinned_mb": round(float(memory.peak_pinned_mb), 2),
                        "cuda_peak_allocated_mb": round(float(cuda_peak_allocated_mb), 2),
                        "image_tokens_per_second": round(image_tokens_per_second, 2),
                        "megapixels_per_second": round(megapixels_per_second, 4),
                        "sample_mean_abs": round(float(samples.detach().abs().mean().item()), 6),
                    }
                    measured_runs.append(run_payload)
                    _append_jsonl(output_dir / "benchmark_results.jsonl", run_payload)
                    print(json.dumps(run_payload, default=_json_default))

                bucket_summary = _summarize_bucket(measured_runs)
                bucket_summary.update(
                    {
                        "bucket_label": bucket_label,
                        "width": width,
                        "height": height,
                        "megapixels": round(float(width * height) / 1_000_000.0, 4),
                        "latent_w": metrics["latent_w"],
                        "latent_h": metrics["latent_h"],
                        "token_w": metrics["token_w"],
                        "token_h": metrics["token_h"],
                        "image_tokens": metrics["image_tokens"],
                    }
                )
                results.append(bucket_summary)

            results.sort(key=lambda item: float(item.get("avg_seconds_per_step", float("inf"))))
            best_seconds_per_step = float(results[0]["avg_seconds_per_step"]) if results else 0.0
            for item in results:
                relative = float(item["avg_seconds_per_step"]) / max(best_seconds_per_step, 1e-9)
                item["relative_to_best"] = round(relative, 4)
                item["recommended"] = bool(relative <= float(args.recommend_within_ratio))

            _print_ranked_table(results)

            summary = {
                "status": "ok",
                "output_dir": str(output_dir),
                "spine_load_wall_seconds": round(spine_load_wall_seconds, 4),
                "requested_attention_backend": args.attention_backend,
                "effective_attention_backend": effective_attention_backend,
                "device": device_label,
                "tier": args.tier,
                "assets": assets,
                "steps": int(args.steps),
                "warmup_runs": int(args.warmup_runs),
                "repeats": int(args.repeats),
                "recommend_within_ratio": float(args.recommend_within_ratio),
                "results": results,
                "recommended_buckets": [item["bucket_label"] for item in results if item.get("recommended")],
                "notes": (
                    "Denoise-only sweep over the current Flux Fill v3 streaming spine. "
                    "Use exact-shape results to choose a Flux-specific bucket allowlist."
                ),
            }
            _write_json(output_dir / "summary.json", summary)
            print(json.dumps({"summary": str(output_dir / "summary.json"), "output_dir": str(output_dir)}, default=_json_default))
            return 0
        finally:
            try:
                spine.end()
            except Exception:
                pass
            try:
                resources.soft_empty_cache(force=True)
            except Exception:
                pass
            gc.collect()
    except Exception as exc:
        payload = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "error.json", payload)
        print(json.dumps(payload, indent=2, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
