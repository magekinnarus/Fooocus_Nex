from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import psutil
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRunConfig
from modules.gguf_headless_runner import (
    HeadlessGGUFRunner,
    append_metrics_jsonl,
    collect_environment,
    scenario_library,
    write_environment_report,
)
from tools.gguf_true_streaming_runtime import TrueStreamingSDXLGGUFRuntime


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _prompt_hash(prompt: str, negative_prompt: str) -> str:
    payload = f"{prompt}\n---\n{negative_prompt}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


@dataclass
class MemorySnapshot:
    peak_rss_bytes: int = 0
    peak_vram_allocated_bytes: int = 0
    peak_vram_reserved_bytes: int = 0


@dataclass
class StreamingRunMetrics:
    scenario: str
    route_label: str
    run_label: str
    prompt_hash: str
    quant_model: str
    resolution: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    batch_size: int
    process_start: float
    cold_model_load_cpu: float
    clip_residency_attach: float
    clip_residency_offload: float
    clip_gpu_load: float
    clip_encode: float
    adm_build: float
    sampler_model_attach: float
    unet_gpu_load_or_patch: float
    cond_prepare_explicit: float
    cond_prep: float
    denoise_wall: float
    denoise_s_per_it: float
    denoise_cpu_proc: float
    gguf_dequant: float
    gguf_dequant_cpu_proc: float
    vae_attach: float
    vae_gpu_load: float
    vae_decode: float
    cleanup_reset: float
    image_save: float
    total_wall: float
    image_path: str
    clip_residency_mode: str = ""
    warm_state_annotation: dict[str, Any] = field(default_factory=dict)
    checkpoint_records_path: str = ""
    checkpoint_record_count: int = 0
    glass_ancestral_noise_policy: str = ""
    gguf_trace_stats: dict[str, Any] = field(default_factory=dict)
    apply_model_trace_stats: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    peak_rss_bytes: int = 0
    peak_vram_allocated_bytes: int = 0
    peak_vram_reserved_bytes: int = 0
    mmap_released_before: Optional[bool] = None
    mmap_released_after: Optional[bool] = None
    host_streaming_note: str = ""


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        self.snapshot = MemorySnapshot()

    def __enter__(self) -> "MemorySampler":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = int(self._process.memory_info().rss)
                if rss > self.snapshot.peak_rss_bytes:
                    self.snapshot.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.snapshot.peak_vram_allocated_bytes = int(torch.cuda.max_memory_allocated())
            self.snapshot.peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the tools-only SDXL true-streaming GGUF prototype.")
    parser.add_argument("--runs", type=int, default=2, help="Total runs including the cold run.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M12.5-W05"))
    parser.add_argument("--scenario", default="mission_q8_acceptance", choices=sorted(scenario_library().keys()))
    parser.add_argument(
        "--mode",
        default="true_streaming",
        choices=("true_streaming", "baseline", "both"),
        help="Select which benchmark path to run.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _clone_scenario(args: argparse.Namespace):
    scenario = scenario_library()[args.scenario]
    if args.notes:
        scenario = replace(scenario, notes=(scenario.notes + " " + args.notes).strip())
    return scenario


def _save_png(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def _normalize_direct_result(metrics: Any, memory: MemorySnapshot) -> dict[str, Any]:
    payload = asdict(metrics)
    payload.update(
        {
            "peak_rss_bytes": memory.peak_rss_bytes,
            "peak_vram_allocated_bytes": memory.peak_vram_allocated_bytes,
            "peak_vram_reserved_bytes": memory.peak_vram_reserved_bytes,
            "mmap_released_before": payload.get("warm_state_annotation", {}).get("gguf_mmap_released_before"),
            "mmap_released_after": payload.get("warm_state_annotation", {}).get("gguf_mmap_released_after"),
            "host_streaming_note": "baseline_direct_partial_load",
        }
    )
    return payload


def _normalize_streaming_result(
    *,
    scenario_name: str,
    runtime: TrueStreamingSDXLGGUFRuntime,
    run_label: str,
    result: Any,
    image_path: Path,
    memory: MemorySnapshot,
    total_wall: float,
) -> dict[str, Any]:
    benchmark = dict(result.benchmark)
    payload = StreamingRunMetrics(
        scenario=scenario_name,
        route_label=runtime.route_label,
        run_label=run_label,
        prompt_hash=_prompt_hash(runtime.config.prompt, runtime.config.negative_prompt),
        quant_model=Path(runtime.config.unet_path).name,
        resolution=f"{runtime.config.width}x{runtime.config.height}",
        steps=runtime.config.steps,
        cfg=runtime.config.cfg,
        sampler=runtime.config.sampler,
        scheduler=runtime.config.scheduler,
        seed=runtime.config.seed,
        batch_size=runtime.config.batch_size,
        process_start=getattr(runtime, "_process_started", time.perf_counter()),
        cold_model_load_cpu=float(benchmark.get("cold_model_load_cpu", 0.0)),
        clip_residency_attach=float(benchmark.get("clip_residency_attach", 0.0)),
        clip_residency_offload=float(benchmark.get("clip_residency_offload", 0.0)),
        clip_gpu_load=float(benchmark.get("clip_gpu_load", 0.0)),
        clip_encode=float(benchmark.get("clip_encode", 0.0)),
        adm_build=float(benchmark.get("adm_build", 0.0)),
        sampler_model_attach=float(benchmark.get("sampler_model_attach", 0.0)),
        unet_gpu_load_or_patch=float(benchmark.get("sampler_model_attach", 0.0)),
        cond_prepare_explicit=float(benchmark.get("cond_prepare_explicit", 0.0)),
        cond_prep=float(benchmark.get("cond_prepare_explicit", 0.0)),
        denoise_wall=float(benchmark.get("denoise_wall", 0.0)),
        denoise_s_per_it=float(benchmark.get("denoise_s_per_it", 0.0)),
        denoise_cpu_proc=float(benchmark.get("denoise_cpu_proc", 0.0)),
        gguf_dequant=float(benchmark.get("gguf_dequant", 0.0)),
        gguf_dequant_cpu_proc=float(benchmark.get("gguf_dequant_cpu_proc", 0.0)),
        vae_attach=float(benchmark.get("vae_attach", 0.0)),
        vae_gpu_load=float(benchmark.get("vae_attach", 0.0)),
        vae_decode=float(benchmark.get("vae_decode", 0.0)),
        cleanup_reset=0.0,
        image_save=0.0,
        total_wall=total_wall,
        image_path=str(image_path),
        clip_residency_mode="gpu_then_offload",
        warm_state_annotation={
            "gguf_mmap_released_before": getattr(runtime.unet, "mmap_released", None),
            "gguf_mmap_released_after": getattr(runtime.unet, "mmap_released", None),
            "streaming_source_retained": True,
        },
        gguf_trace_stats=dict(benchmark.get("gguf_trace_stats", {})),
        notes=getattr(runtime.config, "notes", ""),
        peak_rss_bytes=memory.peak_rss_bytes,
        peak_vram_allocated_bytes=memory.peak_vram_allocated_bytes,
        peak_vram_reserved_bytes=memory.peak_vram_reserved_bytes,
        mmap_released_before=getattr(runtime.unet, "mmap_released", None),
        mmap_released_after=getattr(runtime.unet, "mmap_released", None),
        host_streaming_note="tools_only_true_streaming",
    )
    result_payload = asdict(payload)
    result_payload["image_path"] = str(image_path)
    result_payload["total_wall"] = total_wall
    return result_payload


def _run_direct_baseline(scenario, args: argparse.Namespace, run_dir: Path, run_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = HeadlessGGUFRunner(
        scenario,
        "direct_sdxl_gguf",
        clip_residency_mode="gpu_then_offload",
    )
    image_path = run_dir / f"{scenario.name}_direct_sdxl_gguf_{run_label}.png"
    with MemorySampler() as memory:
        metrics = runner.run_once(run_label, run_dir)
    runner.close()
    payload = _normalize_direct_result(metrics, memory.snapshot)
    payload["image_path"] = str(image_path)
    return payload, asdict(metrics)


def _run_streaming_case(scenario, args: argparse.Namespace, run_dir: Path, run_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = DirectSDXLGGUFRunConfig(
        unet_path=scenario.unet_path,
        clip_l_path=scenario.clip_l_path,
        clip_g_path=scenario.clip_g_path,
        vae_path=scenario.vae_path,
        prompt=scenario.prompt,
        negative_prompt=scenario.negative_prompt,
        width=scenario.width,
        height=scenario.height,
        steps=scenario.steps,
        cfg=scenario.cfg,
        sampler=scenario.sampler,
        scheduler=scenario.scheduler,
        seed=scenario.seed,
        clip_layer=scenario.clip_layer,
        denoise=scenario.denoise,
        batch_size=scenario.batch_size,
        quality=scenario.quality.as_sampling_dict(),
    )
    runtime = TrueStreamingSDXLGGUFRuntime(config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.perf_counter()
    with MemorySampler() as memory:
        result = runtime.run()
    total_wall = time.perf_counter() - start

    image_path = run_dir / f"{scenario.name}_tools_only_true_streaming_{run_label}.png"
    image_save_start = time.perf_counter()
    _save_png(image_path, result.images[0])
    image_save = time.perf_counter() - image_save_start

    payload = _normalize_streaming_result(
        scenario_name=scenario.name,
        runtime=runtime,
        run_label=run_label,
        result=result,
        image_path=image_path,
        memory=memory.snapshot,
        total_wall=total_wall,
    )
    payload["image_save"] = image_save
    runtime.close()
    return payload, dict(result.benchmark)


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    scenario = _clone_scenario(args)
    run_dir_root = Path(args.output_dir)
    run_dir_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir_root / f"{scenario.name}_{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(
        "tools_only_true_streaming" if args.mode != "baseline" else "direct_sdxl_gguf",
        scenario,
    )
    write_environment_report(environment, run_dir)

    run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]
    results: list[dict[str, Any]] = []
    try:
        for run_label in run_labels:
            if args.mode == "baseline":
                payload, raw_result = _run_direct_baseline(scenario, args, run_dir, run_label)
            else:
                payload, raw_result = _run_streaming_case(scenario, args, run_dir, run_label)
            _append_jsonl(run_dir / "benchmark_results.jsonl", payload)
            results.append(payload)
            print(
                json.dumps(
                    {
                        "run": run_label,
                        "route": payload["route_label"],
                        "denoise_s_per_it": round(float(payload.get("denoise_s_per_it", 0.0)), 4),
                        "total_wall": round(float(payload.get("total_wall", 0.0)), 4),
                        "peak_rss_mb": round(float(payload.get("peak_rss_bytes", 0)) / (1024 * 1024), 2),
                        "peak_vram_mb": round(float(payload.get("peak_vram_reserved_bytes", 0)) / (1024 * 1024), 2),
                        "image_path": payload.get("image_path", ""),
                    },
                    default=_json_default,
                )
            )

        summary = {
            "environment": asdict(environment),
            "scenario": asdict(scenario),
            "results": results,
            "output_dir": str(run_dir),
            "comparison_baseline": "Use tools/bench_headless_gguf_txt2img.py --route direct_sdxl_gguf with the same scenario for the baseline control.",
            "mode": args.mode,
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps({"summary": str(run_dir / "summary.json"), "output_dir": str(run_dir)}, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "output_dir": str(run_dir),
        }
        if args.traceback:
            import traceback

            error["traceback"] = traceback.format_exc()
        _write_json(run_dir / "error.json", error)
        print(json.dumps(error, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
