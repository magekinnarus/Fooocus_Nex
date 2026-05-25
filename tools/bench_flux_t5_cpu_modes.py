from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
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

from tools.flux_t5_benchmark_contract import (  # noqa: E402
    FLUX_T5_BENCHMARK_MODES,
    FluxT5BenchmarkAssets,
    FluxT5BenchmarkPrompt,
    normalize_flux_t5_benchmark_mode,
)

DEFAULT_CLIP_ROOT = Path(r"D:\AI\Imagine\models\clip")
DEFAULT_CLIP_L_PATH = DEFAULT_CLIP_ROOT / "clip_l.safetensors"
DEFAULT_Q8_T5_PATH = DEFAULT_CLIP_ROOT / "t5-v1_1-xxl-encoder-Q8_0.gguf"
DEFAULT_FP16_T5_PATH = DEFAULT_CLIP_ROOT / "t5xxl_fp16.safetensors"
DEFAULT_FP8_T5_PATH = DEFAULT_CLIP_ROOT / "t5xxl_fp8_e4m3fn.safetensors"


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


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if psutil is not None else None
        self.peak_rss_bytes = 0
        self.peak_vms_bytes = 0
        self.peak_pagefile_bytes = 0
        self.peak_private_bytes = 0
        self.peak_uss_bytes = 0

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set() and self._process is not None:
            try:
                info = self._process.memory_info()
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(getattr(info, "rss", 0)))
                self.peak_vms_bytes = max(self.peak_vms_bytes, int(getattr(info, "vms", 0)))
                self.peak_pagefile_bytes = max(self.peak_pagefile_bytes, int(getattr(info, "pagefile", 0)))
                self.peak_private_bytes = max(
                    self.peak_private_bytes,
                    int(getattr(info, "private", getattr(info, "private_bytes", 0))),
                )
            except Exception:
                pass
            try:
                full = self._process.memory_full_info()
                self.peak_uss_bytes = max(self.peak_uss_bytes, int(getattr(full, "uss", 0)))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _parse_prompts(args: argparse.Namespace) -> list[FluxT5BenchmarkPrompt]:
    prompts: list[FluxT5BenchmarkPrompt] = []
    for item in args.prompt or ():
        prompts.append(FluxT5BenchmarkPrompt(text=str(item)))
    if args.prompts_file:
        payload = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--prompts-file must contain a JSON list.")
        for index, item in enumerate(payload):
            if isinstance(item, str):
                prompts.append(FluxT5BenchmarkPrompt(text=item, label=f"prompt_{index}"))
            elif isinstance(item, dict):
                prompts.append(
                    FluxT5BenchmarkPrompt(
                        text=str(item.get("text", "")),
                        label=str(item.get("label")) if item.get("label") is not None else f"prompt_{index}",
                    )
                )
            else:
                raise ValueError(f"Unsupported prompt entry type: {type(item).__name__}.")
    prompts = [prompt for prompt in prompts if str(prompt.text).strip()]
    if not prompts:
        raise ValueError("At least one non-empty prompt is required.")
    return prompts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tools-only Flux T5 fp16 streaming benchmark.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M13-W08b1"))
    parser.add_argument(
        "--clip-l",
        default=str(DEFAULT_CLIP_L_PATH),
        help="Path to Flux CLIP-L weights. Defaults to the local text-encoder model area, not models\\clip\\flux.",
    )
    parser.add_argument(
        "--q8-t5",
        default=str(DEFAULT_Q8_T5_PATH),
        help="Path to the Q8 GGUF T5 weights. Defaults to the local text-encoder model area.",
    )
    parser.add_argument(
        "--fp16-t5",
        default=str(DEFAULT_FP16_T5_PATH),
        help="Path to the fp16 safetensors T5 weights. Defaults to the local text-encoder model area.",
    )
    parser.add_argument(
        "--fp8-t5",
        default=str(DEFAULT_FP8_T5_PATH),
        help="Path to the fp8 safetensors T5 weights. Defaults to the local text-encoder model area.",
    )
    parser.add_argument("--embedding-directory", default=None, help="Optional embedding directory.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to encode. Repeat for multiple prompts.")
    parser.add_argument("--prompts-file", default=None, help="Optional JSON file containing prompt rows.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["fp16_t5_stream_runtime"], choices=list(FLUX_T5_BENCHMARK_MODES))
    parser.add_argument("--notes", default="")
    parser.add_argument("--traceback", action="store_true")
    parser.add_argument("--disable-subprocess-isolation", action="store_true", help="Run all modes in the main process instead of isolating non-resident cases in worker subprocesses.")
    parser.add_argument("--worker-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _assets_from_args(args: argparse.Namespace) -> FluxT5BenchmarkAssets:
    return FluxT5BenchmarkAssets(
        clip_l_path=Path(args.clip_l),
        q8_t5_path=Path(args.q8_t5),
        fp16_t5_path=Path(args.fp16_t5),
        fp8_t5_path=Path(args.fp8_t5),
        embedding_directory=Path(args.embedding_directory) if args.embedding_directory else None,
    )


def _mode_status(mode: str) -> tuple[str, str | None]:
    normalized = normalize_flux_t5_benchmark_mode(mode)
    if normalized == "fp8_t5_paged_cpu_dequant":
        return ("ok", None)
    return "ok", None


def _mode_realization(mode: str) -> str:
    normalized = normalize_flux_t5_benchmark_mode(mode)
    if normalized == "q8_t5_paged_cpu_dequant":
        return "gguf_mmap_cpu_dequant"
    if normalized == "fp8_t5_cpu_resident":
        return "native_fp8_cpu_resident"
    if normalized == "fp8_t5_paged_cpu_dequant":
        return "native_fp8_safetensors_iterative_assign"
    if normalized == "fp16_t5_stream_runtime":
        return "native_fp16_safetensors_stream_runtime"
    if normalized == "fp16_t5_lazy_runtime":
        return "native_fp16_safetensors_lazy_runtime"
    raise ValueError(f"Unsupported mode: {mode!r}")


def _t5_loader_policy_for_mode(mode: str) -> str:
    normalized = normalize_flux_t5_benchmark_mode(mode)
    if normalized == "fp8_t5_paged_cpu_dequant":
        return "direct_safetensors_iterative"
    if normalized in {"fp16_t5_stream_runtime", "fp16_t5_lazy_runtime"}:
        return "stream_safetensors_runtime"
    return "eager"


def _memory_snapshot() -> dict[str, int]:
    if psutil is None:
        return {}
    process = psutil.Process()
    snapshot: dict[str, int] = {}
    try:
        info = process.memory_info()
        snapshot["rss"] = int(getattr(info, "rss", 0))
        snapshot["vms"] = int(getattr(info, "vms", 0))
        snapshot["pagefile"] = int(getattr(info, "pagefile", 0))
        snapshot["private"] = int(getattr(info, "private", getattr(info, "private_bytes", 0)))
    except Exception:
        return snapshot
    try:
        full = process.memory_full_info()
        snapshot["uss"] = int(getattr(full, "uss", 0))
    except Exception:
        pass
    return snapshot


def _memory_delta_mb(before: dict[str, int], after: dict[str, int]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in {"rss", "vms", "pagefile", "private", "uss"}:
        if key in before and key in after:
            deltas[f"{key}_delta_mb"] = float(after[key] - before[key]) / (1024 * 1024)
    return deltas


def _clear_resident_state() -> None:
    from tools import flux_text_conditioning_experiments as experiments

    experiments.clear_flux_prompt_text_encoder_cache()
    gc.collect()
    try:
        from backend import resources

        resources.soft_empty_cache(force=True)
    except Exception:
        pass


def _should_isolate_mode(mode: str) -> bool:
    normalized = normalize_flux_t5_benchmark_mode(mode)
    return normalized != "fp8_t5_cpu_resident"


def _worker_payload_path(output_dir: Path, *, mode: str, prompt_label: str, run_label: str) -> tuple[Path, Path]:
    worker_dir = output_dir / "_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{mode}__{prompt_label}__{run_label}"
    return worker_dir / f"{stem}.request.json", worker_dir / f"{stem}.result.json"


def _run_case(
    *,
    mode: str,
    prompt: FluxT5BenchmarkPrompt,
    prompt_index: int,
    run_label: str,
    assets: FluxT5BenchmarkAssets,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from backend.flux.flux_fill_pipeline import save_flux_empty_conditioning_cache
    from tools import flux_text_conditioning_experiments as experiments

    normalized = normalize_flux_t5_benchmark_mode(mode)
    status, gap_reason = _mode_status(normalized)
    prompt_label = prompt.normalized_label(prompt_index)
    t5_path = assets.t5_path_for_mode(normalized)
    keep_resident = normalized == "fp8_t5_cpu_resident"
    t5_loader_policy = _t5_loader_policy_for_mode(normalized)

    if keep_resident and run_label == "cold":
        _clear_resident_state()

    if status != "ok":
        return {
            "status": status,
            "mode": normalized,
            "mode_realization": _mode_realization(normalized),
            "run_label": run_label,
            "prompt_label": prompt_label,
            "prompt": prompt.text,
            "clip_l_path": str(assets.clip_l_path),
            "t5_path": str(t5_path),
            "gap_reason": gap_reason,
        }

    experiments.reset_stream_trace_stats()

    with MemorySampler() as memory:
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        load_memory_before = _memory_snapshot()
        encoder = experiments.get_flux_prompt_text_encoder(
            clip_l_path=assets.clip_l_path,
            t5_path=t5_path,
            embedding_directory=assets.embedding_directory,
            load_device="cpu",
            offload_device="cpu",
            keep_resident=keep_resident,
            t5_loader_policy=t5_loader_policy,
        )
        load_memory_after = _memory_snapshot()
        model_load_wall = time.perf_counter() - load_start
        load_metadata = dict(getattr(encoder, "_nex_load_metadata", {}) or {})

        encode_start = time.perf_counter()
        encode_cpu_start = time.process_time()
        cross_attn, pooled_output = encoder.encode(prompt.text)
        encode_wall = time.perf_counter() - encode_start
        encode_cpu_proc = time.process_time() - encode_cpu_start
        stream_trace = experiments.consume_stream_trace_stats()

        save_path = output_dir / "conditioning" / normalized / prompt_label / f"{run_label}.pt"
        save_start = time.perf_counter()
        conditioning = save_flux_empty_conditioning_cache(
            save_path,
            cross_attn=cross_attn.to(device="cpu"),
            pooled_output=pooled_output.to(device="cpu"),
            metadata={
                "prompt": prompt.text,
                "clip_l_path": str(assets.clip_l_path),
                "t5_path": str(t5_path),
                "t5_format": "gguf" if str(t5_path).lower().endswith(".gguf") else "safetensors",
                "generator": "tools/bench_flux_t5_cpu_modes.py",
                "conditioning_kind": "prompt",
                "transport": "pt_cache",
                "text_encoder_resident": bool(keep_resident),
                "t5_loader_policy": t5_loader_policy,
                "benchmark_mode": normalized,
                "mode_realization": _mode_realization(normalized),
                "run_label": run_label,
                "prompt_label": prompt_label,
                "loader_metadata": load_metadata,
                "stream_trace": stream_trace,
            },
        )
        save_wall = time.perf_counter() - save_start
        total_wall = time.perf_counter() - total_start

    if not keep_resident:
        try:
            del encoder
        except Exception:
            pass
        gc.collect()
        try:
            from backend import resources

            resources.soft_empty_cache()
        except Exception:
            pass

    return {
        "status": "ok",
        "mode": normalized,
        "mode_realization": _mode_realization(normalized),
        "run_label": run_label,
        "prompt_label": prompt_label,
        "prompt": prompt.text,
        "clip_l_path": str(assets.clip_l_path),
        "t5_path": str(t5_path),
        "embedding_directory": str(assets.embedding_directory) if assets.embedding_directory is not None else None,
        "total_wall": total_wall,
        "model_load_wall": model_load_wall,
        "encode_wall": encode_wall,
        "encode_cpu_proc": encode_cpu_proc,
        "save_wall": save_wall,
        "peak_rss_mb": float(memory.peak_rss_bytes) / (1024 * 1024),
        "peak_vms_mb": float(memory.peak_vms_bytes) / (1024 * 1024),
        "peak_pagefile_mb": float(memory.peak_pagefile_bytes) / (1024 * 1024),
        "peak_private_mb": float(memory.peak_private_bytes) / (1024 * 1024),
        "peak_uss_mb": float(memory.peak_uss_bytes) / (1024 * 1024),
        "conditioning_path": str(save_path),
        "conditioning_shape": list(conditioning.cross_attn.shape),
        "pooled_shape": list(conditioning.pooled_output.shape),
        "conditioning_dtype": str(conditioning.cross_attn.dtype),
        "pooled_dtype": str(conditioning.pooled_output.dtype),
        "text_encoder_resident": bool(keep_resident),
        "t5_loader_policy": t5_loader_policy,
        "loader_metadata": load_metadata,
        "stream_trace": stream_trace,
        "conditioning_metadata": dict(conditioning.metadata),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "load_memory_delta_mb": _memory_delta_mb(load_memory_before, load_memory_after),
        "process_isolated": False,
        "worker_pid": int(os.getpid()),
    }


def _run_case_subprocess(
    *,
    mode: str,
    prompt: FluxT5BenchmarkPrompt,
    prompt_index: int,
    run_label: str,
    assets: FluxT5BenchmarkAssets,
    output_dir: Path,
    traceback_enabled: bool,
) -> dict[str, Any]:
    normalized = normalize_flux_t5_benchmark_mode(mode)
    prompt_label = prompt.normalized_label(prompt_index)
    request_path, result_path = _worker_payload_path(
        output_dir,
        mode=normalized,
        prompt_label=prompt_label,
        run_label=run_label,
    )
    request_payload = {
        "mode": normalized,
        "prompt": asdict(prompt),
        "prompt_index": int(prompt_index),
        "run_label": run_label,
        "assets": asdict(assets),
        "output_dir": str(output_dir),
    }
    request_path.write_text(json.dumps(request_payload, indent=2, default=_json_default), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        str(request_path),
        "--worker-result",
        str(result_path),
    ]
    if traceback_enabled:
        command.append("--traceback")

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["process_isolated"] = True
        payload["worker_exit_code"] = int(completed.returncode)
        if completed.stdout.strip():
            payload["worker_stdout_tail"] = completed.stdout.strip().splitlines()[-3:]
        if completed.stderr.strip():
            payload["worker_stderr_tail"] = completed.stderr.strip().splitlines()[-3:]
        return payload

    error = {
        "status": "error",
        "mode": normalized,
        "run_label": run_label,
        "prompt_label": prompt_label,
        "prompt": prompt.text,
        "process_isolated": True,
        "worker_exit_code": int(completed.returncode),
        "worker_stdout_tail": completed.stdout.strip().splitlines()[-10:] if completed.stdout.strip() else [],
        "worker_stderr_tail": completed.stderr.strip().splitlines()[-10:] if completed.stderr.strip() else [],
        "error": {
            "type": "WorkerSubprocessError",
            "message": f"Worker subprocess failed for {normalized}/{prompt_label}/{run_label}.",
        },
    }
    return error


def _run_worker_from_config(args: argparse.Namespace) -> int:
    if not args.worker_config or not args.worker_result:
        raise ValueError("Worker mode requires --worker-config and --worker-result.")

    payload = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    assets_payload = dict(payload["assets"])
    if assets_payload.get("embedding_directory") is not None:
        assets_payload["embedding_directory"] = Path(assets_payload["embedding_directory"])
    assets = FluxT5BenchmarkAssets(
        clip_l_path=Path(assets_payload["clip_l_path"]),
        q8_t5_path=Path(assets_payload["q8_t5_path"]),
        fp16_t5_path=Path(assets_payload["fp16_t5_path"]),
        fp8_t5_path=Path(assets_payload["fp8_t5_path"]),
        embedding_directory=assets_payload.get("embedding_directory"),
        provenance=dict(assets_payload.get("provenance", {})),
    )
    prompt_payload = dict(payload["prompt"])
    prompt = FluxT5BenchmarkPrompt(
        text=str(prompt_payload["text"]),
        label=prompt_payload.get("label"),
    )
    result = _run_case(
        mode=str(payload["mode"]),
        prompt=prompt,
        prompt_index=int(payload["prompt_index"]),
        run_label=str(payload["run_label"]),
        assets=assets,
        output_dir=Path(payload["output_dir"]),
    )
    result["process_isolated"] = True
    result["worker_pid"] = int(os.getpid())
    Path(args.worker_result).write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(result, default=_json_default))
    return 0


def main() -> int:
    args = _parse_args()
    if args.worker_config or args.worker_result:
        try:
            return _run_worker_from_config(args)
        except Exception as exc:
            error = {
                "status": "error",
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            }
            if args.traceback:
                error["traceback"] = traceback.format_exc()
            if args.worker_result:
                Path(args.worker_result).write_text(json.dumps(error, indent=2, default=_json_default), encoding="utf-8")
            print(json.dumps(error, default=_json_default))
            return 1

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = _assets_from_args(args)
    prompts = _parse_prompts(args)
    modes = [normalize_flux_t5_benchmark_mode(mode) for mode in args.modes]
    run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]

    try:
        results: list[dict[str, Any]] = []
        for mode in modes:
            for prompt_index, prompt in enumerate(prompts):
                for run_label in run_labels:
                    if not args.disable_subprocess_isolation and _should_isolate_mode(mode):
                        payload = _run_case_subprocess(
                            mode=mode,
                            prompt=prompt,
                            prompt_index=prompt_index,
                            run_label=run_label,
                            assets=assets,
                            output_dir=output_dir,
                            traceback_enabled=bool(args.traceback),
                        )
                    else:
                        payload = _run_case(
                            mode=mode,
                            prompt=prompt,
                            prompt_index=prompt_index,
                            run_label=run_label,
                            assets=assets,
                            output_dir=output_dir,
                        )
                    results.append(payload)
                    _append_jsonl(output_dir / "benchmark_results.jsonl", payload)
                    print(json.dumps(payload, default=_json_default))

        summary = {
            "assets": asdict(assets),
            "modes": modes,
            "prompts": [asdict(prompt) for prompt in prompts],
            "runs": args.runs,
            "results": results,
            "notes": args.notes,
            "output_dir": str(output_dir),
            "benchmark_boundary": "flux_conditioning_encode_only",
            "lifecycle_emulation": {
                "non_resident_process_isolation": not bool(args.disable_subprocess_isolation),
                "resident_mode_process_isolation": False,
            },
            "mode_contract": {
                "q8_t5_paged_cpu_dequant": "implemented_as_gguf_mmap_cpu_dequant",
                "fp8_t5_cpu_resident": "implemented",
                "fp8_t5_paged_cpu_dequant": "implemented_as_native_fp8_safetensors_iterative_assign",
                "fp16_t5_stream_runtime": "implemented_as_native_fp16_safetensors_stream_runtime",
                "fp16_t5_lazy_runtime": "implemented_as_native_fp16_safetensors_lazy_runtime",
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
            error["traceback"] = traceback.format_exc()
        _write_json(output_dir / "error.json", error)
        print(json.dumps(error, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
