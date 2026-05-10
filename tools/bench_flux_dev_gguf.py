from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.flux_dev_runtime import FluxDevGGUFRunConfig, FluxDevGGUFRuntime


DEFAULT_POSITIVE_CONDITIONING = r"D:\AI\Fooocus_Nex\models\clip\flux\flux_background_conditioning.pt"
DEFAULT_NEGATIVE_CONDITIONING = r"D:\AI\Fooocus_Nex\models\clip\flux\flux_empty_conditioning.pt"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _git_short_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _collect_environment() -> dict[str, Any]:
    gpu_model = "cpu"
    total_vram_bytes = None
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        total_vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)

    return {
        "gpu_model": gpu_model,
        "total_vram_bytes": total_vram_bytes,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python_version": sys.version.replace("\n", " "),
        "os": platform.platform(),
        "repo_commit": _git_short_head(),
    }


@dataclass
class FluxDevBenchmarkCase:
    name: str
    run_label: str
    positive_conditioning_path: str
    negative_conditioning_path: str
    width: int
    height: int
    steps: int
    cfg: float
    guidance: float
    sampler: str
    scheduler: str
    seed: int
    batch_size: int
    unet_path: str
    vae_path: str
    output_path: str
    notes: str = ""


@dataclass
class FluxDevRunMetrics:
    scenario: str
    run_label: str
    route_label: str
    conditioning_hash: str
    positive_conditioning_path: str
    negative_conditioning_path: str
    resolution: str
    steps: int
    cfg: float
    guidance: float
    sampler: str
    scheduler: str
    seed: int
    batch_size: int
    unet_path: str
    vae_path: str
    image_path: str
    load_components: float
    conditioning_load: float
    unet_load: float
    vae_load: float
    conditioning_prepare: float
    conditioning_prepare_cpu_proc: float
    latent_noise_prep: float
    latent_noise_prep_cpu_proc: float
    sampler_model_attach: float
    sampler_model_offload: float
    denoise_wall: float
    denoise_s_per_it: float
    denoise_cpu_proc: float
    denoise_gguf_dequant: float
    denoise_gguf_dequant_cpu_proc: float
    denoise_gguf_forward: float
    denoise_gguf_forward_cpu_proc: float
    vae_attach: float
    vae_model_attach: float
    vae_decode: float
    vae_decode_cpu_proc: float
    vae_offload: float
    total_wall: float
    bootstrapped: bool
    denoise_gguf_trace_stats: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _hash_conditioning(positive_conditioning_path: str, negative_conditioning_path: str) -> str:
    payload = f"{positive_conditioning_path}\n---\n{negative_conditioning_path}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark plain Flux-dev Q8 on Local Edge.")
    parser.add_argument("--runs", type=int, default=2, help="Total runs including the cold run.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M12.5-W04"))
    parser.add_argument("--unet-path", default=r"D:\AI\Imagine\models\unet\flux\flux1-dev-Q8_0.gguf")
    parser.add_argument("--positive-conditioning-path", default=DEFAULT_POSITIVE_CONDITIONING)
    parser.add_argument("--negative-conditioning-path", default=DEFAULT_NEGATIVE_CONDITIONING)
    parser.add_argument("--vae-path", default=r"D:\AI\Imagine\models\vae\ae.safetensors")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=1)
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


def _build_case(args: argparse.Namespace, run_label: str, output_dir: Path) -> FluxDevBenchmarkCase:
    output_path = output_dir / f"{run_label}.png"
    return FluxDevBenchmarkCase(
        name="flux_dev_q8",
        run_label=run_label,
        positive_conditioning_path=args.positive_conditioning_path,
        negative_conditioning_path=args.negative_conditioning_path,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        guidance=args.guidance,
        sampler=args.sampler,
        scheduler=args.scheduler,
        seed=args.seed,
        batch_size=args.batch_size,
        unet_path=args.unet_path,
        vae_path=args.vae_path,
        output_path=str(output_path),
        notes=args.notes,
    )


def _run_case(case: FluxDevBenchmarkCase) -> tuple[FluxDevRunMetrics, dict[str, Any]]:
    config = FluxDevGGUFRunConfig(
        unet_path=case.unet_path,
        positive_conditioning_path=case.positive_conditioning_path,
        negative_conditioning_path=case.negative_conditioning_path,
        vae_path=case.vae_path,
        width=case.width,
        height=case.height,
        steps=case.steps,
        cfg=case.cfg,
        guidance=case.guidance,
        sampler=case.sampler,
        scheduler=case.scheduler,
        seed=case.seed,
        batch_size=case.batch_size,
        output_path=case.output_path,
    )
    runtime = FluxDevGGUFRuntime(config)
    try:
        result = runtime.run()
        metrics = FluxDevRunMetrics(
            scenario=case.name,
            run_label=case.run_label,
            route_label=runtime.route_label,
            conditioning_hash=_hash_conditioning(case.positive_conditioning_path, case.negative_conditioning_path),
            positive_conditioning_path=case.positive_conditioning_path,
            negative_conditioning_path=case.negative_conditioning_path,
            resolution=f"{case.width}x{case.height}",
            steps=case.steps,
            cfg=case.cfg,
            guidance=case.guidance,
            sampler=case.sampler,
            scheduler=case.scheduler,
            seed=case.seed,
            batch_size=case.batch_size,
            unet_path=case.unet_path,
            vae_path=case.vae_path,
            image_path=str(result.output_path) if result.output_path is not None else "",
            load_components=float(result.timings.get("load_components", 0.0)),
            conditioning_load=float(result.timings.get("conditioning_load", 0.0)),
            unet_load=float(result.timings.get("unet_load", 0.0)),
            vae_load=float(result.timings.get("vae_load", 0.0)),
            conditioning_prepare=float(result.timings.get("conditioning_prepare", 0.0)),
            conditioning_prepare_cpu_proc=float(result.timings.get("conditioning_prepare_cpu_proc", 0.0)),
            latent_noise_prep=float(result.timings.get("latent_noise_prep", 0.0)),
            latent_noise_prep_cpu_proc=float(result.timings.get("latent_noise_prep_cpu_proc", 0.0)),
            sampler_model_attach=float(result.timings.get("sampler_model_attach", 0.0)),
            sampler_model_offload=float(result.timings.get("sampler_model_offload", 0.0)),
            denoise_wall=float(result.timings.get("denoise_wall", 0.0)),
            denoise_s_per_it=float(result.timings.get("denoise_s_per_it", 0.0)),
            denoise_cpu_proc=float(result.timings.get("denoise_cpu_proc", 0.0)),
            denoise_gguf_dequant=float(result.timings.get("denoise_gguf_dequant", 0.0)),
            denoise_gguf_dequant_cpu_proc=float(result.timings.get("denoise_gguf_dequant_cpu_proc", 0.0)),
            denoise_gguf_forward=float(result.timings.get("denoise_gguf_forward", 0.0)),
            denoise_gguf_forward_cpu_proc=float(result.timings.get("denoise_gguf_forward_cpu_proc", 0.0)),
            vae_attach=float(result.timings.get("vae_attach", 0.0)),
            vae_model_attach=float(result.timings.get("vae_model_attach", 0.0)),
            vae_decode=float(result.timings.get("vae_decode", 0.0)),
            vae_decode_cpu_proc=float(result.timings.get("vae_decode_cpu_proc", 0.0)),
            vae_offload=float(result.timings.get("vae_offload", 0.0)),
            total_wall=float(result.timings.get("total_wall", 0.0)),
            bootstrapped=bool(result.metadata.get("bootstrapped", False)),
            denoise_gguf_trace_stats=dict(result.denoise_gguf_trace_stats),
            notes=case.notes,
        )
        return metrics, asdict(result)
    finally:
        runtime.close()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    environment = _collect_environment()
    _write_json(run_dir / "environment.json", environment)

    run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]
    cases = [_build_case(args, run_label, run_dir) for run_label in run_labels]
    base_case = cases[0]

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            metrics, _raw_result = _run_case(case)
            _append_jsonl(run_dir / "benchmark_results.jsonl", asdict(metrics))
            results.append(asdict(metrics))
            print(
                json.dumps(
                    {
                        "run": metrics.run_label,
                        "conditioning_hash": metrics.conditioning_hash,
                        "denoise_s_per_it": round(metrics.denoise_s_per_it, 4),
                        "total_wall": round(metrics.total_wall, 4),
                        "image_path": metrics.image_path,
                    },
                    default=_json_default,
                )
            )

        summary = {
            "environment": environment,
            "case": asdict(base_case),
            "results": results,
            "output_dir": str(run_dir),
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps({"summary": str(run_dir / "summary.json"), "output_dir": str(run_dir)}, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
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
