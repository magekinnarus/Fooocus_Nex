from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psutil
except ImportError:
    psutil = None

try:
    import xformers

    xformers_version = getattr(xformers, "__version__", "unknown")
except ImportError:
    xformers_version = None

# Backup sys.argv to prevent args_manager from parsing CLI args at import time.
original_argv = list(sys.argv)
sys.argv = [original_argv[0]]

try:
    from backend import resources
    from backend.encode import encode_pixels
    from backend.flux.flux_fill_pipeline import (
        FluxEmptyConditioning,
        FluxFillConfig,
        FluxFillPrecomputedDenoiseInput,
        _cleanup_model_patcher,
        _snapshot_first_param_runtime,
        _snapshot_module_runtime,
        decode_flux_fill_latent,
        denoise_flux_fill_precomputed_latent,
        load_flux_ae,
        load_flux_empty_conditioning_cache,
        load_flux_fill_native_unet,
    )
    from modules.core import numpy_to_pytorch
finally:
    sys.argv = original_argv


class ProcessMemorySampler:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if psutil is not None else None
        self.peak_rss_bytes = 0
        self.peak_uss_bytes = 0

    def __enter__(self) -> "ProcessMemorySampler":
        self.sample_once()
        if self._process is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            time.sleep(self.interval_s)

    def sample_once(self) -> None:
        if self._process is None:
            return
        try:
            info = self._process.memory_info()
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(getattr(info, "rss", 0)))
        except Exception:
            pass
        try:
            full = self._process.memory_full_info()
            self.peak_uss_bytes = max(self.peak_uss_bytes, int(getattr(full, "uss", 0)))
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sample_once()


class SafeOpenTracker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._patches: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "SafeOpenTracker":
        import backend.loader as backend_loader
        import safetensors

        self._patch_module(safetensors, "safe_open", "safetensors.safe_open")
        self._patch_module(backend_loader, "safe_open", "backend.loader.safe_open")
        return self

    def _patch_module(self, module: Any, attr_name: str, label: str) -> None:
        original = getattr(module, attr_name, None)
        if original is None:
            return
        self._patches.append((module, attr_name, original))

        def wrapped(*args, **kwargs):
            path = str(args[0]) if args else ""
            self.calls.append(
                {
                    "source": label,
                    "path": path,
                    "device": kwargs.get("device"),
                }
            )
            print(f"SENTINEL WARNING: {label} called during warm run for {path}")
            return original(*args, **kwargs)

        setattr(module, attr_name, wrapped)

    def __exit__(self, exc_type, exc, tb) -> None:
        for module, attr_name, original in reversed(self._patches):
            setattr(module, attr_name, original)


def _mb(byte_count: int | float) -> float:
    return float(byte_count) / (1024 * 1024)


def _torch_peaks_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "peak_vram_mb": 0.0,
            "peak_vram_allocated_mb": 0.0,
        }
    return {
        "peak_vram_mb": _mb(torch.cuda.max_memory_reserved()),
        "peak_vram_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
    }


def _process_memory_snapshot() -> dict[str, float]:
    if psutil is None:
        return {}
    try:
        process = psutil.Process()
        info = process.memory_info()
        full = None
        try:
            full = process.memory_full_info()
        except Exception:
            full = None
        return {
            "rss_mb": _mb(getattr(info, "rss", 0)),
            "uss_mb": _mb(getattr(full, "uss", 0)) if full is not None else 0.0,
        }
    except Exception:
        return {}


def _serialize_device(device: Any) -> str:
    if device is None:
        return ""
    return str(device)


def _is_cuda_device_value(device: Any) -> bool:
    return _serialize_device(device).startswith("cuda")


def _current_loaded_device(model_patcher: Any) -> Any:
    current = getattr(model_patcher, "current_loaded_device", None)
    if callable(current):
        return current()
    return getattr(model_patcher, "load_device", None)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_env_report(output_dir: Path) -> dict[str, Any]:
    env_report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "xformers_version": xformers_version,
    }
    if torch.cuda.is_available():
        env_report["gpu"] = torch.cuda.get_device_name(0)
        env_report["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3)
    _write_json(output_dir / "env_report.json", env_report)
    print(f"Environment report written to {output_dir / 'env_report.json'}")
    return env_report


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {path}")
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def prepare_concat_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bb_image_for_concat = image.copy().astype(np.float32) / 255.0
    bb_mask_2d = mask
    if bb_mask_2d.ndim == 3:
        bb_mask_2d = bb_mask_2d[:, :, 0]
    mask_binary = (bb_mask_2d > 127).astype(np.float32)
    inv_mask = 1.0 - mask_binary
    for ch in range(3):
        bb_image_for_concat[:, :, ch] -= 0.5
        bb_image_for_concat[:, :, ch] *= inv_mask
        bb_image_for_concat[:, :, ch] += 0.5
    return np.clip(bb_image_for_concat * 255.0, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless validation script for the Flux Fill sticky-resident Colab Free path.")
    parser.add_argument("--unet-path", required=True, help="Path to Flux Fill FP8 .safetensors")
    parser.add_argument("--ae-path", required=True, help="Path to VAE ae.safetensors")
    parser.add_argument("--image-path", required=True, help="Path to input image")
    parser.add_argument("--mask-path", required=True, help="Path to input mask")
    parser.add_argument("--conditioning-path", default=None, help="Path to precomputed conditioning .pt")
    parser.add_argument("--empty-conditioning", action="store_true", help="Use zero-filled conditioning instead of a saved prompt-conditioning cache.")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--seed", type=int, default=882699830973928, help="Random seed")
    parser.add_argument("--guidance", type=float, default=15.0, help="Guidance scale")
    parser.add_argument("--run-name", default="run_09", help="Run identifier")
    return parser.parse_args()


def _build_conditioning(args: argparse.Namespace) -> tuple[FluxEmptyConditioning, dict[str, Any]]:
    if args.conditioning_path and not args.empty_conditioning:
        conditioning = load_flux_empty_conditioning_cache(args.conditioning_path)
        return (
            conditioning,
            {
                "mode": "artifact",
                "path": args.conditioning_path,
                "metadata": dict(conditioning.metadata or {}),
            },
        )
    conditioning = FluxEmptyConditioning(
        cross_attn=torch.zeros((1, 256, 4096), dtype=torch.float32),
        pooled_output=torch.zeros((1, 768), dtype=torch.float32),
        metadata={"conditioning_kind": "empty"},
    )
    return (
        conditioning,
        {
            "mode": "empty",
            "path": "",
            "metadata": dict(conditioning.metadata or {}),
        },
    )


def _save_samples(path: Path, samples: torch.Tensor, *, run_label: str, args: argparse.Namespace) -> None:
    torch.save(
        {
            "samples": samples.detach().cpu(),
            "metadata": {
                "run": run_label,
                "steps": args.steps,
                "seed": args.seed,
                "guidance": args.guidance,
            },
        },
        path,
    )


def _describe_tensor_bundle(
    *,
    source_latent: torch.Tensor,
    concat_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    conditioning: FluxEmptyConditioning,
) -> dict[str, Any]:
    return {
        "source_latent": {
            "object_id": id(source_latent),
            "shape": list(source_latent.shape),
            "device": str(source_latent.device),
            "dtype": str(source_latent.dtype),
        },
        "concat_latent": {
            "object_id": id(concat_latent),
            "shape": list(concat_latent.shape),
            "device": str(concat_latent.device),
            "dtype": str(concat_latent.dtype),
        },
        "denoise_mask": {
            "object_id": id(denoise_mask),
            "shape": list(denoise_mask.shape),
            "device": str(denoise_mask.device),
            "dtype": str(denoise_mask.dtype),
        },
        "conditioning": {
            "object_id": id(conditioning),
            "cross_attn_object_id": id(conditioning.cross_attn),
            "pooled_output_object_id": id(conditioning.pooled_output),
            "mode": str(conditioning.metadata.get("conditioning_kind", "")),
        },
    }


def _run_decode_pass(
    *,
    run_label: str,
    config: FluxFillConfig,
    precomputed_input: FluxFillPrecomputedDenoiseInput,
    unet_patcher: Any,
    vae: Any,
    ae_path: str,
    output_dir: Path,
    device: str,
    args: argparse.Namespace,
    bundle_description: dict[str, Any],
    load_models_gpu_wall_ms: float | None = None,
    safe_open_tracker: SafeOpenTracker | None = None,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with ProcessMemorySampler() as memory:
        run_started = _process_memory_snapshot()
        denoise_started = time.perf_counter()
        denoise_cpu_started = time.process_time()
        denoise_result = denoise_flux_fill_precomputed_latent(
            config,
            precomputed_input,
            unet_patcher=unet_patcher,
            load_device=config.device,
            cleanup_unet=False,
            disable_pbar=True,
        )
        denoise_wall = time.perf_counter() - denoise_started
        denoise_cpu_proc = time.process_time() - denoise_cpu_started

        decode_started = time.perf_counter()
        decoded = decode_flux_fill_latent(
            denoise_result.samples,
            ae_path,
            stitch=False,
            load_device=device,
            offload_device=device,
            vae=vae,
            cleanup_vae=False,
        )
        ae_decode = time.perf_counter() - decode_started
        run_finished = _process_memory_snapshot()

    preview_path = output_dir / f"{run_label}_preview.png"
    samples_path = output_dir / f"{run_label}_samples.pt"
    Image.fromarray(decoded.bb_image).save(preview_path)
    _save_samples(samples_path, denoise_result.samples, run_label=run_label, args=args)

    run_metadata = dict(denoise_result.metadata or {})
    decode_metadata = dict(decoded.metadata or {})
    native_diag = dict(run_metadata.get("native_unet_load_diagnostics", {}) or {})

    payload = {
        "timings": {
            "load_models_gpu_wall_ms": load_models_gpu_wall_ms,
            "unet_load": denoise_result.timings.get("unet_load", 0.0),
            "denoise_wall": denoise_wall,
            "denoise_cpu_proc": denoise_cpu_proc,
            "ae_decode": ae_decode,
        },
        "metrics": {
            "peak_rss_mb": _mb(memory.peak_rss_bytes),
            "peak_uss_mb": _mb(memory.peak_uss_bytes),
            **_torch_peaks_mb(),
            "run_started": run_started,
            "run_finished": run_finished,
        },
        "unet": {
            "source_weight_dtype": native_diag.get("source_weight_dtype", ""),
            "resident_weight_dtype": native_diag.get("resident_weight_dtype", ""),
            "manual_cast_dtype": native_diag.get("manual_cast_dtype", ""),
            "compute_weight_dtype": run_metadata.get("compute_weight_dtype", ""),
            "resident_weight_mb": float(run_metadata.get("resident_weight_mb", 0.0)),
            "native_unet_load_diagnostics": native_diag,
            "native_unet_runtime_before_denoise": run_metadata.get("native_unet_runtime_before_denoise", {}),
            "native_unet_runtime_after_denoise": run_metadata.get("native_unet_runtime_after_denoise", {}),
        },
        "vae": {
            "runtime_before_decode": decode_metadata.get("vae_runtime_before_decode", {}),
            "runtime_after_decode": decode_metadata.get("vae_runtime_after_decode", {}),
            "loaded_device_after_run": _serialize_device(_current_loaded_device(vae.patcher)),
        },
        "artifacts": {
            "preview_path": str(preview_path),
            "samples_path": str(samples_path),
        },
        "warm_state": {
            "safe_open_calls": len(safe_open_tracker.calls) if safe_open_tracker is not None else 0,
            "safe_open_call_details": list(safe_open_tracker.calls) if safe_open_tracker is not None else [],
            "disk_reload_detected": bool(safe_open_tracker.calls) if safe_open_tracker is not None else False,
        },
        "artifact_bundle": {
            "reused_precomputed_input": True,
            "precomputed_input_object_id": id(precomputed_input),
            "bundle_description": bundle_description,
        },
    }
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_env_report(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_np = load_rgb_image(Path(args.image_path))
    mask_np = load_mask(Path(args.mask_path))

    print("Phase 1: Loading VAE and performing live encode without detach...")
    vae = load_flux_ae(args.ae_path, load_device=device, offload_device=device)
    resources.load_models_gpu([vae.patcher])

    vae_device = _current_loaded_device(vae.patcher)
    if isinstance(vae_device, str):
        vae_device = torch.device(vae_device)
    move_model = getattr(vae.first_stage_model, "to", None)
    if callable(move_model):
        move_model(device=vae_device, dtype=torch.float32)

    vae_encode_started = time.perf_counter()
    vae_metadata: dict[str, Any] = {
        "vae_runtime_before_source_encode": _snapshot_first_param_runtime(vae.first_stage_model),
    }

    orig_pixels = numpy_to_pytorch(image_np)
    source_latent_res = encode_pixels(vae, orig_pixels)
    source_latent = source_latent_res["samples"]

    bb_image_for_concat = prepare_concat_image(image_np, mask_np)
    pixels_for_vae = (numpy_to_pytorch(bb_image_for_concat).movedim(-1, 1) * 2.0) - 1.0
    if pixels_for_vae.ndim == 3:
        pixels_for_vae = pixels_for_vae.unsqueeze(0)

    vae_param = next(vae.first_stage_model.parameters(), None)
    vae_input_device = vae.patcher.load_device
    vae_input_dtype = torch.float32
    if isinstance(vae_param, torch.Tensor):
        vae_input_device = vae_param.device
        vae_input_dtype = vae_param.dtype
    pixels_for_vae = pixels_for_vae.to(device=vae_input_device, dtype=vae_input_dtype)

    vae_metadata["vae_runtime_before_concat_encode"] = _snapshot_first_param_runtime(vae.first_stage_model)
    vae_metadata["vae_runtime_manual_concat_input"] = {
        "device": str(vae_input_device),
        "dtype": str(vae_input_dtype),
    }

    raw_latent = vae.first_stage_model.encode(pixels_for_vae)
    if hasattr(raw_latent, "sample"):
        raw_latent = raw_latent.sample()
    concat_latent = vae.latent_format.process_in(raw_latent).cpu()

    mask_t = torch.from_numpy(mask_np).float() / 255.0
    if mask_t.ndim == 3:
        mask_t = mask_t[:, :, 0]
    mask_t = mask_t[None, None, :, :]
    denoise_mask = torch.nn.functional.max_pool2d(mask_t, kernel_size=8)
    denoise_mask = (denoise_mask > 0.5).float()
    vae_metadata["vae_runtime_after_encode"] = _snapshot_first_param_runtime(vae.first_stage_model)
    vae_encode_wall = time.perf_counter() - vae_encode_started

    print("Phase 2: Loading UNet directly to the target runtime device...")
    unet_patcher = load_flux_fill_native_unet(
        args.unet_path,
        load_device=device,
        offload_device=device,
    )
    flux_options_unet = dict(getattr(unet_patcher, "model_options", {}).get("flux_fill", {}) or {})
    detected_config = dict(flux_options_unet.get("detected_config", {}) or {})

    attach_started = time.perf_counter()
    resources.load_models_gpu([unet_patcher])
    cold_establish_load_models_gpu_wall_ms = (time.perf_counter() - attach_started) * 1000.0

    vae_device_after_unet_load = _current_loaded_device(vae.patcher)
    vae_evicted_during_unet_attach = device == "cuda" and not _is_cuda_device_value(vae_device_after_unet_load)
    if vae_evicted_during_unet_attach:
        print("VAE was evicted while attaching UNet; re-attaching VAE to restore dual residency.")
        resources.load_models_gpu([vae.patcher])
        vae_device_after_unet_load = _current_loaded_device(vae.patcher)

    unet_runtime_after_attach = _snapshot_module_runtime(getattr(unet_patcher, "model", None))
    cpu_bytes_after_attach = int(unet_runtime_after_attach.get("param_bytes_by_device", {}).get("cpu", 0))

    conditioning, conditioning_info = _build_conditioning(args)

    config = FluxFillConfig(
        unet_path=args.unet_path,
        ae_path=args.ae_path,
        conditioning_cache_path=args.conditioning_path or "",
        tier="fp8_tools_only",
        seed=args.seed,
        steps=args.steps,
        cfg=1.0,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
        guidance=args.guidance,
        device=device,
    )
    precomputed_input = FluxFillPrecomputedDenoiseInput(
        source_latent=source_latent,
        concat_latent=concat_latent,
        denoise_mask=denoise_mask,
        empty_conditioning=conditioning,
        seed=config.seed,
        guidance=config.guidance,
        steps=config.steps,
        cfg=config.cfg,
        sampler=config.sampler,
        scheduler=config.scheduler,
        denoise=config.denoise,
    )
    bundle_description = _describe_tensor_bundle(
        source_latent=source_latent,
        concat_latent=concat_latent,
        denoise_mask=denoise_mask,
        conditioning=conditioning,
    )

    print("Phase 3: Running cold establishment pass...")
    cold_run = _run_decode_pass(
        run_label="cold",
        config=config,
        precomputed_input=precomputed_input,
        unet_patcher=unet_patcher,
        vae=vae,
        ae_path=args.ae_path,
        output_dir=output_dir,
        device=device,
        args=args,
        bundle_description=bundle_description,
        load_models_gpu_wall_ms=cold_establish_load_models_gpu_wall_ms,
    )

    print("Phase 4: Running warm reuse pass with safe_open tracking...")
    with SafeOpenTracker() as safe_open_tracker:
        warm_attach_started = time.perf_counter()
        resources.load_models_gpu([unet_patcher])
        warm_load_models_gpu_wall_ms = (time.perf_counter() - warm_attach_started) * 1000.0
        warm_run = _run_decode_pass(
            run_label="warm",
            config=config,
            precomputed_input=precomputed_input,
            unet_patcher=unet_patcher,
            vae=vae,
            ae_path=args.ae_path,
            output_dir=output_dir,
            device=device,
            args=args,
            bundle_description=bundle_description,
            load_models_gpu_wall_ms=warm_load_models_gpu_wall_ms,
            safe_open_tracker=safe_open_tracker,
        )

    warm_state_reuse_viable = bool(
        warm_run["warm_state"]["safe_open_calls"] == 0 and warm_run["timings"]["load_models_gpu_wall_ms"] < 100.0
    )
    dual_resident_fit = bool(
        (device != "cuda")
        or (
            _is_cuda_device_value(vae_device_after_unet_load)
            and _is_cuda_device_value(cold_run["vae"]["runtime_after_decode"].get("device"))
            and _is_cuda_device_value(warm_run["vae"]["runtime_after_decode"].get("device"))
            and _is_cuda_device_value(cold_run["unet"]["native_unet_runtime_after_denoise"].get("first_param", {}).get("device"))
            and _is_cuda_device_value(warm_run["unet"]["native_unet_runtime_after_denoise"].get("first_param", {}).get("device"))
        )
    )

    summary = {
        "status": "ok",
        "run_name": args.run_name,
        "setup": {
            "device": device,
            "conditioning": conditioning_info,
            "image_path": str(Path(args.image_path)),
            "mask_path": str(Path(args.mask_path)),
            "vae_encode_wall": vae_encode_wall,
            "vae_runtime": vae_metadata,
            "mask_coverage": float(denoise_mask.mean().item()),
            "source_latent_shape": list(source_latent.shape),
            "concat_latent_shape": list(concat_latent.shape),
            "denoise_mask_shape": list(denoise_mask.shape),
            "unet_load": {
                "post_construct_runtime": detected_config.get("post_construct_runtime", {}),
                "post_load_runtime": detected_config.get("post_load_runtime", {}),
                "runtime_after_attach": unet_runtime_after_attach,
                "load_models_gpu_wall_ms": cold_establish_load_models_gpu_wall_ms,
                "cpu_shadow_bytes_after_attach": cpu_bytes_after_attach,
                "cpu_shadow_copy_alive": bool(cpu_bytes_after_attach > 0),
                "vae_device_after_unet_load": _serialize_device(vae_device_after_unet_load),
                "vae_evicted_during_unet_attach": vae_evicted_during_unet_attach,
            },
            "artifact_bundle": {
                "build_count": 1,
                "bundle_recreated_for_warm_run": False,
                "precomputed_input_object_id": id(precomputed_input),
                "bundle_description": bundle_description,
            },
        },
        "cold_run": cold_run,
        "warm_run": warm_run,
        "verdicts": {
            "cpu_shadow_copy_alive": bool(cpu_bytes_after_attach > 0),
            "warm_state_reuse_viable": warm_state_reuse_viable,
            "dual_resident_fit": dual_resident_fit,
            "artifact_bundle_reused_for_warm_run": bool(
                cold_run["artifact_bundle"]["precomputed_input_object_id"]
                == warm_run["artifact_bundle"]["precomputed_input_object_id"]
            ),
            "safe_open_calls_during_warm_run": warm_run["warm_state"]["safe_open_calls"],
            "safe_open_call_details_during_warm_run": warm_run["warm_state"]["safe_open_call_details"],
            "disk_reload_detected_during_warm_run": warm_run["warm_state"]["disk_reload_detected"],
            "warm_run_load_models_gpu_wall_ms": warm_run["timings"]["load_models_gpu_wall_ms"],
        },
    }

    _write_json(output_dir / "summary.json", summary)
    print(f"Summary written to {output_dir / 'summary.json'}")
    print(json.dumps(summary["verdicts"], indent=2))

    _cleanup_model_patcher(vae.patcher, cleanup=True)
    _cleanup_model_patcher(unet_patcher, cleanup=True)
    resources.soft_empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
