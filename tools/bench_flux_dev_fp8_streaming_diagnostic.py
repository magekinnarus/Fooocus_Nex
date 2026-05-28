from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import resources, patching
from backend.flux.text_conditioning import encode_flux_prompt_conditioning
from backend.flux.flux_fill_loader import (
    _load_flux_fill_native_probe,
    _detect_flux_fill_weight_dtype,
    _estimate_flux_fill_parameter_count,
    _parse_torch_dtype,
    _snapshot_module_runtime,
    _snapshot_first_param_runtime,
    _load_flux_fill_native_weights_into_model,
    load_flux_ae,
)
from backend.flux.flux_fill_pipeline import decode_flux_fill_latent
from backend.flux.flux_streaming import (
    FluxDirectStreamModelPatcher,
    FluxAsyncLayerPrefetchScheduler,
    _resolve_streaming_scheduler_policy,
    _pin_module_tensors_for_streaming,
    measure_pinned_module_tensors,
)
from ldm_patched.modules import model_detection, supported_models_base


def _instantiate_flux_dev_native_model(
    path: Path,
    *,
    offload_device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    from ldm_patched.modules import model_management

    state_probe = _load_flux_fill_native_probe(str(path))
    source_weight_dtype = _detect_flux_fill_weight_dtype(state_probe)
    parameter_count = _estimate_flux_fill_parameter_count(state_probe)
    inference_device = resources.get_torch_device()
    resident_weight_dtype = source_weight_dtype
    if source_weight_dtype is not None:
        resident_weight_dtype = model_management.unet_dtype(
            device=inference_device,
            model_params=parameter_count,
            weight_dtype=source_weight_dtype,
        )
    detected_config = model_detection.detect_unet_config(state_probe, "", dtype=resident_weight_dtype)
    if detected_config is None:
        raise ValueError(f"Could not detect Flux model config from {path}.")

    # Bypass Flux Fill contract verification (allowing in_channels=64 for standard Flux Dev)
    model_config = model_detection.model_config_from_unet_config(detected_config, state_probe)
    if model_config is None:
        raise ValueError(f"No supported Flux model config matched {path}.")

    manual_cast_dtype = None
    if resident_weight_dtype is not None:
        manual_cast_dtype = model_management.unet_manual_cast(resident_weight_dtype, inference_device)
        model_config.set_manual_cast(manual_cast_dtype)
    elif isinstance(model_config, supported_models_base.BASE):
        model_config.set_manual_cast(torch.float16 if inference_device.type == "cuda" else torch.float32)

    model = model_config.get_model(
        state_probe,
        "",
        device=offload_device,
        model_options={"custom_operations": None},
    )
    detected_metadata = dict(detected_config)
    detected_metadata["weight_dtype"] = str(source_weight_dtype) if source_weight_dtype is not None else None
    detected_metadata["source_weight_dtype"] = str(source_weight_dtype) if source_weight_dtype is not None else None
    detected_metadata["resident_weight_dtype"] = str(resident_weight_dtype) if resident_weight_dtype is not None else None
    detected_metadata["manual_cast_dtype"] = str(manual_cast_dtype) if manual_cast_dtype is not None else None
    detected_metadata["parameter_count"] = parameter_count
    detected_metadata["post_construct_runtime"] = _snapshot_module_runtime(model)
    return model, detected_metadata


def load_flux_dev_native_unet_streaming(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
    prefetch_depth: int | None = None,
    max_prefetch_bytes: int | None = None,
    vram_guard_bytes: int | None = None,
    vram_guard_margin_bytes: int | None = None,
    prefetch_scan_ahead: int | None = None,
    bandwidth_limit_mb_s: float | None = None,
) -> Any:
    host_load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    host_offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    if host_load_device.type != "cpu" or host_offload_device.type != "cpu":
        raise RuntimeError("Native Flux streaming loads must stage weights on CPU host memory.")

    compute_device = resources.get_torch_device() if torch.cuda.is_available() else torch.device("cpu")
    path = Path(unet_path)
    model, detected_config = _instantiate_flux_dev_native_model(
        path,
        offload_device=host_offload_device,
    )
    direct_load_metadata = _load_flux_fill_native_weights_into_model(
        path,
        model,
        target_device=host_offload_device,
    )
    runtime_patcher = FluxDirectStreamModelPatcher(
        model,
        load_device=compute_device,
        offload_device=host_offload_device,
        preserve_source_artifact=True,
    )
    runtime_weight_dtype = detected_config.get("resident_weight_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_dev"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8_streaming",
        "runtime_weight_dtype": runtime_weight_dtype,
        "compute_weight_dtype": detected_config.get("manual_cast_dtype"),
        **direct_load_metadata,
    }
    pinned_bytes = _pin_module_tensors_for_streaming(getattr(runtime_patcher, "model", None))
    scheduler_policy = _resolve_streaming_scheduler_policy(
        device=compute_device,
        prefetch_depth=prefetch_depth,
        max_prefetch_bytes=max_prefetch_bytes,
        vram_guard_bytes=vram_guard_bytes,
        vram_guard_margin_bytes=vram_guard_margin_bytes,
        prefetch_scan_ahead=prefetch_scan_ahead,
        bandwidth_limit_mb_s=bandwidth_limit_mb_s,
    )
    streaming_scheduler = FluxAsyncLayerPrefetchScheduler(
        prefetch_depth=scheduler_policy["prefetch_depth"],
        max_prefetch_bytes=scheduler_policy["max_prefetch_bytes"],
        vram_guard_bytes=scheduler_policy["vram_guard_bytes"],
        vram_guard_margin_bytes=scheduler_policy["vram_guard_margin_bytes"],
        prefetch_scan_ahead=scheduler_policy["prefetch_scan_ahead"],
        bandwidth_limit_mb_s=scheduler_policy["bandwidth_limit_mb_s"],
    )
    scheduled_module_count = streaming_scheduler.attach(getattr(runtime_patcher, "model", None))
    flux_options = runtime_patcher.model_options.setdefault("flux_dev", {})
    flux_options["host_pinned_bytes"] = int(max(pinned_bytes, measure_pinned_module_tensors(getattr(runtime_patcher, "model", None))))
    flux_options["non_blocking_supported"] = bool(resources.device_supports_non_blocking(torch.device("cuda"))) if torch.cuda.is_available() else False
    flux_options["single_host_artifact"] = bool(flux_options.get("direct_safetensors_load", False))
    flux_options["streaming_scheduler"] = streaming_scheduler
    flux_options["streaming_scheduler_kind"] = "flux_async_layer_prefetch_v1"
    flux_options["scheduled_module_count"] = int(scheduled_module_count)
    flux_options["direct_stream_runtime"] = True
    flux_options["compute_device"] = str(compute_device)
    flux_options["host_load_device"] = str(host_load_device)
    flux_options["host_offload_device"] = str(host_offload_device)
    flux_options["streaming_scheduler_policy"] = dict(scheduler_policy)
    return runtime_patcher


def load_flux_dev_native_unet_resident(
    unet_path: Path | str,
    *,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    execution_class: Any | None = None,
) -> Any:
    path = Path(unet_path)
    if not path.exists():
        raise FileNotFoundError(f"Flux Dev UNet path does not exist: {path}")

    runtime_load_device = torch.device(load_device) if load_device is not None else resources.get_torch_device()
    runtime_offload_device = torch.device(offload_device) if offload_device is not None else resources.unet_offload_device()
    model, detected_config = _instantiate_flux_dev_native_model(
        path,
        offload_device=runtime_offload_device,
    )
    direct_load_metadata = _load_flux_fill_native_weights_into_model(
        path,
        model,
        target_device=runtime_offload_device,
    )
    detected_config["post_load_runtime"] = _snapshot_module_runtime(model)
    runtime_patcher = patching.NexModelPatcher(
        model,
        load_device=runtime_load_device,
        offload_device=runtime_offload_device,
        preserve_source_artifact=False,
        runtime_weight_dtype_override=_parse_torch_dtype(detected_config.get("resident_weight_dtype")),
    )
    runtime_weight_bytes = int(runtime_patcher.model_size())
    runtime_weight_dtype = detected_config.get("resident_weight_dtype") or detected_config.get("weight_dtype")
    runtime_patcher.model_options["flux_dev"] = {
        "path": str(path),
        "arch": "flux",
        "detected_config": dict(detected_config),
        "execution_class": getattr(execution_class, "value", execution_class),
        "mode": "native_fp8_resident",
        "runtime_weight_dtype": runtime_weight_dtype,
        "runtime_weight_bytes": runtime_weight_bytes,
        "compute_weight_dtype": detected_config.get("manual_cast_dtype"),
        **direct_load_metadata,
    }
    return runtime_patcher


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _summarize_latents(samples: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(samples.shape),
        "mean": float(samples.mean().item()),
        "std": float(samples.std().item()),
        "min": float(samples.min().item()),
        "max": float(samples.max().item()),
        "nan_count": int(torch.isnan(samples).sum().item()),
        "inf_count": int(torch.isinf(samples).sum().item()),
    }


def _summarize_latent_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if list(reference.shape) != list(candidate.shape):
        raise ValueError(
            f"Latent shape mismatch: reference={list(reference.shape)} candidate={list(candidate.shape)}."
        )
    diff = (candidate - reference).to(dtype=torch.float32)
    abs_diff = diff.abs()
    channel_mean_abs = abs_diff.mean(dim=(0, 2, 3))
    channel_max_abs = abs_diff.amax(dim=(0, 2, 3))
    return {
        "shape": list(reference.shape),
        "mean_abs": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "channel_mean_abs": [float(value.item()) for value in channel_mean_abs],
        "channel_max_abs": [float(value.item()) for value in channel_max_abs],
    }


def _cleanup_unet_patcher(unet_patcher: Any) -> None:
    try:
        scheduler = unet_patcher.model_options.get("flux_dev", {}).get("streaming_scheduler")
        if scheduler is not None and hasattr(scheduler, "detach"):
            scheduler.detach()
    except Exception:
        pass
    try:
        resources.eject_model(unet_patcher)
    except Exception:
        detach = getattr(unet_patcher, "detach", None)
        if callable(detach):
            detach()
    resources.soft_empty_cache()
    gc.collect()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_run_summary(results: dict[str, Any], mode_name: str) -> dict[str, Any]:
    run_entry = dict(results.get("runs", {}).get(mode_name, {}) or {})
    if not run_entry:
        run_entry = {
            "mode": mode_name,
            "timings": dict(results.get("timings", {}) or {}),
            "scheduler_stats": dict(results.get("scheduler_stats", {}) or {}),
            "latent_stats": dict(results.get("latent_stats", {}) or {}),
        }
    return run_entry


def _run_subprocess_mode(
    args: argparse.Namespace,
    *,
    mode_name: str,
    output_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    child_dir = output_dir / mode_name
    child_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--unet",
        str(args.unet),
        "--vae",
        str(args.vae),
        "--prompt",
        str(args.prompt),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(args.steps),
        "--guidance",
        str(args.guidance),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(child_dir),
        "--scheduler",
        str(args.scheduler),
        "--comparison-mode",
        mode_name,
        "--save-latents",
    ]
    if args.prefetch_depth is not None:
        command.extend(["--prefetch-depth", str(args.prefetch_depth)])
    if args.prefetch_max_mb is not None:
        command.extend(["--prefetch-max-mb", str(args.prefetch_max_mb)])
    if args.vram_guard_mb is not None:
        command.extend(["--vram-guard-mb", str(args.vram_guard_mb)])
    if args.vram_guard_margin_mb is not None:
        command.extend(["--vram-guard-margin-mb", str(args.vram_guard_margin_mb)])
    if args.prefetch_scan_ahead is not None:
        command.extend(["--prefetch-scan-ahead", str(args.prefetch_scan_ahead)])
    if args.bandwidth_limit_mb_s is not None:
        command.extend(["--bandwidth-limit-mb-s", str(args.bandwidth_limit_mb_s)])
    if args.conditioning_cache:
        command.extend(["--conditioning-cache", str(args.conditioning_cache)])
    else:
        if args.clip_l:
            command.extend(["--clip-l", str(args.clip_l)])
        if args.t5:
            command.extend(["--t5", str(args.t5)])
    if args.decode_preview:
        command.append("--decode-preview")

    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{mode_name} subprocess run failed with exit code {completed.returncode}.")

    results_path = child_dir / "diagnostic_results.json"
    latent_path = child_dir / "samples.pt"
    if not results_path.exists():
        raise FileNotFoundError(f"{mode_name} subprocess did not produce {results_path}.")
    if not latent_path.exists():
        raise FileNotFoundError(f"{mode_name} subprocess did not produce {latent_path}.")

    results = _load_json(results_path)
    samples = torch.load(latent_path, map_location="cpu", weights_only=True)
    run_summary = _single_run_summary(results, mode_name)
    artifacts = dict(run_summary.get("artifacts", {}) or {})
    artifacts.setdefault("results_path", str(results_path))
    artifacts.setdefault("latent_path", str(latent_path))
    run_summary["artifacts"] = artifacts
    return run_summary, samples


def _run_combined_comparison(args: argparse.Namespace, *, output_dir: Path) -> int:
    print("Running resident and streaming comparison in isolated subprocesses...")
    compare_start = time.perf_counter()
    resident_summary, resident_samples = _run_subprocess_mode(
        args,
        mode_name="resident",
        output_dir=output_dir,
    )
    streaming_summary, streaming_samples = _run_subprocess_mode(
        args,
        mode_name="streaming",
        output_dir=output_dir,
    )
    latent_diff = _summarize_latent_diff(resident_samples, streaming_samples)
    results = {
        "config": {
            "unet": str(args.unet),
            "clip_l": str(args.clip_l),
            "t5": str(args.t5),
            "vae": str(args.vae),
            "prompt": args.prompt,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "comparison_mode": "both",
            "execution_strategy": "isolated_subprocesses",
            "scheduler_overrides": {
                "prefetch_depth": args.prefetch_depth,
                "prefetch_max_mb": args.prefetch_max_mb,
                "vram_guard_mb": args.vram_guard_mb,
                "vram_guard_margin_mb": args.vram_guard_margin_mb,
                "prefetch_scan_ahead": args.prefetch_scan_ahead,
                "bandwidth_limit_mb_s": args.bandwidth_limit_mb_s,
            },
        },
        "timings": {
            "combined_wall_time_s": float(time.perf_counter() - compare_start),
            "mode_count": 2,
        },
        "runs": {
            "resident": resident_summary,
            "streaming": streaming_summary,
        },
        "latent_diff": latent_diff,
    }
    results_path = output_dir / "diagnostic_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved combined comparison stats to: {results_path}")
    print(
        "Combined latent diff: "
        f"mean_abs={latent_diff['mean_abs']:.6f}, max_abs={latent_diff['max_abs']:.6f}, rmse={latent_diff['rmse']:.6f}"
    )
    return 0


def _run_denoise_mode(
    *,
    mode_name: str,
    unet_patcher: Any,
    device: torch.device,
    latent: torch.Tensor,
    noise: torch.Tensor,
    positive: list[list[Any]],
    negative: list[list[Any]],
    steps: int,
    scheduler_name: str,
    seed: int,
    attach_model: bool,
) -> dict[str, Any]:
    from backend import sampling

    model_options = getattr(unet_patcher, "model_options", {})
    flux_options = model_options.get("flux_dev", {})
    streaming_scheduler = flux_options.get("streaming_scheduler")
    if streaming_scheduler is not None:
        streaming_scheduler.reset_run()

    sampler_instance = sampling.KSampler(
        unet_patcher,
        steps,
        device,
        "euler",
        scheduler_name,
        1.0,
        model_options=model_options,
    )

    guider = sampling.prepare_sampler_conds(
        unet_patcher,
        noise.clone(),
        positive,
        negative,
        1.0,
        sampler_name="euler",
        latent_image=latent.clone(),
        denoise_mask=None,
        seed=seed,
        model_options=model_options,
        quality=getattr(sampler_instance, "quality", {}),
        inner_model=getattr(unet_patcher, "model", None),
    )

    denoise_start = time.perf_counter()
    denoise_cpu_start = time.process_time()
    samples = sampling.sample_prepared_sdxl(
        guider,
        noise.clone(),
        sampler_instance.sigmas,
        sampler=sampling.ksampler("euler"),
        latent_image=latent.clone(),
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=seed,
        attach_model=attach_model,
    )
    denoise_time = time.perf_counter() - denoise_start
    denoise_cpu_time = time.process_time() - denoise_cpu_start

    scheduler_stats = {}
    if streaming_scheduler is not None:
        scheduler_stats = streaming_scheduler.snapshot()

    samples_cpu = samples.detach().cpu()
    del samples

    return {
        "mode": mode_name,
        "attach_model": bool(attach_model),
        "timings": {
            "denoise_wall_time_s": float(denoise_time),
            "denoise_cpu_time_s": float(denoise_cpu_time),
        },
        "scheduler_stats": scheduler_stats,
        "latent_stats": _summarize_latents(samples_cpu),
        "sampling_perf": dict(model_options.get("_nex_sampling_perf", {}) or {}),
        "samples_cpu": samples_cpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tools-only Flux Dev fp8 streaming diagnostic harness.")
    parser.add_argument("--unet", required=True, help="Path to standard Flux Dev fp8 safetensors checkpoint.")
    parser.add_argument("--clip-l", default=None, help="Path to clip_l.safetensors.")
    parser.add_argument("--t5", default=None, help="Path to t5xxl_fp16.safetensors.")
    parser.add_argument("--vae", required=True, help="Path to ae.safetensors VAE.")
    parser.add_argument("--prompt", default="a beautiful face of European girl, detailed face, blonde hair", help="Text prompt.")
    parser.add_argument("--conditioning-cache", default=None, help="Path to precomputed conditioning .pt file (skips T5/CLIP-L encoding).")
    parser.add_argument("--width", type=int, default=256, help="Image width (pixels).")
    parser.add_argument("--height", type=int, default=256, help="Image height (pixels).")
    parser.add_argument("--steps", type=int, default=10, help="Denoise step count.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value for Flux Dev.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs/P4-M13-W08c3_local_diagnostic", help="Output directory.")
    parser.add_argument("--save-latents", action="store_true", help="Save denoised latent PT file.")
    parser.add_argument("--decode-preview", action="store_true", help="Decode preview image.")
    parser.add_argument("--scheduler", default="normal", help="Scheduler to use (e.g., normal, beta, simple, karras).")
    parser.add_argument("--prefetch-depth", type=int, default=None, help="Streaming prefetch depth override.")
    parser.add_argument("--prefetch-max-mb", type=float, default=None, help="Streaming max prefetch window in MB.")
    parser.add_argument("--vram-guard-mb", type=float, default=None, help="Streaming VRAM guard override in MB.")
    parser.add_argument("--vram-guard-margin-mb", type=float, default=None, help="Streaming VRAM guard margin override in MB.")
    parser.add_argument("--prefetch-scan-ahead", type=int, default=None, help="Streaming scheduler scan-ahead override.")
    parser.add_argument("--bandwidth-limit-mb-s", type=float, default=None, help="Streaming bandwidth cap in MB/s; omit for open bandwidth.")
    parser.add_argument(
        "--comparison-mode",
        choices=("streaming", "resident", "both"),
        default="streaming",
        help="Which execution lane to run. Use 'both' to produce resident-vs-streaming latent diffs.",
    )
    args = parser.parse_args()

    if not args.conditioning_cache:
        if not args.clip_l or not args.t5:
            parser.error("Either --conditioning-cache or both --clip-l and --t5 must be specified.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.comparison_mode == "both":
        return _run_combined_comparison(args, output_dir=output_dir)

    device = resources.get_torch_device()
    print(f"Using compute device: {device}")

    # 1. Dynamic Text Conditioning Prepare
    if args.conditioning_cache:
        print(f"Loading precomputed prompt conditioning from {args.conditioning_cache}...")
        from backend.flux.flux_fill_pipeline import load_flux_empty_conditioning_cache
        t_start = time.perf_counter()
        conditioning = load_flux_empty_conditioning_cache(args.conditioning_cache, map_location="cpu")
        encode_time = time.perf_counter() - t_start
        print(f"Conditioning loaded in {encode_time:.2f}s.")
    else:
        print("Encoding prompt conditioning...")
        t_start = time.perf_counter()
        conditioning = encode_flux_prompt_conditioning(
            prompt=args.prompt,
            clip_l_path=args.clip_l,
            t5_path=args.t5,
            t5_loader_policy="stream_safetensors_runtime",
        )
        encode_time = time.perf_counter() - t_start
        print(f"Conditioning encoded in {encode_time:.2f}s.")

    cross_attn = conditioning.cross_attn.to(device=device, dtype=torch.float32)
    pooled_output = conditioning.pooled_output.to(device=device, dtype=torch.float32)

    # 2. Prepare latent shape and noise
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent = torch.zeros((1, 16, latent_h, latent_w), dtype=torch.float32, device=device)
    print(f"Prepared latent of shape {list(latent.shape)} ({args.width}x{args.height} resolution).")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device)

    guidance_tensor = torch.tensor([args.guidance], dtype=torch.float32, device=device)
    positive = [[
        cross_attn,
        {
            "model_conds": {
                "pooled_output": pooled_output,
                "guidance": guidance_tensor,
            }
        }
    ]]
    negative = [[
        torch.zeros_like(cross_attn),
        {
            "model_conds": {
                "pooled_output": torch.zeros_like(pooled_output),
                "guidance": guidance_tensor.clone(),
            }
        }
    ]]

    run_plan: list[tuple[str, Any, bool]] = []
    if args.comparison_mode in ("resident", "both"):
        run_plan.append(
            (
                "resident",
                lambda: load_flux_dev_native_unet_resident(args.unet),
                True,
            )
        )
    if args.comparison_mode in ("streaming", "both"):
        run_plan.append(
            (
                "streaming",
                lambda: load_flux_dev_native_unet_streaming(
                    args.unet,
                    load_device="cpu",
                    offload_device="cpu",
                    prefetch_depth=args.prefetch_depth,
                    max_prefetch_bytes=int(args.prefetch_max_mb * 1024 * 1024) if args.prefetch_max_mb is not None else None,
                    vram_guard_bytes=int(args.vram_guard_mb * 1024 * 1024) if args.vram_guard_mb is not None else None,
                    vram_guard_margin_bytes=int(args.vram_guard_margin_mb * 1024 * 1024) if args.vram_guard_margin_mb is not None else None,
                    prefetch_scan_ahead=args.prefetch_scan_ahead,
                    bandwidth_limit_mb_s=args.bandwidth_limit_mb_s,
                ),
                False,
            )
        )

    mode_results: dict[str, dict[str, Any]] = {}
    sample_artifacts: dict[str, torch.Tensor] = {}
    for mode_name, loader_fn, attach_model in run_plan:
        print(f"Loading Flux Dev FP8 {mode_name} model...")
        unet_load_start = time.perf_counter()
        unet_patcher = loader_fn()
        unet_load_time = time.perf_counter() - unet_load_start
        print(f"{mode_name} model loaded in {unet_load_time:.2f}s.")
        try:
            print(f"Running {mode_name} denoise loop...")
            run_result = _run_denoise_mode(
                mode_name=mode_name,
                unet_patcher=unet_patcher,
                device=device,
                latent=latent,
                noise=noise,
                positive=positive,
                negative=negative,
                steps=args.steps,
                scheduler_name=args.scheduler,
                seed=args.seed,
                attach_model=attach_model,
            )
        finally:
            _cleanup_unet_patcher(unet_patcher)

        run_result["timings"]["unet_load_time_s"] = float(unet_load_time)
        sample_artifacts[mode_name] = run_result.pop("samples_cpu")
        mode_results[mode_name] = run_result

        latent_stats = run_result["latent_stats"]
        print(f"{mode_name} latent quality checks:")
        print(f"  NaN count: {latent_stats['nan_count']}")
        print(f"  Inf count: {latent_stats['inf_count']}")
        print(f"  Latent shape: {latent_stats['shape']}")
        print(f"  Latent mean: {latent_stats['mean']:.4f}")
        print(f"  Latent std: {latent_stats['std']:.4f}")
        scheduler_stats = run_result.get("scheduler_stats", {})
        if scheduler_stats:
            print(f"{mode_name} scheduler stats:")
            print(f"  Prefetch hits: {scheduler_stats.get('prefetch_hits', 0)}")
            print(f"  Prefetch misses: {scheduler_stats.get('prefetch_misses', 0)}")
            print(f"  Waits / stream syncs: {scheduler_stats.get('stream_waits', 0)}")
            print(f"  Direct copy calls: {scheduler_stats.get('direct_copy_calls', 0)}")
            print(f"  Total transfer bytes: {scheduler_stats.get('prefetch_bytes', 0) + scheduler_stats.get('direct_copy_bytes', 0)} bytes")

    latent_diff = None
    if "resident" in sample_artifacts and "streaming" in sample_artifacts:
        latent_diff = _summarize_latent_diff(sample_artifacts["resident"], sample_artifacts["streaming"])

    results = {
        "config": {
            "unet": str(args.unet),
            "clip_l": str(args.clip_l),
            "t5": str(args.t5),
            "vae": str(args.vae),
            "prompt": args.prompt,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "scheduler_overrides": {
                "prefetch_depth": args.prefetch_depth,
                "prefetch_max_mb": args.prefetch_max_mb,
                "vram_guard_mb": args.vram_guard_mb,
                "vram_guard_margin_mb": args.vram_guard_margin_mb,
                "prefetch_scan_ahead": args.prefetch_scan_ahead,
                "bandwidth_limit_mb_s": args.bandwidth_limit_mb_s,
            },
            "comparison_mode": args.comparison_mode,
        },
        "timings": {
            "encode_time_s": encode_time,
            "mode_count": len(mode_results),
        },
        "runs": mode_results,
    }
    if latent_diff is not None:
        results["latent_diff"] = latent_diff

    if len(mode_results) == 1:
        only_mode = next(iter(mode_results.values()))
        results["timings"].update(only_mode.get("timings", {}))
        results["scheduler_stats"] = only_mode.get("scheduler_stats", {})
        results["latent_stats"] = only_mode.get("latent_stats", {})

    results_path = output_dir / "diagnostic_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved stats to: {results_path}")

    for mode_name, samples_cpu in sample_artifacts.items():
        suffix = "" if len(sample_artifacts) == 1 else f"_{mode_name}"
        if args.save_latents:
            latent_path = output_dir / f"samples{suffix}.pt"
            torch.save(samples_cpu, latent_path)
            mode_results[mode_name].setdefault("artifacts", {})["latent_path"] = str(latent_path)
            print(f"Saved {mode_name} latent tensor to: {latent_path}")

        if args.decode_preview:
            print(f"Decoding {mode_name} latent to image...")
            decode_start = time.perf_counter()
            decoded = decode_flux_fill_latent(
                samples_cpu,
                args.vae,
                load_device=device,
                cleanup_vae=True,
            )
            decode_time = time.perf_counter() - decode_start
            mode_results[mode_name]["timings"]["decode_time_s"] = float(decode_time)
            image_path = output_dir / f"preview{suffix}.png"
            Image.fromarray(decoded.bb_image).save(image_path)
            mode_results[mode_name].setdefault("artifacts", {})["preview_path"] = str(image_path)
            print(f"Decoded {mode_name} latent in {decode_time:.2f}s.")
            print(f"Saved {mode_name} preview image to: {image_path}")

    results_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    gc.collect()
    resources.soft_empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
