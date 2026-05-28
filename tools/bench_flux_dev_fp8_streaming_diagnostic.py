from __future__ import annotations

import argparse
import gc
import json
import os
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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


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
    args = parser.parse_args()

    if not args.conditioning_cache:
        if not args.clip_l or not args.t5:
            parser.error("Either --conditioning-cache or both --clip-l and --t5 must be specified.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Load Native Flux Dev FP8 UNet under CPU Streaming
    print("Loading Flux Dev FP8 Streaming model...")
    unet_load_start = time.perf_counter()
    unet_patcher = load_flux_dev_native_unet_streaming(
        args.unet,
        load_device="cpu",
        offload_device="cpu",
    )
    unet_load_time = time.perf_counter() - unet_load_start
    print(f"Model loaded and scheduler attached in {unet_load_time:.2f}s.")

    # 3. Prepare latent shape and noise
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

    # 4. Execute the Denoise Loop
    print("Running denoise loop...")
    from backend import sampling
    model_options = getattr(unet_patcher, "model_options", {})
    flux_options = model_options.get("flux_dev", {})
    streaming_scheduler = flux_options.get("streaming_scheduler")
    if streaming_scheduler is not None:
        streaming_scheduler.reset_run()

    sampler_instance = sampling.KSampler(
        unet_patcher,
        args.steps,
        device,
        "euler",
        args.scheduler,
        1.0,
        model_options=model_options,
    )

    guider = sampling.prepare_sampler_conds(
        unet_patcher,
        noise,
        positive,
        negative,
        1.0, # CFG = 1.0
        sampler_name="euler",
        latent_image=latent,
        denoise_mask=None,
        seed=args.seed,
        model_options=model_options,
        quality=getattr(sampler_instance, "quality", {}),
        inner_model=getattr(unet_patcher, "model", None),
    )

    denoise_start = time.perf_counter()
    denoise_cpu_start = time.process_time()

    samples = sampling.sample_prepared_sdxl(
        guider,
        noise,
        sampler_instance.sigmas,
        sampler=sampling.ksampler("euler"),
        latent_image=latent,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=args.seed,
        attach_model=False,
    )

    denoise_time = time.perf_counter() - denoise_start
    denoise_cpu_time = time.process_time() - denoise_cpu_start
    print(f"Denoise completed: wall_time={denoise_time:.2f}s, cpu_process_time={denoise_cpu_time:.2f}s.")

    # 5. Extract Telemetry & Scheduler stats
    scheduler_stats = {}
    if streaming_scheduler is not None:
        scheduler_stats = streaming_scheduler.snapshot()
        print("\nStreaming Scheduler Stats:")
        print(f"  Prefetch hits: {scheduler_stats.get('prefetch_hits', 0)}")
        print(f"  Prefetch misses: {scheduler_stats.get('prefetch_misses', 0)}")
        print(f"  Waits / stream syncs: {scheduler_stats.get('stream_waits', 0)}")
        print(f"  Direct copy calls: {scheduler_stats.get('direct_copy_calls', 0)}")
        print(f"  Total transfer bytes: {scheduler_stats.get('prefetch_bytes', 0) + scheduler_stats.get('direct_copy_bytes', 0)} bytes")

    # Check for NaNs
    nan_count = int(torch.isnan(samples).sum().item())
    inf_count = int(torch.isinf(samples).sum().item())
    print(f"\nLatent Quality Checks:")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  Latent shape: {list(samples.shape)}")
    print(f"  Latent mean: {samples.mean().item():.4f}")
    print(f"  Latent std: {samples.std().item():.4f}")

    # 6. Save outputs
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
        },
        "timings": {
            "encode_time_s": encode_time,
            "unet_load_time_s": unet_load_time,
            "denoise_wall_time_s": denoise_time,
            "denoise_cpu_time_s": denoise_cpu_time,
        },
        "scheduler_stats": scheduler_stats,
        "latent_stats": {
            "shape": list(samples.shape),
            "mean": float(samples.mean().item()),
            "std": float(samples.std().item()),
            "min": float(samples.min().item()),
            "max": float(samples.max().item()),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    }

    results_path = output_dir / "diagnostic_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved stats to: {results_path}")

    if args.save_latents:
        latent_path = output_dir / "samples.pt"
        torch.save(samples.detach().cpu(), latent_path)
        print(f"Saved latent tensor to: {latent_path}")

    if args.decode_preview:
        print("Decoding final latent to image...")
        decode_start = time.perf_counter()
        decoded = decode_flux_fill_latent(
            samples,
            args.vae,
            load_device=device,
            cleanup_vae=True,
        )
        decode_time = time.perf_counter() - decode_start
        print(f"Decoded latent in {decode_time:.2f}s.")
        image_path = output_dir / "preview.png"
        Image.fromarray(decoded.bb_image).save(image_path)
        print(f"Saved preview image to: {image_path}")

    # Cleanup UNet patcher
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
