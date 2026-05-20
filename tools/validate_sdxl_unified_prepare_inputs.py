import argparse
import contextlib
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate unified SDXL prepare_inputs() against a real CPU-owned checkpoint workload.",
    )
    parser.add_argument("--checkpoint-path", default="", help="Optional fp16 SDXL checkpoint override.")
    parser.add_argument(
        "--lora",
        dest="lora_specs",
        action="append",
        default=[],
        help="Optional LoRA spec in the form path[:weight]. Repeat for multiple LoRAs.",
    )
    parser.add_argument(
        "--auto-lora-counts",
        nargs="+",
        type=int,
        default=(0, 1, 3),
        help="Validation rows to run. Counts above discovered LoRA availability are skipped.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--clip-layer", type=int, default=-2)
    parser.add_argument(
        "--pin-base-unet-without-lora",
        action="store_true",
        help="Explicitly pin the base UNet even when no LoRA stack is requested.",
    )
    parser.add_argument(
        "--conditioning-breakdown",
        action="store_true",
        help="Run a diagnostic-only prompt-conditioning breakdown instead of full prepare_inputs().",
    )
    parser.add_argument(
        "--conditioning-attention-mode",
        choices=("current", "basic"),
        default="current",
        help="Diagnostic-only CLIP attention override for conditioning breakdown runs.",
    )
    parser.add_argument(
        "--conditioning-clip-weight-mode",
        choices=("mixed", "fp32"),
        default="mixed",
        help="Diagnostic-only resident CLIP weight mode for conditioning breakdown runs.",
    )
    parser.add_argument("--report-json", default="", help="Optional output JSON path.")
    return parser.parse_args()


def _resolve_lora_pool(explicit_specs: list[str]) -> list[tuple[str, float]]:
    from tools.bench_sdxl_pinned_residency_matrix import _as_folder_list, _parse_lora_spec

    if explicit_specs:
        return [_parse_lora_spec(spec) for spec in explicit_specs]

    import modules.config as config

    folders = _as_folder_list(getattr(config, "paths_lora_lookup", []))
    candidates: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for suffix in ("*.safetensors", "*.ckpt"):
            for candidate in root.rglob(suffix):
                resolved = str(candidate)
                normalized = resolved.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(resolved)

    preferred = [path for path in candidates if _looks_sdxl_lora(path)]
    selected = preferred if preferred else candidates
    return [(path, 1.0) for path in selected]


def _looks_sdxl_lora(path: str) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    if "/sd15/" in normalized or "sd1.5" in normalized:
        return False
    if "/faceid" in normalized or "ip-adapter" in normalized:
        return False
    return any(
        token in normalized
        for token in (
            "/sdxl/",
            "sd_xl",
            "_xl",
            "/xl/",
            "illustrious",
            "pony",
        )
    )


def _resolve_fp16_checkpoint(override: str = "") -> str:
    import modules.config as config
    from tools.bench_sdxl_pinned_residency_matrix import _as_folder_list, _resolve_local_model_path

    folders = _as_folder_list(getattr(config, "paths_checkpoints", []))
    if override:
        resolved = _resolve_local_model_path(str(override), folders)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Resolved checkpoint does not exist: {resolved}")
        return resolved

    selected = getattr(config, "default_base_model_name", None) or getattr(config, "default_model", None)
    if selected and str(selected).lower() != "none":
        resolved = _resolve_local_model_path(str(selected), folders)
        if os.path.isfile(resolved) and not resolved.lower().endswith(".gguf"):
            taxonomy = config.resolve_model_taxonomy(resolved, root_keys=("checkpoints",), folder_paths=folders)
            if taxonomy.architecture == "sdxl":
                return resolved

    candidates = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        for suffix in ("*.safetensors", "*.ckpt"):
            candidates.extend(root.rglob(suffix))

    seen = set()
    for candidate in candidates:
        resolved = str(candidate)
        normalized = resolved.lower()
        if normalized.endswith(".gguf") or normalized in seen:
            continue
        seen.add(normalized)
        taxonomy = config.resolve_model_taxonomy(resolved, root_keys=("checkpoints",), folder_paths=folders)
        if taxonomy.architecture == "sdxl":
            return resolved
        if "sdxl" in normalized or "sdxl" in str(candidate.parent).lower():
            return resolved

    raise FileNotFoundError("No fp16/non-GGUF SDXL checkpoint was found in config.txt checkpoint roots.")


def _first_param_dtype(module: Any) -> str | None:
    if module is None:
        return None
    for tensor in module.parameters():
        if isinstance(tensor, torch.Tensor):
            return str(tensor.dtype)
    return None


def _float_parameter_dtypes(module: Any) -> list[str]:
    if module is None:
        return []
    dtypes: set[str] = set()
    for tensor in module.parameters():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            dtypes.add(str(tensor.dtype))
    return sorted(dtypes)


def _float_buffer_dtypes(module: Any) -> list[str]:
    if module is None:
        return []
    dtypes: set[str] = set()
    for tensor in module.buffers():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            dtypes.add(str(tensor.dtype))
    return sorted(dtypes)


def _build_runtime_config(
    checkpoint_path: str,
    lora_specs: tuple[tuple[str, float], ...],
    args: argparse.Namespace,
) -> Any:
    from backend.sdxl_unified_runtime import UnifiedSDXLRuntimeConfig
    from modules.gguf_headless_runner import DEFAULT_NEGATIVE_PROMPT, DEFAULT_POSITIVE_PROMPT

    return UnifiedSDXLRuntimeConfig(
        model_variant="sdxl",
        execution_class="cpu-first-validation",
        checkpoint_path=checkpoint_path,
        prompt=DEFAULT_POSITIVE_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        seed=args.seed,
        clip_layer=args.clip_layer,
        batch_size=1,
        lora_specs=lora_specs,
        pin_base_unet_without_lora=bool(args.pin_base_unet_without_lora),
    )


def _run_case(
    checkpoint_path: str,
    lora_specs: tuple[tuple[str, float], ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.conditioning_breakdown:
        return _run_conditioning_breakdown_case(checkpoint_path, lora_specs, args)

    from backend.sdxl_unified_runtime import UnifiedSDXLRuntime
    from tools.bench_sdxl_pinned_residency_matrix import MemorySampler, _capture_phase_memory

    runtime = UnifiedSDXLRuntime(_build_runtime_config(checkpoint_path, lora_specs, args))
    phase_memory = []
    start = time.perf_counter()
    with MemorySampler() as memory:
        load_wall = runtime.load_components()
        phase_memory.append(asdict(_capture_phase_memory("after_load_components")))

        prepared_inputs, metrics = runtime.prepare_inputs()
        phase_memory.append(asdict(_capture_phase_memory("after_prepare_inputs")))

        compiled_unet = prepared_inputs.compiled_unet
        conditioning = prepared_inputs.conditioning
        placeholder = prepared_inputs.injected_features.get("feature_boundary_placeholder")

        result = {
            "status": "ok",
            "checkpoint_path": checkpoint_path,
            "lora_specs": list(lora_specs),
            "lora_count": len(lora_specs),
            "load_wall": float(load_wall),
            "prepare_wall": float(time.perf_counter() - start),
            "metrics": metrics,
            "base_model_loaded": bool(prepared_inputs.base_model and prepared_inputs.base_model.loaded),
            "compiled_unet_present": compiled_unet is not None,
            "conditioning_present": conditioning is not None,
            "gpu_boundary_present": prepared_inputs.gpu_attached_execution_state is not None,
            "injected_feature_placeholder_present": placeholder is not None,
            "injected_feature_placeholder_context": getattr(placeholder, "context_key", None),
            "unet_parameter_dtypes": _float_parameter_dtypes(getattr(runtime.unet, "model", None)),
            "clip_parameter_dtypes": _float_parameter_dtypes(getattr(getattr(runtime.clip, "patcher", None), "model", None)),
            "unet_buffer_dtypes": _float_buffer_dtypes(getattr(runtime.unet, "model", None)),
            "clip_buffer_dtypes": _float_buffer_dtypes(getattr(getattr(runtime.clip, "patcher", None), "model", None)),
            "unet_first_dtype": _first_param_dtype(getattr(runtime.unet, "model", None)),
            "clip_first_dtype": _first_param_dtype(getattr(getattr(runtime.clip, "patcher", None), "model", None)),
            "compiled_unet_fingerprint": getattr(compiled_unet, "artifact_fingerprint", None),
            "conditioning_prompt_fingerprint": getattr(conditioning, "prompt_fingerprint", None),
            "conditioning_payload_fingerprint": getattr(conditioning, "conditioning_fingerprint", None),
            "compiled_unet_cpu_mb": getattr(compiled_unet, "pinned_cpu_mb", None),
            "conditioning_positive_dtype": str(prepared_inputs.payload["encoded_prompt_pair"]["positive"]["cond"].dtype),
            "conditioning_negative_dtype": str(prepared_inputs.payload["encoded_prompt_pair"]["negative"]["cond"].dtype),
            "pooled_positive_dtype": str(prepared_inputs.payload["encoded_prompt_pair"]["positive"]["pooled"].dtype),
            "pooled_negative_dtype": str(prepared_inputs.payload["encoded_prompt_pair"]["negative"]["pooled"].dtype),
            "adm_positive_dtype": str(prepared_inputs.payload["adm_pair"]["positive"].dtype),
            "adm_negative_dtype": str(prepared_inputs.payload["adm_pair"]["negative"].dtype),
        }
        result["fp16_parameter_preserved"] = (
            result["unet_parameter_dtypes"] == ["torch.float16"]
        )

        runtime.close()
        gc.collect()
        phase_memory.append(asdict(_capture_phase_memory("after_close")))

        result["peak_rss_bytes"] = memory.snapshot.peak_rss_bytes
        result["peak_vram_reserved_bytes"] = memory.snapshot.peak_vram_reserved_bytes
        result["phase_memory"] = phase_memory
        return result


def _run_conditioning_breakdown_case(
    checkpoint_path: str,
    lora_specs: tuple[tuple[str, float], ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from backend import clip as backend_clip, conditioning
    from backend.sdxl_unified_runtime import UnifiedSDXLRuntime
    from tools.bench_sdxl_pinned_residency_matrix import MemorySampler, _capture_phase_memory

    runtime = UnifiedSDXLRuntime(_build_runtime_config(checkpoint_path, lora_specs, args))
    phase_memory = []
    start = time.perf_counter()
    with _temporary_clip_attention_override(backend_clip, args.conditioning_attention_mode):
        with MemorySampler() as memory:
            load_wall = runtime.load_components()
            phase_memory.append(asdict(_capture_phase_memory("after_load_components")))

            _apply_clip_weight_mode(runtime, args.conditioning_clip_weight_mode)
            phase_memory.append(asdict(_capture_phase_memory("after_clip_weight_mode")))

            lora_metrics = runtime._materialize_lora_stack()
            phase_memory.append(asdict(_capture_phase_memory("after_lora_materialize")))

            positive_tokens = runtime.clip.tokenize(runtime.config.prompt)
            phase_memory.append(asdict(_capture_phase_memory("after_positive_tokenize")))

            negative_tokens = runtime.clip.tokenize(runtime.config.negative_prompt)
            phase_memory.append(asdict(_capture_phase_memory("after_negative_tokenize")))

            positive_cond, positive_pooled = conditioning.encode_tokens_sdxl(
                runtime.clip,
                positive_tokens,
                use_explicit_residency=True,
            )
            phase_memory.append(asdict(_capture_phase_memory("after_positive_encode")))

            negative_cond, negative_pooled = conditioning.encode_tokens_sdxl(
                runtime.clip,
                negative_tokens,
                use_explicit_residency=True,
            )
            phase_memory.append(asdict(_capture_phase_memory("after_negative_encode")))

            encoded_prompt_pair = {
                "positive": {"cond": positive_cond, "pooled": positive_pooled},
                "negative": {"cond": negative_cond, "pooled": negative_pooled},
            }

            adm_pair = conditioning.build_sdxl_adm_pair(
                encoded_prompt_pair,
                runtime.config.width,
                runtime.config.height,
                target_width=runtime.config.width,
                target_height=runtime.config.height,
            )
            phase_memory.append(asdict(_capture_phase_memory("after_adm_build")))

            prompt_stage = conditioning.build_sdxl_text_conditioning_fingerprint(
                prompt=runtime.config.prompt,
                negative_prompt=runtime.config.negative_prompt,
                model_identity=runtime.base_model.fingerprint or runtime.base_model.source_path or runtime.config.model_variant,
                text_encoder_identity=runtime._clip_identity,
                clip_patch_uuid=runtime._lora_signature(),
                clip_layer_idx=runtime.config.clip_layer,
                lora_artifacts_state=runtime._resolved_lora_specs,
                route_family_reconciliation_signature=(runtime.route_label, runtime.seams.compiled_unet_owner),
                residency_class="cpu",
                route_family=runtime.route_label,
                execution_family=runtime.config.execution_class,
                clip_residency_mode="cpu",
            )
            phase_memory.append(asdict(_capture_phase_memory("after_prompt_fingerprint")))

            conditioning_fingerprint = runtime._hash_payload(
                {
                    "positive": encoded_prompt_pair["positive"],
                    "negative": encoded_prompt_pair["negative"],
                    "adm_pair": adm_pair,
                }
            )
            phase_memory.append(asdict(_capture_phase_memory("after_conditioning_hash")))

            pooled_fingerprint = runtime._hash_payload(
                {
                    "positive": encoded_prompt_pair["positive"]["pooled"],
                    "negative": encoded_prompt_pair["negative"]["pooled"],
                }
            )
            phase_memory.append(asdict(_capture_phase_memory("after_pooled_hash")))

            result = {
                "status": "ok",
                "mode": "conditioning_breakdown",
                "attention_mode": args.conditioning_attention_mode,
                "clip_weight_mode": args.conditioning_clip_weight_mode,
                "checkpoint_path": checkpoint_path,
                "lora_specs": list(lora_specs),
                "lora_count": len(lora_specs),
                "load_wall": float(load_wall),
                "diagnostic_wall": float(time.perf_counter() - start),
                "lora_metrics": lora_metrics,
                "clip_parameter_dtypes": _float_parameter_dtypes(getattr(getattr(runtime.clip, "patcher", None), "model", None)),
                "clip_first_dtype": _first_param_dtype(getattr(getattr(runtime.clip, "patcher", None), "model", None)),
                "positive_cond_dtype": str(positive_cond.dtype),
                "negative_cond_dtype": str(negative_cond.dtype),
                "positive_pooled_dtype": str(positive_pooled.dtype),
                "negative_pooled_dtype": str(negative_pooled.dtype),
                "adm_positive_dtype": str(adm_pair["positive"].dtype),
                "adm_negative_dtype": str(adm_pair["negative"].dtype),
                "positive_cond_shape": list(positive_cond.shape),
                "negative_cond_shape": list(negative_cond.shape),
                "positive_pooled_shape": list(positive_pooled.shape),
                "negative_pooled_shape": list(negative_pooled.shape),
                "prompt_fingerprint": prompt_stage.digest(),
                "conditioning_fingerprint": conditioning_fingerprint,
                "pooled_fingerprint": pooled_fingerprint,
            }

            del adm_pair
            del encoded_prompt_pair
            del positive_cond
            del negative_cond
            del positive_pooled
            del negative_pooled
            gc.collect()
            phase_memory.append(asdict(_capture_phase_memory("after_breakdown_gc")))

            runtime.close()
            gc.collect()
            phase_memory.append(asdict(_capture_phase_memory("after_close")))

            result["peak_rss_bytes"] = memory.snapshot.peak_rss_bytes
            result["peak_vram_reserved_bytes"] = memory.snapshot.peak_vram_reserved_bytes
            result["phase_memory"] = phase_memory
            return result


def _basic_clip_attention(q, k, v, heads, mask=None):
    batch = q.shape[0]
    dim_head = q.shape[2] // heads
    scale = dim_head ** -0.5
    q, k, v = map(lambda t: t.reshape(batch, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2).reshape(batch, -1, heads * dim_head)


@contextlib.contextmanager
def _temporary_clip_attention_override(backend_clip_module, mode: str):
    if mode == "current":
        yield
        return

    original = backend_clip_module.optimized_attention
    try:
        if mode == "basic":
            backend_clip_module.optimized_attention = _basic_clip_attention
        else:
            raise ValueError(f"Unsupported attention mode: {mode}")
        yield
    finally:
        backend_clip_module.optimized_attention = original


def _apply_clip_weight_mode(runtime, mode: str) -> None:
    clip_model = getattr(getattr(runtime, "clip", None), "patcher", None)
    clip_model = getattr(clip_model, "model", None)
    if clip_model is None or mode == "mixed":
        return
    if mode == "fp32":
        clip_model.to(dtype=torch.float32)
        return
    raise ValueError(f"Unsupported CLIP weight mode: {mode}")


def main() -> int:
    args = parse_args()
    sys.argv = [sys.argv[0]]
    from tools.bench_sdxl_pinned_residency_matrix import _json_default

    checkpoint_path = _resolve_fp16_checkpoint(args.checkpoint_path)
    lora_pool = _resolve_lora_pool(args.lora_specs)

    results: list[dict[str, Any]] = []
    had_failure = False
    for requested_count in sorted(set(int(count) for count in args.auto_lora_counts)):
        if requested_count < 0:
            continue
        if requested_count > len(lora_pool):
            results.append(
                {
                    "status": "skipped",
                    "checkpoint_path": checkpoint_path,
                    "lora_count": requested_count,
                    "reason": f"Requested {requested_count} LoRAs but only {len(lora_pool)} were available.",
                }
            )
            continue
        active_loras = tuple(lora_pool[:requested_count])
        try:
            results.append(_run_case(checkpoint_path, active_loras, args))
        except Exception as exc:
            had_failure = True
            results.append(
                {
                    "status": "error",
                    "checkpoint_path": checkpoint_path,
                    "lora_count": requested_count,
                    "lora_specs": list(active_loras),
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
            )

    summary = {
        "checkpoint_path": checkpoint_path,
        "available_lora_pool": [path for path, _ in lora_pool],
        "results": results,
    }

    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=_json_default))
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
