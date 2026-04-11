from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMPORTANT_FLUX_KEYS = (
    "img_in.weight",
    "txt_in.weight",
    "vector_in.in_layer.weight",
    "guidance_in.in_layer.weight",
    "final_layer.linear.weight",
    "double_blocks.0.img_attn.norm.key_norm.scale",
    "double_blocks.0.img_attn.qkv.weight",
    "single_blocks.0.linear1.weight",
)


def _json_default(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Size):
            return list(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _shape_of(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "tensor_shape", None)
    if shape is None:
        shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _qtype_of(tensor: Any) -> str | None:
    tensor_type = getattr(tensor, "tensor_type", None)
    if tensor_type is None:
        return None
    return getattr(tensor_type, "name", str(tensor_type))


def _count_block_indices(state_dict: dict[str, Any], prefix: str) -> int:
    indices = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return len(indices)


def _short_error(exc: BaseException) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Flux Fill GGUF detection/construction and Flux AE encode/decode feasibility."
    )
    parser.add_argument("--unet", default=None, help="Path to the Flux Fill GGUF UNet.")
    parser.add_argument("--ae", default=None, help="Path to Flux ae.safetensors.")
    parser.add_argument(
        "--handle-prefix",
        default="model.diffusion_model.",
        help="GGUF tensor prefix to strip. Use an empty string to strip nothing.",
    )
    parser.add_argument(
        "--skip-unet-construct",
        action="store_true",
        help="Inspect GGUF metadata and detection only; do not construct the Flux module.",
    )
    parser.add_argument(
        "--ae-size",
        type=int,
        default=64,
        help="Synthetic square pixel size for AE encode/decode. Must be divisible by 8.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Include full tracebacks for failed probe sections.",
    )
    return parser.parse_args()


def inspect_unet(path: Path, handle_prefix: str | None, *, construct: bool, include_traceback: bool) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "not_run",
    }
    if not path.exists():
        result["status"] = "missing"
        result["error"] = f"UNet path does not exist: {path}"
        return result, False

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        from backend.gguf.loader import gguf_sd_loader
        from backend.gguf.ops import GGMLOps
        from ldm_patched.modules import model_detection
        import ldm_patched.modules.sd as comfy_sd
    finally:
        sys.argv = original_argv

    try:
        state_dict, arch = gguf_sd_loader(str(path), handle_prefix=handle_prefix, return_arch=True)
        qtypes: dict[str, int] = {}
        for tensor in state_dict.values():
            qtype = _qtype_of(tensor) or "torch"
            qtypes[qtype] = qtypes.get(qtype, 0) + 1

        key_shapes = {
            key: {
                "shape": _shape_of(state_dict[key]),
                "qtype": _qtype_of(state_dict[key]),
            }
            for key in IMPORTANT_FLUX_KEYS
            if key in state_dict
        }
        detected_config = model_detection.detect_unet_config(state_dict, "", dtype=None)
        result.update(
            {
                "status": "inspected",
                "arch": arch,
                "tensor_count": len(state_dict),
                "qtypes": qtypes,
                "important_key_shapes": key_shapes,
                "block_counts": {
                    "double_blocks": _count_block_indices(state_dict, "double_blocks."),
                    "single_blocks": _count_block_indices(state_dict, "single_blocks."),
                },
                "detected_config": detected_config,
            }
        )

        if arch != "flux":
            result["architecture_warning"] = f"Expected arch 'flux', got {arch!r}."

        if construct:
            construct_result: dict[str, Any] = {"status": "not_run"}
            try:
                patcher = comfy_sd.load_diffusion_model_state_dict(
                    dict(state_dict),
                    model_options={"custom_operations": GGMLOps},
                )
                if patcher is None:
                    raise RuntimeError("Flux model detection returned no model config.")
                model = patcher.model
                construct_result.update(
                    {
                        "status": "ok",
                        "model_class": model.__class__.__name__,
                        "diffusion_model_class": model.diffusion_model.__class__.__name__,
                        "model_type": getattr(model.model_type, "name", str(model.model_type)),
                        "sampling_class": model.model_sampling.__class__.__name__,
                        "latent_format": model.latent_format.__class__.__name__,
                    }
                )
            except Exception as exc:
                construct_result.update({"status": "failed", "error": _short_error(exc)})
                if include_traceback:
                    construct_result["traceback"] = traceback.format_exc()
            result["construction"] = construct_result
            result["status"] = "ok" if construct_result["status"] == "ok" else "construction_failed"
        else:
            result["construction"] = {"status": "skipped"}
            result["status"] = "ok"

        return result, result["status"] == "ok"
    except Exception as exc:
        result.update({"status": "failed", "error": _short_error(exc)})
        if include_traceback:
            result["traceback"] = traceback.format_exc()
        return result, False


def inspect_ae(path: Path, size: int, *, include_traceback: bool) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "not_run",
        "synthetic_size": size,
    }
    if not path.exists():
        result["status"] = "missing"
        result["error"] = f"AE path does not exist: {path}"
        return result, False
    if size <= 0 or size % 8 != 0:
        result["status"] = "invalid_args"
        result["error"] = "--ae-size must be a positive multiple of 8."
        return result, False

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        import torch
        from backend import loader
    finally:
        sys.argv = original_argv

    try:
        with torch.no_grad():
            vae = loader.load_vae(str(path), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
            vae.first_stage_model.to(device=torch.device("cpu"), dtype=torch.float32)
            pixels = torch.rand(1, size, size, 3, dtype=torch.float32)
            pixels_nchw = pixels.movedim(-1, 1) * 2.0 - 1.0
            raw_latent = vae.first_stage_model.encode(pixels_nchw)
            scaled_latent = vae.latent_format.process_in(raw_latent)
            restored_latent = vae.latent_format.process_out(scaled_latent)
            decoded = vae.first_stage_model.decode(restored_latent)
            decoded_pixels = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0).movedim(1, -1)

            result.update(
                {
                    "status": "ok",
                    "vae_class": vae.first_stage_model.__class__.__name__,
                    "latent_format": vae.latent_format.__class__.__name__,
                    "latent_channels": int(raw_latent.shape[1]),
                    "raw_latent_shape": list(raw_latent.shape),
                    "scaled_latent_shape": list(scaled_latent.shape),
                    "decoded_shape": list(decoded_pixels.shape),
                    "scale_roundtrip_max_delta": float((restored_latent - raw_latent).abs().max()),
                    "decoded_min": float(decoded_pixels.min()),
                    "decoded_max": float(decoded_pixels.max()),
                }
            )
        return result, True
    except Exception as exc:
        result.update({"status": "failed", "error": _short_error(exc)})
        if include_traceback:
            result["traceback"] = traceback.format_exc()
        return result, False


def main() -> int:
    args = parse_args()
    handle_prefix = args.handle_prefix if args.handle_prefix != "" else None
    ok = True
    report: dict[str, Any] = {
        "probe": "P4-M10-W00 Flux Fill model/AE smoke",
        "unet": {"status": "skipped"},
        "ae": {"status": "skipped"},
    }

    if args.unet:
        unet_report, unet_ok = inspect_unet(
            Path(args.unet),
            handle_prefix,
            construct=not args.skip_unet_construct,
            include_traceback=args.traceback,
        )
        report["unet"] = unet_report
        ok = ok and unet_ok

    if args.ae:
        ae_report, ae_ok = inspect_ae(Path(args.ae), args.ae_size, include_traceback=args.traceback)
        report["ae"] = ae_report
        ok = ok and ae_ok

    if not args.unet and not args.ae:
        report["status"] = "no_inputs"
        report["message"] = "Provide --unet, --ae, or both."
        ok = False
    else:
        report["status"] = "ok" if ok else "failed"

    print(json.dumps(report, indent=2, default=_json_default))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())