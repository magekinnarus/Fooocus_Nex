from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.flux.flux_fill_pipeline import (  # noqa: E402
    FluxFillValidationError,
    load_flux_empty_conditioning_cache,
    prepare_flux_fill_latent_source,
)
from tools.flux_fill_fp8_benchmark_contract import FluxFillBenchmarkArtifactBundle  # noqa: E402

DEFAULT_AE_PATH = Path(r"models\vae\flux\ae.safetensors")
DEFAULT_CONDITIONING_CACHE = Path(r"models\clip\flux\flux_empty_conditioning.pt")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {path}")
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    width: int | None,
    height: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if width is None and height is None:
        return image, mask
    if width is None or height is None:
        raise ValueError("Both resize width and height must be provided together.")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resize width/height must be positive, got {width}x{height}.")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"Resize width/height must be multiples of 8 for Flux latents, got {width}x{height}.")

    image_resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    mask_resized = np.asarray(
        Image.fromarray(mask, mode="L").resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return image_resized, mask_resized


def _prepare_flux_fill_mask(mask: np.ndarray, *, grow: int = 16, blur: int = 6) -> np.ndarray:
    import cv2

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Flux Fill mask must be HW or HWC, got shape {mask_np.shape}.")

    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    if grow > 0:
        kernel_size = max(1, int(grow) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    if blur > 0:
        kernel_size = max(3, int(blur) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    return mask_np.clip(0, 255).astype(np.uint8)


def _save_tensor_artifact(path: Path, key: str, tensor, *, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            key: tensor.detach().cpu(),
            "metadata": dict(metadata),
        },
        path,
    )


def _build_bundle_payload(bundle: FluxFillBenchmarkArtifactBundle) -> dict[str, Any]:
    payload = asdict(bundle)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare precomputed denoise-only artifacts for the tools-only Flux Fill fp8 benchmark."
    )
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--mask", required=True, help="Input mask path. White/regenerate, black/keep.")
    parser.add_argument("--output-dir", required=True, help="Directory where artifact .pt files and bundle JSON will be written.")
    parser.add_argument("--fp8-unet", required=True, help="Native fp8 Flux Fill UNet path for the benchmark bundle.")
    parser.add_argument("--q4-gguf-unet", default=None, help="Optional legacy q4 GGUF path retained only for compatibility metadata.")
    parser.add_argument("--ae", default=str(DEFAULT_AE_PATH), help="Flux AE path.")
    parser.add_argument("--conditioning-cache", default=str(DEFAULT_CONDITIONING_CACHE), help="Flux conditioning cache path, e.g. empty_conditioning.pt.")
    parser.add_argument("--bundle-name", default="bundle.json", help="Bundle JSON filename relative to --output-dir.")
    parser.add_argument("--raw-mask", action="store_true", help="Use the input mask as-is instead of Flux mask prep.")
    parser.add_argument("--grow", type=int, default=16, help="Mask growth radius when prep is enabled.")
    parser.add_argument("--blur", type=int, default=6, help="Mask blur radius when prep is enabled.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional resized image width for benchmark prep. Must be a multiple of 8.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional resized image height for benchmark prep. Must be a multiple of 8.")
    parser.add_argument("--device", default=None, help="Optional torch device override for VAE prep, e.g. cuda or cpu.")
    parser.add_argument("--traceback", action="store_true", help="Include traceback details in JSON error output.")
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_dir = Path(args.output_dir)
    bundle_path = output_dir / args.bundle_name
    fp8_unet_path = Path(args.fp8_unet)
    q4_gguf_unet_path = Path(args.q4_gguf_unet) if args.q4_gguf_unet else None
    ae_path = Path(args.ae)
    conditioning_cache_path = Path(args.conditioning_cache)

    image = _load_rgb_image(image_path)
    mask = _load_mask(mask_path)
    image, mask = _resize_pair(
        image,
        mask,
        width=args.resize_width,
        height=args.resize_height,
    )
    mask_prepared = not bool(args.raw_mask)
    working_mask = _prepare_flux_fill_mask(mask, grow=args.grow, blur=args.blur) if mask_prepared else mask

    conditioning = load_flux_empty_conditioning_cache(conditioning_cache_path, map_location="cpu")
    latent_source = prepare_flux_fill_latent_source(
        image,
        working_mask,
        ae_path,
        load_device=args.device,
        offload_device="cpu",
        cleanup_vae=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_latent_path = output_dir / "source_latent.pt"
    concat_latent_path = output_dir / "concat_latent.pt"
    denoise_mask_path = output_dir / "denoise_mask.pt"
    prep_metadata = {
        "generator": "tools/prepare_flux_fill_fp8_benchmark_bundle.py",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "ae_path": str(ae_path),
        "conditioning_cache_path": str(conditioning_cache_path),
        "mask_prepared": bool(mask_prepared),
        "mask_prep": None if not mask_prepared else {"grow": int(args.grow), "blur": int(args.blur)},
        "resized_image": bool(args.resize_width is not None and args.resize_height is not None),
        "resize": None if args.resize_width is None or args.resize_height is None else {
            "width": int(args.resize_width),
            "height": int(args.resize_height),
        },
        "device": str(args.device or ""),
    }
    _save_tensor_artifact(
        source_latent_path,
        "latent",
        latent_source.source_latent,
        metadata={**prep_metadata, "artifact": "source_latent", "timings": latent_source.timings},
    )
    _save_tensor_artifact(
        concat_latent_path,
        "latent",
        latent_source.concat_latent,
        metadata={**prep_metadata, "artifact": "concat_latent", "timings": latent_source.timings},
    )
    _save_tensor_artifact(
        denoise_mask_path,
        "denoise_mask",
        latent_source.denoise_mask,
        metadata={**prep_metadata, "artifact": "denoise_mask", "timings": latent_source.timings},
    )

    bundle = FluxFillBenchmarkArtifactBundle(
        fp8_unet_path=fp8_unet_path,
        ae_path=ae_path,
        conditioning_cache_path=conditioning_cache_path,
        source_latent_path=source_latent_path,
        concat_latent_path=concat_latent_path,
        denoise_mask_path=denoise_mask_path,
        provenance={
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "mask_prepared": bool(mask_prepared),
            "mask_prep": None if not mask_prepared else {"grow": int(args.grow), "blur": int(args.blur)},
            "conditioning_metadata": dict(conditioning.metadata),
            "conditioning_metadata_present": bool(conditioning.metadata),
            "conditioning_kind": str(conditioning.metadata.get("conditioning_kind", "unknown")),
            "conditioning_prompt": conditioning.metadata.get("prompt", None),
            "clip_l_path": conditioning.metadata.get("clip_l_path", None),
            "t5_path": conditioning.metadata.get("t5_path", None),
            "prep_timings": dict(latent_source.timings),
            "source_latent_shape": [int(dim) for dim in latent_source.source_latent.shape],
            "concat_latent_shape": [int(dim) for dim in latent_source.concat_latent.shape],
            "denoise_mask_shape": [int(dim) for dim in latent_source.denoise_mask.shape],
            "denoise_mask_coverage": float(latent_source.denoise_mask.float().mean().item()) if latent_source.denoise_mask.numel() else 0.0,
            "single_host_artifact_target": True,
            "benchmark_boundary": "denoise_only",
        },
        q4_gguf_unet_path=q4_gguf_unet_path,
    )
    bundle_payload = _build_bundle_payload(bundle)
    bundle_path.write_text(json.dumps(bundle_payload, indent=2, default=_json_default), encoding="utf-8")

    summary = {
        "status": "ok",
        "bundle": str(bundle_path),
        "output_dir": str(output_dir),
        "artifacts": {
            "source_latent": str(source_latent_path),
            "concat_latent": str(concat_latent_path),
            "denoise_mask": str(denoise_mask_path),
        },
        "bundle_payload": bundle_payload,
        "conditioning_cache": {
            "path": str(conditioning_cache_path),
            "metadata": conditioning.metadata,
        },
        "source_image_shape": list(image.shape),
        "mask_shape": list(working_mask.shape),
        "timings": dict(latent_source.timings),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    return summary


def main() -> int:
    args = parse_args()
    try:
        result = run_from_args(args)
    except Exception as exc:
        result = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
        if args.traceback:
            result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2, default=_json_default))
        return 1

    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
