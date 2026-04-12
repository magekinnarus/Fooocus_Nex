from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.flux.flux_fill_pipeline import FluxFillConfig, run_flux_fill  # noqa: E402

DEFAULT_UNET_BY_TIER = {
    "q8_0": Path(r"models\unet\flux\flux1-fill-dev-Q8_0.gguf"),
    "q4_k_s": Path(r"models\unet\flux\flux1-fill-dev-Q4_K_S.gguf"),
}
DEFAULT_AE_PATH = Path(r"models\vae\flux\ae.safetensors")
DEFAULT_CONDITIONING_CACHE = Path(r"models\clip\flux\flux_empty_conditioning.pt")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return str(value)


def _load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {path}")
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _resolve_unet_path(tier: str, unet: str | None) -> Path:
    if unet:
        return Path(unet)
    try:
        return DEFAULT_UNET_BY_TIER[tier]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_UNET_BY_TIER))
        raise ValueError(f"Unknown Flux Fill tier {tier!r}. Supported tiers: {supported}.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless Flux Fill object removal on an image and mask.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--mask", required=True, help="Input mask path. White/regenerate, black/keep.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--metadata-output", default=None, help="Optional JSON metadata output path.")
    parser.add_argument("--unet", default=None, help="Flux Fill GGUF UNet path. Overrides --tier.")
    parser.add_argument("--tier", default="q8_0", choices=sorted(DEFAULT_UNET_BY_TIER), help="Known Flux Fill model tier.")
    parser.add_argument("--ae", default=str(DEFAULT_AE_PATH), help="Flux AE path.")
    parser.add_argument("--conditioning-cache", default=str(DEFAULT_CONDITIONING_CACHE), help="True empty Flux conditioning cache path.")
    parser.add_argument("--seed", type=int, default=882699830973928, help="Seed for initial Flux latent noise.")
    parser.add_argument("--steps", type=int, default=30, help="Euler denoise steps.")
    parser.add_argument("--guidance", type=float, default=15.0, help="Flux guidance embedding value.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda or cpu.")
    parser.add_argument("--extend-factor", type=float, default=1.2, help="Bounding-box context expansion factor.")
    parser.add_argument("--tiled-vae", action="store_true", help="Decode with tiled VAE mode.")
    parser.add_argument("--tile-size", type=int, default=64, help="VAE tile size when --tiled-vae is used.")
    parser.add_argument("--show-progress", action="store_true", help="Show sampler progress if the backend supports it.")
    parser.add_argument("--traceback", action="store_true", help="Include traceback details in JSON error output.")
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    metadata_output = Path(args.metadata_output) if args.metadata_output else None
    unet_path = _resolve_unet_path(args.tier, args.unet)

    image = _load_rgb_image(image_path)
    mask = _load_mask(mask_path)
    config = FluxFillConfig(
        unet_path=unet_path,
        ae_path=Path(args.ae),
        conditioning_cache_path=Path(args.conditioning_cache),
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path,
        tier=args.tier,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        device=args.device,
    )
    result = run_flux_fill(
        config,
        image,
        mask,
        extend_factor=args.extend_factor,
        tiled_decode=args.tiled_vae,
        tile_size=args.tile_size,
        disable_pbar=not args.show_progress,
    )
    _save_png(output_path, result.output_image)

    payload = {
        "status": "ok",
        "output": str(output_path),
        "image": str(image_path),
        "mask": str(mask_path),
        "width": result.width,
        "height": result.height,
        "seed": result.seed,
        "timings": result.timings,
        "metadata": result.metadata,
    }
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        payload["metadata_output"] = str(metadata_output)
    return payload


def main() -> int:
    args = parse_args()
    try:
        payload = run_from_args(args)
    except Exception as exc:
        payload = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
        if args.traceback:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(payload, indent=2, default=_json_default))
        return 1

    print(json.dumps(payload, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
