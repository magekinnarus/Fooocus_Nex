from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.flux.flux_fill_pipeline import (  # noqa: E402
    EMPTY_FLUX_CROSS_ATTN_SHAPE,
    EMPTY_FLUX_POOLED_SHAPE,
    FluxFillValidationError,
    load_flux_empty_conditioning_cache,
    save_flux_empty_conditioning_cache,
)

DEFAULT_COMFY_ROOT = Path(r"D:\AI\Imagine_sup\ComfyUI_reference")
DEFAULT_GGUF_NODE_ROOT = DEFAULT_COMFY_ROOT / "custom_nodes" / "ComfyUI-GGUF"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_package(alias: str, package_root: Path):
    init_path = package_root / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"Package __init__.py not found: {init_path}")
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_gguf_support(gguf_node_root: Path):
    package = _load_package("comfyui_gguf", gguf_node_root)
    from comfyui_gguf.loader import gguf_clip_loader  # type: ignore
    from comfyui_gguf.nodes import GGUFModelPatcher  # type: ignore
    from comfyui_gguf.ops import GGMLOps  # type: ignore

    return package, gguf_clip_loader, GGUFModelPatcher, GGMLOps



def _install_comfy_reference_stubs() -> None:
    if "node_helpers" not in sys.modules:
        module = types.ModuleType("node_helpers")

        def conditioning_set_values(conditioning, values=None):
            values = values or {}
            out = []
            for item in conditioning:
                copied = [item[0], item[1].copy()]
                copied[1].update(values)
                out.append(copied)
            return out

        module.conditioning_set_values = conditioning_set_values
        sys.modules["node_helpers"] = module

    if "nodes" not in sys.modules:
        sys.modules["nodes"] = types.ModuleType("nodes")

    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.folder_names_and_paths = {}
        folder_paths.get_filename_list = lambda _name: []
        folder_paths.get_full_path = lambda _name, filename: filename
        folder_paths.get_folder_paths = lambda _name: []
        sys.modules["folder_paths"] = folder_paths

def _load_flux_clip(comfy_root: Path, gguf_node_root: Path, clip_l_path: Path, t5_path: Path):
    if not comfy_root.exists():
        raise FileNotFoundError(f"ComfyUI reference root does not exist: {comfy_root}")
    use_gguf_t5 = t5_path.suffix.lower() == ".gguf"
    if use_gguf_t5 and not gguf_node_root.exists():
        raise FileNotFoundError(f"ComfyUI-GGUF reference root does not exist: {gguf_node_root}")
    if not clip_l_path.exists():
        raise FileNotFoundError(f"CLIP-L path does not exist: {clip_l_path}")
    if not t5_path.exists():
        raise FileNotFoundError(f"T5 path does not exist: {t5_path}")

    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    _install_comfy_reference_stubs()

    import comfy.model_management  # type: ignore
    import comfy.sd  # type: ignore
    import comfy.utils  # type: ignore

    gguf_clip_loader = None
    GGUFModelPatcher = None
    GGMLOps = None
    if use_gguf_t5:
        _, gguf_clip_loader, GGUFModelPatcher, GGMLOps = _load_gguf_support(gguf_node_root)

    clip_l_sd = comfy.utils.load_torch_file(str(clip_l_path), safe_load=True)
    if use_gguf_t5:
        t5_sd = gguf_clip_loader(str(t5_path))
    else:
        t5_sd = comfy.utils.load_torch_file(str(t5_path), safe_load=True)

    model_options = {
        "initial_device": comfy.model_management.text_encoder_offload_device(),
    }
    if use_gguf_t5:
        model_options["custom_operations"] = GGMLOps

    clip = comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[clip_l_sd, t5_sd],
        clip_type=comfy.sd.CLIPType.FLUX,
        model_options=model_options,
        embedding_directory=None,
    )
    if use_gguf_t5:
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    return clip


def generate_empty_conditioning(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    clip_l_path = Path(args.clip_l)
    t5_path = Path(args.t5)
    output_path = Path(args.output)
    comfy_root = Path(args.comfy_root)
    gguf_node_root = Path(args.gguf_node_root)

    clip = _load_flux_clip(comfy_root, gguf_node_root, clip_l_path, t5_path)
    try:
        tokens = clip.tokenize(args.prompt)
        with torch.inference_mode():
            cross_attn, pooled_output = clip.encode_from_tokens(tokens, return_pooled=True)

        metadata = {
            "prompt": args.prompt,
            "clip_l_path": str(clip_l_path),
            "t5_path": str(t5_path),
            "comfy_root": str(comfy_root),
            "gguf_node_root": str(gguf_node_root),
            "t5_format": "gguf" if t5_path.suffix.lower() == ".gguf" else "safetensors",
            "cross_attn_shape": list(cross_attn.shape),
            "pooled_output_shape": list(pooled_output.shape),
            "cross_attn_dtype": str(cross_attn.dtype),
            "pooled_output_dtype": str(pooled_output.dtype),
            "generator": "tools/generate_flux_empty_conditioning.py",
            "conditioning_kind": "empty" if args.prompt == "" else "prompt",
        }
        conditioning = save_flux_empty_conditioning_cache(
            output_path,
            cross_attn=cross_attn.to(device="cpu"),
            pooled_output=pooled_output.to(device="cpu"),
            metadata=metadata,
        )
        return {
            "status": "ok",
            "output": str(output_path),
            "cross_attn_shape": list(conditioning.cross_attn.shape),
            "pooled_output_shape": list(conditioning.pooled_output.shape),
            "metadata": conditioning.metadata,
        }
    finally:
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_existing(path: Path) -> dict[str, Any]:
    conditioning = load_flux_empty_conditioning_cache(path)
    return {
        "status": "ok",
        "path": str(path),
        "cross_attn_shape": list(conditioning.cross_attn.shape),
        "pooled_output_shape": list(conditioning.pooled_output.shape),
        "metadata": conditioning.metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate a Flux text-conditioning cache for Flux Fill."
    )
    parser.add_argument("--clip-l", default=r"models\clip\flux\clip_l.safetensors", help="Path to Flux CLIP-L weights.")
    parser.add_argument("--t5", default=r"models\clip\flux\t5xxl_fp16.safetensors", help="Path to Flux T5 encoder weights. Defaults to fp16 safetensors for reference-quality cache generation.")
    parser.add_argument("--output", default=r"models\clip\flux\flux_empty_conditioning.pt", help="Output cache path.")
    parser.add_argument("--prompt", default="", help="Prompt text to encode. Use an empty string for true empty conditioning.")
    parser.add_argument("--comfy-root", default=str(DEFAULT_COMFY_ROOT), help="Path to the local ComfyUI reference root.")
    parser.add_argument("--gguf-node-root", default=str(DEFAULT_GGUF_NODE_ROOT), help="Path to the ComfyUI-GGUF reference root.")
    parser.add_argument("--validate-existing", action="store_true", help="Validate an existing cache instead of generating one.")
    parser.add_argument("--traceback", action="store_true", help="Include traceback details in JSON error output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.validate_existing:
            result = validate_existing(Path(args.output))
        else:
            result = generate_empty_conditioning(args)
    except Exception as exc:
        result = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "expected_contract": {
                "cross_attn": list(EMPTY_FLUX_CROSS_ATTN_SHAPE),
                "pooled_output": list(EMPTY_FLUX_POOLED_SHAPE),
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




