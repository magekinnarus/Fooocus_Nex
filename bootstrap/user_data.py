"""Creation and ownership rules for mutable Nexfocus user data."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .layout import RuntimeLayout


PATH_DIRECTORY_MAP = {
    "path_checkpoints": "models/checkpoints",
    "path_loras": "models/loras",
    "path_embeddings": "models/embeddings",
    "path_vae_approx": "models/vae_approx",
    "path_vae": "models/vae",
    "path_unet": "models/unet",
    "path_clip": "models/clip",
    "path_upscale_models": "models/upscale_models",
    "path_inpaint": "models/inpaint",
    "path_controlnet": "models/controlnet",
    "path_vision_support": "models/vision_support",
    "path_clip_vision": "models/vision_support/clip_vision",
    "path_preprocessors": "models/preprocessors",
    "path_insightface": "models/insightface",
    "path_removals": "models/removals",
    "path_outputs": "outputs",
    "path_download_manifests": "catalogs/download_manifests",
    "path_model_catalogs_user": "catalogs/user",
    "path_model_thumbnails": "thumbnails",
    "temp_path": "temp",
}

_SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|authorization|credential)", re.I)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?[A-Za-z0-9_.-]*(?:token|secret|password|api[_-]?key|authorization|credential)[A-Za-z0-9_.-]*[\"']?\s*[:=]\s*[\"']?)([^,\s\"'&}]+)",
)
_SECRET_FLAG = re.compile(r"(?i)(--(?:token|secret|password|api[-_]?key|authorization)\s+)([^\s]+)")


def _absolute_user_path(layout: RuntimeLayout, relative: str) -> str:
    return str((layout.user_data_root / relative).resolve()).replace("\\", "/")


def _replace_path_value(layout: RuntimeLayout, key: str, value: Any) -> Any:
    relative = PATH_DIRECTORY_MAP[key]
    absolute = _absolute_user_path(layout, relative)
    if isinstance(value, list):
        return [absolute]
    return absolute


def load_template(template_path: Path) -> dict[str, Any]:
    with template_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Config template must contain an object: {template_path}")
    return value


def materialize_config(layout: RuntimeLayout, template_path: Path) -> Path:
    """Create the first user config while preserving every later edit."""

    layout.ensure_user_directories()
    if layout.config_path.exists():
        return layout.config_path
    config = load_template(template_path)
    for key in PATH_DIRECTORY_MAP:
        if key in config:
            config[key] = _replace_path_value(layout, key, config[key])
            values = config[key] if isinstance(config[key], list) else [config[key]]
            for value in values:
                Path(str(value)).mkdir(parents=True, exist_ok=True)
    bundled_catalog_root = layout.app_root / "configs" / "model_catalogs"
    if bundled_catalog_root.is_dir():
        # This value is only a first-run hint.  The launcher also exports the
        # relocatable bundled root, and modules.config gives that process-local
        # value precedence over a stale path after application replacement.
        config["path_model_catalogs_preset"] = str(bundled_catalog_root.resolve()).replace("\\", "/")
    layout.config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.config_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, indent=4)
        stream.write("\n")
    os.replace(temporary, layout.config_path)
    return layout.config_path


def materialize_credentials(layout: RuntimeLayout, env_template_path: Path) -> Path:
    layout.ensure_user_directories()
    if not layout.env_path.exists():
        shutil.copyfile(env_template_path, layout.env_path)
    return layout.env_path


def ownership_manifest(layout: RuntimeLayout) -> dict[str, Any]:
    return {
        "schema": 1,
        "owner": "user",
        "root": str(layout.user_data_root),
        "preserve_on_repair": True,
        "paths": {
            "config": str(layout.config_root),
            "credentials": str(layout.credentials_root),
            "models": str(layout.models_root),
            "outputs": str(layout.outputs_root),
            "catalogs": str(layout.catalogs_root),
            "thumbnails": str(layout.thumbnails_root),
            "temp": str(layout.temp_root),
            "logs": str(layout.user_log_root),
        },
    }


def redact_text(text: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Redact common credential assignments before writing bootstrap logs."""

    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    result = _SECRET_ASSIGNMENT.sub(r"\1<redacted>", result)
    result = _SECRET_FLAG.sub(r"\1<redacted>", result)
    return result


def redact_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: ("<redacted>" if _SECRET_KEY.search(key) else value)
        for key, value in environment.items()
    }


__all__ = [
    "PATH_DIRECTORY_MAP",
    "load_template",
    "materialize_config",
    "materialize_credentials",
    "ownership_manifest",
    "redact_environment",
    "redact_text",
]
