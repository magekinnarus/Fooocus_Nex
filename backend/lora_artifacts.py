from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import ldm_patched.modules.weight_adapter as weight_adapter
import torch


TARGET_GROUP_UNET_DOWN = "unet.down"
TARGET_GROUP_UNET_MID = "unet.mid"
TARGET_GROUP_UNET_UP = "unet.up"
TARGET_GROUP_CLIP = "clip"
TARGET_GROUP_UNKNOWN = "unknown"


@dataclass(frozen=True)
class AdapterTargetEntry:
    target_key: str
    target_family: str
    target_group: str
    payload_family: str
    payload: Any
    source_rank: int | None = None
    supports_compact_retention: bool = True
    target_subgroup: str | None = None
    block_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterArtifact:
    artifact_id: str
    source_path: str
    source_family: str
    source_hash: str | None
    default_scale: float
    target_entries: tuple[AdapterTargetEntry, ...]
    artifact_metadata: dict[str, Any] = field(default_factory=dict)


def artifact_registry_signature(artifacts: Any) -> tuple[str, ...]:
    """
    Return the canonical retained-identity signature for a LoRA artifact set.

    The helper accepts a single artifact, any iterable of artifacts, or a
    mapping of artifacts. It is intentionally tolerant because the retained
    registry may be threaded through different call sites during route work.
    """
    if artifacts is None:
        return ()

    if isinstance(artifacts, AdapterArtifact):
        return (artifacts.artifact_id,)

    if isinstance(artifacts, (str, bytes)):
        return (str(artifacts),)

    values = artifacts.values() if isinstance(artifacts, Mapping) else artifacts
    signature: list[str] = []
    for artifact in values:
        if artifact is None:
            continue
        if isinstance(artifact, AdapterArtifact):
            signature.append(artifact.artifact_id)
            continue
        if isinstance(artifact, Mapping) and "artifact_id" in artifact:
            signature.append(str(artifact["artifact_id"]))
            continue
        if hasattr(artifact, "artifact_id"):
            signature.append(str(getattr(artifact, "artifact_id")))
            continue
        signature.append(str(artifact))
    return tuple(signature)


def build_application_patch_dict(
    artifact: AdapterArtifact,
    key_map: Mapping[str, str],
    *,
    target_family: str | None = None,
) -> dict[str, Any]:
    """
    Convert a retained artifact into the patch payload expected by the current
    application backend.

    This is intentionally narrow: it performs application-boundary mapping only
    and leaves retention authority with the artifact record itself.
    """
    patch_dict: dict[str, Any] = {}
    for entry in artifact.target_entries:
        if target_family is not None and entry.target_family not in (target_family, "unknown"):
            continue
        if entry.target_key not in key_map:
            continue
        patch_dict[entry.target_key] = entry.payload
    return patch_dict


def compute_file_hash(path: str, *, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_target_group(target_key: str) -> str:
    normalized = _normalize_key(target_key)
    if _is_clip_key(normalized):
        return TARGET_GROUP_CLIP
    if _contains_any(normalized, ("diffusion_model.input_blocks.", "model.diffusion_model.input_blocks.")):
        return TARGET_GROUP_UNET_DOWN
    if _contains_any(normalized, ("diffusion_model.middle_block.", "model.diffusion_model.middle_block.")):
        return TARGET_GROUP_UNET_MID
    if _contains_any(normalized, ("diffusion_model.output_blocks.", "model.diffusion_model.output_blocks.")):
        return TARGET_GROUP_UNET_UP
    if _contains_any(normalized, ("down_blocks.", "unet.down_blocks.")):
        return TARGET_GROUP_UNET_DOWN
    if _contains_any(normalized, ("mid_block.", "unet.mid_block.")):
        return TARGET_GROUP_UNET_MID
    if _contains_any(normalized, ("up_blocks.", "unet.up_blocks.")):
        return TARGET_GROUP_UNET_UP
    return TARGET_GROUP_UNKNOWN


def classify_target_subgroup(target_key: str, target_group: str | None = None) -> str | None:
    group = target_group or classify_target_group(target_key)
    if group == TARGET_GROUP_UNKNOWN:
        return None

    normalized = _normalize_key(target_key)
    local = "mlp" if group == TARGET_GROUP_CLIP else None
    if _contains_any(normalized, ("attn", "attention", "self_attn", ".to_q.", ".to_k.", ".to_v.", ".to_out.")):
        local = "attn"
    elif _contains_any(normalized, ("ff.net", "feed_forward", ".mlp.", "mlp_", ".fc1.", ".fc2.")):
        local = "mlp" if group == TARGET_GROUP_CLIP else "ff"
    elif _contains_any(normalized, ("proj", "projection")):
        local = "proj"
    elif _contains_any(
        normalized,
        (
            ".conv.",
            "conv_in",
            "conv_out",
            "in_layers.2",
            "out_layers.3",
            "skip_connection",
            "input_blocks.0.0",
            "out.2",
        ),
    ):
        local = "conv"

    if local is None:
        return None
    return f"{group}.{local}"


def extract_block_tag(target_key: str, target_group: str | None = None) -> str | None:
    group = target_group or classify_target_group(target_key)
    normalized = _normalize_key(target_key)

    if group == TARGET_GROUP_UNET_DOWN:
        match = re.search(r"(?:unet\.)?down_blocks\.(\d+)\.", normalized)
        if match:
            return f"down.block.{match.group(1)}"
    if group == TARGET_GROUP_UNET_MID:
        return "mid.block.0"
    if group == TARGET_GROUP_UNET_UP:
        match = re.search(r"(?:unet\.)?up_blocks\.(\d+)\.", normalized)
        if match:
            return f"up.block.{match.group(1)}"
    return None


def normalize_loaded_lora_artifact(
    *,
    source_path: str,
    loaded_patches: Mapping[Any, Any],
    default_scale: float = 1.0,
    source_family: str = "lora",
    source_hash: str | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> AdapterArtifact:
    resolved_source_hash, source_hash_origin = _resolve_source_hash(
        source_path=source_path,
        source_hash=source_hash,
        loaded_patches=loaded_patches,
    )
    entries = tuple(
        _build_target_entry(target_key, payload)
        for target_key, payload in sorted(loaded_patches.items(), key=lambda item: str(item[0]))
    )
    metadata = dict(artifact_metadata or {})
    metadata.setdefault("target_count", len(entries))
    metadata.setdefault("source_hash_origin", source_hash_origin)
    artifact_id = _artifact_id(
        source_path=source_path,
        source_hash=resolved_source_hash,
        default_scale=default_scale,
        entries=entries,
    )
    return AdapterArtifact(
        artifact_id=artifact_id,
        source_path=os.path.normpath(source_path),
        source_family=source_family,
        source_hash=resolved_source_hash,
        default_scale=float(default_scale),
        target_entries=entries,
        artifact_metadata=metadata,
    )


def _build_target_entry(target_key: Any, payload: Any) -> AdapterTargetEntry:
    key = _target_key_to_string(target_key)
    group = classify_target_group(key)
    payload_family = _payload_family(payload)
    return AdapterTargetEntry(
        target_key=key,
        target_family=_target_family(group),
        target_group=group,
        payload_family=payload_family,
        payload=payload,
        source_rank=_source_rank(payload),
        supports_compact_retention=_supports_compact_retention(payload_family),
        target_subgroup=classify_target_subgroup(key, group),
        block_tag=extract_block_tag(key, group),
        metadata={"raw_target_key_type": type(target_key).__name__},
    )


def _artifact_id(
    *,
    source_path: str,
    source_hash: str | None,
    default_scale: float,
    entries: tuple[AdapterTargetEntry, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(os.path.normcase(os.path.normpath(source_path)).encode("utf-8"))
    digest.update(str(source_hash or "").encode("utf-8"))
    digest.update(repr(float(default_scale)).encode("ascii"))
    for entry in entries:
        digest.update(entry.target_key.encode("utf-8"))
        digest.update(entry.payload_family.encode("utf-8"))
        digest.update(str(entry.source_rank).encode("ascii"))
    return digest.hexdigest()


def _resolve_source_hash(
    *,
    source_path: str,
    source_hash: str | None,
    loaded_patches: Mapping[Any, Any],
) -> tuple[str, str]:
    if source_hash:
        return source_hash, "provided"
    if os.path.isfile(source_path):
        return compute_file_hash(source_path), "file"
    return _payload_content_hash(loaded_patches), "payload"


def _target_key_to_string(target_key: Any) -> str:
    if isinstance(target_key, str):
        return target_key
    if isinstance(target_key, tuple) and target_key:
        return str(target_key[0])
    return str(target_key)


def _payload_family(payload: Any) -> str:
    if hasattr(payload, "name"):
        return str(payload.name)
    if isinstance(payload, tuple) and payload:
        if isinstance(payload[0], str):
            return payload[0]
        if len(payload) == 1:
            return "diff"
    return type(payload).__name__


def _source_rank(payload: Any) -> int | None:
    weights = getattr(payload, "weights", None)
    if weights is not None:
        return _rank_from_weights(weights)
    if isinstance(payload, tuple) and len(payload) >= 2 and isinstance(payload[1], tuple):
        return _rank_from_weights(payload[1])
    return None


def _rank_from_weights(weights: Any) -> int | None:
    try:
        if len(weights) >= 2 and isinstance(weights[1], torch.Tensor) and weights[1].ndim >= 1:
            return int(weights[1].shape[0])
    except TypeError:
        return None
    return None


def _supports_compact_retention(payload_family: str) -> bool:
    return payload_family not in {"diff", "set", "fooocus", "model_as_lora"}


def _target_family(target_group: str) -> str:
    if target_group.startswith("unet."):
        return "unet"
    if target_group == TARGET_GROUP_CLIP:
        return "clip"
    return TARGET_GROUP_UNKNOWN


def _is_clip_key(normalized_key: str) -> bool:
    return normalized_key.startswith(
        (
            "clip_",
            "clip.",
            "text_encoder",
            "text_encoders.",
            "conditioner.embedders.",
            "cond_stage_model.",
            "t5xxl.",
            "hydit_clip.",
        )
    )


def _normalize_key(key: str) -> str:
    return key.replace("\\", "/").lower()


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _payload_content_hash(loaded_patches: Mapping[Any, Any]) -> str:
    digest = hashlib.sha256()
    for target_key, payload in sorted(loaded_patches.items(), key=lambda item: str(item[0])):
        digest.update(_target_key_to_string(target_key).encode("utf-8"))
        _update_digest(digest, payload)
    return digest.hexdigest()


def _update_digest(digest, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, weight_adapter.WeightAdapterBase):
        digest.update(value.name.encode("utf-8"))
        digest.update(",".join(sorted(value.loaded_keys)).encode("utf-8"))
        _update_digest(digest, value.weights)
        return
    if isinstance(value, Mapping):
        for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0])):
            digest.update(str(item_key).encode("utf-8"))
            _update_digest(digest, item_value)
        return
    if isinstance(value, (tuple, list)):
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, set):
        for item in sorted(str(entry) for entry in value):
            digest.update(item.encode("utf-8"))
        return
    digest.update(repr(value).encode("utf-8"))
