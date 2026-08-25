"""Atomic runtime manifests and ready/current markers."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import TransactionError
from .layout import RuntimeLayout, is_within


MANIFEST_FILENAME = "runtime-manifest.json"
READY_FILENAME = "READY.json"
MANIFEST_SCHEMA = 3


@dataclass(frozen=True)
class RuntimeManifest:
    version: str
    profile: str
    python_version: str
    packages: tuple[dict[str, Any], ...]
    artifact_version: str = "development"
    build_id: str = "development"
    python_identity: dict[str, Any] | None = None
    uv_identity: dict[str, Any] | None = None
    cuda_family: str | None = None
    gpu: dict[str, Any] | None = None
    source_revision: str | None = None
    shared_packages: tuple[dict[str, Any], ...] = ()
    build_manifest_sha256: str | None = None
    source_content_sha256: str | None = None
    shared_lock_sha256: str | None = None
    profile_lock_sha256: str | None = None
    wheel_audit_sha256: str | None = None
    wheel_artifact_manifest_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        artifact_version = self.artifact_version or self.source_revision or "development"
        if artifact_version == "working-tree":
            raise TransactionError("Runtime manifests cannot use working-tree as artifact provenance")
        if artifact_version != "development":
            for label, identity in (("Python", self.python_identity), ("uv", self.uv_identity)):
                if not isinstance(identity, dict) or not identity.get("version") or not identity.get("sha256"):
                    raise TransactionError(f"Release runtime manifest is missing exact {label} identity")
            if not self.cuda_family:
                raise TransactionError("Release runtime manifest is missing CUDA family provenance")
            for label, digest in (
                ("build manifest", self.build_manifest_sha256),
                ("source content", self.source_content_sha256),
                ("shared lock", self.shared_lock_sha256),
                ("profile lock", self.profile_lock_sha256),
                ("wheel audit", self.wheel_audit_sha256),
                ("wheel artifact manifest", self.wheel_artifact_manifest_sha256),
            ):
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdefABCDEF" for character in digest
                ):
                    raise TransactionError(f"Release runtime manifest is missing exact {label} provenance")
        return {
            "schema": MANIFEST_SCHEMA,
            "version": self.version,
            "profile": self.profile,
            "python_version": self.python_version,
            "artifact_version": artifact_version,
            "build_id": self.build_id or artifact_version,
            "source_revision": self.source_revision or artifact_version,
            "python_identity": self.python_identity or {},
            "uv_identity": self.uv_identity or {},
            "cuda_family": self.cuda_family,
            "gpu": self.gpu or {},
            "packages": list(self.packages),
            "shared_packages": list(self.shared_packages),
            "build_manifest_sha256": self.build_manifest_sha256,
            "source_content_sha256": self.source_content_sha256,
            "shared_lock_sha256": self.shared_lock_sha256,
            "profile_lock_sha256": self.profile_lock_sha256,
            "wheel_audit_sha256": self.wheel_audit_sha256,
            "wheel_artifact_manifest_sha256": self.wheel_artifact_manifest_sha256,
        }


def canonical_manifest_digest(payload: Mapping[str, Any]) -> str:
    """Hash the manifest payload without its self-referential digest field."""

    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise TransactionError(f"Could not atomically write {path}") from exc


def write_runtime_markers(version_dir: Path, manifest: RuntimeManifest) -> None:
    """Write the manifest and ready marker only after all validation succeeds."""

    manifest_payload = manifest.as_dict()
    manifest_digest = canonical_manifest_digest(manifest_payload)
    manifest_payload["manifest_sha256"] = manifest_digest
    _atomic_json(version_dir / MANIFEST_FILENAME, manifest_payload)
    _atomic_json(
        version_dir / READY_FILENAME,
        {
            "schema": 2,
            "version": manifest.version,
            "profile": manifest.profile,
            "artifact_version": manifest_payload.get("artifact_version"),
            "build_id": manifest_payload.get("build_id"),
            "manifest_sha256": manifest_digest,
            "source_content_sha256": manifest_payload.get("source_content_sha256"),
        },
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise TransactionError(f"Could not read runtime metadata: {path}") from exc
    if not isinstance(value, dict):
        raise TransactionError(f"Runtime metadata is not an object: {path}")
    return value


def point_current(
    layout: RuntimeLayout,
    version: str,
    profile: str,
    *,
    expected_build_manifest: Mapping[str, Any] | None = None,
) -> None:
    version_dir = layout.version_root / version
    if not is_within(version_dir, layout.version_root):
        raise TransactionError(f"Refusing to point outside the runtime directory: {version}")
    if not is_runtime_ready(version_dir, expected_build_manifest=expected_build_manifest):
        raise TransactionError(f"Refusing to point current at an unready runtime: {version_dir}")
    manifest = read_json(version_dir / MANIFEST_FILENAME)
    if manifest.get("profile") != profile:
        raise TransactionError(f"Runtime profile does not match current pointer: {version_dir}")
    _atomic_json(
        layout.current_pointer,
        {
            "schema": 2,
            "version": version,
            "profile": profile,
            "artifact_version": manifest.get("artifact_version"),
            "build_id": manifest.get("build_id"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "source_content_sha256": manifest.get("source_content_sha256"),
        },
    )


def _identity_matches(actual: object, expected: object, fields: tuple[str, ...]) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    return all(actual.get(field) == expected.get(field) for field in fields)


def _packages_match(actual: object, expected: object) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    fields = ("name", "version", "index_url", "wheel_filename", "sha256", "no_deps")
    actual_by_name = {str(item.get("name", "")).lower().replace("_", "-"): item for item in actual if isinstance(item, dict)}
    expected_by_name = {str(item.get("name", "")).lower().replace("_", "-"): item for item in expected if isinstance(item, dict)}
    if (
        len(actual_by_name) != len(actual)
        or len(expected_by_name) != len(expected)
        or set(actual_by_name) != set(expected_by_name)
    ):
        return False
    return all(
        _identity_matches(actual_by_name[name], expected_by_name[name], fields)
        for name in expected_by_name
    )


def _manifest_matches_build_contract(
    manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any] | None,
) -> bool:
    if not build_manifest:
        return True
    if manifest.get("artifact_version") != build_manifest.get("artifact_version"):
        return False
    if manifest.get("build_id") != build_manifest.get("build_id"):
        return False
    for field in (
        "source_content_sha256",
        "shared_lock_sha256",
        "wheel_audit_sha256",
        "wheel_artifact_manifest_sha256",
    ):
        expected = build_manifest.get(field)
        if expected is not None and manifest.get(field) != expected:
            return False
    expected_build_digest = build_manifest.get("build_manifest_sha256")
    if expected_build_digest is not None and manifest.get("build_manifest_sha256") != expected_build_digest:
        return False
    expected_python = build_manifest.get("python")
    expected_uv = build_manifest.get("uv")
    if not _identity_matches(
        manifest.get("python_identity"),
        expected_python,
        ("version", "archive_name", "sha256", "executable_sha256"),
    ):
        return False
    if not _identity_matches(
        manifest.get("uv_identity"),
        expected_uv,
        ("version", "archive_name", "sha256", "binary_sha256"),
    ):
        return False
    profiles = build_manifest.get("profiles")
    if not isinstance(profiles, list):
        return False
    profile_entry = next(
        (entry for entry in profiles if isinstance(entry, dict) and entry.get("name") == manifest.get("profile")),
        None,
    )
    if profile_entry is None:
        return False
    if manifest.get("cuda_family") != profile_entry.get("cuda_family"):
        return False
    if not _packages_match(manifest.get("packages"), profile_entry.get("packages")):
        return False
    profile_lock_digests = build_manifest.get("profile_lock_sha256")
    if isinstance(profile_lock_digests, dict):
        profile_digest = profile_lock_digests.get(str(manifest.get("profile")))
        if profile_digest is None or manifest.get("profile_lock_sha256") != profile_digest:
            return False
    return True


def _discover_build_manifest(version_dir: Path) -> dict[str, Any] | None:
    """Find the immutable packaged build contract for a release runtime."""

    try:
        install_root = version_dir.resolve().parents[2]
    except (IndexError, OSError):
        return None
    path = install_root / "app" / "bootstrap" / "build_manifest.json"
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except TransactionError:
        return None
    return value


def current_runtime(
    layout: RuntimeLayout,
    *,
    expected_artifact_version: str | None = None,
    expected_profile: str | None = None,
    expected_build_manifest: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    if not layout.current_pointer.is_file():
        return None
    metadata = read_json(layout.current_pointer)
    version = metadata.get("version")
    if not isinstance(version, str) or not version:
        return None
    version_dir = layout.version_root / version
    if not is_within(version_dir, layout.version_root) or not is_runtime_ready(
        version_dir,
        expected_artifact_version=expected_artifact_version,
        expected_profile=expected_profile,
        expected_build_manifest=expected_build_manifest,
    ):
        return None
    manifest = read_json(version_dir / MANIFEST_FILENAME)
    if metadata.get("profile") != manifest.get("profile"):
        return None
    if metadata.get("artifact_version") not in (None, manifest.get("artifact_version")):
        return None
    if metadata.get("build_id") not in (None, manifest.get("build_id")):
        return None
    if metadata.get("manifest_sha256") != manifest.get("manifest_sha256"):
        return None
    if metadata.get("source_content_sha256") != manifest.get("source_content_sha256"):
        return None
    return version_dir, metadata


def is_runtime_ready(
    version_dir: Path,
    *,
    expected_artifact_version: str | None = None,
    expected_profile: str | None = None,
    expected_build_manifest: Mapping[str, Any] | None = None,
) -> bool:
    if not version_dir.is_dir():
        return False
    marker = version_dir / READY_FILENAME
    manifest = version_dir / MANIFEST_FILENAME
    if not marker.is_file() or not manifest.is_file():
        return False
    try:
        marker_data = read_json(marker)
        manifest_data = read_json(manifest)
    except TransactionError:
        return False
    if manifest_data.get("artifact_version") not in {None, "development"} and expected_build_manifest is None:
        expected_build_manifest = _discover_build_manifest(version_dir)
        if expected_build_manifest is None:
            return False
    ready = (
        manifest_data.get("schema") == MANIFEST_SCHEMA
        and marker_data.get("schema") == 2
        and marker_data.get("version") == manifest_data.get("version")
        and marker_data.get("profile") == manifest_data.get("profile")
        and bool(manifest_data.get("artifact_version"))
        and manifest_data.get("artifact_version") != "working-tree"
        and isinstance(manifest_data.get("manifest_sha256"), str)
        and manifest_data.get("manifest_sha256") == canonical_manifest_digest(manifest_data)
        and marker_data.get("manifest_sha256") == manifest_data.get("manifest_sha256")
        and marker_data.get("artifact_version") == manifest_data.get("artifact_version")
        and marker_data.get("build_id") == manifest_data.get("build_id")
        and marker_data.get("source_content_sha256") == manifest_data.get("source_content_sha256")
        and _manifest_matches_build_contract(manifest_data, expected_build_manifest)
    )
    if expected_artifact_version is not None:
        ready = ready and manifest_data.get("artifact_version") == expected_artifact_version
    if expected_profile is not None:
        ready = ready and manifest_data.get("profile") == expected_profile
    return ready


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "READY_FILENAME",
    "RuntimeManifest",
    "canonical_manifest_digest",
    "current_runtime",
    "is_runtime_ready",
    "point_current",
    "read_json",
    "write_runtime_markers",
]
