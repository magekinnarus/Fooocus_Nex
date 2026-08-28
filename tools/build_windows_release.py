"""Build and audit the Nexfocus Windows x64/NVIDIA one-click archive.

The builder is deliberately stdlib-only. Large runtime artifacts are release
inputs, not repository files: provide the official Python embeddable archive,
uv binary, completed shared lock, repository-detached wheel manifest,
structured dependency notices, and the approved InsightFace wheel at build
time. No dependency resolver runs while creating the ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

# Running ``python tools/build_windows_release.py`` puts ``tools`` rather than
# the repository root on sys.path. Keep the build command self-contained.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from bootstrap.errors import ProfileError
from bootstrap.locks import LockSet, load_lock, profile_lock_from_profile, validate_lock, write_lock
from bootstrap.profiles import (
    APPROVED_PROFILES,
    DependencyProfile,
    PYPI_INDEX,
    normalize_package_name,
    validate_approved_profile,
)


ARCHIVE_NAME = "Nexfocus-OneClick-Windows-x64.zip"
EXCLUDED_DIRECTORIES = {
    ".agent",
    ".codex",
    ".github",
    ".git",
    ".pytest_cache",
    ".pytest_full_temp",
    ".pytest_w03_temp",
    ".pytest_w03_suite_current",
    ".pytest_w03_finalcheck",
    ".ssl",
    ".mypy_cache",
    "__pycache__",
    "dist",
    "models",
    "outputs",
    "release",
    "runtime",
    "tests",
    "tools",
    "venv",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".whl", ".zip", ".7z", ".log"}
EXCLUDED_FILES = {".env", "config.txt", "THIRD-PARTY-NOTICES.txt"}

# W03 source files were intentionally kept out of the end-user app tree while
# this order was developed in the working tree.  They are explicit generated
# release inputs, not a permission to walk arbitrary untracked content.
EXPLICIT_RELEASE_FILES = frozenset(
    {
        "bootstrap/__init__.py",
        "bootstrap/__main__.py",
        "bootstrap/build_contract.json",
        "bootstrap/errors.py",
        "bootstrap/gpu.py",
        "bootstrap/installer.py",
        "bootstrap/launcher.py",
        "bootstrap/layout.py",
        "bootstrap/locks.py",
        "bootstrap/manifest.py",
        "bootstrap/ownership.json",
        "bootstrap/profiles.py",
        "bootstrap/release_inputs.json",
        "bootstrap/wheel_artifact_trust.json",
        "bootstrap/runtime.py",
        "bootstrap/user_data.py",
        "bootstrap/licenses/uv-MIT.txt",
        "bootstrap/locks/shared-py312-win-amd64.txt",
        "bootstrap/locks/legacy-cu124-py312-win-amd64.txt",
        "bootstrap/locks/modern-cu128-py312-win-amd64.txt",
        "windows/Nexfocus.bat.in",
        "THIRD-PARTY-NOTICES.txt",
        "tests/test_w03_windows_bootstrap.py",
        "tools/build_windows_release.py",
    }
)
ALLOWED_NON_RELEASE_UNTRACKED_ROOTS = frozenset({".agent", ".codex", ".git"})
SENSITIVE_PACKAGE_NAMES = frozenset({"torch", "torchvision", "xformers"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SECRET_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:sk|ghp|github_pat|hf|xox[baprs])-[-_A-Za-z0-9]{16,}"),
    re.compile(r"(?i)\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)(?:[A-Z]:[\\/]+Users[\\/]|[\\/]home[\\/]|[\\/]Users[\\/])"),
)
_NONEMPTY_ENV_CREDENTIAL = re.compile(
    r'''(?m)^[ \t]*[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API[_-]?KEY|CREDENTIAL)[A-Z0-9_]*[ \t]*[:=][ \t]*'''
)
_WHEEL_MANIFEST_SCHEMA = 1
_UNSUPPORTED_SHARED_PACKAGES = frozenset({"gguf"})
_UNKNOWN_LICENSE_IDENTITIES = frozenset({"UNKNOWN", "NOASSERTION"})
_MAX_LICENSE_IDENTITY_LENGTH = 200
_WHEEL_METADATA_MEMBERS = ("METADATA", "WHEEL", "RECORD")
_TARGET_PYTHON_VERSION = (3, 12)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if not _SHA256_RE.fullmatch(str(expected)):
        raise ValueError(f"Trusted SHA-256 is malformed for {path.name}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def load_release_inputs(source_root: Path) -> dict[str, Any]:
    contract_path = source_root / "bootstrap" / "release_inputs.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read trusted release-input contract: {contract_path}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        raise ValueError("Trusted release-input contract has an unsupported schema")
    for section_name in ("python", "uv", "insightface"):
        section = contract.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Trusted release-input contract is missing {section_name}")
    python = contract["python"]
    uv = contract["uv"]
    for section_name, section in (("python", python), ("uv", uv)):
        if not _SHA256_RE.fullmatch(str(section.get("sha256", ""))):
            raise ValueError(f"Trusted {section_name} digest is missing or malformed")
        if not section.get("archive_name") or not section.get("source_url"):
            raise ValueError(f"Trusted {section_name} source identity is incomplete")
    license_source = str(uv.get("license_source", ""))
    if not license_source or PurePosixPath(license_source).is_absolute() or ".." in PurePosixPath(license_source).parts:
        raise ValueError("Trusted uv license source must remain inside the repository")
    if not _SHA256_RE.fullmatch(str(uv.get("license_sha256", ""))):
        raise ValueError("Trusted uv license digest is missing or malformed")
    lock_source = str(contract.get("shared_lock_source", ""))
    if not lock_source or PurePosixPath(lock_source).is_absolute() or ".." in PurePosixPath(lock_source).parts:
        raise ValueError("Trusted shared-lock source must remain inside the repository")
    wheel_manifest_contract = contract.get("wheel_artifact_manifest")
    if not isinstance(wheel_manifest_contract, dict) or wheel_manifest_contract.get("schema") != _WHEEL_MANIFEST_SCHEMA:
        raise ValueError("Trusted wheel artifact manifest contract is missing or unsupported")
    trust_root = str(wheel_manifest_contract.get("trust_root", ""))
    if (
        wheel_manifest_contract.get("authentication_method") != "detached-repository-sha256"
        or not trust_root
        or PurePosixPath(trust_root).is_absolute()
        or ".." in PurePosixPath(trust_root).parts
    ):
        raise ValueError("Trusted wheel artifact manifest must use a repository-owned detached trust root")
    if not wheel_manifest_contract.get("manifest_filename"):
        raise ValueError("Trusted wheel artifact manifest filename is missing")
    origins = wheel_manifest_contract.get("approved_source_origins")
    if not isinstance(origins, list) or not origins or not all(
        isinstance(origin, str) and origin.startswith("https://") and origin.endswith("/")
        for origin in origins
    ):
        raise ValueError("Trusted wheel artifact manifest source-origin policy is incomplete")
    notice_contract = contract.get("dependency_notice_contract")
    if not isinstance(notice_contract, dict) or notice_contract.get("schema") != 1 or notice_contract.get("requires_shared_and_profile_artifacts") is not True:
        raise ValueError("Trusted dependency notice contract is missing or unsupported")
    return contract


def _git_output(source_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise ValueError("Release assembly requires a Git checkout with tracked source files") from exc
    if result.returncode != 0:
        raise ValueError(f"Git source inventory failed: {(result.stderr or result.stdout).strip()[:400]}")
    return result.stdout


def tracked_source_inventory(source_root: Path) -> tuple[str, ...]:
    """Return the exact tracked source inventory used for application assembly."""

    raw = _git_output(source_root, "ls-files", "-z")
    paths = tuple(sorted(path for path in raw.split("\0") if path))
    if not paths:
        raise ValueError("Git source inventory is empty")
    return paths


def untracked_source_inventory(source_root: Path) -> tuple[str, ...]:
    """Return untracked files, including files below untracked directories."""

    raw = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    paths: list[str] = []
    for record in raw.split("\0"):
        if not record or not record.startswith("?? "):
            continue
        path = record[3:].replace("\\", "/")
        if path:
            paths.append(path)
    return tuple(sorted(paths))


def _validate_untracked_workspace(source_root: Path, tracked: set[str]) -> None:
    unexpected: list[str] = []
    for relative in untracked_source_inventory(source_root):
        normalized = relative.replace("\\", "/")
        first = PurePosixPath(normalized).parts[0] if PurePosixPath(normalized).parts else ""
        if normalized in EXPLICIT_RELEASE_FILES:
            continue
        if first in ALLOWED_NON_RELEASE_UNTRACKED_ROOTS:
            continue
        # Test/tool files are allowed only when they are already part of the
        # explicitly listed W03 development surface.  A newly seeded file in
        # either tree must still fail the release gate.
        if normalized not in tracked:
            unexpected.append(normalized)
    if unexpected:
        raise ValueError(
            "Unexpected untracked release input(s): "
            + ", ".join(unexpected[:12])
        )


def _reject_source_symlinks(source_root: Path) -> None:
    for current_root, directory_names, file_names in os.walk(source_root, topdown=True, followlinks=False):
        current_path = Path(current_root)
        symlinked_directories = [name for name in directory_names if (current_path / name).is_symlink()]
        if symlinked_directories:
            raise ValueError(f"Symlinked source directory is not allowed: {current_path / symlinked_directories[0]}")
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"Symlinked source file is not allowed: {path}")


def _validate_source_state(source_root: Path, tracked: set[str]) -> None:
    """Fail closed on ignored artifacts even though they are not tracked."""

    skipped_directories = {
        ".git",
        ".agent",
        ".codex",
        ".pytest_cache",
        ".pytest_full_temp",
        ".pytest_w03_temp",
        ".pytest_w03_suite_current",
        ".pytest_w03_finalcheck",
        ".pytest_run_temp",
        ".mypy_cache",
        "__pycache__",
        "venv",
    }
    forbidden_names = {".env", "config.txt", "credentials.json"}
    forbidden_suffixes = {".whl", ".zip", ".7z", ".ckpt", ".safetensors", ".gguf", ".onnx", ".partial"}
    for current_root, directory_names, file_names in os.walk(source_root, topdown=True, followlinks=False):
        current_path = Path(current_root)
        relative_root = current_path.relative_to(source_root)
        directory_names[:] = [name for name in directory_names if name not in skipped_directories]
        for name in file_names:
            relative = (relative_root / name).as_posix()
            lower = relative.lower()
            if name.lower() in {".env", "config.txt"}:
                # A developer's credential/config file is intentionally
                # ignored and never read/copied.  The release inventory
                # excludes it.
                continue
            if (name in forbidden_names or name.lower().startswith(".env.")
                    or Path(name).suffix.lower() in forbidden_suffixes):
                raise ValueError(f"Personal or generated artifact is present in the source tree: {relative}")
            parts = PurePosixPath(lower).parts
            if parts and parts[0] in {"models", "outputs", "credentials", "runtime", "release", "cache"}:
                if parts[0] == "models" and relative in tracked:
                    continue
                raise ValueError(f"User/runtime state is present in the source tree: {relative}")


def _is_excluded(relative: Path) -> bool:
    if relative.name in EXCLUDED_FILES or relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return any(part in EXCLUDED_DIRECTORIES for part in relative.parts)


def copy_application(source_root: Path, app_root: Path) -> list[str]:
    """Copy only tracked/explicit release inputs and reject local state."""

    source_root = source_root.resolve()
    _reject_source_symlinks(source_root)
    tracked = set(tracked_source_inventory(source_root))
    _validate_source_state(source_root, tracked)
    _validate_untracked_workspace(source_root, tracked)
    inventory = sorted(tracked | {path for path in EXPLICIT_RELEASE_FILES if (source_root / path).is_file()})
    copied: list[str] = []
    for relative_text in inventory:
        relative = Path(relative_text)
        source = source_root / relative
        if not source.is_file() or _is_excluded(relative) or relative.parts[0] in {".agent", ".codex", ".git"}:
            continue
        destination = app_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(relative.as_posix())
    required = {"launch.py", "config_template.txt", ".env_template", "bootstrap/profiles.py"}
    missing = required - set(copied)
    if missing:
        raise ValueError(f"Source tree is missing required release inputs: {sorted(missing)}")
    return copied


def extract_python_embed(
    python_zip: Path,
    destination: Path,
    expected_sha256: str | None = None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Extract only the repository-approved CPython embeddable archive."""

    python_contract = dict(contract or {})
    trusted_sha256 = str(python_contract.get("sha256", ""))
    if not trusted_sha256:
        raise ValueError("Python extraction requires the repository-pinned runtime contract")
    if expected_sha256 is not None and expected_sha256.lower() != trusted_sha256.lower():
        raise ValueError("Caller-supplied Python digest does not match the repository contract")
    if python_zip.name != python_contract.get("archive_name"):
        raise ValueError(
            f"Unexpected Python archive name: {python_zip.name}; expected {python_contract.get('archive_name')}"
        )
    verify_sha256(python_zip, trusted_sha256)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(python_zip) as archive:
        members = {PurePosixPath(info.filename).as_posix() for info in archive.infolist()}
        required_members = {
            str(python_contract.get("executable_member", "python.exe")),
            str(python_contract.get("pth_member", "python312._pth")),
            *(str(value) for value in python_contract.get("license_members", [])),
        }
        missing = required_members - members
        if missing:
            raise ValueError(f"Approved Python archive is missing required member(s): {sorted(missing)}")
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or not relative.as_posix():
                raise ValueError(f"Unsafe Python archive member: {info.filename}")
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Symlinked Python archive member is not allowed: {info.filename}")
            target = destination.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    if not (destination / str(python_contract.get("executable_member", "python.exe"))).is_file():
        raise ValueError("Python embeddable archive did not contain python.exe")
    for member in python_contract.get("license_members", []):
        if not (destination / str(member)).is_file() or (destination / str(member)).stat().st_size == 0:
            raise ValueError(f"Python runtime license material is missing or empty: {member}")


def add_embedded_app_path(python_root: Path) -> None:
    """Allow the first bootstrap process to import app/bootstrap from _pth."""

    pth_files = tuple(python_root.glob("*_._pth")) + tuple(python_root.glob("*_pth"))
    for pth_path in pth_files:
        lines = pth_path.read_text(encoding="utf-8").splitlines()
        if "../../app" not in [line.strip() for line in lines]:
            lines.append("../../app")
        if not any(line.strip() == "import site" for line in lines):
            lines.append("import site")
        pth_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def copy_uv(
    uv_binary: Path,
    destination: Path,
    expected_sha256: str | None = None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Verify the pinned uv release ZIP and extract exactly uv.exe."""

    uv_contract = dict(contract or {})
    trusted_sha256 = str(uv_contract.get("sha256", ""))
    if not trusted_sha256:
        raise ValueError("uv extraction requires the repository-pinned runtime contract")
    if expected_sha256 is not None and expected_sha256.lower() != trusted_sha256.lower():
        raise ValueError("Caller-supplied uv digest does not match the repository contract")
    if uv_binary.name != uv_contract.get("archive_name"):
        raise ValueError(
            f"Unexpected uv archive name: {uv_binary.name}; expected {uv_contract.get('archive_name')}"
        )
    verify_sha256(uv_binary, trusted_sha256)
    try:
        archive = zipfile.ZipFile(uv_binary)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Pinned uv input must be the official Windows release ZIP") from exc
    member_name = str(uv_contract.get("binary_member", "uv.exe"))
    with archive:
        members = {PurePosixPath(info.filename).as_posix(): info for info in archive.infolist()}
        info = members.get(member_name)
        if info is None or info.is_dir():
            raise ValueError(f"Approved uv archive is missing {member_name}")
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe uv archive member: {info.filename}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
    if destination.stat().st_size == 0:
        raise ValueError("Approved uv executable is empty")


def _lock_identity(lock: LockSet) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        normalize_package_name(pin.name): (
            pin.version,
            tuple(sorted(value.lower() for value in pin.hashes)),
        )
        for pin in lock.pins
    }


def _validate_complete_shared_lock(shared: LockSet, canonical: LockSet) -> None:
    expected_identity = _lock_identity(canonical)
    actual_identity = _lock_identity(shared)
    expected = {name: version for name, (version, _hashes) in expected_identity.items()}
    actual = {name: version for name, (version, _hashes) in actual_identity.items()}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise ValueError(
            "Shared lock does not match the checked-in complete graph: "
            f"missing={missing[:8]}, extra={extra[:8]}, changed={changed[:8]}"
        )
    unsupported = sorted(set(actual) & _UNSUPPORTED_SHARED_PACKAGES)
    if unsupported:
        raise ValueError(
            "Shared lock contains unsupported retired package(s): "
            + ", ".join(unsupported)
        )
    # A future repository-owned canonical lock may carry hashes as well as
    # versions.  When it does, the caller's completed lock must preserve those
    # identities; otherwise the external lock remains the sole authority.
    for name, (_version, expected_hashes) in expected_identity.items():
        if expected_hashes:
            actual_hashes = actual_identity[name][1]
            if actual_hashes != expected_hashes:
                raise ValueError(f"Shared lock digest drift for {name}")
    if shared.name != canonical.name and shared.name not in {"shared", "shared-py312-win-amd64"}:
        raise ValueError(f"Shared lock has an unexpected identity: {shared.name}")


def _wheel_filename_parts(filename: str) -> tuple[str, str, str, str, str, str] | None:
    if not filename.lower().endswith(".whl"):
        return None
    stem = filename[:-4]
    try:
        distribution_and_version, python_tag, abi_tag, platform_tag = stem.rsplit("-", 3)
    except ValueError:
        return None
    prefix_parts = distribution_and_version.split("-")
    if len(prefix_parts) < 2:
        return None
    build = ""
    version = prefix_parts[-1]
    distribution_parts = prefix_parts[:-1]
    # A build tag is optional and is separated from the distribution/version
    # pair.  Look for the unambiguous ``version-build`` shape without breaking
    # normal hyphenated distribution names such as opencv-contrib-python.
    if len(prefix_parts) >= 3 and prefix_parts[-1][:1].isdigit() and prefix_parts[-2][:1].isdigit():
        build = prefix_parts[-1]
        version = prefix_parts[-2]
        distribution_parts = prefix_parts[:-2]
    distribution = "-".join(distribution_parts)
    if not distribution:
        return None
    return distribution, version, build, python_tag, abi_tag, platform_tag


def _wheel_tag_is_cp312_compatible(python_tag: str, abi_tag: str, platform_tag: str) -> bool:
    platform_tags = set(platform_tag.split("."))
    if not platform_tags.intersection({"any", "win_amd64"}):
        return False
    python_tags = set(python_tag.split("."))
    abi_tags = set(abi_tag.split("."))
    if "py3" in python_tags and "none" in abi_tags:
        return True
    for tag in python_tags:
        if tag.startswith("py") and tag[2:].isdigit() and "none" in abi_tags:
            # A pure-Python wheel tagged for an older Python minor remains
            # usable on CPython 3.12.  Future-only tags do not.
            if int(tag[2:]) <= 312:
                return True
        if tag == "cp312" and abi_tags.intersection({"cp312", "abi3", "none"}):
            return True
        if tag.startswith("cp") and tag[2:].isdigit() and "abi3" in abi_tags:
            # abi3 is forward compatible, never backward compatible from a
            # future interpreter.  This specifically rejects cp313-abi3 for
            # the CPython 3.12 release target.
            if int(tag[2:]) <= 312:
                return True
    return False


def _wheel_matches_pin(path: Path, pin: Any) -> bool:
    parsed = _wheel_filename_parts(path.name)
    if parsed is None:
        return False
    distribution, version, _build, python_tag, abi_tag, platform_tag = parsed
    normalized_distribution = normalize_package_name(distribution)
    normalized_pin = normalize_package_name(pin.name)
    if normalized_distribution != normalized_pin or version != pin.version:
        return False
    return _wheel_tag_is_cp312_compatible(python_tag, abi_tag, platform_tag)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _approved_source_url(source_url: str, origins: Iterable[str]) -> bool:
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    return any(source_url.startswith(origin) for origin in origins)


def load_wheel_artifact_trust(path: Path) -> dict[str, Any]:
    """Load the repository-owned detached trust root for a wheel manifest."""

    if not path.is_file():
        raise ValueError(f"Repository-owned wheel artifact trust root is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Wheel artifact trust root is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("Wheel artifact trust root has an unsupported schema")
    manifest_filename = str(payload.get("manifest_filename", ""))
    manifest_digest = str(payload.get("manifest_sha256", ""))
    if not manifest_filename or "/" in manifest_filename or "\\" in manifest_filename:
        raise ValueError("Wheel artifact trust root has an invalid manifest filename")
    if not _SHA256_RE.fullmatch(manifest_digest) or manifest_digest.lower() == "0" * 64:
        raise ValueError(
            "Wheel artifact trust root has no approved detached manifest digest; "
            "obtain and record the Director-approved manifest identity first"
        )
    origins = payload.get("approved_source_origins")
    if not isinstance(origins, list) or not origins or not all(
        isinstance(origin, str) and origin.startswith("https://") and origin.endswith("/")
        for origin in origins
    ):
        raise ValueError("Wheel artifact trust root has no approved HTTPS source origins")
    return payload


def load_wheel_artifact_manifest(
    path: Path,
    *,
    trust_root: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    """Load a wheel manifest bound to a detached repository-owned trust root.

    The manifest's self-digest is only an internal consistency check.  The
    detached raw-file digest in the repository-owned trust root is the actual
    authenticity boundary; fields in the manifest cannot create or replace
    that trust decision.
    """

    if not path.is_file():
        raise ValueError(f"Wheel artifact manifest is missing: {path}")
    trusted = load_wheel_artifact_trust(trust_root) if isinstance(trust_root, Path) else dict(trust_root)
    if path.name != str(trusted.get("manifest_filename", "")):
        raise ValueError("Wheel artifact manifest filename does not match the detached trust root")
    detached_digest = str(trusted.get("manifest_sha256", "")).lower()
    if not _SHA256_RE.fullmatch(detached_digest) or sha256_file(path).lower() != detached_digest:
        raise ValueError("Wheel artifact manifest does not match the repository-owned detached trust root")
    approved_origins = tuple(str(origin) for origin in trusted.get("approved_source_origins", ()))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Wheel artifact manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _WHEEL_MANIFEST_SCHEMA:
        raise ValueError("Wheel artifact manifest has an unsupported schema")
    authentication = payload.get("authentication")
    if not isinstance(authentication, dict):
        raise ValueError("Wheel artifact manifest is missing authentication metadata")
    if authentication.get("method") != "source-attested-sha256":
        raise ValueError("Wheel artifact manifest has no approved authentication method")
    source = str(authentication.get("source_url", ""))
    if not _approved_source_url(source, approved_origins):
        raise ValueError("Wheel artifact manifest source_url is outside the approved trust-root origins")
    if not authentication.get("retrieved_utc") or not authentication.get("attestation"):
        raise ValueError("Wheel artifact manifest is missing retrieval/attestation provenance")
    claimed_digest = str(authentication.get("manifest_sha256", ""))
    unsigned = dict(payload)
    unsigned_authentication = dict(authentication)
    unsigned_authentication.pop("manifest_sha256", None)
    unsigned["authentication"] = unsigned_authentication
    if not _SHA256_RE.fullmatch(claimed_digest) or claimed_digest.lower() != _payload_sha256(unsigned):
        raise ValueError("Wheel artifact manifest self-digest is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Wheel artifact manifest has no artifact records")
    seen: set[tuple[str, str, str]] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("Wheel artifact manifest contains a malformed artifact record")
        name = normalize_package_name(str(record.get("name", "")))
        version = str(record.get("version", ""))
        filename = str(record.get("filename", ""))
        digest = str(record.get("sha256", ""))
        source_url = str(record.get("source_url", ""))
        if not name or not version or not filename.endswith(".whl"):
            raise ValueError("Wheel artifact manifest record is missing name/version/filename")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Wheel artifact manifest has an invalid digest for {name}=={version}")
        if not _approved_source_url(source_url, approved_origins):
            raise ValueError(f"Wheel artifact source is outside the approved origins for {name}=={version}")
        if not str(record.get("metadata_sha256", "")) or not str(record.get("wheel_metadata_sha256", "")):
            raise ValueError(
                f"Wheel artifact manifest lacks METADATA/WHEEL digests for {name}=={version}"
            )
        license_value = record.get("license")
        if not isinstance(license_value, dict):
            raise ValueError(f"Wheel artifact manifest lacks trusted license evidence for {name}=={version}")
        license_id = str(license_value.get("id", "")).strip()
        evidence = str(license_value.get("evidence", "")).strip()
        if not license_id or license_id.upper() in {"UNKNOWN", "NOASSERTION"}:
            raise ValueError(f"Wheel artifact manifest has no trusted license identity for {name}=={version}")
        if evidence not in {"wheel-member", "authenticated-source"}:
            raise ValueError(f"Wheel artifact manifest has unsupported license evidence for {name}=={version}")
        if evidence == "authenticated-source" and not _approved_source_url(
            str(license_value.get("source_url", "")), approved_origins
        ):
            raise ValueError(f"Wheel artifact manifest has untrusted license source for {name}=={version}")
        license_files = license_value.get("files")
        if not isinstance(license_files, list) or not license_files:
            raise ValueError(f"Wheel artifact manifest has no license material for {name}=={version}")
        seen_license_paths: set[str] = set()
        for license_file in license_files:
            if not isinstance(license_file, dict):
                raise ValueError(f"Wheel artifact manifest has malformed license material for {name}=={version}")
            license_path = str(license_file.get("path", "")).strip()
            license_text = str(license_file.get("text", ""))
            license_digest = str(license_file.get("sha256", "")).lower()
            path_parts = PurePosixPath(license_path)
            if (
                not license_path
                or path_parts.is_absolute()
                or ".." in path_parts.parts
                or license_path in seen_license_paths
                or not license_text.strip()
                or not _SHA256_RE.fullmatch(license_digest)
                or hashlib.sha256(license_text.encode("utf-8")).hexdigest() != license_digest
            ):
                raise ValueError(f"Wheel artifact manifest has invalid license material for {name}=={version}")
            seen_license_paths.add(license_path)
        key = (name, version, filename)
        if key in seen:
            raise ValueError(f"Duplicate wheel artifact record: {filename}")
        seen.add(key)
    return payload


def _artifact_records(manifest: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (
            normalize_package_name(str(record["name"])),
            str(record["version"]),
            str(record["filename"]),
        ): record
        for record in manifest["artifacts"]
    }


def _safe_wheel_member(info: zipfile.ZipInfo) -> PurePosixPath:
    relative = PurePosixPath(info.filename)
    if relative.is_absolute() or ".." in relative.parts or not relative.as_posix():
        raise ValueError(f"Unsafe wheel archive member: {info.filename}")
    if info.external_attr >> 16 & 0o170000 == 0o120000:
        raise ValueError(f"Symlinked wheel archive member is not allowed: {info.filename}")
    return relative


def _read_wheel_member_text(path: Path, member_path: str) -> str:
    """Read one UTF-8 wheel member after applying the archive safety rules."""

    requested = PurePosixPath(member_path)
    if requested.is_absolute() or ".." in requested.parts or not requested.as_posix():
        raise ValueError(f"Unsafe license member path: {member_path}")
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                relative = _safe_wheel_member(info)
                if relative.as_posix() == requested.as_posix():
                    return archive.read(info.filename).decode("utf-8")
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise ValueError(f"Wheel license member is unreadable: {path.name}:{member_path}") from exc
    raise ValueError(f"Wheel does not contain the declared license member: {path.name}:{member_path}")


def _declared_license_identities(metadata: Any) -> tuple[str, ...]:
    identities = [
        str(value).strip()
        for value in metadata.get_all("License-Expression", [])
        if str(value).strip()
    ]
    identities.extend(
        text
        for value in metadata.get_all("License", [])
        if (text := str(value).strip())
        and "\n" not in text
        and len(text) <= _MAX_LICENSE_IDENTITY_LENGTH
    )
    return tuple(
        value
        for value in identities
        if value.upper() not in _UNKNOWN_LICENSE_IDENTITIES
    )


def _inspect_wheel(path: Path, pin: Any) -> dict[str, Any]:
    """Parse and validate the wheel container and its CP312 metadata."""

    parsed_filename = _wheel_filename_parts(path.name)
    if parsed_filename is None or not _wheel_matches_pin(path, pin):
        raise ValueError(f"Wheel filename is incompatible with {pin.requirement}: {path.name}")
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Wheel ZIP integrity check failed: {path.name}")
            infos = [_safe_wheel_member(info) for info in archive.infolist()]
            metadata_names = [
                info for info in infos
                if len(info.parts) >= 2 and info.parts[-1] == "METADATA" and info.parts[-2].endswith(".dist-info")
            ]
            wheel_names = [
                info for info in infos
                if len(info.parts) >= 2 and info.parts[-1] == "WHEEL" and info.parts[-2].endswith(".dist-info")
            ]
            record_names = [
                info for info in infos
                if len(info.parts) >= 2 and info.parts[-1] == "RECORD" and info.parts[-2].endswith(".dist-info")
            ]
            if len(metadata_names) != 1 or len(wheel_names) != 1 or len(record_names) != 1:
                raise ValueError(f"Wheel is missing unique METADATA/WHEEL/RECORD members: {path.name}")
            metadata_bytes = archive.read(metadata_names[0].as_posix())
            wheel_bytes = archive.read(wheel_names[0].as_posix())
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"Wheel is not a valid ZIP archive: {path.name}") from exc
    try:
        metadata = Parser().parsestr(metadata_bytes.decode("utf-8"))
        wheel_metadata = Parser().parsestr(wheel_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Wheel metadata is not valid UTF-8: {path.name}") from exc
    name = normalize_package_name(str(metadata.get("Name", "")))
    version = str(metadata.get("Version", ""))
    if name != normalize_package_name(pin.name) or version != pin.version:
        raise ValueError(
            f"Wheel METADATA identity mismatch for {path.name}: {name}=={version}"
        )
    tags = [str(tag).strip() for tag in wheel_metadata.get_all("Tag", []) if str(tag).strip()]
    if not tags:
        raise ValueError(f"Wheel WHEEL metadata has no Tag entries: {path.name}")
    filename_parts = parsed_filename
    filename_tags = {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in filename_parts[3].split(".")
        for abi_tag in filename_parts[4].split(".")
        for platform_tag in filename_parts[5].split(".")
    }
    if not any(
        len(tag.split("-")) == 3
        and _wheel_tag_is_cp312_compatible(*tag.split("-"))
        for tag in tags
    ):
        raise ValueError(f"Wheel WHEEL tags are not CPython 3.12 Windows compatible: {path.name}")
    if not _wheel_tag_is_cp312_compatible(*filename_parts[3:]):
        raise ValueError(f"Wheel filename tag is not CPython 3.12 Windows compatible: {path.name}")
    if not filename_tags.issubset(tags):
        raise ValueError(f"Wheel filename tag set is absent from WHEEL metadata: {path.name}")
    license_metadata = _declared_license_identities(metadata)
    return {
        "name": name,
        "version": version,
        "filename": path.name,
        "sha256": sha256_file(path),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "wheel_metadata_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "requires_dist": tuple(str(value) for value in metadata.get_all("Requires-Dist", [])),
        "wheel_tags": tuple(tags),
        "filename_tags": tuple(sorted(filename_tags)),
        "license_metadata": license_metadata,
    }


def _source_matches_index(source_url: str, index_url: str | None) -> bool:
    if not index_url:
        return True
    if index_url.rstrip("/") == PYPI_INDEX.rstrip("/"):
        return source_url.startswith("https://files.pythonhosted.org/")
    return source_url.rstrip("/").startswith(index_url.rstrip("/") + "/")


def _version_key(value: str, *, include_local: bool = True) -> tuple[object, ...]:
    """Return a comparable key for the PEP 440 forms admitted by release locks."""

    normalized = value.strip().lower().replace("-", ".").replace("_", ".")
    public, separator, local = normalized.partition("+")
    match = re.fullmatch(
        r"(?:(?P<epoch>\d+)!)?"
        r"(?P<release>\d+(?:\.\d+)*)"
        r"(?:(?:\.)?(?P<pre>a|b|rc|alpha|beta|pre|preview)(?P<pre_n>\d*)?)?"
        r"(?:(?:\.)?(?P<post>post|rev|r)(?P<post_n>\d*)?)?"
        r"(?:(?:\.)?dev(?P<dev_n>\d*)?)?",
        public,
    )
    if not match:
        raise ValueError(f"Unsupported wheel dependency version: {value}")
    release = tuple(int(part) for part in match.group("release").split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    pre_name = match.group("pre")
    pre_number = int(match.group("pre_n") or 0)
    pre_rank = {
        "a": 0,
        "alpha": 0,
        "b": 1,
        "beta": 1,
        "rc": 2,
        "pre": 2,
        "preview": 2,
    }
    dev_number = match.group("dev_n")
    if pre_name is None and dev_number is not None:
        pre = (-1, 0)
    elif pre_name is None:
        pre = (3, 0)
    else:
        pre = (pre_rank[pre_name], pre_number)
    post = (0, int(match.group("post_n") or 0)) if match.group("post") else (-1, 0)
    dev = (0, int(dev_number or 0)) if dev_number is not None else (1, 0)
    local_key: tuple[tuple[int, object], ...] = ()
    if include_local and separator:
        local_key = tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in local.split(".")
        )
    return int(match.group("epoch") or 0), release, pre, post, dev, local_key


def _specifier_satisfied(version: str, specifier: str) -> bool:
    for raw_part in (part.strip() for part in specifier.split(",")):
        if not raw_part:
            continue
        match = re.fullmatch(r"(===|==|!=|<=|>=|~=|<|>)[ ]*([^ ]+)", raw_part)
        if not match:
            raise ValueError(f"Unsupported wheel dependency specifier: {specifier}")
        operator, expected = match.groups()
        wildcard = expected.endswith(".*")
        if wildcard:
            expected_prefix = tuple(int(part) for part in expected[:-2].split("."))
            actual_release = _version_key(version)[1]
            wildcard_match = tuple(actual_release[:len(expected_prefix)]) == expected_prefix
            if operator == "==" and not wildcard_match:
                return False
            if operator == "!=" and wildcard_match:
                return False
            if operator in {"==", "!="}:
                continue
        include_local = "+" in expected
        actual_key = _version_key(version, include_local=include_local)
        expected_key = _version_key(expected, include_local=include_local)
        if operator == "==" and actual_key != expected_key:
            return False
        if operator == "===" and version != expected:
            return False
        if operator == "!=" and actual_key == expected_key:
            return False
        if operator == "<" and not actual_key < expected_key:
            return False
        if operator == "<=" and not actual_key <= expected_key:
            return False
        if operator == ">" and not actual_key > expected_key:
            return False
        if operator == ">=" and not actual_key >= expected_key:
            return False
        if operator == "~=":
            expected_release = expected_key[1]
            prefix_length = max(1, len(expected_release) - 1)
            if not (
                actual_key >= expected_key
                and actual_key[1][:prefix_length] == expected_release[:prefix_length]
            ):
                return False
    return True


def _split_marker_expression(expression: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    token = f" {operator} "
    index = 0
    while index < len(expression):
        character = expression[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth == 0 and expression[index:index + len(token)].lower() == token:
            parts.append(expression[start:index])
            start = index + len(token)
            index = start
            continue
        index += 1
    parts.append(expression[start:])
    return parts


def _marker_applies(marker: str | None) -> bool:
    if not marker:
        return True
    expression = marker.strip()
    while expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1].strip()
    or_parts = _split_marker_expression(expression, "or")
    if len(or_parts) > 1:
        return any(_marker_applies(part) for part in or_parts)
    and_parts = _split_marker_expression(expression, "and")
    if len(and_parts) > 1:
        return all(_marker_applies(part) for part in and_parts)
    match = re.fullmatch(
        r"(?P<name>python_version|python_full_version|sys_platform|platform_system|platform_machine|platform_python_implementation|implementation_name|extra)\s*"
        r"(?P<operator>not in|in|==|!=|<=|>=|<|>)\s*['\"]?(?P<value>[^'\"]+)['\"]?",
        expression,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unsupported wheel dependency marker: {marker}")
    values = {
        "python_version": "3.12",
        "python_full_version": "3.12.10",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "extra": "",
    }
    actual = values[match.group("name").lower()]
    operator = match.group("operator").lower()
    expected = match.group("value").strip()
    if operator == "in":
        return actual in expected.split()
    if operator == "not in":
        return actual not in expected.split()
    if operator in {"<", "<=", ">", ">="} and match.group("name").lower().startswith("python"):
        actual_value: object = _version_key(actual)
        expected_value: object = _version_key(expected)
    else:
        actual_value = actual
        expected_value = expected
    return {
        "==": actual_value == expected_value,
        "!=": actual_value != expected_value,
        "<": actual_value < expected_value,
        "<=": actual_value <= expected_value,
        ">": actual_value > expected_value,
        ">=": actual_value >= expected_value,
    }[operator]


def _requirement_from_metadata(raw_requirement: str) -> tuple[str, str, str | None]:
    requirement, _, marker = raw_requirement.partition(";")
    requirement = requirement.strip()
    match = re.match(r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?(?P<specifier>.*)$", requirement)
    if not match:
        raise ValueError(f"Malformed wheel dependency: {raw_requirement}")
    specifier = match.group("specifier").strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1].strip()
    return (
        normalize_package_name(match.group("name")),
        specifier,
        marker.strip() or None,
    )


def validate_wheelhouse(
    lock: LockSet,
    wheelhouse: Path,
    *,
    label: str,
    artifact_manifest: Mapping[str, Any] | None = None,
    expected_wheels: Mapping[object, str] | None = None,
    expected_indexes: Mapping[object, str] | None = None,
) -> dict[str, Path]:
    """Prove each lock pin maps to one independently identified CP312 wheel."""

    if not wheelhouse.is_dir():
        raise ValueError(f"{label} wheelhouse is missing: {wheelhouse}")
    if artifact_manifest is None:
        raise ValueError("A repository-detached wheel artifact manifest is required")
    records = _artifact_records(artifact_manifest)
    wheels = sorted(path for path in wheelhouse.glob("*.whl") if path.is_file())
    selected: dict[str, Path] = {}
    for pin in lock.pins:
        normalized_name = normalize_package_name(pin.name)
        matches = [path for path in wheels if _wheel_matches_pin(path, pin)]
        valid: list[Path] = []
        for path in matches:
            audit = _inspect_wheel(path, pin)
            key = (normalized_name, pin.version, path.name)
            record = records.get(key)
            if record is None:
                continue
            actual_digest = str(audit["sha256"]).lower()
            if actual_digest != str(record.get("sha256", "")).lower():
                continue
            if actual_digest not in {value.lower() for value in pin.hashes}:
                continue
            if str(record.get("metadata_sha256", "")).lower() != str(audit["metadata_sha256"]).lower():
                continue
            if str(record.get("wheel_metadata_sha256", "")).lower() != str(audit["wheel_metadata_sha256"]).lower():
                continue
            expected_index = (expected_indexes or {}).get((normalized_name, pin.version))
            if expected_index is None:
                expected_index = (expected_indexes or {}).get(normalized_name)
            if not _source_matches_index(str(record.get("source_url", "")), expected_index):
                continue
            expected_filename = (expected_wheels or {}).get((normalized_name, pin.version))
            if expected_filename is None:
                expected_filename = (expected_wheels or {}).get(normalized_name)
            if expected_filename is not None and path.name != expected_filename:
                continue
            valid.append(path)
        if len(valid) != 1:
            raise ValueError(
                f"{label} lacks exactly one compatible independently identified wheel for {pin.requirement}"
            )
        selected[normalized_name] = valid[0]
    return selected


def _validate_wheel_graph_closure(
    locks: Sequence[LockSet],
    selected: Mapping[tuple[str, str], Path],
) -> None:
    pins: dict[tuple[str, str], Any] = {}
    pins_by_name: dict[str, list[Any]] = {}
    for lock in locks:
        for pin in lock.pins:
            name = normalize_package_name(pin.name)
            key = (name, pin.version)
            existing = pins.get(key)
            if existing is not None and (existing.version != pin.version or existing.hashes != pin.hashes):
                raise ValueError(f"Conflicting locked identities for {name}")
            pins[key] = pin
            if pin not in pins_by_name.setdefault(name, []):
                pins_by_name[name].append(pin)
    audits = {
        key: _inspect_wheel(path, pins[key])
        for key, path in selected.items()
    }
    for (name, _version), audit in audits.items():
        for raw_requirement in audit["requires_dist"]:
            dependency_name, specifier, marker = _requirement_from_metadata(raw_requirement)
            if not _marker_applies(marker):
                continue
            dependency_candidates = pins_by_name.get(dependency_name, [])
            matching_candidates = [
                candidate
                for candidate in dependency_candidates
                if not specifier or _specifier_satisfied(candidate.version, specifier)
            ]
            if not matching_candidates:
                raise ValueError(
                    f"Wheel dependency closure is incomplete: {name} requires {raw_requirement}"
                )


def validate_wheelhouse_bundle(
    locks: Sequence[LockSet],
    wheelhouse: Path,
    *,
    artifact_manifest: Mapping[str, Any],
    expected_wheels: Mapping[object, str] | None = None,
    expected_indexes: Mapping[object, str] | None = None,
    closure_groups: Sequence[Sequence[LockSet]] | None = None,
) -> dict[tuple[str, str], Path]:
    """Validate the complete shared/profile wheel inventory and graph closure."""

    selected: dict[tuple[str, str], Path] = {}
    for lock in locks:
        current = validate_wheelhouse(
            lock,
            wheelhouse,
            label=lock.name,
            artifact_manifest=artifact_manifest,
            expected_wheels=expected_wheels,
            expected_indexes=expected_indexes,
        )
        for name, path in current.items():
            key = (name, next(pin.version for pin in lock.pins if normalize_package_name(pin.name) == name))
            if key in selected and selected[key].name != path.name:
                raise ValueError(f"Multiple wheel identities selected for {name}=={key[1]}")
            selected[key] = path
    for group in (closure_groups or (locks,)):
        group_selected = {
            (normalize_package_name(pin.name), pin.version): selected[(normalize_package_name(pin.name), pin.version)]
            for lock in group
            for pin in lock.pins
        }
        _validate_wheel_graph_closure(group, group_selected)
    wheel_names = {path.name for path in wheelhouse.glob("*.whl") if path.is_file()}
    selected_names = {path.name for path in selected.values()}
    extra = sorted(wheel_names - selected_names)
    if extra:
        raise ValueError("Wheelhouse contains unapproved extra wheel(s): " + ", ".join(extra[:8]))
    manifest_names = {
        str(record["filename"])
        for record in artifact_manifest.get("artifacts", [])
    }
    if manifest_names != selected_names:
        missing = sorted(selected_names - manifest_names)
        extra_manifest = sorted(manifest_names - selected_names)
        raise ValueError(
            f"Wheel artifact manifest does not match the approved inventory: missing={missing[:8]}, "
            f"extra={extra_manifest[:8]}"
        )
    return selected


def prepare_lock_inputs(
    app_root: Path,
    profile: DependencyProfile,
    *,
    shared_lock: Path,
    canonical_shared_lock: Path,
    profile_lock: Path | None,
    artifact_hashes: dict[str, str],
    wheelhouse: Path,
    artifact_manifest: Mapping[str, Any],
) -> None:
    destination = app_root / "bootstrap" / "locks"
    destination.mkdir(parents=True, exist_ok=True)
    shared = load_lock(shared_lock)
    validate_lock(shared, require_hashes=True)
    canonical = load_lock(canonical_shared_lock, require_hashes=False)
    _validate_complete_shared_lock(shared, canonical)
    try:
        if shared.pin("insightface").version != "0.7.3":
            raise ValueError("InsightFace must be pinned to 0.7.3 in the shared lock")
    except ProfileError:
        raise ValueError("The shared lock must contain the approved insightface==0.7.3 resolution")
    forbidden_shared = {"torch", "torchvision", "xformers"}
    if any(pin.name.lower().replace("_", "-") in forbidden_shared for pin in shared.pins):
        raise ValueError("CUDA-sensitive packages belong only in the selected profile lock")
    validate_wheelhouse(
        shared,
        wheelhouse,
        label="Shared",
        artifact_manifest=artifact_manifest,
    )
    if profile_lock is not None:
        selected = load_lock(profile_lock)
    else:
        selected = profile_lock_from_profile(profile, hash_overrides=artifact_hashes)
    validate_lock(selected, profile=profile, require_hashes=True)
    expected = {package.normalized_name for package in profile.packages}
    actual = {pin.name.lower().replace("_", "-") for pin in selected.pins}
    if actual != expected:
        raise ValueError(f"Profile lock contains unapproved packages: {profile.name}")
    validate_wheelhouse(
        selected,
        wheelhouse,
        label=profile.name,
        artifact_manifest=artifact_manifest,
        expected_wheels={package.normalized_name: package.wheel_filename for package in profile.packages},
        expected_indexes={package.normalized_name: package.index_url for package in profile.packages},
    )
    write_lock(destination / "shared-py312-win-amd64.txt", shared)
    write_lock(destination / f"{profile.name}-py312-win-amd64.txt", selected)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_required_text(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"Required {label} is missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Required {label} is empty: {path}")
    return text


def _notice_locks(
    shared: LockSet,
    profiles: Iterable[DependencyProfile],
    profile_locks: Mapping[str, LockSet] | None,
) -> list[LockSet]:
    locks = [shared]
    profile_locks = profile_locks or {}
    for profile in profiles:
        locks.append(
            profile_locks.get(profile.name)
            or profile_lock_from_profile(profile)
        )
    return locks


def validate_dependency_notice(
    lock: LockSet,
    path: Path,
    *,
    profiles: Iterable[DependencyProfile] = (),
    profile_locks: Mapping[str, LockSet] | None = None,
    artifact_manifest: Mapping[str, Any] | None = None,
    audited_wheels: Mapping[tuple[str, str], Path] | None = None,
) -> dict[str, Any]:
    """Validate the structured redistribution notice/SBOM contract.

    A package-name inventory is deliberately not accepted.  Every exact
    shared/profile wheel must be bound to the audited filename and digest,
    HTTPS source, declared SPDX-like license identity, and actual included
    license material.  The same contract is used to generate the archive's
    deterministic human-readable notice index.
    """

    if not path.is_file():
        raise ValueError(f"Required dependency license notice report is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Dependency license notice report must be structured JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1 or payload.get("format") != "nexfocus-wheel-notices":
        raise ValueError("Dependency license notice report has an unsupported schema")
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("Dependency license notice report has no artifact records")

    locks = _notice_locks(lock, profiles, profile_locks)
    expected: dict[tuple[str, str], Any] = {}
    for candidate_lock in locks:
        validate_lock(candidate_lock, require_hashes=True)
        for pin in candidate_lock.pins:
            key = (normalize_package_name(pin.name), pin.version)
            if key in expected:
                raise ValueError(f"Duplicate notice identity in lock graph: {pin.requirement}")
            expected[key] = pin
    manifest_records = _artifact_records(artifact_manifest) if artifact_manifest is not None else {}
    actual: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Dependency license notice report contains a malformed artifact record")
        name = normalize_package_name(str(record.get("name", "")))
        version = str(record.get("version", ""))
        key = (name, version)
        if not name or not version or key in actual:
            raise ValueError(f"Dependency license notice report has a duplicate/invalid identity: {key}")
        pin = expected.get(key)
        if pin is None:
            raise ValueError(f"Dependency license notice report contains an unapproved package: {name}=={version}")
        filename = str(record.get("filename", ""))
        digest = str(record.get("sha256", "")).lower()
        if not filename.endswith(".whl") or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Notice record is missing a wheel filename or valid SHA-256: {name}=={version}")
        if digest not in {value.lower() for value in pin.hashes}:
            raise ValueError(f"Notice digest does not match the completed lock: {pin.requirement}")
        manifest_record = manifest_records.get((name, version, filename))
        if artifact_manifest is None or manifest_record is None:
            raise ValueError(f"Notice record is not bound to an authenticated wheel identity: {filename}")
        if digest != str(manifest_record.get("sha256", "")).lower():
            raise ValueError(f"Notice digest does not match the authenticated wheel identity: {filename}")
        if str(record.get("source_url", "")) != str(manifest_record.get("source_url", "")):
            raise ValueError(f"Notice source does not match the authenticated wheel identity: {filename}")
        source_parts = urlparse(str(record.get("source_url", "")))
        if source_parts.scheme != "https" or not source_parts.netloc:
            raise ValueError(f"Notice source is not an HTTPS artifact URL: {filename}")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("source") or not provenance.get("retrieved_utc"):
            raise ValueError(f"Notice provenance is incomplete: {filename}")
        license_value = record.get("license") or record.get("license_id")
        if isinstance(license_value, dict):
            license_id = str(license_value.get("id", "")).strip()
        else:
            license_id = str(license_value or "").strip()
        if not license_id or license_id.upper() in {"UNKNOWN", "NOASSERTION"}:
            raise ValueError(f"Notice record has no declared license identity: {filename}")
        trusted_license = manifest_record.get("license")
        if not isinstance(trusted_license, dict):
            raise ValueError(f"Notice record has no detached license evidence: {filename}")
        trusted_license_id = str(trusted_license.get("id", "")).strip()
        evidence = str(trusted_license.get("evidence", "")).strip()
        if license_id != trusted_license_id:
            raise ValueError(f"Notice license identity does not match detached evidence: {filename}")
        trusted_files = trusted_license.get("files")
        if not isinstance(trusted_files, list) or not trusted_files:
            raise ValueError(f"Detached license evidence has no material: {filename}")
        license_files = record.get("license_files")
        if not isinstance(license_files, list) or not license_files:
            raise ValueError(f"Notice record has no included license file material: {filename}")
        def _material_tuple(items: list[Any]) -> tuple[tuple[str, str, str], ...]:
            material: list[tuple[str, str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"Notice license file is malformed: {filename}")
                material.append(
                    (
                        str(item.get("path", "")).strip(),
                        str(item.get("sha256", "")).lower(),
                        str(item.get("text", item.get("content", ""))),
                    )
                )
            return tuple(sorted(material))

        if _material_tuple(license_files) != _material_tuple(trusted_files):
            raise ValueError(f"Notice license material does not match detached evidence: {filename}")
        for license_file in license_files:
            if not isinstance(license_file, dict):
                raise ValueError(f"Notice license file is malformed: {filename}")
            license_path = str(license_file.get("path", "")).strip()
            license_text = str(license_file.get("text", license_file.get("content", "")))
            license_digest = str(license_file.get("sha256", "")).lower()
            if not license_path or not license_text.strip() or not _SHA256_RE.fullmatch(license_digest):
                raise ValueError(f"Notice license file material is incomplete: {filename}")
            if hashlib.sha256(license_text.encode("utf-8")).hexdigest() != license_digest:
                raise ValueError(f"Notice license file digest is incorrect: {filename}:{license_path}")
        if audited_wheels is None:
            raise ValueError(f"Notice record is not bound to an audited wheel: {filename}")
        wheel_path = audited_wheels.get(key)
        if wheel_path is None or wheel_path.name != filename:
            raise ValueError(f"Notice record is not bound to the audited wheel bytes: {filename}")
        wheel_audit = _inspect_wheel(wheel_path, pin)
        declared_license = tuple(wheel_audit.get("license_metadata", ()))
        if declared_license and license_id not in declared_license:
            raise ValueError(f"Notice license identity conflicts with wheel metadata: {filename}")
        if evidence == "wheel-member":
            for license_file in license_files:
                license_path = str(license_file["path"])
                wheel_text = _read_wheel_member_text(wheel_path, license_path)
                if wheel_text != str(license_file.get("text", license_file.get("content", ""))):
                    raise ValueError(f"Notice license material does not match wheel member: {filename}:{license_path}")
        elif evidence == "authenticated-source":
            source_url = str(trusted_license.get("source_url", ""))
            if urlparse(source_url).scheme != "https" or not urlparse(source_url).netloc:
                raise ValueError(f"Detached license source is not authenticated: {filename}")
        else:
            raise ValueError(f"Detached license evidence has an unsupported type: {filename}")
        if not str(record.get("license_text", "")).strip() and not license_files:
            raise ValueError(f"Notice record has no license text: {filename}")
        actual[key] = record

    missing = sorted(set(expected) - set(actual))
    if missing:
        display = ", ".join(f"{name}=={version}" for name, version in missing[:8])
        raise ValueError("Dependency license notice report is incomplete; missing artifact records: " + display)
    return payload


def write_generated_notices(
    destination: Path,
    source_notice: Path,
    shared_lock_path: Path,
    profiles: Iterable[DependencyProfile],
    dependency_notice: Path,
    *,
    profile_locks: Mapping[str, LockSet] | None = None,
    artifact_manifest: Mapping[str, Any] | None = None,
    audited_wheels: Mapping[tuple[str, str], Path] | None = None,
) -> None:
    """Add deterministic exact-package license material to the notice index."""

    shared = load_lock(shared_lock_path)
    source_text = _read_required_text(source_notice, label="source notice index")
    profiles = tuple(profiles)
    contract = validate_dependency_notice(
        shared,
        dependency_notice,
        profiles=profiles,
        profile_locks=profile_locks,
        artifact_manifest=artifact_manifest,
        audited_wheels=audited_wheels,
    )
    lines = [source_text, "", "Generated locked package license inventory", "--------------------------------------------"]
    for record in sorted(contract["artifacts"], key=lambda item: (normalize_package_name(item["name"]), item["version"], item["filename"])):
        license_value = record.get("license") or record.get("license_id")
        license_id = license_value.get("id") if isinstance(license_value, dict) else license_value
        lines.extend(
            [
                "",
                f"Package: {record['name']}=={record['version']}",
                f"Wheel: {record['filename']}",
                f"SHA256: {record['sha256']}",
                f"Source: {record['source_url']}",
                f"License: {license_id}",
                "License material:",
            ]
        )
        for license_file in record["license_files"]:
            lines.extend(
                [
                    f"  {license_file['path']} (sha256:{license_file['sha256']})",
                    str(license_file.get("text", license_file.get("content", ""))).rstrip(),
                ]
            )
    _write_text(destination, "\n".join(lines) + "\n")


def build_archive_manifest(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def _content_inventory_digest(root: Path, inventory: Iterable[str]) -> str:
    entries = []
    for relative in sorted(inventory):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Copied source inventory member disappeared: {relative}")
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return _payload_sha256(entries)


def _wheel_audit_records(
    artifact_manifest: Mapping[str, Any],
    selected: Mapping[object, Path],
) -> list[dict[str, Any]]:
    records = _artifact_records(artifact_manifest)
    output = []
    unique_paths = {path.resolve(): path for path in selected.values()}
    for path in sorted(unique_paths.values(), key=lambda item: item.name):
        matching = [
            record
            for (_record_name, _version, filename), record in records.items()
            if filename == path.name
        ]
        if len(matching) != 1:
            raise ValueError(f"Wheel audit record is missing for {path.name}")
        record = matching[0]
        output.append(
            {
                "name": record["name"],
                "version": record["version"],
                "filename": record["filename"],
                "sha256": record["sha256"],
                "metadata_sha256": record["metadata_sha256"],
                "wheel_metadata_sha256": record["wheel_metadata_sha256"],
                "source_url": record["source_url"],
                "license": record["license"],
            }
        )
    return output


def scan_staged_content(root: Path) -> None:
    """Reject credentials, personal paths, and user/runtime state before ZIP creation."""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        normalized = f"/{relative.lower()}/"
        path_parts = set(PurePosixPath(relative.lower()).parts)
        if (
            path.name.lower() == ".env"
            or path.name.lower().startswith(".env.")
            or path.name.lower() in {"config.txt", "credentials.json"}
        ):
            raise ValueError(f"Credential/config state leaked into staging: {relative}")
        if path_parts.intersection({"models", "outputs", "credentials", "venv", "runtime-cache"}):
            raise ValueError(f"User state leaked into staging: {relative}")
        if any(fragment in normalized for fragment in ("/cache/", "/__pycache__/", "/.git/", "/release/")):
            raise ValueError(f"Cache or release state leaked into staging: {relative}")
        if path.suffix.lower() == ".whl" and relative.lower().startswith("app/bootstrap/wheelhouse/") \
                and path.name.lower().startswith("insightface-"):
            pass
        elif path.suffix.lower() in {".whl", ".zip", ".7z", ".ckpt", ".safetensors", ".gguf", ".onnx"}:
            raise ValueError(f"Generated binary input leaked into the public archive: {relative}")
        data = path.read_bytes()
        if b"\x00" in data[:4096]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in _SECRET_CONTENT_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"Secret or personal path detected in staged content: {relative}")
        for match in _NONEMPTY_ENV_CREDENTIAL.finditer(text):
            remainder = text[match.end():].lstrip()
            if not remainder or remainder.startswith(("#", "\r", "\n")):
                continue
            if remainder.startswith(('"', "'")):
                quote = remainder[0]
                if len(remainder) > 1 and remainder[1] == quote:
                    continue
                if remainder[1:].split(quote, 1)[0].strip():
                    raise ValueError(f"Secret or credential assignment detected in staged content: {relative}")
                continue
            if remainder.splitlines()[0].strip():
                raise ValueError(f"Secret or credential assignment detected in staged content: {relative}")
        if re.search(
            r"(?i)(?:api[_-]?key|access[_-]?token|password|client_secret)\s*[:=]\s*[\"']?[^\s\"'<>{}]+",
            text,
        ):
            raise ValueError(f"Credential assignment detected in staged content: {relative}")


def audit_archive_manifest(
    entries: Iterable[dict[str, object]],
    *,
    require_both_profiles: bool = False,
    root: Path | None = None,
) -> None:
    paths = {str(entry["path"]) for entry in entries}
    forbidden_fragments = ("/models/", "/outputs/", "/venv/", "/__pycache__/", "/cache/")
    for path in paths:
        normalized = f"/{path}/"
        path_parts = set(PurePosixPath(path).parts)
        if path_parts.intersection({".env", "config.txt", "credentials"}):
            raise ValueError(f"Credential/config state leaked into archive: {path}")
        if any(fragment in normalized for fragment in forbidden_fragments):
            raise ValueError(f"User state leaked into archive: {path}")
    required = {"Nexfocus.bat", "runtime/python312/python.exe", "runtime/uv.exe", "app/bootstrap/launcher.py"}
    if not any(path.startswith("app/bootstrap/wheelhouse/insightface-") and path.endswith(".whl") for path in paths):
        raise ValueError("Archive is missing the approved InsightFace wheelhouse input")
    if require_both_profiles:
        required.update(
            {
                "app/bootstrap/locks/legacy-cu124-py312-win-amd64.txt",
                "app/bootstrap/locks/modern-cu128-py312-win-amd64.txt",
            }
        )
    missing = required - paths
    if missing:
        raise ValueError(f"Archive is missing required one-click files: {sorted(missing)}")
    if root is not None:
        scan_staged_content(root)


def deterministic_zip(root: Path, destination: Path, *, require_both_profiles: bool = False) -> None:
    entries = build_archive_manifest(root)
    audit_archive_manifest(entries, require_both_profiles=require_both_profiles, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in entries:
            path = root / str(entry["path"])
            info = zipfile.ZipInfo(str(entry["path"]))
            info.date_time = (2020, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_release(
    source_root: Path,
    output_dir: Path,
    *,
    python_zip: Path,
    python_sha256: str | None = None,
    uv_binary: Path,
    uv_sha256: str | None = None,
    shared_lock: Path,
    wheelhouse: Path | None = None,
    profile_locks: dict[str, Path] | None = None,
    legacy_xformers_sha256: str | None = None,
    insightface_wheel: Path | None = None,
    insightface_sha256: str | None = None,
    wheel_manifest: Path | None = None,
    dependency_notices: Path | None = None,
    profile_name: str = "both",
    release_version: str | None = None,
) -> Path:
    source_root = source_root.resolve()
    contract = load_release_inputs(source_root)
    if not release_version or release_version in {"working-tree", "unversioned"} or "/" in release_version or "\\" in release_version:
        raise ValueError("A concrete release version is required; working-tree is not an artifact identity")
    profiles = list(APPROVED_PROFILES.values()) if profile_name == "both" else [validate_approved_profile(profile_name)]
    for profile in profiles:
        profile.validate(require_hashes=True)
    if wheelhouse is None:
        raise ValueError("A complete external Windows CP312 wheelhouse is required")
    if insightface_wheel is None:
        raise ValueError("The approved Windows CP312 InsightFace wheel is required")
    if dependency_notices is None:
        raise ValueError("A generated dependency license notice report is required")
    if wheel_manifest is None:
        raise ValueError("A repository-detached wheel artifact manifest is required")
    wheel_manifest_contract = contract["wheel_artifact_manifest"]
    trust_root_path = source_root / str(wheel_manifest_contract["trust_root"])
    artifact_manifest = load_wheel_artifact_manifest(wheel_manifest, trust_root=trust_root_path)
    profile_locks = profile_locks or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nexfocus-release-") as temporary:
        staging = Path(temporary)
        app_root = staging / "app"
        copied_inventory = copy_application(source_root, app_root)
        runtime_python = staging / "runtime" / "python312"
        extract_python_embed(
            python_zip,
            runtime_python,
            python_sha256,
            contract=contract["python"],
        )
        add_embedded_app_path(runtime_python)
        copy_uv(
            uv_binary,
            staging / "runtime" / "uv.exe",
            uv_sha256,
            contract=contract["uv"],
        )
        python_executable_sha256 = sha256_file(runtime_python / str(contract["python"].get("executable_member", "python.exe")))
        uv_binary_sha256 = sha256_file(staging / "runtime" / "uv.exe")
        batch_source = source_root / "windows" / "Nexfocus.bat.in"
        if not batch_source.is_file():
            raise ValueError(f"Windows launcher template is missing: {batch_source}")
        shutil.copyfile(batch_source, staging / "Nexfocus.bat")
        shutil.copyfile(source_root / "bootstrap" / "build_contract.json", staging / "build_contract.json")
        shutil.copyfile(source_root / "bootstrap" / "ownership.json", staging / "ownership.json")

        uv_license_source = source_root / str(contract["uv"]["license_source"])
        verify_sha256(uv_license_source, str(contract["uv"]["license_sha256"]))
        shutil.copyfile(uv_license_source, staging / "runtime" / "UV-LICENSE.txt")

        canonical_shared_lock = source_root / str(contract["shared_lock_source"])
        if not canonical_shared_lock.is_file():
            raise ValueError(f"Canonical shared-lock source is missing: {canonical_shared_lock}")

        shared_input = load_lock(shared_lock)
        insightface_pin = shared_input.pin("insightface")
        insightface_expected = str(contract["insightface"]["version"])
        if insightface_pin.version != insightface_expected:
            raise ValueError(f"InsightFace must be pinned to {insightface_expected}")

        # A single archive carries both profile locks. Each is validated before
        # it enters the ZIP; installers never resolve or choose a third profile.
        selected_locks: dict[str, LockSet] = {}
        for profile in profiles:
            if (
                legacy_xformers_sha256 is not None
                and profile.name == "legacy-cu124"
                and legacy_xformers_sha256.lower() != profile.package("xformers").sha256.lower()
            ):
                raise ValueError("Caller-supplied legacy xformers digest does not match the profile contract")
            prepare_lock_inputs(
                app_root,
                profile,
                shared_lock=shared_lock,
                canonical_shared_lock=canonical_shared_lock,
                profile_lock=profile_locks.get(profile.name),
                artifact_hashes={},
                wheelhouse=wheelhouse,
                artifact_manifest=artifact_manifest,
            )

            selected_locks[profile.name] = load_lock(
                profile_locks.get(profile.name)
                if profile_locks.get(profile.name) is not None
                else app_root / "bootstrap" / "locks" / f"{profile.name}-py312-win-amd64.txt"
            )

        expected_wheels = {
            (package.normalized_name, package.version): package.wheel_filename
            for profile in profiles
            for package in profile.packages
        }
        expected_indexes = {
            (package.normalized_name, package.version): package.index_url
            for profile in profiles
            for package in profile.packages
        }
        wheel_inventory = validate_wheelhouse_bundle(
            [shared_input, *selected_locks.values()],
            wheelhouse,
            artifact_manifest=artifact_manifest,
            expected_wheels=expected_wheels,
            expected_indexes=expected_indexes,
            closure_groups=[
                [shared_input, selected_locks[profile.name]]
                for profile in profiles
            ],
        )
        insightface_paths = [
            path
            for (name, _version), path in wheel_inventory.items()
            if name == normalize_package_name("insightface")
        ]
        if len(insightface_paths) != 1 or insightface_paths[0].resolve() != insightface_wheel.resolve():
            raise ValueError("The approved InsightFace wheel must be present in the validated wheelhouse")
        insightface_actual_sha256 = sha256_file(insightface_wheel)
        if insightface_sha256 is not None and insightface_sha256.lower() != insightface_actual_sha256.lower():
            raise ValueError("Caller-supplied InsightFace digest does not match the wheel bytes")
        wheel_target = app_root / "bootstrap" / "wheelhouse" / insightface_wheel.name
        wheel_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(insightface_wheel, wheel_target)
        write_generated_notices(
            staging / "THIRD-PARTY-NOTICES.txt",
            source_root / "THIRD-PARTY-NOTICES.txt",
            shared_lock,
            profiles,
            dependency_notices,
            profile_locks=selected_locks,
            artifact_manifest=artifact_manifest,
            audited_wheels=wheel_inventory,
        )

        source_content_digest = _content_inventory_digest(app_root, copied_inventory)
        shared_lock_destination = app_root / "bootstrap" / "locks" / "shared-py312-win-amd64.txt"
        profile_lock_digests = {
            name: sha256_file(app_root / "bootstrap" / "locks" / f"{name}-py312-win-amd64.txt")
            for name in selected_locks
        }
        wheel_audit = _wheel_audit_records(artifact_manifest, wheel_inventory)
        build_manifest = {
            "schema": 2,
            "artifact_version": release_version,
            "build_id": release_version,
            "archive_name": ARCHIVE_NAME,
            # This is content-addressed source provenance, not merely a list
            # of filenames that could remain unchanged while file contents
            # drift underneath the release builder.
            "source_inventory_sha256": source_content_digest,
            "source_content_sha256": source_content_digest,
            "shared_lock_sha256": sha256_file(shared_lock_destination),
            "profile_lock_sha256": profile_lock_digests,
            "wheel_artifact_manifest_sha256": str(artifact_manifest["authentication"]["manifest_sha256"]),
            "wheel_audit": wheel_audit,
            "wheel_audit_sha256": _payload_sha256(wheel_audit),
            "dependency_notice_sha256": sha256_file(dependency_notices),
            "python": {
                "version": contract["python"]["version"],
                "archive_name": contract["python"]["archive_name"],
                "source_url": contract["python"]["source_url"],
                "sha256": contract["python"]["sha256"],
                "executable_sha256": python_executable_sha256,
                "license_members": contract["python"]["license_members"],
            },
            "uv": {
                "version": contract["uv"]["version"],
                "archive_name": contract["uv"]["archive_name"],
                "source_url": contract["uv"]["source_url"],
                "sha256": contract["uv"]["sha256"],
                "binary_sha256": uv_binary_sha256,
                "license_sha256": contract["uv"]["license_sha256"],
            },
            "profiles": [profile.as_manifest() for profile in profiles],
        }
        build_manifest["build_manifest_sha256"] = _payload_sha256(build_manifest)
        _write_text(
            app_root / "bootstrap" / "build_manifest.json",
            json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        )
        # If both profiles are present, the second loop has overwritten the
        # shared lock only identically and both profile locks remain.
        manifest = build_archive_manifest(staging)
        _write_text(
            staging / "ARCHIVE-MANIFEST.json",
            json.dumps({"schema": 1, "entries": manifest}, indent=2, sort_keys=True) + "\n",
        )
        archive_path = output_dir / ARCHIVE_NAME
        deterministic_zip(staging, archive_path, require_both_profiles=profile_name == "both")
        checksum = sha256_file(archive_path)
        _write_text(output_dir / "SHA256SUMS.txt", f"{checksum}  {archive_path.name}\n")
        _write_text(
            output_dir / "RELEASE-NOTES.md",
            "# Nexfocus Windows one-click release\n\n"
            "This candidate targets Windows x64 with an externally installed NVIDIA driver. "
            "The archive contains private Python 3.12 and uv, but not models or user state.\n\n"
            "The bootstrap selects legacy-cu124 below compute capability 7.5 and modern-cu128 "
            "at or above 7.5. Dependency installation is wheel-only, hash-verified, and retryable.\n\n"
            f"Artifact version: `{release_version}`.\n"
            "First launch requires a supported NVIDIA driver and network access for the locked wheel cache. "
            "Models are downloaded separately through Nexfocus.\n",
        )
        return archive_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("dist") / "windows")
    parser.add_argument("--python-embed-zip", type=Path, required=True)
    parser.add_argument("--python-sha256", help="Deprecated assertion; must equal the checked-in digest")
    parser.add_argument("--uv-binary", type=Path, required=True, help="Official pinned uv Windows ZIP")
    parser.add_argument("--uv-sha256", help="Deprecated assertion; must equal the checked-in digest")
    parser.add_argument("--shared-lock", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True, help="Complete external Windows CP312 wheelhouse")
    parser.add_argument("--legacy-xformers-sha256", help="Deprecated assertion; must equal the checked-in digest")
    parser.add_argument("--insightface-wheel", type=Path, required=True)
    parser.add_argument("--insightface-sha256", help="Optional byte assertion; the completed lock remains authoritative")
    parser.add_argument("--wheel-manifest", type=Path, required=True, help="Wheel identity manifest bound to the repository trust root")
    parser.add_argument("--dependency-notices", type=Path, required=True)
    parser.add_argument("--release-version", required=True, help="Concrete artifact version, for example 0.1.0")
    parser.add_argument("--profile", choices=["both", *APPROVED_PROFILES], default="both")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_release(
        args.source_root.resolve(),
        args.output_dir.resolve(),
        python_zip=args.python_embed_zip.resolve(),
        python_sha256=args.python_sha256,
        uv_binary=args.uv_binary.resolve(),
        uv_sha256=args.uv_sha256,
        shared_lock=args.shared_lock.resolve(),
        wheelhouse=args.wheelhouse.resolve(),
        legacy_xformers_sha256=args.legacy_xformers_sha256,
        insightface_wheel=args.insightface_wheel.resolve() if args.insightface_wheel else None,
        insightface_sha256=args.insightface_sha256,
        wheel_manifest=args.wheel_manifest.resolve() if args.wheel_manifest else None,
        dependency_notices=args.dependency_notices.resolve() if args.dependency_notices else None,
        profile_name=args.profile,
        release_version=args.release_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
