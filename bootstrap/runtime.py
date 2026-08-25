"""Validation of an installed private runtime without importing the app."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from .errors import InstallError, ProfileError
from .layout import RuntimeLayout
from .manifest import current_runtime, read_json
from .profiles import DependencyProfile, normalize_package_name


def installed_versions(site_packages: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    if not site_packages.is_dir():
        return versions
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            versions[normalize_package_name(name)] = version
    return versions


def installed_sensitive_versions(site_packages: Path, profile: DependencyProfile) -> dict[str, tuple[str, ...]]:
    """Return every CUDA-sensitive distribution, preserving duplicate metadata."""

    names = {package.normalized_name for package in profile.packages}
    found: dict[str, list[str]] = {name: [] for name in names}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = normalize_package_name(name)
        if normalized in names:
            found.setdefault(normalized, []).append(distribution.version)
    return {name: tuple(values) for name, values in found.items()}


def validate_site_packages(site_packages: Path, profile: DependencyProfile) -> dict[str, str]:
    """Check the three CUDA-sensitive packages exactly before committing."""

    versions = installed_versions(site_packages)
    sensitive = installed_sensitive_versions(site_packages, profile)
    for package in profile.packages:
        actual = versions.get(package.normalized_name)
        records = sensitive.get(package.normalized_name, ())
        if len(records) != 1 or records[0] != package.version or actual != package.version:
            raise InstallError(
                f"Installed runtime does not match {profile.name}: "
                f"{package.name} expected exactly one {package.version}, got {records or 'missing'}"
            )
    return versions


def validate_private_python(
    python_executable: Path,
    site_packages: Path,
    profile: DependencyProfile,
    *,
    runner: Any = subprocess.run,
    check_cuda: bool = False,
) -> None:
    if not python_executable.is_file():
        raise InstallError(f"Private Python is missing: {python_executable}")
    validate_site_packages(site_packages, profile)
    code = (
        f"import sys; sys.path.insert(0, {str(site_packages)!r}); "
        "import importlib.metadata as m; "
        f"assert m.version('torch') == {profile.package('torch').version!r}; "
        f"assert m.version('torchvision') == {profile.package('torchvision').version!r}; "
        f"assert m.version('xformers') == {profile.package('xformers').version!r}; "
        "import torch, xformers"
    )
    if check_cuda:
        code += (
            "; assert torch.cuda.is_available(); assert torch.version.cuda; "
            "from xformers.ops import memory_efficient_attention; "
            "q=torch.randn((1,4,1,64), device='cuda', dtype=torch.float16); "
            "y=memory_efficient_attention(q,q,q); assert tuple(y.shape)==tuple(q.shape)"
        )
    try:
        result = runner(
            [str(python_executable), "-s", "-c", code],
            cwd=str(python_executable.parent),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise InstallError(f"Private Python could not validate the runtime: {python_executable}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown validation error").strip()
        raise InstallError(f"Runtime import validation failed: {detail[:500]}")


def current_profile(layout: RuntimeLayout) -> str | None:
    current = current_runtime(layout)
    if current is None:
        return None
    return str(current[1].get("profile")) if current[1].get("profile") else None


__all__ = [
    "current_profile",
    "installed_sensitive_versions",
    "installed_versions",
    "validate_private_python",
    "validate_site_packages",
]
