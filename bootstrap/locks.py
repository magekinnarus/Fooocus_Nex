"""Strict lock-file parsing for wheel-only private runtime installs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import ProfileError
from .profiles import DependencyProfile, normalize_package_name


_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)(?P<rest>.*)$"
)
_HASH = re.compile(r"--hash\s*=\s*sha256:([0-9a-fA-F]{64})")


@dataclass(frozen=True)
class LockPin:
    name: str
    version: str
    hashes: tuple[str, ...]
    index_url: str | None = None
    wheel_filename: str | None = None

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"

    def as_line(self) -> str:
        hashes = " ".join(f"--hash=sha256:{value}" for value in self.hashes)
        return f"{self.requirement} {hashes}".rstrip()


@dataclass(frozen=True)
class LockSet:
    name: str
    python: str
    platform: str
    pins: tuple[LockPin, ...]
    wheel_only: bool = True
    require_hashes: bool = True

    def pin(self, name: str) -> LockPin:
        wanted = normalize_package_name(name)
        for pin in self.pins:
            if normalize_package_name(pin.name) == wanted:
                return pin
        raise ProfileError(f"Lock {self.name!r} has no package {name!r}")

    def as_requirements(self) -> str:
        header = [
            f"# Nexfocus lock: {self.name}",
            f"# Python: {self.python}; platform: {self.platform}",
            "# Generated release input; do not edit pins by hand.",
        ]
        return "\n".join(header + [pin.as_line() for pin in self.pins]) + "\n"


def parse_requirements_lock(
    path: Path,
    *,
    name: str | None = None,
    require_hashes: bool = True,
) -> LockSet:
    pins: list[LockPin] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("--"):
            raise ProfileError(f"Unsupported global option in lock {path}:{line_number}")
        match = _REQUIREMENT.match(line)
        if not match or ";" in match.group("rest"):
            raise ProfileError(f"Lock must contain exact wheel pins: {path}:{line_number}")
        rest = match.group("rest")
        hashes = tuple(_HASH.findall(rest))
        if any(token.startswith("--hash") is False for token in rest.split() if token):
            # A non-hash token may be an accidental URL, editable, or index
            # option.  Reject it instead of letting the resolver reinterpret it.
            leftovers = [token for token in rest.split() if not token.startswith("--hash")]
            if leftovers:
                raise ProfileError(f"Unsupported lock token(s) at {path}:{line_number}: {leftovers}")
        pins.append(
            LockPin(
                name=match.group(1),
                version=match.group(2),
                hashes=hashes,
            )
        )
    lock = LockSet(
        name=name or path.stem,
        python="3.12",
        platform="win_amd64",
        pins=tuple(pins),
    )
    validate_lock(lock, require_hashes=require_hashes)
    return lock


def load_json_lock(path: Path, *, require_hashes: bool = True) -> LockSet:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Could not read lock: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), list):
        raise ProfileError(f"Invalid JSON lock: {path}")
    pins = []
    for item in payload["packages"]:
        if not isinstance(item, dict) or not item.get("name") or not item.get("version"):
            raise ProfileError(f"Invalid package entry in lock: {path}")
        hashes = tuple(item.get("hashes", ()))
        pins.append(
            LockPin(
                name=str(item["name"]),
                version=str(item["version"]),
                hashes=tuple(str(value).removeprefix("sha256:") for value in hashes),
                index_url=item.get("index_url"),
                wheel_filename=item.get("wheel_filename"),
            )
        )
    lock = LockSet(
        name=str(payload.get("name", path.stem)),
        python=str(payload.get("python", "3.12")),
        platform=str(payload.get("platform", "win_amd64")),
        pins=tuple(pins),
        wheel_only=bool(payload.get("wheel_only", True)),
        require_hashes=bool(payload.get("require_hashes", True)),
    )
    validate_lock(lock, require_hashes=require_hashes)
    return lock


def load_lock(path: Path, *, require_hashes: bool = True) -> LockSet:
    if path.suffix.lower() == ".json":
        return load_json_lock(path, require_hashes=require_hashes)
    return parse_requirements_lock(path, require_hashes=require_hashes)


def validate_lock(
    lock: LockSet,
    *,
    profile: DependencyProfile | None = None,
    require_hashes: bool | None = None,
) -> None:
    if lock.python != "3.12" or lock.platform != "win_amd64":
        raise ProfileError(f"Lock {lock.name} is not a Windows CP312 lock")
    if not lock.pins:
        raise ProfileError(f"Lock {lock.name} is empty")
    if not lock.wheel_only:
        raise ProfileError(f"Lock {lock.name} must be wheel-only")
    hashes_required = lock.require_hashes if require_hashes is None else require_hashes
    seen: set[str] = set()
    for pin in lock.pins:
        normalized = normalize_package_name(pin.name)
        if normalized in seen:
            raise ProfileError(f"Duplicate package in lock {lock.name}: {pin.name}")
        seen.add(normalized)
        if not pin.version or any(character in pin.version for character in "<>~=!* "):
            raise ProfileError(f"Non-exact version in lock {lock.name}: {pin.requirement}")
        if hashes_required:
            if not pin.hashes or any(not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in pin.hashes):
                raise ProfileError(f"Missing or invalid SHA-256 for {pin.requirement}")
        if profile is not None:
            expected = {normalize_package_name(package.name): package for package in profile.packages}
            actual_profile_names = {normalize_package_name(candidate.name) for candidate in lock.pins}
            if actual_profile_names != set(expected):
                raise ProfileError(f"Profile lock does not contain exactly the approved package set: {lock.name}")
            for name, package in expected.items():
                pin = next((candidate for candidate in lock.pins if normalize_package_name(candidate.name) == name), None)
                if pin is None or pin.version != package.version:
                    raise ProfileError(f"Profile lock drift for {package.name}: expected {package.version}")
                if pin.index_url and pin.index_url.rstrip("/") != package.index_url.rstrip("/"):
                    raise ProfileError(f"Profile lock index drift for {package.name}")
                if package.sha256 and tuple(pin.hashes) != (package.sha256,):
                    raise ProfileError(f"Profile lock hash drift for {package.name}")
                if pin.wheel_filename and pin.wheel_filename != package.wheel_filename:
                    raise ProfileError(f"Profile lock wheel drift for {package.name}")


def profile_lock_from_profile(
    profile: DependencyProfile,
    *,
    hash_overrides: dict[str, str] | None = None,
) -> LockSet:
    overrides = {normalize_package_name(key): value for key, value in (hash_overrides or {}).items()}
    pins = []
    for package in profile.packages:
        digest = package.sha256 or overrides.get(package.normalized_name)
        pins.append(
            LockPin(
                name=package.name,
                version=package.version,
                hashes=(digest,) if digest else (),
                index_url=package.index_url,
                wheel_filename=package.wheel_filename,
            )
        )
    lock = LockSet(
        name=f"{profile.name}-py312-win-amd64",
        python="3.12",
        platform="win_amd64",
        pins=tuple(pins),
    )
    validate_lock(lock, profile=profile, require_hashes=False)
    return lock


def write_lock(path: Path, lock: LockSet) -> None:
    validate_lock(lock, require_hashes=lock.require_hashes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lock.as_requirements(), encoding="utf-8", newline="\n")


__all__ = [
    "LockPin",
    "LockSet",
    "load_json_lock",
    "load_lock",
    "parse_requirements_lock",
    "profile_lock_from_profile",
    "validate_lock",
    "write_lock",
]
