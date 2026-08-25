"""The closed Windows x64/NVIDIA dependency profile contract.

Keeping this contract in a small stdlib-only module makes the release builder,
installer, launcher, and tests agree on the same pins.  A profile is selected
from the GPU's compute capability, never from the locally installed torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping
from urllib.parse import quote

from .errors import ProfileError, UnsupportedHardwareError


PYTHON_VERSION = "3.12"
PYTHON_TAG = "cp312"
PLATFORM_TAG = "win_amd64"
COMPUTE_CAPABILITY_BOUNDARY = (7, 5)
PYTORCH_INDEX_ROOT = "https://download.pytorch.org/whl"
PYPI_INDEX = "https://pypi.org/simple"


def normalize_package_name(name: str) -> str:
    """Normalize a PEP 503 package name for comparisons."""

    return name.strip().lower().replace("_", "-").replace(".", "-")


@dataclass(frozen=True)
class PackagePin:
    """One exact wheel pin used by a release profile."""

    name: str
    version: str
    index_url: str
    wheel_filename: str
    sha256: str | None
    no_deps: bool = True

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"

    @property
    def wheel_url(self) -> str:
        return f"{self.index_url.rstrip('/')}/{quote(self.wheel_filename)}"

    @property
    def normalized_name(self) -> str:
        return normalize_package_name(self.name)

    def as_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "requirement": self.requirement,
            "index_url": self.index_url,
            "wheel_filename": self.wheel_filename,
            "wheel_url": self.wheel_url,
            "sha256": self.sha256,
            "no_deps": self.no_deps,
        }


@dataclass(frozen=True)
class DependencyProfile:
    """An approved, exact PyTorch/xformers profile."""

    name: str
    cuda_family: str
    compute_capability_min: tuple[int, int] | None
    compute_capability_max_exclusive: tuple[int, int] | None
    pytorch_index: str
    packages: tuple[PackagePin, ...]

    def package(self, name: str) -> PackagePin:
        wanted = normalize_package_name(name)
        for package in self.packages:
            if package.normalized_name == wanted:
                return package
        raise ProfileError(f"Profile {self.name!r} has no package {name!r}")

    @property
    def package_names(self) -> frozenset[str]:
        return frozenset(package.normalized_name for package in self.packages)

    def supports_compute_capability(self, capability: tuple[int, int]) -> bool:
        if self.compute_capability_min is not None and capability < self.compute_capability_min:
            return False
        if (
            self.compute_capability_max_exclusive is not None
            and capability >= self.compute_capability_max_exclusive
        ):
            return False
        return True

    def validate(self, *, require_hashes: bool = False) -> None:
        """Reject drift from the two-profile release contract."""

        try:
            canonical = _CANONICAL_PROFILE_SPECS[self.name]
        except KeyError as exc:
            raise ProfileError(f"Unapproved profile: {self.name}") from exc
        if self.cuda_family != canonical["cuda_family"]:
            raise ProfileError(f"Profile {self.name} has an unapproved CUDA family")
        if self.compute_capability_min != canonical["compute_capability_min"]:
            raise ProfileError(f"Profile {self.name} has an altered minimum capability boundary")
        if self.compute_capability_max_exclusive != canonical["compute_capability_max_exclusive"]:
            raise ProfileError(f"Profile {self.name} has an altered maximum capability boundary")
        if self.pytorch_index.rstrip("/") != canonical["pytorch_index"].rstrip("/"):
            raise ProfileError(f"Profile {self.name} has a mismatched PyTorch index")
        if len(self.packages) != len(canonical["packages"]):
            raise ProfileError(f"Profile {self.name} must contain exactly the approved package set")

        expected = canonical["packages"]
        seen: set[str] = set()
        for package in self.packages:
            name = package.normalized_name
            if name in seen or name not in expected:
                raise ProfileError(f"Profile {self.name} contains an unapproved package set")
            seen.add(name)
            expected_package = expected[name]
            if package.name != expected_package["name"]:
                raise ProfileError(f"Profile {self.name} has a renamed package pin: {package.name}")
            if package.version != expected_package["version"]:
                raise ProfileError(
                    f"Profile {self.name} drifted: {package.name}=={package.version}; "
                    f"expected {expected_package['version']}"
                )
            if package.index_url.rstrip("/") != expected_package["index_url"].rstrip("/"):
                raise ProfileError(f"{package.name} must use {expected_package['index_url']}")
            if package.wheel_filename != expected_package["wheel_filename"]:
                raise ProfileError(f"{package.name} has an unapproved wheel filename")
            if package.no_deps is not expected_package["no_deps"]:
                raise ProfileError(f"{package.name} has an altered dependency-install posture")
            if package.sha256 != expected_package["sha256"]:
                raise ProfileError(f"{package.name} has an unapproved SHA-256 digest")
            if require_hashes and not _is_sha256(package.sha256):
                raise ProfileError(
                    f"No approved SHA-256 is recorded for {self.name}/{package.wheel_filename}"
                )
        if seen != set(expected):
            raise ProfileError(f"Profile {self.name} is missing an approved package")

    @property
    def expected_index(self) -> str:
        try:
            return str(_CANONICAL_PROFILE_SPECS[self.name]["pytorch_index"])
        except KeyError as exc:
            raise ProfileError(f"Unapproved profile: {self.name}") from exc

    def as_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "cuda_family": self.cuda_family,
            "python": PYTHON_VERSION,
            "python_tag": PYTHON_TAG,
            "platform": PLATFORM_TAG,
            "pytorch_index": self.pytorch_index,
            "compute_capability_min": _format_capability(self.compute_capability_min),
            "compute_capability_max_exclusive": _format_capability(
                self.compute_capability_max_exclusive
            ),
            "packages": [package.as_manifest() for package in self.packages],
        }


def _is_sha256(value: str | None) -> bool:
    if value is None or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _format_capability(value: tuple[int, int] | None) -> str | None:
    if value is None:
        return None
    return f"{value[0]}.{value[1]}"


def parse_compute_capability(value: object) -> tuple[int, int]:
    """Parse ``7.5``/``(7, 5)`` without floating point boundary errors."""

    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) != 2:
            raise ProfileError(f"Invalid compute capability: {value!r}")
        try:
            major, minor = int(value[0]), int(value[1])
        except (TypeError, ValueError) as exc:
            raise ProfileError(f"Invalid compute capability: {value!r}") from exc
    else:
        text = str(value).strip().replace(",", ".")
        parts = text.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ProfileError(f"Invalid compute capability: {value!r}")
        minor_text = parts[1].rstrip("0") or "0"
        if len(minor_text) > 2:
            raise ProfileError(f"Invalid compute capability: {value!r}")
        major, minor = int(parts[0]), int(minor_text)
    if major < 0 or minor < 0 or minor > 99:
        raise ProfileError(f"Invalid compute capability: {value!r}")
    return major, minor


def select_profile(compute_capability: object) -> DependencyProfile:
    capability = parse_compute_capability(compute_capability)
    if capability < COMPUTE_CAPABILITY_BOUNDARY:
        return LEGACY_PROFILE
    return MODERN_PROFILE


def validate_approved_profile(profile: DependencyProfile | str, *, require_hashes: bool = False) -> DependencyProfile:
    if isinstance(profile, str):
        try:
            profile = APPROVED_PROFILES[profile]
        except KeyError as exc:
            raise ProfileError(f"Unapproved profile: {profile}") from exc
    profile.validate(require_hashes=require_hashes)
    return profile


def reject_unapproved_packages(packages: Iterable[tuple[str, str]]) -> None:
    """Validate an installed package/version set against the closed contract."""

    normalized = {(normalize_package_name(name), version) for name, version in packages}
    approved = {
        (normalize_package_name(package.name), package.version)
        for profile in (LEGACY_PROFILE, MODERN_PROFILE)
        for package in profile.packages
    }
    unexpected = normalized - approved
    if unexpected:
        display = ", ".join(f"{name}=={version}" for name, version in sorted(unexpected))
        raise ProfileError(f"Unapproved dependency pin(s): {display}")


def _pin(
    name: str,
    version: str,
    cuda_family: str,
    wheel_filename: str,
    sha256: str | None,
) -> PackagePin:
    return PackagePin(
        name=name,
        version=version,
        index_url=f"{PYTORCH_INDEX_ROOT}/{cuda_family}",
        wheel_filename=wheel_filename,
        sha256=sha256,
    )


# All package digests are part of the checked-in release contract.  The
# PyTorch simple index does not publish a hash fragment for every wheel, so the
# digest was independently recorded from the official wheel before being
# admitted here.
LEGACY_PROFILE = DependencyProfile(
    name="legacy-cu124",
    cuda_family="cu124",
    compute_capability_min=None,
    compute_capability_max_exclusive=COMPUTE_CAPABILITY_BOUNDARY,
    pytorch_index=f"{PYTORCH_INDEX_ROOT}/cu124",
    packages=(
        _pin(
            "torch",
            "2.5.1+cu124",
            "cu124",
            "torch-2.5.1+cu124-cp312-cp312-win_amd64.whl",
            "3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6",
        ),
        _pin(
            "torchvision",
            "0.20.1+cu124",
            "cu124",
            "torchvision-0.20.1+cu124-cp312-cp312-win_amd64.whl",
            "0f6c7b3b0e13663fb3359e64f3604c0ab74c2b4809ae6949ace5635a5240f0e5",
        ),
        _pin(
            "xformers",
            "0.0.28.post3",
            "cu124",
            "xformers-0.0.28.post3-cp312-cp312-win_amd64.whl",
            "3a3a0b8b12fadff6a111effcdc3573c6a584a8d71fbd5ead77999fec658ab919",
        ),
    ),
)

MODERN_PROFILE = DependencyProfile(
    name="modern-cu128",
    cuda_family="cu128",
    compute_capability_min=COMPUTE_CAPABILITY_BOUNDARY,
    compute_capability_max_exclusive=None,
    pytorch_index=f"{PYTORCH_INDEX_ROOT}/cu128",
    packages=(
        PackagePin(
            name="sympy",
            version="1.13.3",
            index_url=PYPI_INDEX,
            wheel_filename="sympy-1.13.3-py3-none-any.whl",
            sha256="54612cf55a62755ee71824ce692986f23c88ffa77207b30c1368eda4a7060f73",
        ),
        _pin(
            "torch",
            "2.11.0+cu128",
            "cu128",
            "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl",
            "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c",
        ),
        _pin(
            "torchvision",
            "0.26.0+cu128",
            "cu128",
            "torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl",
            "8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b",
        ),
        _pin(
            "xformers",
            "0.0.35",
            "cu128",
            "xformers-0.0.35-py39-none-win_amd64.whl",
            "57381ce3cbb79b593e6b62cb20a937885345fad2796de2aa6fbb66c033601179",
        ),
    ),
)

APPROVED_PROFILES: Mapping[str, DependencyProfile] = MappingProxyType(
    {
        LEGACY_PROFILE.name: LEGACY_PROFILE,
        MODERN_PROFILE.name: MODERN_PROFILE,
    }
)


# This is deliberately independent from the dataclass instances above.  The
# builder and installer must reject a mutated object rather than deriving the
# expected answer from the object's own mutable fields.
_CANONICAL_PROFILE_SPECS: Mapping[str, dict[str, object]] = MappingProxyType(
    {
        "legacy-cu124": {
            "cuda_family": "cu124",
            "compute_capability_min": None,
            "compute_capability_max_exclusive": COMPUTE_CAPABILITY_BOUNDARY,
            "pytorch_index": f"{PYTORCH_INDEX_ROOT}/cu124",
            "packages": {
                "torch": {
                    "name": "torch",
                    "version": "2.5.1+cu124",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu124",
                    "wheel_filename": "torch-2.5.1+cu124-cp312-cp312-win_amd64.whl",
                    "sha256": "3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6",
                    "no_deps": True,
                },
                "torchvision": {
                    "name": "torchvision",
                    "version": "0.20.1+cu124",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu124",
                    "wheel_filename": "torchvision-0.20.1+cu124-cp312-cp312-win_amd64.whl",
                    "sha256": "0f6c7b3b0e13663fb3359e64f3604c0ab74c2b4809ae6949ace5635a5240f0e5",
                    "no_deps": True,
                },
                "xformers": {
                    "name": "xformers",
                    "version": "0.0.28.post3",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu124",
                    "wheel_filename": "xformers-0.0.28.post3-cp312-cp312-win_amd64.whl",
                    "sha256": "3a3a0b8b12fadff6a111effcdc3573c6a584a8d71fbd5ead77999fec658ab919",
                    "no_deps": True,
                },
            },
        },
        "modern-cu128": {
            "cuda_family": "cu128",
            "compute_capability_min": COMPUTE_CAPABILITY_BOUNDARY,
            "compute_capability_max_exclusive": None,
            "pytorch_index": f"{PYTORCH_INDEX_ROOT}/cu128",
            "packages": {
                "sympy": {
                    "name": "sympy",
                    "version": "1.13.3",
                    "index_url": PYPI_INDEX,
                    "wheel_filename": "sympy-1.13.3-py3-none-any.whl",
                    "sha256": "54612cf55a62755ee71824ce692986f23c88ffa77207b30c1368eda4a7060f73",
                    "no_deps": True,
                },
                "torch": {
                    "name": "torch",
                    "version": "2.11.0+cu128",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu128",
                    "wheel_filename": "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl",
                    "sha256": "7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c",
                    "no_deps": True,
                },
                "torchvision": {
                    "name": "torchvision",
                    "version": "0.26.0+cu128",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu128",
                    "wheel_filename": "torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl",
                    "sha256": "8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b",
                    "no_deps": True,
                },
                "xformers": {
                    "name": "xformers",
                    "version": "0.0.35",
                    "index_url": f"{PYTORCH_INDEX_ROOT}/cu128",
                    "wheel_filename": "xformers-0.0.35-py39-none-win_amd64.whl",
                    "sha256": "57381ce3cbb79b593e6b62cb20a937885345fad2796de2aa6fbb66c033601179",
                    "no_deps": True,
                },
            },
        },
    }
)

for _profile in (LEGACY_PROFILE, MODERN_PROFILE):
    _profile.validate()


__all__ = [
    "APPROVED_PROFILES",
    "COMPUTE_CAPABILITY_BOUNDARY",
    "DependencyProfile",
    "LEGACY_PROFILE",
    "MODERN_PROFILE",
    "PackagePin",
    "PLATFORM_TAG",
    "PYPI_INDEX",
    "PYTHON_TAG",
    "PYTHON_VERSION",
    "normalize_package_name",
    "parse_compute_capability",
    "reject_unapproved_packages",
    "select_profile",
    "validate_approved_profile",
]
