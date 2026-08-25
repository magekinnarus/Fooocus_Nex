"""Retryable, versioned, transactional private-runtime installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .errors import InstallError, ProfileError, TransactionError
from .layout import RuntimeLayout, is_within
from .locks import LockSet, load_lock, validate_lock
from .manifest import RuntimeManifest, current_runtime, point_current, read_json, write_runtime_markers
from .profiles import DependencyProfile, PYPI_INDEX, PYTHON_VERSION, validate_approved_profile
from .runtime import validate_private_python, validate_site_packages
from .user_data import redact_text


Runner = Callable[..., subprocess.CompletedProcess[str]]
ProgressCallback = Callable[[str], None]
MAX_READY_RUNTIMES = 2


@dataclass(frozen=True)
class InstallResult:
    version: str
    profile: str
    installed: bool
    runtime_dir: Path
    log_path: Path


class PrivateRuntimeInstaller:
    """Install only into a versioned directory and publish it last."""

    def __init__(
        self,
        layout: RuntimeLayout,
        profile: DependencyProfile,
        *,
        app_root: Path | None = None,
        python_executable: Path | None = None,
        uv_executable: Path | None = None,
        shared_lock: Path | None = None,
        profile_lock: Path | None = None,
        artifact_hashes: dict[str, str] | None = None,
        build_manifest: dict[str, object] | None = None,
        gpu_info: dict[str, object] | None = None,
        runner: Runner = subprocess.run,
        retries: int = 3,
        release_version: str = "development",
        check_cuda: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.layout = layout
        self.profile = validate_approved_profile(profile, require_hashes=True)
        self.app_root = app_root or layout.app_root
        self.python_executable = python_executable or layout.python_executable
        self.uv_executable = uv_executable or layout.uv_executable
        lock_root = self.app_root / "bootstrap" / "locks"
        self.shared_lock = shared_lock or lock_root / "shared-py312-win-amd64.txt"
        self.profile_lock = profile_lock or lock_root / f"{self.profile.name}-py312-win-amd64.txt"
        self.wheelhouse = self.app_root / "bootstrap" / "wheelhouse"
        self.artifact_hashes = artifact_hashes or {}
        if self.artifact_hashes:
            raise ProfileError("Caller-supplied profile digests are not accepted; use the packaged profile lock")
        self.runner = runner
        self.retries = max(1, retries)
        self.build_manifest = build_manifest or self._load_build_manifest()
        self.gpu_info = gpu_info or {}
        self.release_version = str(
            self.build_manifest.get("artifact_version", release_version)
        )
        if self.release_version == "working-tree":
            raise InstallError("The packaged runtime has no traceable artifact version")
        self.check_cuda = check_cuda
        self.progress = progress or self._default_progress
        self._log_path = self.layout.install_log_root / "bootstrap.log"

    def _load_build_manifest(self) -> dict[str, object]:
        path = self.app_root / "bootstrap" / "build_manifest.json"
        if not path.is_file():
            return {}
        try:
            value = read_json(path)
        except Exception as exc:
            raise InstallError(f"The packaged build manifest is unreadable: {path}") from exc
        if not value.get("artifact_version") or value.get("artifact_version") == "working-tree":
            raise InstallError("The packaged build manifest has no concrete artifact version")
        claimed_manifest_digest = value.get("build_manifest_sha256")
        unsigned_manifest = {key: item for key, item in value.items() if key != "build_manifest_sha256"}
        encoded_manifest = json.dumps(
            unsigned_manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if not isinstance(claimed_manifest_digest, str) or hashlib.sha256(encoded_manifest).hexdigest() != claimed_manifest_digest:
            raise InstallError("The packaged build manifest failed its content-integrity check")
        profiles = value.get("profiles")
        if not isinstance(profiles, list) or not any(
            isinstance(item, dict) and item.get("name") == self.profile.name for item in profiles
        ):
            raise ProfileError(f"Build manifest does not contain profile {self.profile.name}")
        return value

    def ensure(self, *, repair: bool = False) -> InstallResult:
        """Reuse a valid current runtime; package resolution happens once."""

        self.layout.ensure_runtime_directories()
        # Reuse is still a release-integrity gate: a changed bundled Python,
        # uv binary, or lock must not be hidden by an otherwise-valid runtime
        # directory and READY marker.
        self._validate_inputs()
        current = current_runtime(
            self.layout,
            expected_artifact_version=self.release_version,
            expected_profile=self.profile.name,
            expected_build_manifest=self.build_manifest,
        )
        if not repair and current is not None and current[1].get("profile") == self.profile.name:
            runtime_dir = current[0]
            try:
                validate_site_packages(runtime_dir / "site-packages", self.profile)
            except InstallError:
                self._log("Current runtime failed metadata validation; repairing it")
            else:
                return InstallResult(
                    version=str(current[1]["version"]),
                    profile=self.profile.name,
                    installed=False,
                    runtime_dir=runtime_dir,
                    log_path=self._log_path,
                )
        return self.install()

    def install(self) -> InstallResult:
        self.layout.ensure_runtime_directories()
        self._validate_inputs()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version = f"{self.profile.name}-{timestamp}-{uuid.uuid4().hex[:8]}"
        staging = Path(tempfile.mkdtemp(prefix=f"{version}-", dir=self.layout.staging_root))
        final_dir = self.layout.version_root / version
        log_path = self._log_path
        current_published = False
        try:
            site_packages = staging / "site-packages"
            site_packages.mkdir(parents=True, exist_ok=True)
            shared_lock = load_lock(self.shared_lock)
            self._validate_shared_lock(shared_lock)
            profile_lock = self._load_profile_lock(staging)
            self._validate_profile_lock(profile_lock)
            self._install_profile_core(profile_lock, site_packages)
            # The shared graph is complete and already has the exact CUDA
            # packages in place.  No resolver may inspect its metadata and
            # replace them with CPU or another CUDA family.
            self._install_lock(shared_lock, site_packages, index_url=PYPI_INDEX, no_deps=True)
            self._install_lock(
                self._single_pin_lock(profile_lock, "xformers"),
                site_packages,
                index_url=self.profile.pytorch_index,
                no_deps=True,
            )
            validate_site_packages(site_packages, self.profile)
            if self.check_cuda:
                validate_private_python(
                    self.python_executable,
                    site_packages,
                    self.profile,
                    check_cuda=True,
                )
            manifest = RuntimeManifest(
                version=version,
                profile=self.profile.name,
                python_version=PYTHON_VERSION,
                packages=tuple(
                    package.as_manifest() for package in self.profile.packages
                ),
                artifact_version=self.release_version,
                build_id=str(self.build_manifest.get("build_id", self.release_version)),
                python_identity=dict(self.build_manifest.get("python", {})),
                uv_identity=dict(self.build_manifest.get("uv", {})),
                cuda_family=self.profile.cuda_family,
                gpu=self.gpu_info,
                shared_packages=tuple(
                    {
                        "name": pin.name,
                        "version": pin.version,
                        "hashes": list(pin.hashes),
                        "index_url": pin.index_url,
                        "wheel_filename": pin.wheel_filename,
                    }
                    for pin in shared_lock.pins
                ),
                build_manifest_sha256=self.build_manifest.get("build_manifest_sha256"),
                source_content_sha256=self.build_manifest.get("source_content_sha256"),
                shared_lock_sha256=self.build_manifest.get("shared_lock_sha256"),
                profile_lock_sha256=(
                    self.build_manifest.get("profile_lock_sha256", {}).get(self.profile.name)
                    if isinstance(self.build_manifest.get("profile_lock_sha256"), dict)
                    else None
                ),
                wheel_audit_sha256=self.build_manifest.get("wheel_audit_sha256"),
                wheel_artifact_manifest_sha256=self.build_manifest.get("wheel_artifact_manifest_sha256"),
            )
            write_runtime_markers(staging, manifest)
            # The staging directory is now internally ready.  Moving it under
            # versions is the only publication step before current.json.
            os.replace(staging, final_dir)
            point_current(
                self.layout,
                version,
                self.profile.name,
                expected_build_manifest=self.build_manifest,
            )
            current_published = True
            self._prune_ready_runtimes()
            self._log(f"Committed runtime {version} ({self.profile.name})")
            return InstallResult(
                version=version,
                profile=self.profile.name,
                installed=True,
                runtime_dir=final_dir,
                log_path=log_path,
            )
        except Exception as exc:
            self._log(f"Install failed for {self.profile.name}: {type(exc).__name__}: {exc}")
            # If publication happened but the pointer did not, remove only the
            # newly created unreferenced runtime. Older/current runtimes survive.
            if final_dir.exists() and not current_published:
                self._remove_path(final_dir)
            if staging.exists():
                self._remove_path(staging)
            if isinstance(exc, (InstallError, ProfileError, TransactionError)):
                raise
            raise InstallError(f"Private runtime installation failed: {exc}") from exc

    def repair(self) -> InstallResult:
        """Remove incomplete staging and install a fresh version, preserving user data."""

        self.cleanup_incomplete()
        result = self.install()
        self._prune_ready_runtimes()
        return result

    def cleanup_incomplete(self) -> None:
        self.layout.ensure_runtime_directories()
        for path in self.layout.staging_root.iterdir():
            self._remove_path(path)
        self._cleanup_download_cache()
        current = current_runtime(self.layout)
        for path in self.layout.version_root.iterdir():
            if current is not None and path.resolve() == current[0].resolve():
                continue
            if not (path / "READY.json").is_file() or not (path / "runtime-manifest.json").is_file():
                self._remove_path(path)

    def _cleanup_download_cache(self) -> None:
        """Leave uv's persistent cache intact so its verifier can resume/reuse it.

        uv owns the cache layout and validates cached wheel bytes against the
        hash-required requirements file.  Repair is allowed to remove only
        Nexfocus-owned incomplete runtime staging; it must not delete uv's
        partial/download state and thereby turn a retry into a fresh download.
        """

        cache_root = self.layout.download_cache_root
        if not cache_root.is_dir():
            return
        if not is_within(cache_root, self.layout.runtime_root):
            raise TransactionError(f"Refusing to inspect a cache outside the private runtime: {cache_root}")
        files = [path for path in cache_root.rglob("*") if path.is_file() and is_within(path, cache_root)]
        self._log(f"Preserving {len(files)} uv cache file(s) for verified retry/reuse")

    def _prune_ready_runtimes(self, *, max_ready: int = MAX_READY_RUNTIMES) -> None:
        """Keep the current runtime plus one rollback runtime, at most."""

        current = current_runtime(self.layout)
        ready = [
            path
            for path in self.layout.version_root.iterdir()
            if path.is_dir() and (path / "READY.json").is_file() and (path / "runtime-manifest.json").is_file()
        ]
        ready.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        keep: set[Path] = set(ready[:max(1, max_ready)])
        if current is not None:
            keep.add(current[0].resolve())
        for path in ready:
            if path.resolve() not in keep:
                self._remove_path(path)

    def _validate_inputs(self) -> None:
        if not self.python_executable.is_file():
            raise InstallError(
                f"Private Python 3.12 is missing from the distribution: {self.python_executable}"
            )
        if not self.uv_executable.is_file():
            raise InstallError(
                f"Private uv is missing from the distribution: {self.uv_executable}"
            )
        if not self.shared_lock.is_file():
            raise InstallError(f"Shared Windows lock is missing: {self.shared_lock}")
        if not self.profile_lock.is_file():
            raise InstallError(f"Profile lock is missing: {self.profile_lock}")
        if not any(self.wheelhouse.glob("insightface-*.whl")):
            raise InstallError(
                "The approved InsightFace Windows wheel is missing from the release wheelhouse"
            )
        self._validate_packaged_provenance_inputs()

    def _validate_packaged_provenance_inputs(self) -> None:
        """Check release-critical files before a ready runtime can be built."""

        if not self.build_manifest:
            return
        python_identity = self.build_manifest.get("python")
        uv_identity = self.build_manifest.get("uv")
        if isinstance(python_identity, dict) and python_identity.get("executable_sha256"):
            if _sha256_file(self.python_executable) != str(python_identity["executable_sha256"]).lower():
                raise InstallError("Private Python bytes do not match the packaged provenance contract")
        if isinstance(uv_identity, dict) and uv_identity.get("binary_sha256"):
            if _sha256_file(self.uv_executable) != str(uv_identity["binary_sha256"]).lower():
                raise InstallError("Private uv bytes do not match the packaged provenance contract")
        expected_shared = self.build_manifest.get("shared_lock_sha256")
        if expected_shared and _sha256_file(self.shared_lock).lower() != str(expected_shared).lower():
            raise InstallError("Shared lock bytes do not match the packaged provenance contract")
        expected_profile_map = self.build_manifest.get("profile_lock_sha256")
        if isinstance(expected_profile_map, dict):
            expected_profile = expected_profile_map.get(self.profile.name)
            if expected_profile and _sha256_file(self.profile_lock).lower() != str(expected_profile).lower():
                raise InstallError("Profile lock bytes do not match the packaged provenance contract")

    def _load_profile_lock(self, staging: Path) -> LockSet:
        lock = load_lock(self.profile_lock)
        self._validate_profile_lock(lock)
        return lock

    def _validate_profile_lock(self, lock: LockSet) -> None:
        validate_lock(lock, profile=self.profile, require_hashes=True)
        expected_names = {package.normalized_name for package in self.profile.packages}
        actual_names = {pin.name.lower().replace("_", "-") for pin in lock.pins}
        if actual_names != expected_names:
            raise ProfileError(f"Profile lock contains packages outside the closed profile: {lock.name}")
        for package in self.profile.packages:
            pin = lock.pin(package.name)
            if pin.index_url and pin.index_url.rstrip("/") != self.profile.pytorch_index.rstrip("/"):
                raise ProfileError(f"{package.name} is pinned to the wrong PyTorch index")
            if package.sha256 not in pin.hashes:
                raise ProfileError(f"{package.name} does not carry the canonical profile digest")

    @staticmethod
    def _single_pin_lock(lock: LockSet, name: str) -> LockSet:
        pin = lock.pin(name)
        return LockSet(
            name=f"{lock.name}-{name}",
            python=lock.python,
            platform=lock.platform,
            pins=(pin,),
            wheel_only=lock.wheel_only,
            require_hashes=lock.require_hashes,
        )

    def _install_profile_core(self, lock: LockSet, site_packages: Path) -> None:
        core = LockSet(
            name=f"{lock.name}-torch-core",
            python=lock.python,
            platform=lock.platform,
            pins=(lock.pin("torch"), lock.pin("torchvision")),
            wheel_only=lock.wheel_only,
            require_hashes=lock.require_hashes,
        )
        self._install_lock(
            core,
            site_packages,
            index_url=self.profile.pytorch_index,
            no_deps=True,
        )

    def _validate_shared_lock(self, lock: LockSet) -> None:
        validate_lock(lock, require_hashes=True)
        if lock.pin("insightface").version != "0.7.3":
            raise ProfileError("Shared lock must pin insightface==0.7.3")
        profile_names = {
            package.normalized_name
            for profile in (self.profile,)
            for package in profile.packages
        }
        overlap = {
            pin.name.lower().replace("_", "-")
            for pin in lock.pins
            if pin.name.lower().replace("_", "-") in profile_names
        }
        if overlap:
            raise ProfileError(
                "Shared lock contains CUDA-sensitive packages that must be installed "
                f"from {self.profile.pytorch_index}: {sorted(overlap)}"
            )

    def _install_lock(
        self,
        lock: LockSet,
        site_packages: Path,
        *,
        index_url: str,
        no_deps: bool = False,
    ) -> None:
        validate_lock(lock, require_hashes=True)
        requirement_file = site_packages.parent / f"{lock.name}.txt"
        requirement_file.write_text(lock.as_requirements(), encoding="utf-8", newline="\n")
        command = [
            str(self.uv_executable),
            "pip",
            "install",
            "--python",
            str(self.python_executable),
            "--target",
            str(site_packages),
            "--only-binary",
            ":all:",
            "--no-build",
            "--require-hashes",
            "--index-url",
            index_url,
            "--cache-dir",
            str(self.layout.download_cache_root),
            "--link-mode",
            "copy",
        ]
        if self.wheelhouse.is_dir():
            command.extend(["--find-links", str(self.wheelhouse)])
        if no_deps:
            command.append("--no-deps")
        command.extend(["-r", str(requirement_file)])
        self._run_with_retries(command)

    def _run_with_retries(self, command: Sequence[str]) -> None:
        safe_command = " ".join(redact_text(str(part)) for part in command)
        for attempt in range(1, self.retries + 1):
            self._log(f"uv attempt {attempt}/{self.retries}: {safe_command}")
            try:
                result = self._run_attempt(command)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._log(f"uv invocation failed: {redact_text(str(exc))}")
                self._emit_progress(f"uv invocation failed; retrying with the persistent cache: {type(exc).__name__}")
                if attempt == self.retries:
                    raise InstallError("Private dependency installer could not be started") from exc
                time.sleep(min(attempt, 2))
                continue
            self._record_result_output(result)
            if result.returncode == 0:
                self._emit_progress("uv completed; verified wheel cache remains available for the next launch")
                return
            if attempt < self.retries:
                self._emit_progress(
                    f"uv attempt {attempt} failed; retrying in place with the persistent verified-cache path"
                )
                time.sleep(min(attempt, 2))
        raise InstallError("Private dependency installation failed after retries")

    def _run_attempt(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if self.runner is not subprocess.run:
            return self.runner(
                list(command),
                cwd=str(self.layout.install_root),
                capture_output=True,
                text=True,
                check=False,
            )
        # The real release path streams uv's merged output line-by-line.  This
        # keeps first-run progress useful while the persistent log still gets
        # the same redacted lines.  Injected runners remain available for
        # deterministic tests and failure simulation.
        process = subprocess.Popen(
            list(command),
            cwd=str(self.layout.install_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                captured.append(line)
                safe_line = redact_text(line.rstrip())
                if safe_line:
                    self._emit_progress(safe_line)
        return_code = process.wait()
        return subprocess.CompletedProcess(list(command), return_code, "".join(captured), "")

    def _record_result_output(self, result: subprocess.CompletedProcess[str]) -> None:
        for stream in (result.stdout or "", result.stderr or ""):
            for line in redact_text(str(stream)).splitlines():
                if line.strip():
                    self._emit_progress(line.rstrip())

    def _emit_progress(self, message: str) -> None:
        safe_message = redact_text(message)
        self._log(safe_message)
        self.progress(safe_message)

    @staticmethod
    def _default_progress(message: str) -> None:
        print(f"[Nexfocus] {message}", flush=True)

    def _log(self, message: str) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{timestamp} {redact_text(message)}\n")

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["InstallResult", "MAX_READY_RUNTIMES", "PrivateRuntimeInstaller"]
