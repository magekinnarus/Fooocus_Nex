from __future__ import annotations

import http.server
import shutil
import subprocess
import hashlib
import json
import sys
import threading
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from bootstrap.errors import HardwareProbeError, InstallError, ProfileError, TransactionError
from bootstrap.gpu import GpuSelection, NvidiaGpu, child_environment, parse_nvidia_smi_csv, select_gpu
from bootstrap.installer import PrivateRuntimeInstaller
from bootstrap.launcher import build_child_environment, parse_bootstrap_args
from bootstrap.layout import make_layout
from bootstrap.locks import LockPin, LockSet, load_lock, validate_lock
from bootstrap.manifest import (
    RuntimeManifest,
    canonical_manifest_digest,
    current_runtime,
    is_runtime_ready,
    point_current,
    write_runtime_markers,
)
from bootstrap.profiles import LEGACY_PROFILE, MODERN_PROFILE, PackagePin, select_profile
from bootstrap.user_data import materialize_config, redact_text
from tools.build_windows_release import (
    _marker_applies,
    _payload_sha256,
    _is_excluded,
    _source_matches_index,
    _specifier_satisfied,
    _validate_complete_shared_lock,
    _validate_untracked_workspace,
    build_release,
    copy_uv,
    extract_python_embed,
    load_wheel_artifact_manifest,
    load_release_inputs,
    scan_staged_content,
    validate_dependency_notice,
    validate_wheelhouse,
    validate_wheelhouse_bundle,
)


def test_profile_boundary_and_closed_profile_versions() -> None:
    assert select_profile("7.4").name == "legacy-cu124"
    assert select_profile("7.5").name == "modern-cu128"
    assert select_profile("7.50").name == "modern-cu128"
    assert LEGACY_PROFILE.package("torch").version == "2.5.1+cu124"
    assert MODERN_PROFILE.package("torch").version == "2.11.0+cu128"
    assert MODERN_PROFILE.package("sympy").version == "1.13.3"
    with pytest.raises(ProfileError):
        select_profile("cpu")


def test_wheel_marker_supports_target_python_implementation() -> None:
    assert _marker_applies('platform_python_implementation == "CPython"')
    assert not _marker_applies('platform_python_implementation != "CPython"')
    assert _marker_applies('platform_machine == "AMD64"')
    assert not _marker_applies('platform_machine == "amd64"')
    assert _source_matches_index(
        "https://files.pythonhosted.org/packages/demo.whl",
        "https://pypi.org/simple",
    )
    assert not _source_matches_index(
        "https://example.invalid/demo.whl",
        "https://pypi.org/simple",
    )


@pytest.mark.parametrize(
    ("version", "specifier", "expected"),
    [
        ("0.47.0", "<0.48,>=0.47.0dev0", True),
        ("2.5.1+cu124", "==2.5.1", True),
        ("2.5.1+cu124", "==2.5.1+cu124", True),
        ("0.0.28.post3", ">=0.0.28", True),
        ("3.2.0", "!=3.2.0b1", True),
        ("8.1.1", "!=8.1.*", False),
        ("2.4.0", "~=2.3", True),
        ("3.0.0", "~=2.3", False),
    ],
)
def test_wheel_specifier_supports_release_graph_pep440_forms(
    version: str,
    specifier: str,
    expected: bool,
) -> None:
    assert _specifier_satisfied(version, specifier) is expected


@pytest.mark.parametrize(
    "mutation",
    [
        {"cuda_family": "cu130"},
        {"pytorch_index": "https://download.pytorch.org/whl/cu130"},
        {"compute_capability_max_exclusive": (9, 9)},
        {"packages": replace(LEGACY_PROFILE.packages[0], version="2.5.1+cpu")},
        {"packages": replace(LEGACY_PROFILE.packages[0], wheel_filename="torch-mutated.whl")},
        {"packages": replace(LEGACY_PROFILE.packages[0], sha256="a" * 64)},
    ],
)
def test_profile_contract_rejects_one_field_drift(mutation: dict[str, object]) -> None:
    profile = LEGACY_PROFILE
    if "packages" in mutation:
        packages = list(profile.packages)
        packages[0] = mutation["packages"]
        candidate = replace(profile, packages=tuple(packages))
    else:
        candidate = replace(profile, **mutation)
    with pytest.raises(ProfileError):
        candidate.validate(require_hashes=True)


def test_profile_contract_has_fixed_hashes_and_no_third_profile() -> None:
    LEGACY_PROFILE.validate(require_hashes=True)
    MODERN_PROFILE.validate(require_hashes=True)
    with pytest.raises(ProfileError):
        replace(LEGACY_PROFILE, name="experimental-cu130").validate()


def test_gpu_csv_selection_and_process_local_mapping() -> None:
    gpus = parse_nvidia_smi_csv(
        "0, NVIDIA GeForce GTX 1050, 6.1, 551.23, 3072\n"
        "1, NVIDIA RTX 4090, 8.9, 551.23, 24564\n"
    )
    selection = select_gpu(gpus, requested_index=1)
    assert selection.gpu.index == 1
    assert selection.profile.name == "modern-cu128"
    assert selection.visible_devices == "1"
    assert child_environment(selection, {"PATH": "safe"})["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(HardwareProbeError):
        select_gpu(gpus, requested_index=5)


def test_bootstrap_args_remove_physical_gpu_flag_from_child() -> None:
    options = parse_bootstrap_args(["--repair", "--gpu-device-id", "1", "--preset", "anime"])
    assert options.repair is True
    assert options.gpu_device_id == 1
    assert options.app_args == ("--preset", "anime")


def test_user_config_is_absolute_and_secret_logs_are_redacted(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    config_path = materialize_config(layout, Path("config_template.txt"))
    config_text = config_path.read_text(encoding="utf-8")
    assert str(layout.models_root).replace("\\", "/") in config_text
    config_path.write_text('{"path_outputs": "custom"}\n', encoding="utf-8")
    assert materialize_config(layout, Path("config_template.txt")) == config_path
    assert config_path.read_text(encoding="utf-8") == '{"path_outputs": "custom"}\n'
    assert "secret-value" not in redact_text('HUGGINGFACE_TOKEN="secret-value"')
    assert "secret-value" not in redact_text('{"api_key": "secret-value"}')


def _write_test_lock(path: Path, name: str, package_lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([f"# Nexfocus lock: {name}", "# Python: 3.12; platform: win_amd64", *package_lines]) + "\n",
        encoding="utf-8",
    )
    return path


def _make_test_wheel(
    path: Path,
    name: str,
    version: str,
    *,
    tag: str = "cp312-cp312-win_amd64",
    requires: tuple[str, ...] = (),
    metadata_name: str | None = None,
    metadata_license: str = "MIT",
    metadata_tags: tuple[str, ...] | None = None,
) -> tuple[bytes, bytes]:
    normalized_name = name.replace("-", "_")
    dist_info = f"{normalized_name}-{version}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.3",
        f"Name: {metadata_name or name}",
        f"Version: {version}",
        f"License: {metadata_license}",
        *[f"Requires-Dist: {requirement}" for requirement in requires],
        "",
    ]
    metadata_bytes = "\n".join(metadata_lines).encode("utf-8")
    wheel_bytes = (
        "Wheel-Version: 1.0\n"
        "Generator: nexfocus-test\n"
        "Root-Is-Purelib: false\n"
        + "".join(f"Tag: {wheel_tag}\n" for wheel_tag in (metadata_tags or (tag,)))
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    license_text = f"{name} redistribution license\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata_bytes)
        archive.writestr(f"{dist_info}/WHEEL", wheel_bytes)
        archive.writestr(f"{dist_info}/RECORD", "")
        archive.writestr("LICENSE", license_text)
    return metadata_bytes, wheel_bytes


def _make_test_wheel_manifest(wheels: list[Path]) -> dict[str, object]:
    artifacts = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            wheel_name = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
            metadata = archive.read(metadata_name).decode("utf-8")
            wheel_metadata = archive.read(wheel_name).decode("utf-8")
            license_text = archive.read("LICENSE").decode("utf-8")
        values = dict(
            line.split(": ", 1)
            for line in metadata.splitlines()
            if ": " in line and not line.startswith("Requires-Dist:")
        )
        artifacts.append(
            {
                "name": values["Name"],
                "version": values["Version"],
                "filename": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata.encode("utf-8")).hexdigest(),
                "wheel_metadata_sha256": hashlib.sha256(wheel_metadata.encode("utf-8")).hexdigest(),
                "source_url": f"https://files.pythonhosted.org/packages/{wheel.name}",
                "license": {
                    "id": values.get("License", "MIT"),
                    "evidence": "wheel-member",
                    "files": [{
                        "path": "LICENSE",
                        "sha256": hashlib.sha256(license_text.encode("utf-8")).hexdigest(),
                        "text": license_text,
                    }],
                },
            }
        )
    payload: dict[str, object] = {
        "schema": 1,
        "authentication": {
            "method": "source-attested-sha256",
            "source_url": "https://files.pythonhosted.org/",
            "retrieved_utc": "2026-08-24T00:00:00Z",
            "attestation": "test-attestation",
        },
        "artifacts": artifacts,
    }
    authentication = dict(payload["authentication"])
    authentication["manifest_sha256"] = _payload_sha256(payload)
    payload["authentication"] = authentication
    return payload


def _write_trusted_test_manifest(path: Path, payload: dict[str, object], trust_root: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trust_root.write_text(
        json.dumps(
            {
                "schema": 1,
                "manifest_filename": path.name,
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "approved_source_origins": [
                    "https://files.pythonhosted.org/",
                    "https://download.pytorch.org/whl/cu124/",
                    "https://download.pytorch.org/whl/cu128/",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_test_notice(path: Path, pins: list[tuple[str, str, str, str, str]]) -> None:
    records = []
    for name, version, filename, digest, source_url in pins:
        license_text = f"{name} redistribution license\n"
        records.append(
            {
                "name": name,
                "version": version,
                "filename": filename,
                "sha256": digest,
                "source_url": source_url,
                "provenance": {"source": source_url, "retrieved_utc": "2026-08-24T00:00:00Z"},
                "license": "MIT",
                "license_text": license_text,
                "license_files": [
                    {
                        "path": "LICENSE",
                        "sha256": hashlib.sha256(license_text.encode("utf-8")).hexdigest(),
                        "text": license_text,
                    }
                ],
            }
        )
    path.write_text(
        json.dumps({"schema": 1, "format": "nexfocus-wheel-notices", "artifacts": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def _install_runner_factory(*, fail: bool = False):
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        if fail:
            return subprocess.CompletedProcess(command, 1, "", "network down")
        target = Path(command[command.index("--target") + 1])
        target.mkdir(parents=True, exist_ok=True)
        for name, version in (
            ("sympy", "1.13.3"),
            ("torch", "2.11.0+cu128"),
            ("torchvision", "0.26.0+cu128"),
            ("xformers", "0.0.35"),
        ):
            dist = target / f"{name}-{version}.dist-info"
            dist.mkdir(exist_ok=True)
            (dist / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return runner, calls


def _prepare_installer_inputs(layout, lock_root: Path) -> tuple[Path, Path]:
    layout.python_executable.parent.mkdir(parents=True, exist_ok=True)
    layout.python_executable.write_bytes(b"private python")
    layout.uv_executable.parent.mkdir(parents=True, exist_ok=True)
    layout.uv_executable.write_bytes(b"private uv")
    wheelhouse = layout.app_root / "bootstrap" / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    (wheelhouse / "insightface-0.7.3-cp312-cp312-win_amd64.whl").write_bytes(b"approved test wheel")
    shared = _write_test_lock(
        lock_root / "shared.txt",
        "shared",
        [
            "packaging==24.1 --hash=sha256:" + "a" * 64,
            "insightface==0.7.3 --hash=sha256:" + "b" * 64,
            "sympy==1.13.1 --hash=sha256:" + "c" * 64,
        ],
    )
    profile = _write_test_lock(
        lock_root / "modern.txt",
        "modern-cu128-py312-win-amd64",
        [
            f"sympy==1.13.3 --hash=sha256:{MODERN_PROFILE.package('sympy').sha256}",
            f"torch==2.11.0+cu128 --hash=sha256:{MODERN_PROFILE.package('torch').sha256}",
            f"torchvision==0.26.0+cu128 --hash=sha256:{MODERN_PROFILE.package('torchvision').sha256}",
            f"xformers==0.0.35 --hash=sha256:{MODERN_PROFILE.package('xformers').sha256}",
        ],
    )
    return shared, profile


def test_failed_install_has_no_ready_marker_and_success_reuses_runtime(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    app_root = layout.app_root
    lock_root = tmp_path / "locks"
    shared, profile = _prepare_installer_inputs(layout, lock_root)
    runner, calls = _install_runner_factory(fail=True)
    installer = PrivateRuntimeInstaller(
        layout,
        MODERN_PROFILE,
        app_root=app_root,
        shared_lock=shared,
        profile_lock=profile,
        runner=runner,
        retries=2,
    )
    with pytest.raises(InstallError):
        installer.install()
    assert current_runtime(layout) is None
    assert not any(layout.version_root.glob("*/READY.json"))
    assert len(calls) == 2

    runner, calls = _install_runner_factory()
    installer = PrivateRuntimeInstaller(
        layout,
        MODERN_PROFILE,
        app_root=app_root,
        shared_lock=shared,
        profile_lock=profile,
        runner=runner,
    )
    shared_without_overrides = installer._shared_without_profile_overrides(
        load_lock(shared),
        load_lock(profile),
    )
    assert {pin.name for pin in shared_without_overrides.pins} == {"packaging", "insightface"}
    assert installer._profile_overrides(load_lock(profile)).pin("sympy").version == "1.13.3"
    result = installer.ensure()
    assert result.installed is True
    assert is_runtime_ready(result.runtime_dir)
    previous_call_count = len(calls)
    reused = installer.ensure()
    assert reused.installed is False
    assert len(calls) == previous_call_count
    assert len(calls) == 4
    assert all("--no-deps" in call for call in calls)
    assert all("--cache-dir" in call and "--no-cache" not in call for call in calls)
    assert calls[0][calls[0].index("--index-url") + 1] == MODERN_PROFILE.pytorch_index
    assert calls[1][calls[1].index("--index-url") + 1] == "https://pypi.org/simple"
    assert calls[2][calls[2].index("--index-url") + 1] == "https://pypi.org/simple"
    assert "sympy" in calls[2][-1]
    assert calls[3][calls[3].index("--index-url") + 1] == MODERN_PROFILE.pytorch_index
    assert "xformers" in calls[3][-1]


def test_interrupted_attempt_reuses_persistent_cache_and_emits_progress(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    app_root = layout.app_root
    shared, profile = _prepare_installer_inputs(layout, tmp_path / "locks")
    events: list[str] = []
    attempts = 0

    def runner(command, **kwargs):
        nonlocal attempts
        attempts += 1
        cache_root = Path(command[command.index("--cache-dir") + 1])
        cache_root.mkdir(parents=True, exist_ok=True)
        retained = cache_root / "uv-reused-wheel.whl"
        if attempts == 1:
            retained.write_bytes(b"partial-but-uv-owned")
            return subprocess.CompletedProcess(command, 1, "download progress 42%", "network interrupted")
        assert retained.read_bytes() == b"partial-but-uv-owned"
        target = Path(command[command.index("--target") + 1])
        target.mkdir(parents=True, exist_ok=True)
        for name, version in (
            ("sympy", "1.13.3"),
            ("torch", "2.11.0+cu128"),
            ("torchvision", "0.26.0+cu128"),
            ("xformers", "0.0.35"),
        ):
            dist = target / f"{name}-{version}.dist-info"
            dist.mkdir(exist_ok=True)
            (dist / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "download progress 100%", "")

    installer = PrivateRuntimeInstaller(
        layout,
        MODERN_PROFILE,
        app_root=app_root,
        shared_lock=shared,
        profile_lock=profile,
        runner=runner,
        retries=2,
        progress=events.append,
    )
    result = installer.install()
    assert result.installed is True
    assert attempts == 5
    assert any("42%" in event for event in events)
    assert (layout.download_cache_root / "uv-reused-wheel.whl").is_file()


def test_real_uv_cache_reuses_verified_wheel_after_failed_first_attempt(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("real uv executable is unavailable in this validation environment")
    index_root = tmp_path / "index"
    alpha = index_root / "alpha-1.0-py3-none-any.whl"
    beta = index_root / "beta-1.0-py3-none-any.whl"
    _make_test_wheel(alpha, "alpha", "1.0", tag="py3-none-any")
    _make_test_wheel(beta, "beta", "1.0", tag="py3-none-any")
    request_paths: list[str] = []
    beta_started = threading.Event()
    release_beta = threading.Event()
    first_attempt = True

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(index_root), **kwargs)

        def do_GET(self):  # noqa: N802 - stdlib handler API
            request_paths.append(self.path)
            if first_attempt and self.path.endswith(beta.name):
                beta_started.set()
                release_beta.wait(timeout=30)
            return super().do_GET()

        def log_message(self, _format, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        index_url = f"http://127.0.0.1:{server.server_port}/"
        requirements = tmp_path / "requirements.txt"
        alpha_digest = hashlib.sha256(alpha.read_bytes()).hexdigest()
        beta_digest = hashlib.sha256(beta.read_bytes()).hexdigest()
        requirements.write_text(
            f"alpha==1.0 --hash=sha256:{alpha_digest}\n",
            encoding="utf-8",
        )
        warm_target = tmp_path / "warm-target"
        warm = subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(warm_target),
                "--no-index",
                "--find-links",
                index_url,
                "--require-hashes",
                "--cache-dir",
                str(tmp_path / "uv-cache"),
                "-r",
                str(requirements),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert warm.returncode == 0, warm.stdout + warm.stderr
        assert any(path.endswith(alpha.name) for path in request_paths)

        requirements.write_text(
            "\n".join(
                [
                    f"alpha==1.0 --hash=sha256:{alpha_digest}",
                    f"beta==1.0 --hash=sha256:{beta_digest}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cache_dir = tmp_path / "uv-cache"
        first_target = tmp_path / "first-target"
        command = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(first_target),
            "--no-index",
            "--find-links",
            index_url,
            "--require-hashes",
            "--cache-dir",
            str(cache_dir),
            "-r",
            str(requirements),
        ]
        first_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert beta_started.wait(timeout=30), "uv did not reach the interrupted artifact request"
        first_attempt = False
        release_beta.set()
        first_process.terminate()
        first_output, _ = first_process.communicate(timeout=15)
        assert first_process.returncode != 0, first_output

        alpha_requests_before_retry = sum(path.endswith(alpha.name) for path in request_paths)
        second_target = tmp_path / "second-target"
        second = subprocess.run(
            [*command[: command.index("--target")], "--target", str(second_target), *command[command.index("--target") + 2 :]],
            capture_output=True,
            text=True,
            check=False,
        )
        assert second.returncode == 0, second.stdout + second.stderr
        alpha_requests_after_retry = sum(path.endswith(alpha.name) for path in request_paths)
        assert alpha_requests_after_retry == alpha_requests_before_retry
        assert (second_target / "alpha-1.0.dist-info").is_dir()
        assert (second_target / "beta-1.0.dist-info").is_dir()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_repair_preserves_user_data_and_profile_lock_rejects_cpu_drift(tmp_path: Path) -> None:
    bad = LockSet(
        name="bad",
        python="3.12",
        platform="win_amd64",
        pins=(
            LockPin("torch", "2.11.0+cpu", ("a" * 64,)),
            LockPin("torchvision", "0.26.0+cu128", ("b" * 64,)),
            LockPin("xformers", "0.0.35", ("c" * 64,)),
            LockPin("sympy", "1.13.3", ("d" * 64,)),
        ),
    )
    with pytest.raises(ProfileError):
        validate_lock(bad, profile=MODERN_PROFILE)
    user_model = tmp_path / "user" / "models" / "keep.safetensors"
    user_model.parent.mkdir(parents=True)
    user_model.write_bytes(b"user-owned")
    assert user_model.read_bytes() == b"user-owned"


def test_repair_keeps_at_most_two_ready_runtimes_and_preserves_user_state(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    shared, profile = _prepare_installer_inputs(layout, tmp_path / "locks")
    user_file = layout.models_root / "keep.safetensors"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_bytes(b"keep")
    runner, _calls = _install_runner_factory()
    installer = PrivateRuntimeInstaller(
        layout,
        MODERN_PROFILE,
        app_root=layout.app_root,
        shared_lock=shared,
        profile_lock=profile,
        runner=runner,
    )
    installer.install()
    installer.repair()
    installer.repair()
    assert len(list(layout.version_root.glob("*/READY.json"))) <= 2
    assert user_file.read_bytes() == b"keep"
    assert not any(layout.staging_root.iterdir())


def test_runtime_manifest_requires_release_provenance() -> None:
    manifest = RuntimeManifest(
        version="modern-cu128-test",
        profile="modern-cu128",
        python_version="3.12.10",
        packages=tuple(package.as_manifest() for package in MODERN_PROFILE.packages),
        artifact_version="0.1.0-test",
        build_id="0.1.0-test",
        python_identity={"version": "3.12.10", "sha256": "a" * 64},
        uv_identity={"version": "0.12.5", "sha256": "b" * 64},
        cuda_family="cu128",
        build_manifest_sha256="c" * 64,
        source_content_sha256="d" * 64,
        shared_lock_sha256="e" * 64,
        profile_lock_sha256="f" * 64,
        wheel_audit_sha256="1" * 64,
        wheel_artifact_manifest_sha256="2" * 64,
    ).as_dict()
    assert manifest["artifact_version"] == "0.1.0-test"
    assert manifest["cuda_family"] == "cu128"
    assert manifest["python_identity"]["version"] == "3.12.10"
    with pytest.raises(TransactionError):
        RuntimeManifest(
            version="bad",
            profile="modern-cu128",
            python_version="3.12",
            packages=(),
            artifact_version="working-tree",
        ).as_dict()


def test_runtime_marker_rejects_tampered_python_identity_and_pointer_reuse(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    version_dir = layout.version_root / "modern-cu128-test"
    version_dir.mkdir(parents=True)
    manifest = RuntimeManifest(
        version="modern-cu128-test",
        profile="modern-cu128",
        python_version="3.12.10",
        packages=tuple(package.as_manifest() for package in MODERN_PROFILE.packages),
        artifact_version="development",
        build_id="development",
        python_identity={"version": "3.12.10", "archive_name": "python.zip", "sha256": "a" * 64},
        uv_identity={"version": "0.12.5", "archive_name": "uv.zip", "sha256": "b" * 64},
        cuda_family="cu128",
        build_manifest_sha256="c" * 64,
        source_content_sha256="d" * 64,
        shared_lock_sha256="e" * 64,
        profile_lock_sha256="f" * 64,
        wheel_audit_sha256="1" * 64,
        wheel_artifact_manifest_sha256="2" * 64,
    )
    write_runtime_markers(version_dir, manifest)
    point_current(layout, version_dir.name, manifest.profile)
    assert current_runtime(layout, expected_artifact_version="development", expected_profile="modern-cu128")
    manifest_path = version_dir / "runtime-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["python_identity"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not is_runtime_ready(version_dir)
    assert current_runtime(layout, expected_artifact_version="development", expected_profile="modern-cu128") is None


def test_runtime_rejects_recomputed_wheel_audit_mismatch(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    version_dir = layout.version_root / "modern-cu128-release"
    version_dir.mkdir(parents=True)
    build_manifest = {
        "artifact_version": "0.1.0-test",
        "build_id": "0.1.0-test",
        "build_manifest_sha256": "c" * 64,
        "source_content_sha256": "d" * 64,
        "shared_lock_sha256": "e" * 64,
        "profile_lock_sha256": {"modern-cu128": "f" * 64},
        "wheel_audit_sha256": "1" * 64,
        "wheel_artifact_manifest_sha256": "2" * 64,
        "python": {
            "version": "3.12.10",
            "archive_name": "python.zip",
            "sha256": "a" * 64,
            "executable_sha256": "a" * 64,
        },
        "uv": {
            "version": "0.12.5",
            "archive_name": "uv.zip",
            "sha256": "b" * 64,
            "binary_sha256": "b" * 64,
        },
        "profiles": [MODERN_PROFILE.as_manifest()],
    }
    manifest = RuntimeManifest(
        version="modern-cu128-release",
        profile="modern-cu128",
        python_version="3.12.10",
        packages=tuple(package.as_manifest() for package in MODERN_PROFILE.packages),
        artifact_version="0.1.0-test",
        build_id="0.1.0-test",
        python_identity=dict(build_manifest["python"]),
        uv_identity=dict(build_manifest["uv"]),
        cuda_family="cu128",
        build_manifest_sha256="c" * 64,
        source_content_sha256="d" * 64,
        shared_lock_sha256="e" * 64,
        profile_lock_sha256="f" * 64,
        wheel_audit_sha256="1" * 64,
        wheel_artifact_manifest_sha256="2" * 64,
    )
    write_runtime_markers(version_dir, manifest)
    point_current(layout, version_dir.name, manifest.profile, expected_build_manifest=build_manifest)
    assert current_runtime(layout, expected_build_manifest=build_manifest) is not None

    manifest_path = version_dir / "runtime-manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["wheel_audit_sha256"] = "9" * 64
    manifest_payload["manifest_sha256"] = canonical_manifest_digest(manifest_payload)
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    ready_path = version_dir / "READY.json"
    ready_payload = json.loads(ready_path.read_text(encoding="utf-8"))
    ready_payload["manifest_sha256"] = manifest_payload["manifest_sha256"]
    ready_path.write_text(json.dumps(ready_payload), encoding="utf-8")
    assert not is_runtime_ready(version_dir, expected_build_manifest=build_manifest)
    assert current_runtime(layout, expected_build_manifest=build_manifest) is None


def test_packaged_config_keeps_bundled_catalogue_and_user_overlay(tmp_path: Path) -> None:
    layout = make_layout(tmp_path / "install", tmp_path / "user")
    bundled_catalog = layout.app_root / "configs" / "model_catalogs"
    bundled_catalog.mkdir(parents=True)
    (bundled_catalog / "github_main_catalog.json").write_text("{}\n", encoding="utf-8")
    config_path = materialize_config(layout, Path("config_template.txt"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["path_model_catalogs_preset"].replace("\\", "/") == str(bundled_catalog.resolve()).replace("\\", "/")
    assert config["path_model_catalogs_user"].replace("\\", "/") == str(layout.catalogs_root.joinpath("user")).replace("\\", "/")
    user_catalog = layout.catalogs_root / "user" / "my.catalog.json"
    user_catalog.write_text("{\"catalog_id\": \"user\"}\n", encoding="utf-8")
    assert user_catalog.read_text(encoding="utf-8") == "{\"catalog_id\": \"user\"}\n"


def test_bundled_catalogue_index_and_thumbnail_fallback_survive_empty_user_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from modules.model_catalog_index import ModelCatalogIndex
    import modules.model_thumbnails as model_thumbnails

    bundled_catalog_root = Path("configs/model_catalogs")
    index = ModelCatalogIndex.from_directories([bundled_catalog_root])
    assert {Path(source.path).name for source in index.list_sources()} == {
        "civitai_main_catalog.json",
        "github_main_catalog.json",
        "huggingface_main_catalog.json",
    }

    user_thumbnail_root = tmp_path / "user-thumbnails"
    bundled_thumbnail_root = Path("thumbnails")
    monkeypatch.setattr(
        model_thumbnails.config,
        "get_model_thumbnail_directories",
        lambda: [str(user_thumbnail_root), str(bundled_thumbnail_root)],
    )
    bundled_relative = "thumbnails/vae/sdxl/sdxl_vae_sdxl_vae.jpg"
    assert model_thumbnails.resolve_thumbnail_absolute_path(bundled_relative) == (
        bundled_thumbnail_root / "vae/sdxl/sdxl_vae_sdxl_vae.jpg"
    ).resolve()
    assert model_thumbnails.resolve_thumbnail_absolute_path("thumbnails/new/user.png") == (
        user_thumbnail_root / "new/user.png"
    ).resolve()


def test_checked_in_legacy_source_lock_is_hash_complete() -> None:
    lock = load_lock(Path("bootstrap/locks/legacy-cu124-py312-win-amd64.txt"))
    assert lock.pin("xformers").hashes == (LEGACY_PROFILE.package("xformers").sha256,)


def test_shared_lock_completeness_and_notice_inputs_fail_closed(tmp_path: Path) -> None:
    canonical = LockSet(
        name="shared-py312-win-amd64",
        python="3.12",
        platform="win_amd64",
        pins=(
            LockPin("packaging", "24.1", ("a" * 64,)),
            LockPin("insightface", "0.7.3", ("b" * 64,)),
        ),
    )
    incomplete = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("packaging", "24.1", ("a" * 64,)),),
    )
    with pytest.raises(ValueError):
        _validate_complete_shared_lock(incomplete, canonical)
    notice = tmp_path / "licenses.txt"
    notice.write_text("packaging==24.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_dependency_notice(canonical, notice)
    packaging_wheel = tmp_path / "packaging-24.1-py3-none-any.whl"
    insightface_wheel = tmp_path / "insightface-0.7.3-cp312-cp312-win_amd64.whl"
    _make_test_wheel(packaging_wheel, "packaging", "24.1", tag="py3-none-any")
    _make_test_wheel(insightface_wheel, "insightface", "0.7.3")
    manifest = _make_test_wheel_manifest([packaging_wheel, insightface_wheel])
    canonical = replace(
        canonical,
        pins=tuple(
            LockPin(record["name"], record["version"], (record["sha256"],))
            for record in manifest["artifacts"]
        ),
    )
    notice_records = [
        (
            record["name"],
            record["version"],
            record["filename"],
            record["sha256"],
            record["source_url"],
        )
        for record in manifest["artifacts"]
    ]
    _write_test_notice(notice, notice_records)
    validated = validate_dependency_notice(
        canonical,
        notice,
        artifact_manifest=manifest,
        audited_wheels={
            ("packaging", "24.1"): packaging_wheel,
            ("insightface", "0.7.3"): insightface_wheel,
        },
    )
    assert validated["format"] == "nexfocus-wheel-notices"


def test_wheel_audit_rejects_fake_future_abi_metadata_and_digest_substitution(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    good = wheelhouse / "demo-1.0-cp312-cp312-win_amd64.whl"
    _make_test_wheel(good, "demo", "1.0")
    manifest = _make_test_wheel_manifest([good])
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    lock = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("demo", "1.0", (digest,)),),
    )
    assert validate_wheelhouse(lock, wheelhouse, label="demo", artifact_manifest=manifest)["demo"] == good

    substituted = replace(lock, pins=(LockPin("demo", "1.0", ("f" * 64,)),))
    with pytest.raises(ValueError, match="independently identified"):
        validate_wheelhouse(substituted, wheelhouse, label="demo", artifact_manifest=manifest)

    future_root = tmp_path / "future-wheelhouse"
    future = future_root / "demo-1.0-cp313-abi3-win_amd64.whl"
    _make_test_wheel(future, "demo", "1.0", tag="cp313-abi3-win_amd64")
    future_manifest = _make_test_wheel_manifest([future])
    future_lock = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("demo", "1.0", (future_manifest["artifacts"][0]["sha256"],)),),
    )
    with pytest.raises(ValueError):
        validate_wheelhouse(future_lock, future_root, label="demo", artifact_manifest=future_manifest)

    mismatch_root = tmp_path / "mismatch-wheelhouse"
    mismatch = mismatch_root / "demo-1.0-cp312-cp312-win_amd64.whl"
    _make_test_wheel(mismatch, "demo", "1.0", metadata_name="other")
    mismatch_manifest = _make_test_wheel_manifest([mismatch])
    mismatch_lock = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("demo", "1.0", (hashlib.sha256(mismatch.read_bytes()).hexdigest(),)),),
    )
    with pytest.raises(ValueError, match="METADATA identity"):
        validate_wheelhouse(mismatch_lock, mismatch_root, label="demo", artifact_manifest=mismatch_manifest)


def test_wheel_audit_expands_compressed_filename_tags(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheel = wheelhouse / "demo-1.0-py2.py3-none-any.whl"
    _make_test_wheel(
        wheel,
        "demo",
        "1.0",
        tag="py2.py3-none-any",
        metadata_tags=("py2-none-any", "py3-none-any"),
    )
    manifest = _make_test_wheel_manifest([wheel])
    lock = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("demo", "1.0", (manifest["artifacts"][0]["sha256"],)),),
    )

    assert validate_wheelhouse(lock, wheelhouse, label="demo", artifact_manifest=manifest)


def test_wheel_manifest_requires_detached_trust_root(tmp_path: Path) -> None:
    wheel = tmp_path / "demo-1.0-cp312-cp312-win_amd64.whl"
    _make_test_wheel(wheel, "demo", "1.0")
    manifest = _make_test_wheel_manifest([wheel])
    manifest_path = tmp_path / "wheel-artifact-manifest.json"
    trust_root = tmp_path / "wheel-artifact-trust.json"
    _write_trusted_test_manifest(manifest_path, manifest, trust_root)
    assert load_wheel_artifact_manifest(manifest_path, trust_root=trust_root)["schema"] == 1

    for field, value in (("source_url", "https://evil.example/demo.whl"), ("attestation", "anything")):
        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        if field == "source_url":
            altered["authentication"][field] = value
        else:
            altered["authentication"][field] = value
        unsigned = dict(altered)
        authentication = dict(unsigned["authentication"])
        authentication.pop("manifest_sha256", None)
        unsigned["authentication"] = authentication
        altered["authentication"]["manifest_sha256"] = _payload_sha256(unsigned)
        manifest_path.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="detached trust root"):
            load_wheel_artifact_manifest(manifest_path, trust_root=trust_root)
        _write_trusted_test_manifest(manifest_path, manifest, trust_root)


def test_wheel_graph_closure_and_extra_inventory_fail_closed(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    dependency = wheelhouse / "dependency-1.0-py3-none-any.whl"
    root = wheelhouse / "root-1.0-py3-none-any.whl"
    extra = wheelhouse / "extra-1.0-py3-none-any.whl"
    _make_test_wheel(dependency, "dependency", "1.0", tag="py3-none-any")
    _make_test_wheel(root, "root", "1.0", tag="py3-none-any", requires=("dependency>=1.0",))
    _make_test_wheel(extra, "extra", "1.0", tag="py3-none-any")
    manifest = _make_test_wheel_manifest([dependency, root, extra])
    locks = [
        LockSet(
            name="shared",
            python="3.12",
            platform="win_amd64",
            pins=tuple(
                LockPin(
                    record["name"],
                    record["version"],
                    (record["sha256"],),
                )
                for record in manifest["artifacts"][:2]
            ),
        )
    ]
    with pytest.raises(ValueError, match="extra wheel"):
        validate_wheelhouse_bundle(locks, wheelhouse, artifact_manifest=manifest)

    wheelhouse_extra_removed = tmp_path / "wheelhouse-no-extra"
    wheelhouse_extra_removed.mkdir()
    for source in (dependency, root):
        (wheelhouse_extra_removed / source.name).write_bytes(source.read_bytes())
    no_extra_manifest = _make_test_wheel_manifest(
        [wheelhouse_extra_removed / dependency.name, wheelhouse_extra_removed / root.name]
    )
    valid_locks = [
        LockSet(
            name="shared",
            python="3.12",
            platform="win_amd64",
            pins=tuple(
                LockPin(record["name"], record["version"], (record["sha256"],))
                for record in no_extra_manifest["artifacts"]
            ),
        )
    ]
    assert validate_wheelhouse_bundle(valid_locks, wheelhouse_extra_removed, artifact_manifest=no_extra_manifest)

    missing_dependency = replace(valid_locks[0], pins=(valid_locks[0].pins[1],))
    with pytest.raises(ValueError, match="dependency closure"):
        validate_wheelhouse_bundle([missing_dependency], wheelhouse_extra_removed, artifact_manifest=no_extra_manifest)


def test_structured_notices_cover_profile_wheels_and_reject_placeholder(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    shared_wheel = wheelhouse / "sharedpkg-1.0-py3-none-any.whl"
    _make_test_wheel(shared_wheel, "sharedpkg", "1.0", tag="py3-none-any")
    profile_wheels: list[Path] = []
    for package in MODERN_PROFILE.packages:
        wheel = wheelhouse / package.wheel_filename
        wheel_tag = "-".join(package.wheel_filename[:-4].rsplit("-", 3)[-3:])
        _make_test_wheel(wheel, package.name, package.version, tag=wheel_tag)
        profile_wheels.append(wheel)
    manifest = _make_test_wheel_manifest([shared_wheel, *profile_wheels])
    manifest_records = {
        (str(record["name"]), str(record["version"])): record
        for record in manifest["artifacts"]
    }
    shared = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("sharedpkg", "1.0", (str(manifest_records[("sharedpkg", "1.0")]["sha256"]),)),),
    )
    profile_lock = LockSet(
        name="modern-cu128",
        python="3.12",
        platform="win_amd64",
        pins=tuple(
            LockPin(
                package.name,
                package.version,
                (str(manifest_records[(package.name, package.version)]["sha256"]),),
                package.index_url,
                package.wheel_filename,
            )
            for package in MODERN_PROFILE.packages
        ),
    )
    records = []
    audited_wheels = {}
    for record in manifest["artifacts"]:
        wheel_path = wheelhouse / str(record["filename"])
        audited_wheels[(str(record["name"]), str(record["version"]))] = wheel_path
        records.append(
            {
                "name": record["name"],
                "version": record["version"],
                "filename": record["filename"],
                "sha256": record["sha256"],
                "source_url": record["source_url"],
                "provenance": {"source": record["source_url"], "retrieved_utc": "2026-08-24T00:00:00Z"},
                "license": record["license"]["id"],
                "license_files": record["license"]["files"],
            }
        )
    notice = tmp_path / "notices.json"
    notice.write_text(json.dumps({"schema": 1, "format": "nexfocus-wheel-notices", "artifacts": records}), encoding="utf-8")
    assert validate_dependency_notice(
        shared,
        notice,
        profiles=(MODERN_PROFILE,),
        profile_locks={MODERN_PROFILE.name: profile_lock},
        artifact_manifest=manifest,
        audited_wheels=audited_wheels,
    )["schema"] == 1

    fabricated = json.loads(notice.read_text(encoding="utf-8"))
    fabricated_text = "fabricated license text\n"
    fabricated["artifacts"][0]["license_files"][0] = {
        "path": "LICENSE",
        "sha256": hashlib.sha256(fabricated_text.encode("utf-8")).hexdigest(),
        "text": fabricated_text,
    }
    fabricated_path = tmp_path / "fabricated-notices.json"
    fabricated_path.write_text(json.dumps(fabricated), encoding="utf-8")
    with pytest.raises(ValueError, match="detached evidence"):
        validate_dependency_notice(
            shared,
            fabricated_path,
            profiles=(MODERN_PROFILE,),
            profile_locks={MODERN_PROFILE.name: profile_lock},
            artifact_manifest=manifest,
            audited_wheels=audited_wheels,
        )

    fabricated_identity = json.loads(notice.read_text(encoding="utf-8"))
    fabricated_identity["artifacts"][0]["license"] = "Apache-2.0"
    fabricated_identity_path = tmp_path / "fabricated-identity-notices.json"
    fabricated_identity_path.write_text(json.dumps(fabricated_identity), encoding="utf-8")
    with pytest.raises(ValueError, match="license identity"):
        validate_dependency_notice(
            shared,
            fabricated_identity_path,
            profiles=(MODERN_PROFILE,),
            profile_locks={MODERN_PROFILE.name: profile_lock},
            artifact_manifest=manifest,
            audited_wheels=audited_wheels,
        )

    placeholder = tmp_path / "placeholder.txt"
    placeholder.write_text("sharedpkg==1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="structured JSON"):
        validate_dependency_notice(shared, placeholder, artifact_manifest=manifest)


@pytest.mark.parametrize(
    "metadata_license",
    ["UNKNOWN", "NOASSERTION", "complete license text " * 20],
)
def test_notice_accepts_evidence_backed_license_when_wheel_metadata_is_placeholder(
    tmp_path: Path,
    metadata_license: str,
) -> None:
    wheel = tmp_path / "demo-1.0-py3-none-any.whl"
    _make_test_wheel(
        wheel,
        "demo",
        "1.0",
        tag="py3-none-any",
        metadata_license=metadata_license,
    )
    manifest = _make_test_wheel_manifest([wheel])
    record = manifest["artifacts"][0]
    record["license"]["id"] = "MIT"
    lock = LockSet(
        name="shared",
        python="3.12",
        platform="win_amd64",
        pins=(LockPin("demo", "1.0", (record["sha256"],)),),
    )
    notice = tmp_path / "notices.json"
    _write_test_notice(
        notice,
        [("demo", "1.0", record["filename"], record["sha256"], record["source_url"])],
    )

    assert validate_dependency_notice(
        lock,
        notice,
        artifact_manifest=manifest,
        audited_wheels={("demo", "1.0"): wheel},
    )["schema"] == 1


def test_release_builder_rejects_fake_or_unversioned_external_inputs(tmp_path: Path) -> None:
    python_zip = tmp_path / "python-embed.zip"
    with zipfile.ZipFile(python_zip, "w") as archive:
        archive.writestr("python.exe", b"private python")
        archive.writestr("python312._pth", "python312.zip\n.\n")
    uv = tmp_path / "uv-x86_64-pc-windows-msvc.zip"
    with zipfile.ZipFile(uv, "w") as archive:
        archive.writestr("uv.exe", b"private uv")
    shared = _write_test_lock(
        tmp_path / "shared.txt",
        "shared",
        [
            "packaging==24.1 --hash=sha256:" + "a" * 64,
            "insightface==0.7.3 --hash=sha256:" + "b" * 64,
        ],
    )
    with pytest.raises(ValueError, match="concrete release version"):
        build_release(
            Path("."),
            tmp_path / "output",
            python_zip=python_zip,
            python_sha256=hashlib.sha256(python_zip.read_bytes()).hexdigest(),
            uv_binary=uv,
            uv_sha256=hashlib.sha256(uv.read_bytes()).hexdigest(),
            shared_lock=shared,
            profile_name="modern-cu128",
        )


def test_release_input_contract_is_repository_pinned() -> None:
    contract = load_release_inputs(Path("."))
    assert contract["python"]["version"] == "3.12.10"
    assert contract["python"]["archive_name"] == "python-3.12.10-embed-amd64.zip"
    assert contract["uv"]["version"] == "0.12.5"
    assert contract["uv"]["archive_name"] == "uv-x86_64-pc-windows-msvc.zip"
    assert contract["python"]["sha256"] != "a" * 64
    uv_license = Path(str(contract["uv"]["license_source"]))
    uv_license_bytes = uv_license.read_bytes()
    assert hashlib.sha256(uv_license_bytes).hexdigest() == contract["uv"]["license_sha256"]
    assert b"\r\n" not in uv_license_bytes
    assert "bootstrap/licenses/uv-MIT.txt text eol=lf" in Path(".gitattributes").read_text(
        encoding="utf-8"
    ).splitlines()
    shared_lock = load_lock(Path(str(contract["shared_lock_source"])), require_hashes=False)
    assert shared_lock.pin("markdown-it-py").version == "4.2.0"
    assert shared_lock.pin("mdurl").version == "0.1.2"


def test_fake_python_and_uv_bytes_fail_identity_validation(tmp_path: Path) -> None:
    contract = load_release_inputs(Path("."))
    python_zip = tmp_path / str(contract["python"]["archive_name"])
    with zipfile.ZipFile(python_zip, "w") as archive:
        archive.writestr("python.exe", b"fake")
        archive.writestr("python312._pth", "import site\n")
        archive.writestr("LICENSE.txt", "fake\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_python_embed(python_zip, tmp_path / "python", contract=contract["python"])

    uv_zip = tmp_path / str(contract["uv"]["archive_name"])
    with zipfile.ZipFile(uv_zip, "w") as archive:
        archive.writestr("uv.exe", b"fake")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        copy_uv(uv_zip, tmp_path / "uv.exe", contract=contract["uv"])


def test_builder_requires_insightface_wheelhouse_and_dependency_notices(tmp_path: Path) -> None:
    shared = _write_test_lock(tmp_path / "shared.txt", "shared", [
        "packaging==24.1 --hash=sha256:" + "a" * 64,
        "insightface==0.7.3 --hash=sha256:" + "b" * 64,
    ])
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    with pytest.raises(ValueError, match="InsightFace wheel"):
        build_release(
            Path("."),
            tmp_path / "output",
            python_zip=tmp_path / "python.zip",
            uv_binary=tmp_path / "uv.zip",
            shared_lock=shared,
            wheelhouse=wheelhouse,
            release_version="0.1.0-test",
        )
    insightface = wheelhouse / "insightface-0.7.3-cp312-cp312-win_amd64.whl"
    insightface.write_bytes(b"test wheel")
    with pytest.raises(ValueError, match="dependency license notice"):
        build_release(
            Path("."),
            tmp_path / "output",
            python_zip=tmp_path / "python.zip",
            uv_binary=tmp_path / "uv.zip",
            shared_lock=shared,
            wheelhouse=wheelhouse,
            insightface_wheel=insightface,
            release_version="0.1.0-test",
        )


def test_python_license_and_staged_secret_gates_fail_closed(tmp_path: Path) -> None:
    python_zip = tmp_path / "python-test.zip"
    with zipfile.ZipFile(python_zip, "w") as archive:
        archive.writestr("python.exe", b"private python")
        archive.writestr("python312._pth", "import site\n")
    contract = {
        "archive_name": python_zip.name,
        "sha256": hashlib.sha256(python_zip.read_bytes()).hexdigest(),
        "executable_member": "python.exe",
        "pth_member": "python312._pth",
        "license_members": ["LICENSE.txt"],
    }
    with pytest.raises(ValueError, match="required member"):
        extract_python_embed(python_zip, tmp_path / "python", contract=contract)

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "notes.txt").write_text("HUGGINGFACE_TOKEN=do-not-ship\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Secret or credential"):
        scan_staged_content(staged)


def test_unexpected_untracked_release_input_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tools.build_windows_release.untracked_source_inventory",
        lambda _source_root: ("unexpected-release-note.txt",),
    )
    with pytest.raises(ValueError, match="Unexpected untracked"):
        _validate_untracked_workspace(tmp_path, set())


def test_release_inventory_excludes_repository_automation() -> None:
    assert _is_excluded(Path(".github/workflows/build_container.yml"))
    assert _is_excluded(Path(".ssl/key.pem"))
    assert not _is_excluded(Path("launch.py"))
