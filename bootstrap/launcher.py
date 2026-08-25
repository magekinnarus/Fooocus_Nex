"""One-click Windows launcher for the private Nexfocus runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

if __package__ in (None, ""):
    # Keep ``python bootstrap/launcher.py`` useful in a source checkout and
    # in diagnostics; the release batch normally invokes ``-m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bootstrap.errors import BootstrapError
    from bootstrap.gpu import GpuSelection, child_environment, probe_nvidia_gpus, select_gpu
    from bootstrap.installer import InstallResult, PrivateRuntimeInstaller
    from bootstrap.layout import RuntimeLayout, make_layout
    from bootstrap.user_data import materialize_config, materialize_credentials
else:
    from .errors import BootstrapError
    from .gpu import GpuSelection, child_environment, probe_nvidia_gpus, select_gpu
    from .installer import InstallResult, PrivateRuntimeInstaller
    from .layout import RuntimeLayout, make_layout
    from .user_data import materialize_config, materialize_credentials


@dataclass(frozen=True)
class BootstrapOptions:
    repair: bool
    no_launch: bool
    user_data_dir: Path | None
    gpu_device_id: int | None
    app_args: tuple[str, ...]


def parse_bootstrap_args(argv: Sequence[str]) -> BootstrapOptions:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repair", "--bootstrap-repair", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--user-data-dir")
    parser.add_argument("--gpu-device-id", type=int)
    known, app_args = parser.parse_known_args(list(argv))
    return BootstrapOptions(
        repair=bool(known.repair),
        no_launch=bool(known.no_launch),
        user_data_dir=Path(known.user_data_dir).expanduser() if known.user_data_dir else None,
        gpu_device_id=known.gpu_device_id,
        app_args=tuple(app_args),
    )


def archive_root(launcher_file: Path | None = None) -> Path:
    file_path = (launcher_file or Path(__file__)).resolve()
    # Source tree: <root>/bootstrap/launcher.py. Release: <root>/app/bootstrap/launcher.py.
    if file_path.parent.parent.name == "app":
        return file_path.parent.parent.parent
    return file_path.parent.parent


def build_child_environment(
    layout: RuntimeLayout,
    selection: GpuSelection,
    runtime_dir: Path,
    *,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = child_environment(selection, base)
    site_packages = runtime_dir / "site-packages"
    environment["NEXFOCUS_BOOTSTRAP"] = "1"
    environment["NEXFOCUS_USER_DATA_DIR"] = str(layout.user_data_root)
    environment["NEXFOCUS_CONFIG_PATH"] = str(layout.config_path)
    environment["NEXFOCUS_ENV_FILE"] = str(layout.env_path)
    environment["NEXFOCUS_RUNTIME_DIR"] = str(runtime_dir)
    environment["NEXFOCUS_APP_ROOT"] = str(layout.app_root)
    environment["NEXFOCUS_BUNDLED_CATALOG_ROOT"] = str(layout.app_root / "configs" / "model_catalogs")
    environment["NEXFOCUS_BUNDLED_THUMBNAIL_ROOT"] = str(layout.app_root / "thumbnails")
    environment["PYTHONNOUSERSITE"] = "1"
    # Do not inherit a caller's system/user Python path into the private
    # runtime. The app and the selected site-packages are the complete import
    # surface for a one-click child process.
    environment["PYTHONPATH"] = os.pathsep.join([str(site_packages), str(layout.app_root)])
    environment["PATH"] = os.pathsep.join(
        [str(layout.runtime_root), str(layout.python_root)]
        + ([environment["PATH"]] if environment.get("PATH") else [])
    )
    return environment


def configure_embedded_python_paths(layout: RuntimeLayout, runtime_dir: Path) -> None:
    """Make the embeddable Python see the app and the selected site-packages.

    The official embeddable distribution uses a ``*_._pth`` file and may
    ignore PYTHONPATH. Updating that private file is process/package-local and
    avoids falling back to a system interpreter or user site directory.
    """

    relative_site = os.path.relpath(runtime_dir / "site-packages", layout.python_root).replace("\\", "/")
    pth_files = tuple(layout.python_root.glob("*_._pth")) + tuple(layout.python_root.glob("*_pth"))
    for pth_path in pth_files:
        lines = pth_path.read_text(encoding="utf-8").splitlines()
        entries = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        for entry in ("../../app", relative_site):
            if entry not in entries:
                lines.append(entry)
                entries.append(entry)
        if not any(line.strip() == "import site" for line in lines):
            lines.append("import site")
        handle, temporary_name = tempfile.mkstemp(prefix=f".{pth_path.name}.", suffix=".tmp", dir=pth_path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(lines).rstrip() + "\n")
            os.replace(temporary_name, pth_path)
        except OSError:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise


class WindowsBootstrap:
    """Coordinate probe, install/reuse, user-data setup, and child launch."""

    def __init__(
        self,
        install_root: Path,
        *,
        user_data_root: Path | None = None,
        probe: Callable[[], tuple] = probe_nvidia_gpus,
        installer_factory: Callable[..., PrivateRuntimeInstaller] = PrivateRuntimeInstaller,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.install_root = install_root.resolve()
        self.layout = make_layout(self.install_root, user_data_root)
        self.probe = probe
        self.installer_factory = installer_factory
        self.runner = runner

    def run(self, options: BootstrapOptions) -> int:
        selection = select_gpu(self.probe(), options.gpu_device_id)
        installer = self.installer_factory(
            self.layout,
            selection.profile,
            app_root=self.layout.app_root,
            runner=self.runner,
            check_cuda=True,
            gpu_info={
                "index": selection.gpu.index,
                "name": selection.gpu.name,
                "compute_capability": selection.gpu.compute_capability_text,
                "driver_version": selection.gpu.driver_version,
                "memory_total_mb": selection.gpu.memory_total_mb,
            },
        )
        result = installer.repair() if options.repair else installer.ensure()
        template_path = self.layout.app_root / "config_template.txt"
        env_template_path = self.layout.app_root / ".env_template"
        if not template_path.is_file() or not env_template_path.is_file():
            raise BootstrapError("The release is incomplete: user-data templates are missing")
        materialize_config(self.layout, template_path)
        materialize_credentials(self.layout, env_template_path)
        configure_embedded_python_paths(self.layout, result.runtime_dir)
        if options.no_launch:
            return 0
        environment = build_child_environment(self.layout, selection, result.runtime_dir)
        child_args = [argument for argument in options.app_args if argument not in {"--bootstrap-repair"}]
        command = [str(self.layout.python_executable), "-s", str(self.layout.app_root / "launch.py")]
        command.extend(child_args)
        completed = self.runner(
            command,
            cwd=str(self.layout.app_root),
            env=environment,
            check=False,
        )
        return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_bootstrap_args(argv if argv is not None else sys.argv[1:])
    try:
        return WindowsBootstrap(archive_root()).run(options)
    except BootstrapError as exc:
        print(f"Nexfocus setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapOptions",
    "WindowsBootstrap",
    "archive_root",
    "build_child_environment",
    "configure_embedded_python_paths",
    "main",
    "parse_bootstrap_args",
]
