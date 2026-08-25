"""Private runtime and user-data layout for the Windows distribution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_user_data_root() -> Path:
    """Return the per-user root without requiring administrator access."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Nexfocus"
    return Path.home() / "AppData" / "Local" / "Nexfocus"


@dataclass(frozen=True)
class RuntimeLayout:
    """All paths owned by either the immutable app or the mutable user data."""

    install_root: Path
    user_data_root: Path

    @property
    def app_root(self) -> Path:
        return self.install_root / "app"

    @property
    def runtime_root(self) -> Path:
        return self.install_root / "runtime"

    @property
    def python_root(self) -> Path:
        return self.runtime_root / "python312"

    @property
    def python_executable(self) -> Path:
        return self.python_root / "python.exe"

    @property
    def uv_executable(self) -> Path:
        return self.runtime_root / "uv.exe"

    @property
    def version_root(self) -> Path:
        return self.runtime_root / "versions"

    @property
    def staging_root(self) -> Path:
        return self.runtime_root / ".staging"

    @property
    def download_cache_root(self) -> Path:
        """Persistent uv cache retained across retry and repair attempts."""

        return self.runtime_root / "download-cache"

    @property
    def current_pointer(self) -> Path:
        return self.runtime_root / "current.json"

    @property
    def install_log_root(self) -> Path:
        return self.runtime_root / "logs"

    @property
    def config_root(self) -> Path:
        return self.user_data_root / "config"

    @property
    def config_path(self) -> Path:
        return self.config_root / "config.txt"

    @property
    def credentials_root(self) -> Path:
        return self.user_data_root / "credentials"

    @property
    def env_path(self) -> Path:
        return self.credentials_root / ".env"

    @property
    def models_root(self) -> Path:
        return self.user_data_root / "models"

    @property
    def outputs_root(self) -> Path:
        return self.user_data_root / "outputs"

    @property
    def catalogs_root(self) -> Path:
        return self.user_data_root / "catalogs"

    @property
    def thumbnails_root(self) -> Path:
        return self.user_data_root / "thumbnails"

    @property
    def temp_root(self) -> Path:
        return self.user_data_root / "temp"

    @property
    def user_log_root(self) -> Path:
        return self.user_data_root / "logs"

    def ensure_user_directories(self) -> None:
        for path in (
            self.config_root,
            self.credentials_root,
            self.models_root,
            self.outputs_root,
            self.catalogs_root,
            self.thumbnails_root,
            self.temp_root,
            self.user_log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_runtime_directories(self) -> None:
        for path in (
            self.runtime_root,
            self.version_root,
            self.staging_root,
            self.download_cache_root,
            self.install_log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def make_layout(install_root: str | os.PathLike[str], user_data_root: str | os.PathLike[str] | None = None) -> RuntimeLayout:
    return RuntimeLayout(
        install_root=Path(install_root).expanduser().resolve(),
        user_data_root=(Path(user_data_root) if user_data_root is not None else default_user_data_root())
        .expanduser()
        .resolve(),
    )


def is_within(path: Path, parent: Path) -> bool:
    """Return true when ``path`` is contained by ``parent`` after resolution."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


__all__ = ["RuntimeLayout", "default_user_data_root", "is_within", "make_layout"]
