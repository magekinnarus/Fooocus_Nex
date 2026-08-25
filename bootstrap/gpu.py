"""Stdlib-only NVIDIA discovery used before a private runtime is installed."""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .errors import HardwareProbeError, UnsupportedHardwareError
from .profiles import DependencyProfile, parse_compute_capability, select_profile


@dataclass(frozen=True)
class NvidiaGpu:
    index: int
    name: str
    compute_capability: tuple[int, int]
    driver_version: str
    memory_total_mb: int | None = None

    @property
    def compute_capability_text(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


@dataclass(frozen=True)
class GpuSelection:
    gpu: NvidiaGpu
    profile: DependencyProfile
    visible_devices: str


def _parse_memory(value: str) -> int | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    try:
        return int(float(match.group(0)))
    except ValueError:
        return None


def parse_nvidia_smi_csv(output: str) -> tuple[NvidiaGpu, ...]:
    """Parse the fixed ``nvidia-smi`` query output.

    The normal query uses ``noheader,nounits`` and five columns.  Header-aware
    parsing is accepted as well so captured diagnostics and tests can be fed
    back through the same parser.
    """

    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(output))
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        raise HardwareProbeError("nvidia-smi returned no GPU records")

    header = {cell.lower().replace(" ", "_"): index for index, cell in enumerate(rows[0])}
    has_header = "index" in header or "compute_cap" in header
    if has_header:
        rows = rows[1:]
        index_col = header.get("index", 0)
        name_col = header.get("name", 1)
        compute_col = header.get("compute_cap", 2)
        driver_col = header.get("driver_version", 3)
        memory_col = header.get("memory.total", header.get("memory_total", 4))
    else:
        index_col, name_col, compute_col, driver_col, memory_col = 0, 1, 2, 3, 4

    gpus: list[NvidiaGpu] = []
    for row in rows:
        if len(row) < 4:
            raise HardwareProbeError(f"Malformed nvidia-smi record: {row!r}")
        try:
            index = int(row[index_col])
            capability = parse_compute_capability(row[compute_col])
        except (IndexError, ValueError) as exc:
            raise HardwareProbeError(f"Malformed nvidia-smi record: {row!r}") from exc
        except Exception as exc:
            raise HardwareProbeError(f"Malformed nvidia-smi record: {row!r}") from exc
        driver = row[driver_col] if driver_col < len(row) else "unknown"
        memory = row[memory_col] if memory_col < len(row) else ""
        gpus.append(
            NvidiaGpu(
                index=index,
                name=row[name_col] if name_col < len(row) else "Unknown NVIDIA GPU",
                compute_capability=capability,
                driver_version=driver or "unknown",
                memory_total_mb=_parse_memory(memory),
            )
        )
    if not gpus:
        raise HardwareProbeError("nvidia-smi returned no GPU records")
    return tuple(sorted(gpus, key=lambda gpu: gpu.index))


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )


def probe_nvidia_gpus(
    *,
    executable: str | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[NvidiaGpu, ...]:
    """Run the bounded NVIDIA probe and return deterministic GPU records."""

    executable = executable or shutil.which("nvidia-smi") or "nvidia-smi"
    runner = runner or _default_runner
    command = [
        executable,
        "--query-gpu=index,name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HardwareProbeError(
            "NVIDIA driver probe failed: nvidia-smi is unavailable or timed out. "
            "Install a supported NVIDIA driver and retry."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown nvidia-smi error").strip()
        raise HardwareProbeError(
            "NVIDIA driver probe failed. Install or update the NVIDIA driver, "
            f"then retry. Details: {detail[:400]}"
        )
    try:
        return parse_nvidia_smi_csv(result.stdout)
    except HardwareProbeError:
        raise
    except Exception as exc:
        raise HardwareProbeError("NVIDIA driver probe returned an unreadable response") from exc


def select_gpu(
    gpus: Iterable[NvidiaGpu],
    requested_index: int | None = None,
) -> GpuSelection:
    """Select one physical GPU and map it to logical device zero in the child."""

    available = tuple(sorted(gpus, key=lambda gpu: gpu.index))
    if not available:
        raise HardwareProbeError("No NVIDIA GPU was detected")
    if requested_index is None:
        gpu = available[0]
    else:
        if requested_index < 0:
            raise HardwareProbeError("--gpu-device-id must be a non-negative physical GPU index")
        matches = [candidate for candidate in available if candidate.index == requested_index]
        if not matches:
            ids = ", ".join(str(candidate.index) for candidate in available)
            raise HardwareProbeError(
                f"GPU {requested_index} was not found. Available physical GPU IDs: {ids}"
            )
        gpu = matches[0]
    try:
        profile = select_profile(gpu.compute_capability)
    except Exception as exc:
        raise UnsupportedHardwareError(
            f"GPU {gpu.name} has unsupported compute capability {gpu.compute_capability_text}"
        ) from exc
    return GpuSelection(gpu=gpu, profile=profile, visible_devices=str(gpu.index))


def child_environment(selection: GpuSelection, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return process-local environment; never mutate the machine environment."""

    environment = dict(base if base is not None else os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = selection.visible_devices
    return environment

__all__ = [
    "GpuSelection",
    "NvidiaGpu",
    "child_environment",
    "parse_nvidia_smi_csv",
    "probe_nvidia_gpus",
    "select_gpu",
]
