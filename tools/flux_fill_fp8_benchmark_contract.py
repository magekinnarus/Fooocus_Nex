from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FLUX_FILL_BENCHMARK_MODES = (
    "fp8_stream_async",
    "fp8_resident",
)

FLUX_FILL_BENCHMARK_NEGATIVE_CONTROL = "fp8_stream_sync"


@dataclass(frozen=True)
class FluxFillBenchmarkArtifactBundle:
    fp8_unet_path: Path
    ae_path: Path
    conditioning_cache_path: Path
    source_latent_path: Path
    concat_latent_path: Path
    denoise_mask_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)
    q4_gguf_unet_path: Path | None = None

    def unet_path_for_mode(self, mode: str) -> Path:
        normalized = normalize_flux_fill_benchmark_mode(mode)
        if normalized in {"fp8_stream_async", "fp8_resident", FLUX_FILL_BENCHMARK_NEGATIVE_CONTROL}:
            return self.fp8_unet_path
        raise ValueError(f"Unsupported Flux Fill benchmark mode: {mode!r}.")


@dataclass(frozen=True)
class FluxFillBenchmarkRunMetrics:
    mode: str
    run_label: str
    denoise_s_per_it: float
    total_wall: float
    peak_vram_mb: float
    peak_rss_mb: float
    host_pinned_mb: float
    denoise_wall: float = 0.0
    denoise_cpu_proc: float = 0.0
    overlap_note: str = ""
    gpu_busy_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FluxFillBenchmarkRunInput:
    mode: str
    bundle: FluxFillBenchmarkArtifactBundle
    steps: int
    seed: int
    guidance: float
    sampler: str
    scheduler: str
    execution_class: str | None = None
    denoise_only: bool = True
    notes: str = ""

    def validate(self) -> None:
        if self.mode not in FLUX_FILL_BENCHMARK_MODES:
            raise ValueError(
                f"Unsupported Flux Fill benchmark mode: {self.mode!r}. "
                f"Expected one of {list(FLUX_FILL_BENCHMARK_MODES)}."
            )
        if not self.denoise_only:
            raise ValueError("Flux Fill W08b benchmark must remain denoise-only.")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}.")
        if self.guidance <= 0:
            raise ValueError(f"guidance must be > 0, got {self.guidance}.")


def normalize_flux_fill_benchmark_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in FLUX_FILL_BENCHMARK_MODES:
        return value
    if value == FLUX_FILL_BENCHMARK_NEGATIVE_CONTROL:
        return value
    raise ValueError(
        f"Unsupported Flux Fill benchmark mode: {mode!r}. "
        f"Expected one of {list(FLUX_FILL_BENCHMARK_MODES)}."
    )
