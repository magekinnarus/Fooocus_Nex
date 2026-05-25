from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FLUX_T5_BENCHMARK_MODES = (
    "q8_t5_paged_cpu_dequant",
    "fp8_t5_cpu_resident",
    "fp8_t5_paged_cpu_dequant",
    "fp16_t5_stream_runtime",
    "fp16_t5_lazy_runtime",
)


@dataclass(frozen=True)
class FluxT5BenchmarkAssets:
    clip_l_path: Path
    q8_t5_path: Path
    fp16_t5_path: Path
    fp8_t5_path: Path
    embedding_directory: Path | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def t5_path_for_mode(self, mode: str) -> Path:
        normalized = normalize_flux_t5_benchmark_mode(mode)
        if normalized == "q8_t5_paged_cpu_dequant":
            return self.q8_t5_path
        if normalized in {"fp16_t5_stream_runtime", "fp16_t5_lazy_runtime"}:
            return self.fp16_t5_path
        if normalized in {"fp8_t5_cpu_resident", "fp8_t5_paged_cpu_dequant"}:
            return self.fp8_t5_path
        raise ValueError(f"Unsupported Flux T5 benchmark mode: {mode!r}.")


@dataclass(frozen=True)
class FluxT5BenchmarkPrompt:
    text: str
    label: str | None = None

    def normalized_label(self, index: int) -> str:
        raw = str(self.label or f"prompt_{index}").strip().lower().replace(" ", "_").replace("-", "_")
        return raw or f"prompt_{index}"


@dataclass(frozen=True)
class FluxT5BenchmarkRunMetrics:
    mode: str
    run_label: str
    prompt_label: str
    total_wall: float
    encode_wall: float
    encode_cpu_proc: float
    peak_rss_mb: float
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_flux_t5_benchmark_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value not in FLUX_T5_BENCHMARK_MODES:
        raise ValueError(
            f"Unsupported Flux T5 benchmark mode: {mode!r}. Expected one of {list(FLUX_T5_BENCHMARK_MODES)}."
        )
    return value
