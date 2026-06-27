from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np
import torch


class UNetSpineKind(str, Enum):
    STREAMING = "streaming"


class T5PostureKind(str, Enum):
    DISK_PAGED = "disk_paged"


class VAEPostureKind(str, Enum):
    TRANSIENT = "transient"


class FluxFillCategory(str, Enum):
    INPAINT = "inpaint"
    REMOVAL = "removal"


def normalize_category(value: Any) -> FluxFillCategory | None:
    if value is None:
        return None
    if isinstance(value, FluxFillCategory):
        return value
    val_str = str(value).strip().lower()
    if val_str in {"inpaint", "flux_inpaint"}:
        return FluxFillCategory.INPAINT
    if val_str in {"removal", "remove", "remove_obj"}:
        return FluxFillCategory.REMOVAL
    raise ValueError(f"Unknown Flux Fill category: {value}")


@dataclass(frozen=True)
class FluxRuntimeIdentity:
    unet_spine: UNetSpineKind
    t5_posture: T5PostureKind
    vae_posture: VAEPostureKind

    def as_dict(self) -> dict[str, str | None]:
        return {
            "unet_spine": self.unet_spine.value,
            "t5_posture": self.t5_posture.value,
            "vae_posture": self.vae_posture.value,
        }


@dataclass
class FluxFillRequest:
    unet_path: Path | str
    ae_path: Path | str
    conditioning_cache_path: Path | str
    seed: int
    steps: int
    guidance: float = 15.0
    sampler: str = "euler"
    scheduler: str = "simple"
    prefetch_depth: int = 1
    prefetch_chunk_mb: int = 64
    unet_spine: UNetSpineKind = UNetSpineKind.STREAMING
    device: str | torch.device | None = None
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    prompt: str | None = None
    blend_mode: str = "morphological"
    target_megapixels: float = 1.15
    clip_l_path: Path | str | None = None
    t5_path: Path | str | None = None
    category: FluxFillCategory | str | None = None

    def __post_init__(self) -> None:
        if self.category is not None:
            self.category = normalize_category(self.category)

    def validate_static(self, *, require_existing_assets: bool = True) -> None:
        from backend.flux_fill_v2.contracts import load_flux_fill_sampler_scheduler_allowlist, _normalize_flux_fill_sampling_name

        policy = load_flux_fill_sampler_scheduler_allowlist()
        original_sampler = str(self.sampler or "")
        original_scheduler = str(self.scheduler or "")
        normalized_sampler = _normalize_flux_fill_sampling_name(original_sampler)
        normalized_scheduler = _normalize_flux_fill_sampling_name(original_scheduler)

        resolved_sampler = (
            normalized_sampler
            if normalized_sampler in policy["allowed_samplers"]
            else str(policy["fallback_sampler"])
        )
        resolved_scheduler = (
            normalized_scheduler
            if normalized_scheduler in policy["allowed_schedulers"]
            else str(policy["fallback_scheduler"])
        )

        if resolved_sampler != original_sampler or resolved_scheduler != original_scheduler:
            print(
                f"[Flux Fill] Adjusted sampler/scheduler to "
                f"{resolved_sampler}/{resolved_scheduler} (was {original_sampler}/{original_scheduler})"
            )
            self.sampler = resolved_sampler
            self.scheduler = resolved_scheduler

        if self.steps < 1:
            raise ValueError(f"Steps must be >= 1, got {self.steps}.")
        if self.guidance <= 0:
            raise ValueError(f"Guidance must be > 0, got {self.guidance}.")
        if not str(self.sampler or "").strip():
            raise ValueError("Sampler must be a non-empty string.")
        if not str(self.scheduler or "").strip():
            raise ValueError("Scheduler must be a non-empty string.")
        if self.prefetch_depth < 0:
            raise ValueError(f"prefetch_depth must be >= 0, got {self.prefetch_depth}.")
        if self.prefetch_chunk_mb < 1:
            raise ValueError(f"prefetch_chunk_mb must be >= 1, got {self.prefetch_chunk_mb}.")

        if require_existing_assets:
            is_empty_cond = not str(self.prompt or "").strip()
            assets_to_check = [
                ("UNet", self.unet_path),
                ("AE", self.ae_path),
            ]
            if is_empty_cond:
                assets_to_check.append(("conditioning cache", self.conditioning_cache_path))

            if not is_empty_cond:
                if self.clip_l_path:
                    assets_to_check.append(("CLIP-L", self.clip_l_path))
                if self.t5_path:
                    assets_to_check.append(("T5", self.t5_path))

            for label, value in assets_to_check:
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(f"{label} path does not exist: {path}")

    def validate_dispatch_ready(self, *, require_existing_assets: bool = True) -> None:
        self.validate_static(require_existing_assets=require_existing_assets)
        if self.image is None:
            raise ValueError("Flux Fill dispatch requires an input image.")
        if self.mask is None:
            raise ValueError("Flux Fill dispatch requires an input mask.")
        if not isinstance(self.image, np.ndarray) or self.image.ndim != 3 or int(self.image.shape[2]) != 3:
            raise ValueError(f"image must be an HWC RGB numpy array, got {getattr(self.image, 'shape', None)}.")
        if not isinstance(self.mask, np.ndarray) or self.mask.ndim not in (2, 3):
            raise ValueError(f"mask must be an HW or HWC numpy array, got {getattr(self.mask, 'shape', None)}.")
        if self.mask.shape[0] != self.image.shape[0] or self.mask.shape[1] != self.image.shape[1]:
            raise ValueError(
                f"mask spatial shape {tuple(self.mask.shape[:2])} must match image shape {tuple(self.image.shape[:2])}."
            )


@dataclass(frozen=True)
class FluxLatentArtifactBundle:
    source_latent: torch.Tensor
    concat_latent: torch.Tensor
    denoise_mask: torch.Tensor
    fingerprint: str
    vae_load_time: float = 0.0
    vae_encode_time: float = 0.0


@dataclass(frozen=True)
class FluxFillResult:
    output_image: np.ndarray
    seed: int
    width: int
    height: int
    runtime_identity: FluxRuntimeIdentity | None = None
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_default_vae_approx_path() -> str | None:
    try:
        from modules.config import path_vae_approx
    except Exception:
        return None

    if isinstance(path_vae_approx, (list, tuple)):
        return str(path_vae_approx[0]) if path_vae_approx else None
    return str(path_vae_approx) if path_vae_approx else None


class FluxFillPreviewContext:
    """Encapsulates the preview generation capability of the active runtime posture."""
    def __init__(self, latent_format: Any, device: torch.device, vae_approx_path: str | None = None) -> None:
        self.latent_format = latent_format
        self.device = device
        self.vae_approx_path = vae_approx_path or _resolve_default_vae_approx_path()
        self._previewer = None
        self._resolved = False

    def decode(self, latent: torch.Tensor) -> np.ndarray | None:
        if not isinstance(latent, torch.Tensor):
            return latent if isinstance(latent, np.ndarray) else None

        if not self._resolved:
            try:
                from backend.preview import resolve_best_available_previewer
                self._previewer = resolve_best_available_previewer(
                    self.device,
                    self.latent_format,
                    vae_approx_path=self.vae_approx_path,
                )
            except Exception:
                pass
            finally:
                self._resolved = True

        if self._previewer is None:
            return None

        try:
            from backend.preview import decode_preview_payload
            return decode_preview_payload(self._previewer, self.latent_format, latent)
        except Exception:
            return None
