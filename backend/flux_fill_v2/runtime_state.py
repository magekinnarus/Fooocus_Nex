from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from backend.flux_fill_v2.contracts import FluxFillRequest, UNetSpineKind
from backend.flux_fill_v2.resident_spine import FluxResidentUNetSpine


@dataclass(frozen=True)
class FluxResidentSpineKey:
    unet_path: str
    device: str


class FluxResidentRuntimeState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: FluxResidentSpineKey | None = None
        self._spine: FluxResidentUNetSpine | None = None

    @staticmethod
    def _build_key(request: FluxFillRequest) -> FluxResidentSpineKey:
        return FluxResidentSpineKey(
            unet_path=str(request.unet_path),
            device=str(request.device or ""),
        )

    def acquire(self, request: FluxFillRequest) -> tuple[FluxResidentUNetSpine, bool]:
        if request.unet_spine != UNetSpineKind.RESIDENT:
            raise ValueError("Resident runtime state can only acquire resident UNet requests.")

        requested_key = self._build_key(request)
        stale_spine: FluxResidentUNetSpine | None = None

        with self._lock:
            if self._spine is not None and self._key == requested_key:
                self._spine.request = request
                if not self._spine.started or self._spine.unet_patcher is None:
                    self._spine.start()
                return self._spine, True

            stale_spine = self._spine
            self._spine = None
            self._key = None

        if stale_spine is not None:
            stale_spine.end()

        spine = FluxResidentUNetSpine(request)
        spine.start()

        with self._lock:
            self._spine = spine
            self._key = requested_key
        return spine, False

    def release(self, *, reason: str | None = None) -> bool:
        del reason  # Reserved for future telemetry.

        with self._lock:
            spine = self._spine
            self._spine = None
            self._key = None

        if spine is None:
            return False

        spine.end()
        return True

    def get_active_key(self) -> FluxResidentSpineKey | None:
        with self._lock:
            return self._key


_RESIDENT_RUNTIME_STATE = FluxResidentRuntimeState()


def acquire_resident_spine(request: FluxFillRequest) -> tuple[FluxResidentUNetSpine, bool]:
    return _RESIDENT_RUNTIME_STATE.acquire(request)


def release_active_flux_resident_spine(*, reason: str | None = None) -> bool:
    return _RESIDENT_RUNTIME_STATE.release(reason=reason)


def get_active_flux_resident_spine_key() -> FluxResidentSpineKey | None:
    return _RESIDENT_RUNTIME_STATE.get_active_key()
