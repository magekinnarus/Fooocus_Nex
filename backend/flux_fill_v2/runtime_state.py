from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from backend.flux_fill_v2.contracts import FluxFillRequest, UNetSpineKind, FluxLatentArtifactBundle
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


@dataclass(frozen=True)
class FluxResidentT5Key:
    clip_l_path: str
    t5_path: str


class FluxResidentT5State:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: FluxResidentT5Key | None = None
        self._posture: Any = None

    @staticmethod
    def _build_key(request: FluxFillRequest) -> FluxResidentT5Key:
        return FluxResidentT5Key(
            clip_l_path=str(request.clip_l_path or ""),
            t5_path=str(request.t5_path or ""),
        )

    def acquire(self, request: FluxFillRequest) -> Any:
        from backend.flux_fill_v2.t5_posture import FluxCpuFp16ResidentT5Posture

        requested_key = self._build_key(request)
        stale_posture: Any = None

        with self._lock:
            if self._posture is not None and self._key == requested_key:
                self._posture.request = request
                return self._posture

            stale_posture = self._posture
            self._posture = None
            self._key = None

        if stale_posture is not None:
            stale_posture.teardown()

        posture = FluxCpuFp16ResidentT5Posture(request)

        with self._lock:
            self._posture = posture
            self._key = requested_key
        return posture

    def release(self) -> bool:
        with self._lock:
            posture = self._posture
            self._posture = None
            self._key = None

        if posture is None:
            return False

        posture.teardown()
        return True

    def get_active_key(self) -> FluxResidentT5Key | None:
        with self._lock:
            return self._key


_RESIDENT_T5_STATE = FluxResidentT5State()


def acquire_resident_t5(request: FluxFillRequest) -> Any:
    return _RESIDENT_T5_STATE.acquire(request)


def release_active_flux_resident_t5() -> bool:
    return _RESIDENT_T5_STATE.release()


def get_active_flux_resident_t5_key() -> FluxResidentT5Key | None:
    return _RESIDENT_T5_STATE.get_active_key()


class FluxLatentArtifactState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._bundle: FluxLatentArtifactBundle | None = None

    def get_bundle(self, fingerprint: str) -> FluxLatentArtifactBundle | None:
        with self._lock:
            if self._bundle is not None and self._bundle.fingerprint == fingerprint:
                return self._bundle
            return None

    def set_bundle(self, bundle: FluxLatentArtifactBundle) -> None:
        with self._lock:
            self._bundle = bundle

    def release(self) -> bool:
        with self._lock:
            if self._bundle is None:
                return False
            self._bundle = None
            return True


_LATENT_ARTIFACT_STATE = FluxLatentArtifactState()


def get_cached_latent_artifact_bundle(fingerprint: str) -> FluxLatentArtifactBundle | None:
    return _LATENT_ARTIFACT_STATE.get_bundle(fingerprint)


def set_cached_latent_artifact_bundle(bundle: FluxLatentArtifactBundle) -> None:
    _LATENT_ARTIFACT_STATE.set_bundle(bundle)


def release_flux_latent_artifacts() -> bool:
    return _LATENT_ARTIFACT_STATE.release()
