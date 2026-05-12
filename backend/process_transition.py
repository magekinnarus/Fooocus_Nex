from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional


PROCESS_FAMILY_SDXL = "sdxl"
PROCESS_FAMILY_FLUX_FILL = "flux_fill"

PROCESS_CLASS_STANDARD_SDXL = "standard_sdxl"
PROCESS_CLASS_SDXL_GGUF_STAGED = "sdxl_gguf_staged"
PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING = "sdxl_gguf_true_streaming"
PROCESS_CLASS_FLUX_FILL = "flux_fill"

_TOKEN_ALIASES = {
    "sdxl": PROCESS_FAMILY_SDXL,
    "flux": PROCESS_FAMILY_FLUX_FILL,
    "flux_fill": PROCESS_FAMILY_FLUX_FILL,
    "flux-fill": PROCESS_FAMILY_FLUX_FILL,
    "standard sdxl": PROCESS_CLASS_STANDARD_SDXL,
    "sdxl standard": PROCESS_CLASS_STANDARD_SDXL,
    "full_resident": PROCESS_CLASS_STANDARD_SDXL,
    "full resident": PROCESS_CLASS_STANDARD_SDXL,
    "full": PROCESS_CLASS_STANDARD_SDXL,
    "gguf staged": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "sdxl gguf staged": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf_staged_residency": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf staged residency": PROCESS_CLASS_SDXL_GGUF_STAGED,
    "gguf true streaming": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "sdxl gguf true streaming": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "benchmark only": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
    "benchmark-only": PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
}


def _normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _TOKEN_ALIASES.get(token, token)


def normalize_process_family(value: Any) -> str:
    return _normalize_token(value)


def resolve_process_class(
    value: Any = None,
    *,
    family: Any = None,
    execution_family: Any = None,
    residency_class: Any = None,
) -> str:
    if value is not None:
        return normalize_process_class(value, family=family)
    if execution_family is not None:
        return normalize_process_class(execution_family, family=family)
    if residency_class is not None:
        return normalize_process_class(residency_class, family=family)
    normalized_family = normalize_process_family(family)
    if normalized_family == PROCESS_FAMILY_FLUX_FILL:
        return PROCESS_CLASS_FLUX_FILL
    return PROCESS_CLASS_STANDARD_SDXL if normalized_family == PROCESS_FAMILY_SDXL else _normalize_token(normalized_family)


def normalize_process_class(value: Any, *, family: Any = None) -> str:
    token = _normalize_token(value)
    normalized_family = normalize_process_family(family)

    if normalized_family == PROCESS_FAMILY_FLUX_FILL:
        if token in {PROCESS_CLASS_FLUX_FILL, "flux", "flux_fill"}:
            return PROCESS_CLASS_FLUX_FILL
        return token

    if normalized_family == PROCESS_FAMILY_SDXL:
        if token in {
            PROCESS_CLASS_STANDARD_SDXL,
            "full_resident",
            "full",
            "standard",
            "standard_sdxl",
        }:
            return PROCESS_CLASS_STANDARD_SDXL
        if token in {
            PROCESS_CLASS_SDXL_GGUF_STAGED,
            "gguf_staged_residency",
            "staged",
            "gguf_staged",
        }:
            return PROCESS_CLASS_SDXL_GGUF_STAGED
        if token in {
            PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING,
            "gguf_true_streaming",
            "true_streaming",
        }:
            return PROCESS_CLASS_SDXL_GGUF_TRUE_STREAMING
    return token


@dataclass(frozen=True)
class ProcessKey:
    family: str
    process_class: str
    authoritative_identity: Any
    execution_family: Optional[str] = None
    residency_class: Optional[str] = None
    route_family: Optional[str] = None

    def normalized(self) -> "ProcessKey":
        return ProcessKey(
            family=normalize_process_family(self.family),
            process_class=normalize_process_class(self.process_class, family=self.family),
            authoritative_identity=self.authoritative_identity,
            execution_family=self.execution_family if self.execution_family is None else str(self.execution_family),
            residency_class=self.residency_class if self.residency_class is None else str(self.residency_class),
            route_family=self.route_family if self.route_family is None else str(self.route_family),
        )


@dataclass(frozen=True)
class ProcessTransitionDecision:
    action: str
    reason: str
    reset_required: bool
    current_key: ProcessKey | None
    requested_key: ProcessKey

    @property
    def reuse_allowed(self) -> bool:
        return not self.reset_required


class SharedProcessRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._active_key: ProcessKey | None = None

    def get_active_key(self) -> ProcessKey | None:
        with self._lock:
            return self._active_key

    def set_active_key(self, key: ProcessKey | None) -> ProcessKey | None:
        normalized = key.normalized() if key is not None else None
        with self._lock:
            self._active_key = normalized
            return self._active_key

    def clear_active_key(self) -> None:
        with self._lock:
            self._active_key = None

    def evaluate_transition(self, requested_key: ProcessKey) -> ProcessTransitionDecision:
        requested = requested_key.normalized()
        current = self.get_active_key()
        if current is None:
            return ProcessTransitionDecision(
                action="start",
                reason="no_active_process",
                reset_required=False,
                current_key=None,
                requested_key=requested,
            )

        if current == requested:
            return ProcessTransitionDecision(
                action="reuse",
                reason="same_process_identity",
                reset_required=False,
                current_key=current,
                requested_key=requested,
            )

        if current.family != requested.family:
            reason = "family_change"
        elif current.process_class != requested.process_class:
            reason = "process_class_change"
        elif current.authoritative_identity != requested.authoritative_identity:
            reason = "identity_change"
        else:
            reason = "same_process_identity"
            return ProcessTransitionDecision(
                action="reuse",
                reason=reason,
                reset_required=False,
                current_key=current,
                requested_key=requested,
            )

        return ProcessTransitionDecision(
            action="reset",
            reason=reason,
            reset_required=True,
            current_key=current,
            requested_key=requested,
        )


_DEFAULT_REGISTRY = SharedProcessRegistry()


def get_active_process_key() -> ProcessKey | None:
    return _DEFAULT_REGISTRY.get_active_key()


def set_active_process_key(key: ProcessKey | None) -> ProcessKey | None:
    return _DEFAULT_REGISTRY.set_active_key(key)


def clear_active_process_key() -> None:
    _DEFAULT_REGISTRY.clear_active_key()


def evaluate_process_transition(requested_key: ProcessKey) -> ProcessTransitionDecision:
    return _DEFAULT_REGISTRY.evaluate_transition(requested_key)


def build_process_key(
    *,
    family: Any,
    process_class: Any = None,
    authoritative_identity: Any,
    execution_family: Any = None,
    residency_class: Any = None,
    route_family: Any = None,
) -> ProcessKey:
    resolved_family = normalize_process_family(family)
    resolved_process_class = resolve_process_class(
        process_class,
        family=resolved_family,
        execution_family=execution_family,
        residency_class=residency_class,
    )
    return ProcessKey(
        family=resolved_family,
        process_class=resolved_process_class,
        authoritative_identity=authoritative_identity,
        execution_family=None if execution_family is None else str(execution_family),
        residency_class=None if residency_class is None else str(residency_class),
        route_family=None if route_family is None else str(route_family),
    )


def describe_process_key(key: ProcessKey | None) -> str:
    if key is None:
        return "<none>"
    return (
        f"family={key.family} "
        f"class={key.process_class} "
        f"identity={key.authoritative_identity!r}"
    )
