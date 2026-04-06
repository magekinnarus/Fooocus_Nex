"""
Central memory governance scaffold inspired by ComfyUI Dynamic VRAM.

The goal here is not to reproduce ComfyUI's allocator internals. Instead, we
provide a single place where phase transitions, cache policy, environment-aware
thresholds, and lightweight telemetry can live so the rest of the app stops
making ad hoc memory decisions.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import logging
import platform
import threading
import time
from typing import Any, Deque, Dict, Optional

import psutil


class MemoryPhase(str, Enum):
    IDLE = 'idle'
    TASK = 'task'
    ROUTE_SELECT = 'route_select'
    MODEL_REFRESH = 'model_refresh'
    PROMPT_ENCODE = 'prompt_encode'
    IMAGE_INPUT_PREPARE = 'image_input_prepare'
    VAE_ENCODE = 'vae_encode'
    STRUCTURAL_PREPROCESS = 'structural_preprocess'
    CONTEXTUAL_PREPROCESS = 'contextual_preprocess'
    CONTROL_APPLY = 'control_apply'
    REMOVAL = 'removal'
    DIFFUSION = 'diffusion'
    DECODE = 'decode'
    STITCH = 'stitch'
    UPSCALE = 'upscale'
    TILED_REFINE = 'tiled_refine'
    FINALIZE = 'finalize'


PHASE_ALIASES = {
    'prepare': MemoryPhase.MODEL_REFRESH.value,
    'image_input': MemoryPhase.IMAGE_INPUT_PREPARE.value,
    'control': MemoryPhase.CONTROL_APPLY.value,
    'postprocess': MemoryPhase.FINALIZE.value,
}


@dataclass
class MemorySnapshot:
    timestamp: float
    phase: str
    total_vram_mb: Optional[float]
    free_vram_mb: Optional[float]
    total_ram_mb: Optional[float]
    free_ram_mb: Optional[float]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResidencyPlan:
    pinned: tuple[str, ...] = ()
    warm: tuple[str, ...] = ()
    evictable: tuple[str, ...] = ()
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPolicy:
    low_vram_threshold_mb: float = 8192.0
    medium_vram_threshold_mb: float = 16384.0
    low_vram_cache_cooldown_s: float = 0.5
    medium_vram_cache_cooldown_s: float = 1.5
    high_vram_cache_cooldown_s: float = 4.0
    low_ram_headroom_mb: float = 2048.0
    critical_ram_headroom_mb: float = 1024.0
    checkpoint_switch_ram_headroom_mb: float = 4096.0
    minimum_cache_cooldown_s: float = 0.25
    linux_malloc_trim_enabled: bool = False
    linux_malloc_trim_trigger_mb: float = 2048.0
    aggressive_checkpoint_switch_reclaim: bool = False


@dataclass
class MemoryAffordance:
    allowed: bool
    phase: str
    required_ram_mb: float
    required_vram_mb: float
    minimum_free_ram_mb: float
    minimum_free_vram_mb: float
    free_ram_mb: Optional[float]
    free_vram_mb: Optional[float]
    free_ram_after_mb: Optional[float]
    free_vram_after_mb: Optional[float]
    reason: str
    notes: Dict[str, Any] = field(default_factory=dict)


class MemoryGovernor:
    def __init__(self, policy: MemoryPolicy | None = None):
        self._lock = threading.RLock()
        self._phase = MemoryPhase.IDLE.value
        self._phase_started_at = time.time()
        self._last_cache_flush = 0.0
        self._history: Deque[MemorySnapshot] = deque(maxlen=64)
        self._phase_stack: list[tuple[str, float]] = []
        self._base_policy = policy or MemoryPolicy()
        self.policy = MemoryPolicy(**vars(self._base_policy))
        self._environment_profile = None

    def configure_environment(self, profile=None, policy: MemoryPolicy | None = None):
        with self._lock:
            if profile is not None:
                self._environment_profile = profile
                merged = dict(vars(self._base_policy))
                merged.update(getattr(profile, 'policy_overrides', {}) or {})
                if policy is not None:
                    merged.update(vars(policy))
                self.policy = MemoryPolicy(**merged)
            elif policy is not None:
                self.policy = MemoryPolicy(**vars(policy))

    def environment_profile(self):
        return self._environment_profile

    def profile_name(self):
        profile = self.environment_profile()
        return getattr(profile, 'name', 'unconfigured')

    def policy_summary(self):
        return {
            'profile': self.profile_name(),
            'low_ram_headroom_mb': self.policy.low_ram_headroom_mb,
            'critical_ram_headroom_mb': self.policy.critical_ram_headroom_mb,
            'checkpoint_switch_ram_headroom_mb': self.policy.checkpoint_switch_ram_headroom_mb,
            'low_vram_threshold_mb': self.policy.low_vram_threshold_mb,
            'medium_vram_threshold_mb': self.policy.medium_vram_threshold_mb,
            'linux_malloc_trim_enabled': self.policy.linux_malloc_trim_enabled,
            'linux_malloc_trim_trigger_mb': self.policy.linux_malloc_trim_trigger_mb,
            'aggressive_checkpoint_switch_reclaim': self.policy.aggressive_checkpoint_switch_reclaim,
        }

    def begin_phase(self, phase: str | MemoryPhase, task=None, notes: Dict[str, Any] | None = None):
        phase_name = self._normalize_phase(phase)
        with self._lock:
            started_at = time.time()
            self._phase_stack.append((phase_name, started_at))
            self._phase = phase_name
            self._phase_started_at = started_at
            snapshot = self.capture_snapshot(notes=notes, task=task)
            self._history.append(snapshot)
            return snapshot

    def end_phase(self, phase: str | MemoryPhase | None = None, notes: Dict[str, Any] | None = None):
        with self._lock:
            if phase is None:
                if self._phase_stack:
                    self._phase_stack.pop()
            else:
                phase_name = self._normalize_phase(phase)
                for index in range(len(self._phase_stack) - 1, -1, -1):
                    if self._phase_stack[index][0] == phase_name:
                        del self._phase_stack[index]
                        break

            if self._phase_stack:
                self._phase, self._phase_started_at = self._phase_stack[-1]
            else:
                self._phase = MemoryPhase.IDLE.value
                self._phase_started_at = time.time()
            snapshot = self.capture_snapshot(notes=notes)
            self._history.append(snapshot)
            return snapshot

    @contextmanager
    def phase_scope(
        self,
        phase: str | MemoryPhase,
        task=None,
        notes: Dict[str, Any] | None = None,
        end_notes: Dict[str, Any] | None = None,
    ):
        phase_name = self._normalize_phase(phase)
        self.begin_phase(phase_name, task=task, notes=notes)
        try:
            yield phase_name
        finally:
            self.end_phase(phase_name, notes=end_notes)

    def capture_snapshot(self, notes: Dict[str, Any] | None = None, task=None):
        total_vram_mb = None
        free_vram_mb = None
        total_ram_mb = float(psutil.virtual_memory().total) / (1024 * 1024)
        free_ram_mb = float(psutil.virtual_memory().available) / (1024 * 1024)

        try:
            from backend import resources as resource_state

            total_vram_mb = getattr(resource_state, 'total_vram', None)
            if total_vram_mb is not None:
                total_vram_mb = float(total_vram_mb)

            try:
                device = resource_state.get_torch_device()
                free_vram_mb = float(resource_state.get_free_memory(device)) / (1024 * 1024)
            except Exception:
                free_vram_mb = None
        except Exception:
            logging.debug('MemoryGovernor could not import backend.resources for snapshot capture.', exc_info=True)

        payload = dict(notes or {})
        payload.setdefault('profile', self.profile_name())
        if task is not None:
            payload.setdefault('task_type', task.__class__.__name__)

        return MemorySnapshot(
            timestamp=time.time(),
            phase=self.current_phase(),
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            total_ram_mb=total_ram_mb,
            free_ram_mb=free_ram_mb,
            notes=payload,
        )

    def plan_for_task(self, task=None, phase: str | MemoryPhase | None = None):
        notes = {'phase': self._normalize_phase(phase) if phase is not None else self.current_phase()}
        if task is not None:
            notes['task_type'] = task.__class__.__name__
        notes['profile'] = self.profile_name()
        return ResidencyPlan(notes=notes)

    def can_afford(
        self,
        *,
        required_ram_mb: float = 0.0,
        required_vram_mb: float = 0.0,
        minimum_free_ram_mb: float | None = None,
        minimum_free_vram_mb: float = 0.0,
        phase: str | MemoryPhase | None = None,
        notes: Dict[str, Any] | None = None,
    ):
        phase_name = self._normalize_phase(phase) if phase is not None else self.current_phase()
        snapshot = self.capture_snapshot(notes=notes)
        ram_floor = self.policy.low_ram_headroom_mb if minimum_free_ram_mb is None else float(minimum_free_ram_mb)
        vram_floor = float(minimum_free_vram_mb)

        free_ram_after = None if snapshot.free_ram_mb is None else float(snapshot.free_ram_mb) - float(required_ram_mb)
        free_vram_after = None if snapshot.free_vram_mb is None else float(snapshot.free_vram_mb) - float(required_vram_mb)

        ram_ok = free_ram_after is None or free_ram_after >= ram_floor
        vram_ok = free_vram_after is None or free_vram_after >= vram_floor
        allowed = ram_ok and vram_ok

        reason_parts = []
        if not ram_ok:
            reason_parts.append(
                f"ram_after={free_ram_after:.1f}MB below floor={ram_floor:.1f}MB"
            )
        if not vram_ok:
            reason_parts.append(
                f"vram_after={free_vram_after:.1f}MB below floor={vram_floor:.1f}MB"
            )
        if not reason_parts:
            reason_parts.append('headroom_ok')

        return MemoryAffordance(
            allowed=allowed,
            phase=phase_name,
            required_ram_mb=float(required_ram_mb),
            required_vram_mb=float(required_vram_mb),
            minimum_free_ram_mb=ram_floor,
            minimum_free_vram_mb=vram_floor,
            free_ram_mb=snapshot.free_ram_mb,
            free_vram_mb=snapshot.free_vram_mb,
            free_ram_after_mb=free_ram_after,
            free_vram_after_mb=free_vram_after,
            reason='; '.join(reason_parts),
            notes=dict(notes or {}),
        )

    def needs_host_cleanup(self, *, required_ram_mb: float = 0.0, minimum_free_ram_mb: float | None = None, aggressive: bool = False):
        affordance = self.can_afford(
            required_ram_mb=required_ram_mb,
            minimum_free_ram_mb=minimum_free_ram_mb,
        )
        if aggressive:
            return True
        if affordance.free_ram_after_mb is not None and affordance.free_ram_after_mb < self.policy.critical_ram_headroom_mb:
            return True
        return not affordance.allowed

    def should_trim_host_memory(self, snapshot: MemorySnapshot | None = None, *, aggressive: bool = False):
        if not self.policy.linux_malloc_trim_enabled:
            return False
        if platform.system() != 'Linux':
            return False
        if aggressive:
            return True
        snapshot = snapshot or self.capture_snapshot()
        return snapshot.free_ram_mb is not None and snapshot.free_ram_mb < self.policy.linux_malloc_trim_trigger_mb

    def should_flush_cache(self, force: bool = False):
        if force:
            return True

        snapshot = self.capture_snapshot()
        if snapshot.total_vram_mb is None:
            return True

        total_vram = snapshot.total_vram_mb
        if total_vram < self.policy.low_vram_threshold_mb:
            cooldown = self.policy.low_vram_cache_cooldown_s
        elif total_vram < self.policy.medium_vram_threshold_mb:
            cooldown = self.policy.medium_vram_cache_cooldown_s
        else:
            cooldown = self.policy.high_vram_cache_cooldown_s

        if snapshot.phase in {MemoryPhase.DIFFUSION.value, MemoryPhase.DECODE.value}:
            cooldown *= 1.25
        elif snapshot.phase in {MemoryPhase.REMOVAL.value, MemoryPhase.UPSCALE.value}:
            cooldown *= 0.75

        if snapshot.free_ram_mb is not None and snapshot.free_ram_mb < self.policy.low_ram_headroom_mb:
            cooldown = max(self.policy.minimum_cache_cooldown_s, cooldown * 0.5)

        elapsed = time.time() - self._last_cache_flush
        return elapsed >= max(self.policy.minimum_cache_cooldown_s, cooldown)

    def note_cache_flush(self):
        with self._lock:
            self._last_cache_flush = time.time()

    def current_phase(self):
        return self._phase

    def phase_age(self):
        return time.time() - self._phase_started_at

    def history(self):
        return tuple(self._history)

    @staticmethod
    def _normalize_phase(phase: str | MemoryPhase):
        if isinstance(phase, MemoryPhase):
            return phase.value
        phase_name = str(phase).strip().lower()
        return PHASE_ALIASES.get(phase_name, phase_name)


governor = MemoryGovernor()


def configure_environment(profile=None, policy: MemoryPolicy | None = None):
    governor.configure_environment(profile=profile, policy=policy)


def environment_profile():
    return governor.environment_profile()


def policy_summary():
    return governor.policy_summary()


def begin_phase(phase: str | MemoryPhase, task=None, notes: Dict[str, Any] | None = None):
    return governor.begin_phase(phase, task=task, notes=notes)


def end_phase(phase: str | MemoryPhase | None = None, notes: Dict[str, Any] | None = None):
    return governor.end_phase(phase, notes=notes)


def phase_scope(
    phase: str | MemoryPhase,
    task=None,
    notes: Dict[str, Any] | None = None,
    end_notes: Dict[str, Any] | None = None,
):
    return governor.phase_scope(phase, task=task, notes=notes, end_notes=end_notes)


def capture_snapshot(notes: Dict[str, Any] | None = None, task=None):
    return governor.capture_snapshot(notes=notes, task=task)


def can_afford(**kwargs):
    return governor.can_afford(**kwargs)


def needs_host_cleanup(**kwargs):
    return governor.needs_host_cleanup(**kwargs)


def should_trim_host_memory(snapshot: MemorySnapshot | None = None, *, aggressive: bool = False):
    return governor.should_trim_host_memory(snapshot=snapshot, aggressive=aggressive)


def should_flush_cache(force: bool = False):
    return governor.should_flush_cache(force=force)


def note_cache_flush():
    governor.note_cache_flush()


def plan_for_task(task=None, phase: str | MemoryPhase | None = None):
    return governor.plan_for_task(task=task, phase=phase)


def current_phase():
    return governor.current_phase()


def normalize_phase(phase: str | MemoryPhase):
    return governor._normalize_phase(phase)
