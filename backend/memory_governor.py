"""
Central memory governance scaffold inspired by ComfyUI Dynamic VRAM.

The goal here is not to reproduce ComfyUI's allocator internals. Instead, we
provide a single place where phase transitions, cache policy, and lightweight
telemetry can live so the rest of the app stops making ad hoc memory decisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
from typing import Any, Deque, Dict, Optional

import psutil


class MemoryPhase(str, Enum):
    IDLE = 'idle'
    PREPARE = 'prepare'
    IMAGE_INPUT = 'image_input'
    CONTROL = 'control'
    REMOVAL = 'removal'
    DIFFUSION = 'diffusion'
    DECODE = 'decode'
    UPSCALE = 'upscale'
    POSTPROCESS = 'postprocess'


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
    minimum_cache_cooldown_s: float = 0.25


class MemoryGovernor:
    def __init__(self, policy: MemoryPolicy | None = None):
        self._lock = threading.RLock()
        self._phase = MemoryPhase.IDLE.value
        self._phase_started_at = time.time()
        self._last_cache_flush = 0.0
        self._history: Deque[MemorySnapshot] = deque(maxlen=64)
        self._phase_stack: list[tuple[str, float]] = []
        self.policy = policy or MemoryPolicy()

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
        return ResidencyPlan(notes=notes)

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
        return str(phase).strip().lower()


governor = MemoryGovernor()


def begin_phase(phase: str | MemoryPhase, task=None, notes: Dict[str, Any] | None = None):
    return governor.begin_phase(phase, task=task, notes=notes)


def end_phase(phase: str | MemoryPhase | None = None, notes: Dict[str, Any] | None = None):
    return governor.end_phase(phase, notes=notes)


def capture_snapshot(notes: Dict[str, Any] | None = None, task=None):
    return governor.capture_snapshot(notes=notes, task=task)


def should_flush_cache(force: bool = False):
    return governor.should_flush_cache(force=force)


def note_cache_flush():
    governor.note_cache_flush()


def plan_for_task(task=None, phase: str | MemoryPhase | None = None):
    return governor.plan_for_task(task=task, phase=phase)


def current_phase():
    return governor.current_phase()
