from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict

import psutil


PROFILE_COLAB_PRO = 'colab_pro'
PROFILE_COLAB_FREE = 'colab_free'
PROFILE_LOCAL_LOW_VRAM = 'local_low_vram'
PROFILE_LOCAL_NORMAL = 'local_normal'
PROFILE_CUSTOM = 'custom'
PROFILE_AUTO = 'auto'

KNOWN_PROFILE_OVERRIDES = {
    PROFILE_AUTO,
    PROFILE_COLAB_PRO,
    PROFILE_COLAB_FREE,
    PROFILE_LOCAL_LOW_VRAM,
    PROFILE_LOCAL_NORMAL,
    PROFILE_CUSTOM,
}


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    display_name: str
    source: str
    total_ram_mb: float
    total_vram_mb: float
    is_colab: bool
    policy_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'source': self.source,
            'total_ram_mb': round(float(self.total_ram_mb), 1),
            'total_vram_mb': round(float(self.total_vram_mb), 1),
            'is_colab': bool(self.is_colab),
            'policy_overrides': dict(self.policy_overrides),
            'notes': dict(self.notes),
        }

    def startup_message(self) -> str:
        return (
            f"[Startup] Memory environment profile: {self.display_name} "
            f"(name={self.name}, source={self.source}, ram={self.total_ram_mb:.0f}MB, "
            f"vram={self.total_vram_mb:.0f}MB, colab={self.is_colab})"
        )


PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    PROFILE_COLAB_PRO: {
        'display_name': 'Colab Pro',
        'policy_overrides': {
            'low_ram_headroom_mb': 4096.0,
            'critical_ram_headroom_mb': 2048.0,
            'checkpoint_switch_ram_headroom_mb': 8192.0,
            'linux_malloc_trim_enabled': True,
            'linux_malloc_trim_trigger_mb': 4096.0,
            'aggressive_checkpoint_switch_reclaim': False,
        },
    },
    PROFILE_COLAB_FREE: {
        'display_name': 'Colab Free',
        'policy_overrides': {
            'low_ram_headroom_mb': 3072.0,
            'critical_ram_headroom_mb': 1536.0,
            'checkpoint_switch_ram_headroom_mb': 4096.0,
            'linux_malloc_trim_enabled': True,
            'linux_malloc_trim_trigger_mb': 3072.0,
            'aggressive_checkpoint_switch_reclaim': True,
        },
    },
    PROFILE_LOCAL_LOW_VRAM: {
        'display_name': 'Local Low VRAM',
        'policy_overrides': {
            'low_vram_threshold_mb': 6144.0,
            'medium_vram_threshold_mb': 12288.0,
            'low_ram_headroom_mb': 3072.0,
            'critical_ram_headroom_mb': 1536.0,
            'checkpoint_switch_ram_headroom_mb': 4096.0,
            'aggressive_checkpoint_switch_reclaim': True,
        },
    },
    PROFILE_LOCAL_NORMAL: {
        'display_name': 'Local Normal',
        'policy_overrides': {
            'low_ram_headroom_mb': 4096.0,
            'critical_ram_headroom_mb': 2048.0,
            'checkpoint_switch_ram_headroom_mb': 6144.0,
            'aggressive_checkpoint_switch_reclaim': False,
        },
    },
    PROFILE_CUSTOM: {
        'display_name': 'Custom Override',
        'policy_overrides': {},
    },
}


def detect_total_ram_mb() -> float:
    return float(psutil.virtual_memory().total) / (1024 * 1024)


def detect_total_vram_mb() -> float:
    try:
        import torch

        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return float(props.total_memory) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def detect_is_colab() -> bool:
    if any(key in os.environ for key in ('COLAB_GPU', 'COLAB_RELEASE_TAG', 'COLAB_BACKEND_VERSION')):
        return True

    try:
        import args_manager

        return bool(getattr(args_manager.args, 'colab', False))
    except Exception:
        return False


def _merge_policy_overrides(profile_name: str, custom_policy_overrides: Dict[str, Any] | None = None):
    overrides = dict(PROFILE_DEFAULTS[profile_name]['policy_overrides'])
    for key, value in (custom_policy_overrides or {}).items():
        if value is None:
            continue
        overrides[key] = value
    return overrides


def auto_detect_profile_name(*, total_ram_mb: float, total_vram_mb: float, is_colab: bool) -> str:
    if is_colab:
        if total_vram_mb >= 20000 or total_ram_mb >= 40960:
            return PROFILE_COLAB_PRO
        return PROFILE_COLAB_FREE

    if total_vram_mb > 0 and total_vram_mb <= 6144:
        return PROFILE_LOCAL_LOW_VRAM
    return PROFILE_LOCAL_NORMAL


def resolve_environment_profile(
    *,
    override: str = PROFILE_AUTO,
    custom_name: str = PROFILE_CUSTOM,
    total_ram_mb: float | None = None,
    total_vram_mb: float | None = None,
    is_colab: bool | None = None,
    custom_policy_overrides: Dict[str, Any] | None = None,
) -> EnvironmentProfile:
    total_ram_mb = detect_total_ram_mb() if total_ram_mb is None else float(total_ram_mb)
    total_vram_mb = detect_total_vram_mb() if total_vram_mb is None else float(total_vram_mb)
    is_colab = detect_is_colab() if is_colab is None else bool(is_colab)

    requested = str(override or PROFILE_AUTO).strip().lower()
    if requested not in KNOWN_PROFILE_OVERRIDES:
        requested = PROFILE_AUTO

    if requested == PROFILE_AUTO:
        profile_name = auto_detect_profile_name(
            total_ram_mb=total_ram_mb,
            total_vram_mb=total_vram_mb,
            is_colab=is_colab,
        )
        source = 'auto'
    elif requested == PROFILE_CUSTOM:
        profile_name = PROFILE_CUSTOM
        source = 'custom'
    else:
        profile_name = requested
        source = 'override'

    display_name = PROFILE_DEFAULTS[profile_name]['display_name']
    if profile_name == PROFILE_CUSTOM and custom_name:
        display_name = str(custom_name)

    return EnvironmentProfile(
        name=profile_name,
        display_name=display_name,
        source=source,
        total_ram_mb=total_ram_mb,
        total_vram_mb=total_vram_mb,
        is_colab=is_colab,
        policy_overrides=_merge_policy_overrides(profile_name, custom_policy_overrides=custom_policy_overrides),
        notes={
            'requested_override': requested,
        },
    )
