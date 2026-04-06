import os
import sys
from types import SimpleNamespace

import pytest

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import environment_profile, memory_governor, resources


@pytest.fixture
def restore_profile():
    original_profile = memory_governor.environment_profile()
    original_policy = memory_governor.governor.policy
    yield
    memory_governor.configure_environment(original_profile, original_policy)


def test_residency_plan_differs_by_profile_and_phase(restore_profile):
    free_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_COLAB_FREE,
        total_ram_mb=16384,
        total_vram_mb=15360,
        is_colab=True,
    )
    memory_governor.configure_environment(free_profile)
    free_plan = memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.DIFFUSION)

    pro_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_COLAB_PRO,
        total_ram_mb=53248,
        total_vram_mb=23000,
        is_colab=True,
    )
    memory_governor.configure_environment(pro_profile)
    pro_plan = memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.DIFFUSION)

    assert free_plan.mode_for('unet') == 'pinned'
    assert free_plan.mode_for('clip_vision') == 'evictable'
    assert pro_plan.mode_for('unet') == 'pinned'
    assert pro_plan.mode_for('clip_vision') == 'warm'
    assert memory_governor.plan_for_task(phase=memory_governor.MemoryPhase.PROMPT_ENCODE).mode_for('clip') == 'pinned'


def test_cleanup_memory_dispatches_residency_handlers_by_target_phase(monkeypatch, restore_profile):
    low_vram_profile = environment_profile.resolve_environment_profile(
        override=environment_profile.PROFILE_LOCAL_LOW_VRAM,
        total_ram_mb=16384,
        total_vram_mb=4096,
        is_colab=False,
    )
    memory_governor.configure_environment(low_vram_profile)

    calls = []

    def snapshot(*args, **kwargs):
        return SimpleNamespace(free_ram_mb=8192.0, free_vram_mb=2048.0)

    monkeypatch.setattr(resources, 'capture_memory_snapshot', snapshot)
    monkeypatch.setattr(resources, 'soft_empty_cache', lambda force=False: calls.append(('soft_empty_cache', force)))
    monkeypatch.setattr(resources, '_try_malloc_trim', lambda: False)
    monkeypatch.setattr(resources.gc, 'collect', lambda: None)

    import modules.default_pipeline as default_pipeline
    from backend.preprocessors import runtime as preprocessor_runtime
    import backend.ip_adapter as ip_adapter
    import backend.pulid_runtime as pulid_runtime

    monkeypatch.setattr(
        default_pipeline,
        'apply_controlnet_residency',
        lambda mode: calls.append(('controlnet', mode)) or {'mode': mode},
    )
    monkeypatch.setattr(
        preprocessor_runtime,
        'apply_residency_policy',
        lambda mode: calls.append(('preprocessors', mode)) or {'mode': mode},
    )
    monkeypatch.setattr(
        ip_adapter,
        'apply_contextual_residency',
        lambda mode, clip_vision_action=None, insightface_action=None: calls.append(
            ('contextual', mode, clip_vision_action, insightface_action)
        ) or {'mode': mode, 'clip_vision_action': clip_vision_action, 'insightface_action': insightface_action},
    )
    monkeypatch.setattr(
        pulid_runtime,
        'apply_contextual_residency',
        lambda mode: calls.append(('pulid', mode)) or {'mode': mode},
    )

    resources.cleanup_memory(
        'unit_test_finalize',
        gc_collect=False,
        target_phase=resources.MemoryPhase.FINALIZE,
        notes={'test': True},
    )

    assert ('controlnet', 'destroy') in calls
    assert ('preprocessors', 'destroy') in calls
    assert ('pulid', 'destroy') in calls
    assert ('contextual', 'destroy', 'destroy', 'destroy') in calls
    assert any(name == 'soft_empty_cache' for name, *_ in calls)
