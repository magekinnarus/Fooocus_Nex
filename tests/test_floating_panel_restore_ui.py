from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_staging_palette_minimize_control_becomes_restore_action():
    source = (ROOT / 'javascript' / 'modules' / '20_staging_viewer.js').read_text(encoding='utf-8')

    assert "toggleButton.textContent = isMinimized ? 'Restore' : '-'" in source
    assert "toggleButton.title = isMinimized ? 'Restore Palette' : 'Minimize'" in source


def test_monitor_minimize_control_becomes_restore_action():
    source = (ROOT / 'javascript' / 'modules' / '30_nex_monitor.js').read_text(encoding='utf-8')

    assert "toggleButton.textContent = isMinimized ? 'Restore' : '－'" in source
    assert "toggleButton.title = isMinimized ? 'Restore Dashboard' : 'Minimize'" in source


def test_minimized_panel_css_keeps_window_controls_discoverable():
    source = (ROOT / 'css' / 'modules' / '09_floating_panels.css').read_text(encoding='utf-8')

    assert '.floating-panel.minimized .panel-controls > button:not([id$="-toggle"]):not([id$="-close"])' in source
    assert '.floating-panel.minimized .panel-controls [id$="-toggle"]' in source
