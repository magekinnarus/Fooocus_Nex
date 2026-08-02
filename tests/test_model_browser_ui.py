from pathlib import Path


def test_model_browser_hides_sd15_subtabs_from_active_ui():
    source = (Path(__file__).resolve().parents[1] / 'javascript' / 'modules' / '60_nex_model_browser.js').read_text(encoding='utf-8')

    assert "{ key: 'sd15', label: 'SD15'" not in source
    assert "label: 'SD15 VAE'" not in source
    assert "label: 'SD15 Embeddings'" not in source


def test_model_browser_uses_top_only_download_and_preserves_installed_actions():
    source = (Path(__file__).resolve().parents[1] / 'javascript' / 'modules' / '60_nex_model_browser.js').read_text(encoding='utf-8')

    assert 'Download + Apply' not in source
    assert 'Download + Positive' not in source
    assert 'Download + Negative' not in source
    assert 'activate-available' not in source
    assert 'pendingActivations' not in source
    assert '>Apply</button>' in source
    assert '>Review</button>' in source
    assert 'Download Selected' in source
