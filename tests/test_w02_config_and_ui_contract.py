import json
from pathlib import Path

import pytest

from modules.config_metadata import (
    CONFIG_TEMPLATE_METADATA_KEYS,
    split_config_template_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def test_comment_is_reserved_template_metadata():
    runtime_config, metadata = split_config_template_metadata(
        {"_comment": ["Copy this file."], "default_cfg_scale": 7.0}
    )

    assert CONFIG_TEMPLATE_METADATA_KEYS == frozenset({"_comment"})
    assert runtime_config == {"default_cfg_scale": 7.0}
    assert metadata == {"_comment": ["Copy this file."]}


@pytest.mark.parametrize("comment", [42, {"text": "invalid"}, ["valid", 42]])
def test_comment_metadata_rejects_invalid_shapes(comment):
    with pytest.raises(ValueError, match="_comment"):
        split_config_template_metadata({"_comment": comment})


def test_committed_template_uses_the_reserved_comment_contract():
    template = json.loads((ROOT / "config_template.txt").read_text(encoding="utf-8"))

    _, metadata = split_config_template_metadata(template)
    assert metadata["_comment"]
    assert any("config.txt" in line for line in metadata["_comment"])


def test_aspect_ratio_label_caches_launch_value_before_dom_lookup():
    source = (ROOT / "javascript" / "modules" / "00_ui_utils.js").read_text(
        encoding="utf-8"
    )

    cache_assignment = "aspectRatioSelectedValue = htmlDecode(String(value).trim());"
    label_lookup = "const label = root.querySelector('#aspect_ratios_accordion"

    assert "var aspectRatioSelectedValue = '';" in source
    assert source.index(cache_assignment) < source.index(label_lookup)
    assert "get_selected_aspect_ratio_value(aspectRatioSelectedValue)" in source
