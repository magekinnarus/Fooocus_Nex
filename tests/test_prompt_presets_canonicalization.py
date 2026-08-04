import hashlib
import json
from random import Random

import pytest

from modules.pipeline.preprocessing import process_prompt
from modules.prompt_presets import (
    apply_prompt_preset,
    get_random_prompt_preset,
    prompt_presets,
    random_prompt_preset_name,
)
from modules.task_state import TaskState


FIXED_PROMPT = "a glasshouse in winter"

# Frozen from the pre-rename expansion workloads. These cover every bundled
# preset while keeping the test fixture independent of the renamed JSON files.
EXPECTED_EXPANSION_SHA256 = {
    "Dj Digital Illustration": "5612685bb169681df17e82f3f8de911b3e65af3181bf57d5ad6f34a26ad081cb",
    "Dj Dynamic": "a2cb1904d42135b18d41a8c09a1f9a9369417154355ae7fc22d57bdf56d695bf",
    "Dj Dystopian": "bc0316cbaff77bcf6d2fbca63aaf09d212b1cc33f82acc66efff2341928ba965",
    "Dj Fairy Tale": "7c49c69237330ecbd684ab8c16fa981a5173874105437f33926f0dc9b80fd65b",
    "Dj Hyperrealism": "6644bbc1f82f99af5249df3782fafdae5dd2f37492271907363511477b2effa5",
    "Dj Illustrious": "356b6754b1c25f55df289960a79e4da6d45f57622bde47cf1595ddd191041622",
    "Dj Negative Enhance": "3584fc71a41a59dfa709b36938d6b62cacd36b825c3a4dff5c64692653c9f3c5",
    "Dj Pony": "7ea46d607b7d072ffa912412c9f2ea9d2fcf1b98ee7fa165501362a7a5f167b6",
    "Fooocus Cinematic": "5faf4b4a76d9d771ade755645447f0ec6eb01d0bde3f40a88db46b6c773c15fc",
    "Fooocus Enhance": "69eca8954ef095b527c79162714e67c6be9de4da48ab8ed1520428df472846e5",
    "Fooocus Masterpiece": "7ee7cf7677a220f88ce0fe7c4ebd13092d7194e5cbc0b823ed3860d98c4fa7a1",
    "Fooocus Negative": "451d06256d21b8131ff257e3e7825c1194d8e7c0ca20b0509268fbb950ffcf70",
    "Fooocus Photograph": "6d853cdf7cab21c215171de1a9f1416bda669bddd68ac12194f076766d2eb597",
    "Fooocus Pony": "d90cb468a0f741c745f12eb270b10c89bdb6c410a4f6e860e443c70f602ca102",
    "Fooocus Semi Realistic": "6dd227003de40bf5c2414d8f206e82ba870234352210b72a704adb00f808ede2",
    "Fooocus Sharp": "e5e27a6c0adbe02a47aeee4b1dc51cab59dc2b19fe643f56d48303c78ac5e005",
}


def _expansion_digest(prompt_preset_name):
    workload = apply_prompt_preset(prompt_preset_name, positive=FIXED_PROMPT)
    payload = json.dumps(workload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_every_shipped_prompt_preset_matches_pre_rename_expansion_workload():
    actual = {
        name: _expansion_digest(name)
        for name in sorted(prompt_presets)
    }

    assert actual == EXPECTED_EXPANSION_SHA256


def test_multiple_prompt_presets_preserve_selection_order():
    task_state = TaskState(
        prompt="a glasshouse in winter",
        negative_prompt="low quality",
        prompt_preset_selections=["Fooocus Sharp", "Dj Dynamic"],
        seed=123,
    )

    task = process_prompt(task_state, base_model_additional_loras=[])[0]

    assert task["positive"] == [
        "a glasshouse in winter",
        "cinematic still a glasshouse in winter . emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "dynamic pose, interesting angle, eye catching composition, depth of field, forced perspective",
    ]
    assert task["negative"] == [
        "low quality",
        "anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    ]
    assert task["prompt_presets"] == ["Fooocus Sharp", "Dj Dynamic"]


def test_random_prompt_preset_is_deterministic_and_never_the_sentinel():
    expected = [
        "Dj Dynamic",
        "Fooocus Sharp",
        "Dj Dynamic",
        "Dj Fairy Tale",
        "Fooocus Negative",
        "Fooocus Pony",
    ]
    first_rng = Random(2026)
    second_rng = Random(2026)
    first = [get_random_prompt_preset(first_rng) for _ in expected]
    second = [get_random_prompt_preset(second_rng) for _ in expected]

    assert first == second == expected
    assert random_prompt_preset_name not in prompt_presets
    assert all(name in prompt_presets for name in first)

    task_state = TaskState(
        prompt=FIXED_PROMPT,
        prompt_preset_selections=[random_prompt_preset_name],
        image_number=len(expected),
        seed=2026,
    )
    tasks = process_prompt(task_state, base_model_additional_loras=[])

    assert [task["prompt_presets"][0] for task in tasks] == expected


@pytest.mark.parametrize("route_family", ["txt2img", "inpaint", "outpaint", "upscale"])
def test_sdxl_route_prompt_tasks_carry_nonempty_prompt_preset_selection(route_family):
    task_state = TaskState(
        prompt=FIXED_PROMPT,
        negative_prompt="low quality",
        prompt_preset_selections=["Dj Dynamic"],
        seed=123,
    )

    task = process_prompt(
        task_state,
        base_model_additional_loras=[],
        route_family=route_family,
    )[0]

    assert task["prompt_presets"] == ["Dj Dynamic"]
    assert task["positive"][1] == (
        "dynamic pose, interesting angle, eye catching composition, "
        "depth of field, forced perspective"
    )
