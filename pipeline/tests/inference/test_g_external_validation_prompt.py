"""Regression tests for the new external-validation G prompt builder.

Proves build_g_prompt_render() (the frozen target-G path) is byte-for-byte
unchanged by this addition, and that the new builder injects no
target-specific content (climate-scientist context, target material,
target questions/intervention labels, fabricated filler).
"""

from __future__ import annotations

import survey_content as sc

from inference.prompts import (
    G_EXTERNAL_VALIDATION_PROTOCOL_ID,
    G_PROMPT_PROTOCOL,
    G_SYSTEM_PROMPT,
    build_g_external_validation_prompt_render,
    build_g_prompt_render,
    text_hash,
)

# Golden hashes captured from build_g_prompt_render() BEFORE this session's
# addition of build_g_external_validation_prompt_render -- if either changes,
# the target-G prompt compiler's output has changed and this test must fail.
_GOLDEN_SYSTEM_PROMPT_HASH = "bd227a7dd8fbc554711e48172ab92d9b051720d02750d357d9b47893222d1a8f"
_GOLDEN_USER_PROMPT_HASH = "f35bab4d46ee8ef4db197008b178234564089f47d5424cfb15ec15e75590f20f"


def _target_profile(state_abbr: str = "CA") -> dict[str, object]:
    return {
        "age": 34,
        "gender": "Female",
        "race": "White",
        "education": "Bachelor's degree",
        "income": "$56,000 to $99,999",
        "party": "Independent",
        "state": "California",
        "state_abbr": state_abbr,
    }


def test_target_g_prompt_render_byte_identical_after_addition():
    render = build_g_prompt_render(_target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    assert text_hash(render.system_prompt) == _GOLDEN_SYSTEM_PROMPT_HASH
    assert text_hash(render.user_prompt) == _GOLDEN_USER_PROMPT_HASH
    assert render.protocol_id == G_PROMPT_PROTOCOL


def test_target_g_still_injects_climate_scientist_context():
    """Confirms the target path's own known behavior is untouched (not that
    it's desirable) -- the external builder must NOT do this (see below)."""
    render = build_g_prompt_render(_target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    assert sc.get_common_climate_scientist_context() in render.user_prompt


_ATP_ITEM = {
    "qualtrics_label": "livstan",
    "target_label": "livstan",
    "response_key": "Q001",
    "question_text": "Compared to your parents when they were the age you are now, do you think your own standard of living now is...",
    "response_options": "1=Much better 2=Somewhat better 3=About the same 4=Somewhat worse 5=Much worse",
    "scale": "external_native_integer",
    "scale_min": 1,
    "scale_max": 5,
}


def test_external_builder_uses_frozen_system_prompt():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    assert render.system_prompt == G_SYSTEM_PROMPT


def test_external_builder_has_distinct_protocol_and_request_key():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    assert render.protocol_id == G_EXTERNAL_VALIDATION_PROTOCOL_ID
    assert render.protocol_id != G_PROMPT_PROTOCOL
    assert render.request_key.startswith("G_EXTERNAL|ATP1|104210|")


def test_external_builder_never_injects_climate_context():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    assert sc.get_common_climate_scientist_context() not in render.user_prompt
    assert "climate" not in render.user_prompt.lower()


def test_external_builder_renders_no_material_section_when_empty():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    assert "SURVEY MATERIAL" not in render.user_prompt


def test_external_builder_renders_material_when_provided():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="Some real participant-visible material.", source_id="ATP1", respondent_id="104210"
    )
    assert "SURVEY MATERIAL" in render.user_prompt
    assert "Some real participant-visible material." in render.user_prompt


def test_external_builder_never_fabricates_filler_text():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    for forbidden in ["neckties", "baseball", "dances", "no narrative"]:
        assert forbidden not in render.user_prompt.lower()


def test_external_builder_schema_matches_item_bounds():
    render = build_g_external_validation_prompt_render(
        _target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210"
    )
    prop = render.response_schema["properties"]["Q001"]
    assert prop["minimum"] == 1
    assert prop["maximum"] == 5


def test_external_builder_does_not_mutate_build_g_prompt_render_module_state():
    # Calling the external builder before/after must not change target output.
    before = build_g_prompt_render(_target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    build_g_external_validation_prompt_render(_target_profile(), [_ATP_ITEM], external_material="", source_id="ATP1", respondent_id="104210")
    after = build_g_prompt_render(_target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    assert before.user_prompt == after.user_prompt
    assert before.system_prompt == after.system_prompt
