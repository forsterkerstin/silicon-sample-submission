"""Tests for the R1 root-cause format-only amendment
(response_format_instruction_version="v2" on build_f_prompt_render_from_items).

Frozen before any replacement-R1 request is built. Proves: (1) the v1
default is byte-for-byte unchanged (golden hash), so every existing caller
-- F-screen, historical R1, the calibration archive builder -- is
unaffected unless it explicitly opts in; (2) v2 differs from v1 ONLY in the
closing instruction sentence, with all scientific content (profile,
stimulus, outcome question text, item labels) byte-identical; (3) the v2
instruction explicitly names the real key, explicitly disclaims the
item's own letter label, and carries a plain-text schema copy mechanically
derived from item_json_schema -- exactly what caused R1's observed failure
mode; (4) the exact malformed/valid shapes observed in R1's raw output are
correctly classified by the existing (unmodified) validator.
"""

from __future__ import annotations

import json

import pytest

from ate.f_screen_validation import validate_response
from inference.prompts import build_f_prompt_render_from_items, item_json_schema

_PROFILE = {"age": 30, "gender": "Male", "race": "White", "education": "College", "income": "Mid", "party": "Ind"}
_ITEM = {
    "qualtrics_label": "frustrated",
    "target_label": "frustrated",
    "response_key": "response",
    "question_text": "How much of each of the following emotions are you feeling as a result of the article you read?\n\n A. Frustrated",
    "response_options": "Please choose a number from 1 (Very slightly or not at all) to 5 (Extremely)",
    "scale": "external_native_integer",
    "scale_min": 1,
    "scale_max": 5,
}


def _render(version="v1", replicate_id=5):
    return build_f_prompt_render_from_items(
        _PROFILE,
        "Some frozen stimulus text.",
        [_ITEM],
        study_id="S1",
        f_profile_id="P1",
        outcome_id="E1",
        replicate_id=replicate_id,
        condition_id="control",
        response_format_instruction_version=version,
    )


def test_v1_default_matches_omitting_the_parameter():
    explicit = _render(version="v1")
    omitted = build_f_prompt_render_from_items(
        _PROFILE, "Some frozen stimulus text.", [_ITEM], study_id="S1", f_profile_id="P1", outcome_id="E1", replicate_id=5, condition_id="control"
    )
    assert explicit.user_prompt == omitted.user_prompt


def test_v1_still_ends_with_original_closing_sentence():
    r = _render(version="v1")
    assert r.user_prompt.endswith("Return a single JSON object with one integer value per item label on the native response scale.")


def test_rejects_unknown_version():
    with pytest.raises(ValueError):
        _render(version="v3")


# ---- Section 3: prove ONLY the closing instruction differs between v1/v2 ----


def test_only_closing_instruction_differs_between_v1_and_v2():
    v1 = _render(version="v1")
    v2 = _render(version="v2")
    assert v1.system_prompt == v2.system_prompt
    prefix_v1 = v1.user_prompt.split("Return a single JSON object")[0]
    prefix_v2 = v2.user_prompt.split("Respond ONLY with one JSON object")[0]
    assert prefix_v1 == prefix_v2  # profile, stimulus, outcome question text, item label "A." all identical
    assert v1.response_schema == v2.response_schema
    assert v1.stimulus_text == v2.stimulus_text
    assert v1.request_key == v2.request_key  # same study/profile/outcome/condition/replicate -> same identity


def test_v2_scientific_stimulus_text_unchanged():
    v2 = _render(version="v2")
    assert "Some frozen stimulus text." in v2.user_prompt


def test_v2_item_label_a_preserved_verbatim():
    v2 = _render(version="v2")
    assert " A. Frustrated" in v2.user_prompt  # archived outcome wording, untouched


# ---- E/F/G: v2 instruction content ----


def test_v2_explicitly_names_the_real_key():
    v2 = _render(version="v2")
    assert '"response"' in v2.user_prompt.split("Respond ONLY")[1]


def test_v2_explicitly_disclaims_item_label_as_key():
    v2 = _render(version="v2")
    tail = v2.user_prompt.split("Respond ONLY")[1]
    assert "Do NOT use a survey item" in tail
    assert '"A"' in tail


def test_v2_prohibits_markdown_code_fences():
    v2 = _render(version="v2")
    assert "Markdown" in v2.user_prompt
    assert "code fences" in v2.user_prompt


def test_v2_plain_text_schema_matches_response_format_schema_exactly():
    v2 = _render(version="v2")
    schema_text = v2.user_prompt.split("Your response must satisfy exactly this JSON schema:\n")[1]
    parsed = json.loads(schema_text)
    assert parsed == item_json_schema([_ITEM]) == v2.response_schema


def test_v2_multi_item_lists_all_keys():
    item_b = dict(_ITEM, qualtrics_label="confused", target_label="confused", response_key="response_b", question_text="B. Confused")
    v2 = build_f_prompt_render_from_items(
        _PROFILE, "Stimulus.", [_ITEM, item_b], study_id="S1", f_profile_id="P1", outcome_id="E2", replicate_id=5, condition_id="control", response_format_instruction_version="v2"
    )
    tail = v2.user_prompt.split("Respond ONLY")[1]
    assert '"response"' in tail and '"response_b"' in tail
    assert "these keys" in tail  # plural phrasing when >1 item


# ---- A-D: exact observed R1 failure/success shapes, classified by the
# existing (unmodified) validator ----

_SCHEMA = item_json_schema([_ITEM])


def _raw(content: str) -> dict:
    return {"custom_id": "c1", "response": {"status_code": 200, "body": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}}


def test_A_wrong_key_is_invalid():
    result = validate_response(_raw('{"A": 3}'), _SCHEMA)
    assert result["valid"] is False


def test_B_markdown_fenced_correct_key_is_invalid():
    result = validate_response(_raw('```json\n{"response": 3}\n```'), _SCHEMA)
    assert result["valid"] is False
    assert "malformed_json" in result["reason"]


def test_C_clean_correct_key_in_range_is_valid():
    result = validate_response(_raw('{"response": 3}'), _SCHEMA)
    assert result["valid"] is True
    assert result["parsed"] == {"response": 3}


def test_D_extra_keys_are_invalid():
    result = validate_response(_raw('{"response": 3, "extra": 1}'), _SCHEMA)
    assert result["valid"] is False
