"""Regression tests for the G-v2 PROVIDER_SERVING_FORMAT_FAILURE format-only
fix. Proves v1 is unchanged (existing golden hash), v2 differs from v1 ONLY
by an appended closing-instruction suffix (system prompt, schema, and every
character preceding the suffix are byte-identical), and that every existing
call site (response_format_instruction_version="v1" by default) is
unaffected."""

from __future__ import annotations

import survey_content as sc
from inference.prompts import (
    G_FORMAT_INSTRUCTION_V2,
    G_SYSTEM_PROMPT,
    build_g_consensus_stage_a_prompt_render,
    build_g_prompt_render,
    text_hash,
)

P1_GOLDEN_SHA256 = "bd227a7dd8fbc554711e48172ab92d9b051720d02750d357d9b47893222d1a8f"


def _target_profile():
    return {
        "age": 40,
        "gender": "Female",
        "race_ethnicity": "White / Caucasian",
        "education": "Bachelor's degree",
        "household_income": "$56,000 to $99,999",
        "party_id": "Independent",
        "political_ideology": "Moderate",
        "state_abbr": "CA",
        "religion": "Protestant",
    }


def test_v1_still_byte_identical_to_production_golden_hash():
    assert text_hash(G_SYSTEM_PROMPT) == P1_GOLDEN_SHA256


def test_v2_appends_format_instruction_without_altering_anything_before_it():
    items = sc.load_items()
    v1 = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control")
    v2 = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control", response_format_instruction_version="v2")
    assert v1.system_prompt == v2.system_prompt
    assert v1.response_schema == v2.response_schema
    assert v2.user_prompt.startswith(v1.user_prompt)
    assert v2.user_prompt[len(v1.user_prompt) :] == f" {G_FORMAT_INSTRUCTION_V2}"
    # content-specific clause preserved verbatim, not removed
    assert "Return raw item responses only." in v2.user_prompt


def test_v2_request_key_never_collides_with_v1():
    items = sc.load_items()
    v1 = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control")
    v2 = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control", response_format_instruction_version="v2")
    assert v1.request_key != v2.request_key
    assert v2.request_key == "G|D1|control|replicate_1|fmt_v2"


def test_consensus_stage_a_v2_preserves_three_estimates_clause():
    r2 = build_g_consensus_stage_a_prompt_render(_target_profile(), donor_key="D1", response_format_instruction_version="v2")
    assert "Return only the three estimate responses." in r2.user_prompt
    assert G_FORMAT_INSTRUCTION_V2 in r2.user_prompt
    assert r2.request_key == "G|D1|Consensus|stage_a|replicate_1|fmt_v2"


def test_unknown_response_format_instruction_version_rejected():
    import pytest

    items = sc.load_items()
    with pytest.raises(ValueError, match="unknown response_format_instruction_version"):
        build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control", response_format_instruction_version="v3")


def test_default_call_sites_are_v1_and_unaffected():
    items = sc.load_items()
    explicit_v1 = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control", response_format_instruction_version="v1")
    default_call = build_g_prompt_render(_target_profile(), "Stimulus.", items, donor_key="D1", condition_id="control")
    assert explicit_v1.user_prompt == default_call.user_prompt
    assert explicit_v1.request_key == default_call.request_key
