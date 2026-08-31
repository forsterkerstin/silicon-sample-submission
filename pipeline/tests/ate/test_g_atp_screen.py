"""Regression tests for the ATP1/ATP2 G-screen profile mapping and primary
loss, frozen before any DeepSeek/Gemma G-screen results are observed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ate.g_atp_screen import (
    ATP1_SUBSTANTIVE_CHOICES,
    ATP2_SUBSTANTIVE_CHOICES,
    ATP_PROFILE_FIELD_MAP,
    atp1_human_reference_positions,
    atp1_item,
    atp2_human_reference_positions,
    atp2_item,
    atp_row_to_g_profile,
    g_atp_loss,
    item_w1_pp,
    model_response_to_unit_interval,
    position_on_unit_interval,
    select_g_star,
    usable_atp1_respondents,
    usable_atp2_respondents,
)

PIPELINE_ROOT = Path(__file__).resolve().parents[2]


def test_profile_mapping_is_one_representation_per_concept():
    assert set(ATP_PROFILE_FIELD_MAP.values()) == {
        "age", "gender", "race", "education", "income", "party", "political_ideology", "religion",
    }
    assert "state" not in ATP_PROFILE_FIELD_MAP.values()  # never fabricated
    assert len(ATP_PROFILE_FIELD_MAP) == len(set(ATP_PROFILE_FIELD_MAP.values()))  # no duplicated concept


def test_profile_mapping_omits_refused_as_not_present():
    row = {"age_cat": "30-49", "gender": "Refused", "race_ethn": "White non-Hispanic", "edu_cat2": "Refused",
           "family_income": "$100,000 or more", "party": "Democrat", "ideology": "Moderate", "religion_4cat": "Catholic"}
    profile = atp_row_to_g_profile(row)
    assert "gender" not in profile
    assert "education" not in profile
    assert profile["income"] == "$100,000 or more"


def test_profile_mapping_never_includes_attitude_or_outcome_fields():
    row = {"age_cat": "30-49", "gender": "A man", "race_ethn": "White non-Hispanic", "edu_cat2": "Postgraduate",
           "family_income": "$100,000 or more", "party": "Democrat", "ideology": "Moderate", "religion_4cat": "Catholic",
           "persona_description": "leaks attitudes", "answer_label": "Much better than your parents", "correct_answer": "A"}
    profile = atp_row_to_g_profile(row)
    assert set(profile.keys()) <= set(ATP_PROFILE_FIELD_MAP.values())


def test_usable_respondent_counts_match_source_convention():
    a1 = pd.read_csv(PIPELINE_ROOT / "data" / "atp_survey_simulations" / "atp1_human_test.csv")
    a2 = pd.read_csv(PIPELINE_ROOT / "data" / "atp_survey_simulations" / "atp2_human_test.csv")
    assert len(usable_atp1_respondents(a1)) == 650
    assert len(usable_atp2_respondents(a2)) == 576


def test_not_sure_and_refused_excluded_from_position_mapping():
    assert position_on_unit_interval("Not sure", ATP1_SUBSTANTIVE_CHOICES) is None
    assert position_on_unit_interval("Refused/Web blank", ATP1_SUBSTANTIVE_CHOICES) is None


def test_position_mapping_is_equally_spaced_not_source_response_map():
    # 5 substantive ATP1 choices -> {0, .25, .5, .75, 1.0}, NOT the source's
    # own inconsistent 0-4 (ATP1) vs 1-4 (ATP2) RESPONSE_MAP integers.
    positions = [position_on_unit_interval(c, ATP1_SUBSTANTIVE_CHOICES) for c in ATP1_SUBSTANTIVE_CHOICES]
    assert positions == [0.0, 0.25, 0.5, 0.75, 1.0]
    positions2 = [position_on_unit_interval(c, ATP2_SUBSTANTIVE_CHOICES) for c in ATP2_SUBSTANTIVE_CHOICES]
    assert positions2 == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_model_response_position_matches_human_position_convention():
    # A model answering "1" (first choice) must land on the same [0,1] point
    # as a human who chose that same first-listed option.
    human_pos = position_on_unit_interval(ATP1_SUBSTANTIVE_CHOICES[0], ATP1_SUBSTANTIVE_CHOICES)
    model_pos = model_response_to_unit_interval(1, ATP1_SUBSTANTIVE_CHOICES)
    assert human_pos == model_pos


def test_model_response_out_of_range_raises():
    with pytest.raises(ValueError):
        model_response_to_unit_interval(6, ATP1_SUBSTANTIVE_CHOICES)


def test_item_w1_pp_zero_for_identical_distributions():
    assert item_w1_pp([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(0.0)


def test_item_w1_pp_full_range_shift_is_100pp():
    assert item_w1_pp([0.0, 0.0], [1.0, 1.0]) == pytest.approx(100.0)


def test_g_atp_loss_is_equal_item_weight_not_equal_study_weight():
    # If it were weighted by (much larger) ATP1 N vs ATP2 N this wouldn't be
    # a plain average of the two pp values.
    assert g_atp_loss(10.0, 30.0) == pytest.approx(20.0)


def test_items_have_correct_scale_bounds():
    assert atp1_item()["scale_min"] == 1 and atp1_item()["scale_max"] == 5
    assert atp2_item()["scale_min"] == 1 and atp2_item()["scale_max"] == 4


def test_select_g_star_lowest_loss_wins():
    result = select_g_star({"modelA": 5.0, "modelB": 3.0}, invalid_response_rate={"modelA": 0.0, "modelB": 0.0}, realized_cost_usd={"modelA": 1.0, "modelB": 1.0})
    assert result["g_star"] == "modelB"


def test_atp1_human_reference_positions_excludes_non_substantive_and_maps_correctly():
    df = pd.DataFrame({"answer_label": ["Much better than your parents", "Much worse than your parents", "Not sure", "Refused/Web blank"]})
    positions = atp1_human_reference_positions(df)
    assert positions == [0.0, 1.0]


def test_atp2_human_reference_positions_uses_correct_answer_column():
    df = pd.DataFrame({"correct_answer": ["Excellent", "Poor", "Not sure"]})
    positions = atp2_human_reference_positions(df)
    assert positions == pytest.approx([0.0, 1.0])


def test_select_g_star_tie_break_shape_matches_f_screen():
    result = select_g_star(
        {"google/gemma-4-31B-it": 5.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 5.0},
        invalid_response_rate={"google/gemma-4-31B-it": 0.02, "deepseek-ai/DeepSeek-V4-Pro-0813": 0.0},
        realized_cost_usd={"google/gemma-4-31B-it": 1.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 100.0},
    )
    assert result["g_star"] == "deepseek-ai/DeepSeek-V4-Pro-0813"
