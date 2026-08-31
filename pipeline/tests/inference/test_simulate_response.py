"""SurveyResponse: validates native integer answers against each item's own
scale bounds at construction time (Step 10: "validate aggressively").
Uses real item labels/scales from codebook.csv (trust_post=slider,
donation_ams=donation, newsletter_signup=binary) rather than a mock
schema, so this also catches a drift between survey_content.py's real
scale assignment and this module's bounds logic.
"""

from __future__ import annotations

import pytest

import survey_content as sc
from inference.simulate_response import SurveyResponse, simulate_response


def test_valid_values_construct_cleanly():
    resp = SurveyResponse(values={"trust_post": 80, "donation_ams": 5, "newsletter_signup": 1})
    assert resp["trust_post"] == 80
    assert resp["donation_ams"] == 5
    assert resp["newsletter_signup"] == 1


@pytest.mark.parametrize("value", [-1, 101, 100.5])
def test_slider_out_of_bounds_rejected(value):
    with pytest.raises(ValueError):
        SurveyResponse(values={"trust_post": value})


@pytest.mark.parametrize("value", [-1, 11])
def test_donation_out_of_bounds_rejected(value):
    with pytest.raises(ValueError):
        SurveyResponse(values={"donation_ams": value})


@pytest.mark.parametrize("value", [-1, 2, 0.0])
def test_newsletter_must_be_exactly_0_or_1(value):
    with pytest.raises(ValueError):
        SurveyResponse(values={"newsletter_signup": value})


def test_non_int_type_rejected():
    with pytest.raises(ValueError):
        SurveyResponse(values={"trust_post": "80"})


def test_bool_rejected_even_though_bool_is_an_int_subclass():
    with pytest.raises(ValueError):
        SurveyResponse(values={"newsletter_signup": True})


def test_boundary_values_accepted():
    SurveyResponse(values={"trust_post": 0})
    SurveyResponse(values={"trust_post": 100})
    SurveyResponse(values={"donation_ams": 0})
    SurveyResponse(values={"donation_ams": 10})
    SurveyResponse(values={"newsletter_signup": 0})
    SurveyResponse(values={"newsletter_signup": 1})


def test_unknown_label_is_not_validated_not_rejected():
    # a caller-restricted item subset (e.g. build_advocacy_validation.py's 4
    # mapped outcomes) is fine -- unrecognized labels pass through untouched.
    resp = SurveyResponse(values={"not_a_real_item": 12345})
    assert resp["not_a_real_item"] == 12345


def test_simulate_response_maps_neutral_prompt_keys_to_target_labels():
    items = [item for item in sc.load_items() if item["target_label"] in {"trust_post", "newsletter_signup"}]

    class FakeClient:
        def __call__(self, messages, schema):
            assert all(field.startswith("Q") for field in schema["required"])
            assert "trust_post_1:" not in messages[1]["content"]
            out = {}
            for field in schema["required"]:
                spec = schema["properties"][field]
                out[field] = 1 if spec.get("enum") == [0, 1] else 80
            return out

    resp = simulate_response(
        {"age": 40, "gender": "Female", "race": "White / Caucasian"},
        "Stimulus",
        items,
        FakeClient(),
        donor_key="D1",
        condition_id="control",
    )

    assert resp["trust_post"] == 80
    assert resp["newsletter_signup"] == 1
