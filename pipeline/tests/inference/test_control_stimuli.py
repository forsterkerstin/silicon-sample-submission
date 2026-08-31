"""Control-condition handling: preserve the three neutral filler texts from
the organizer survey instead of replacing control with an empty/no-treatment
prompt."""

from __future__ import annotations

import pandas as pd

import survey_content as sc
from generate_responses import simulate_profile


def test_control_stimuli_parse_all_three_neutral_texts():
    texts = sc.get_control_stimuli()
    assert len(texts) == 3
    assert "The History of Neckties" in texts[0]
    assert "The Rules of Baseball" in texts[1]
    assert "Different Types of Dances" in texts[2]
    assert all(text.strip() for text in texts)


def test_control_variant_uses_one_indexed_replicate_and_wraps():
    texts = sc.get_control_stimuli()
    assert sc.get_condition_stimulus("control", control_variant=1) == texts[0]
    assert sc.get_condition_stimulus("control", control_variant=2) == texts[1]
    assert sc.get_condition_stimulus("control", control_variant=3) == texts[2]
    assert sc.get_condition_stimulus("control", control_variant=4) == texts[0]


def test_control_stimulus_is_not_empty_by_default():
    assert sc.get_condition_stimulus("control").strip()


def test_generate_responses_carries_demographics_and_uses_control_replicates(monkeypatch):
    seen_user_prompts = []

    def fake_simulate_response(profile, stimulus, items, client, **kwargs):
        seen_user_prompts.append(stimulus)
        return type("FakeResponse", (), {"items": lambda self: [("trust_post", 50)]})()

    monkeypatch.setattr("generate_responses.simulate_response", fake_simulate_response)
    rows = pd.DataFrame(
        [
            {
                "profile_id": "LP0001__control__R1",
                "condition": "control",
                "condition_replicate": 1,
                "gender": "Male",
                "age_band": "18-29",
                "race": "White / Caucasian",
                "education": "Bachelor's degree",
                "income": "$56,000 to $99,999",
                "party": "Independent",
                "state_abbr": "CA",
            },
            {
                "profile_id": "LP0001__control__R2",
                "condition": "control",
                "condition_replicate": 2,
                "gender": "Male",
                "age_band": "18-29",
                "race": "White / Caucasian",
                "education": "Bachelor's degree",
                "income": "$56,000 to $99,999",
                "party": "Independent",
                "state_abbr": "CA",
            },
        ]
    )

    out = simulate_profile("LP0001", rows, [{"target_label": "trust_post"}], client=None)

    assert out[0]["gender"] == "Male"
    assert out[0]["state_abbr"] == "CA"
    assert "The History of Neckties" in seen_user_prompts[0]
    assert "The Rules of Baseball" in seen_user_prompts[1]
