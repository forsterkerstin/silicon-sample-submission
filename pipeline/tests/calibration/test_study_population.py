from __future__ import annotations

import inspect

import pandas as pd
import pytest

from calibration.study_population import (
    DEFAULT_EXTERNAL_N_F,
    archive_profile_to_prompt_profile,
    effect_panel_from_analytic_sample,
    largest_remainder_allocations,
    validate_profile_fields,
)


FIELDS = ["GENDER", "race_4", "pid_3", "age_5", "EDUC4", "ideo_3"]
FORBIDDEN = {"y", "condition", "condition.name", "treatment", "outcome", "outcome.name", "condition_id"}


def _analytic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "respondent_row_id": 10,
                "GENDER": "Female",
                "race_4": "White",
                "pid_3": "Democrat",
                "age_5": "18-29",
                "EDUC4": "College",
                "ideo_3": "Liberal",
                "y": 1,
                "condition.name": "control",
            },
            {
                "respondent_row_id": 20,
                "GENDER": "Male",
                "race_4": "Black",
                "pid_3": "Republican",
                "age_5": "30-44",
                "EDUC4": "HS or less",
                "ideo_3": None,
                "y": 2,
                "condition.name": "treatment",
            },
            {
                "respondent_row_id": 30,
                "GENDER": "Female",
                "race_4": "Asian",
                "pid_3": None,
                "age_5": "45-59",
                "EDUC4": "Some college",
                "ideo_3": "Moderate",
                "y": 3,
                "condition.name": "control",
            },
        ]
    )


def _panel(effect_id: str = "study:outcome:hyp1") -> tuple[pd.DataFrame, dict[str, object]]:
    return effect_panel_from_analytic_sample(
        _analytic(),
        study_id="study",
        effect_id=effect_id,
        fields=FIELDS,
        n_f=DEFAULT_EXTERNAL_N_F,
    )


def test_effect_panel_has_exactly_500_profiles_and_pass_status():
    panel, audit = _panel()

    assert len(panel) == DEFAULT_EXTERNAL_N_F
    assert panel["f_profile_id"].nunique() == DEFAULT_EXTERNAL_N_F
    assert audit["n_f"] == DEFAULT_EXTERNAL_N_F
    assert audit["status"] == "PASS"


def test_missing_demographics_do_not_exclude_human_analytic_rows():
    panel, audit = _panel()

    assert audit["analytic_n"] == 3
    assert audit["profiles_with_any_missing_demographic"] == 2
    assert panel["missing_demographic_fields"].astype(str).str.contains("pid_3|ideo_3", regex=True).any()


def test_prompt_profile_renderer_omits_missing_fields_without_unknown_imputation():
    prompt_profile = archive_profile_to_prompt_profile(_analytic().iloc[1])

    assert prompt_profile == {
        "gender": "Male",
        "race": "Black",
        "party": "Republican",
        "age": "30-44",
        "education": "HS or less",
    }
    assert "political_ideology" not in prompt_profile
    assert "unknown" not in {str(v).lower() for v in prompt_profile.values()}


def test_forbidden_outcome_and_assignment_fields_cannot_enter_f_profile():
    panel, _ = _panel()

    assert not (FORBIDDEN & set(panel.columns))
    with pytest.raises(ValueError, match="forbidden"):
        validate_profile_fields(panel.assign(y=1))


def test_observed_joint_demographic_signatures_are_preserved_with_missingness_pattern():
    panel, _ = _panel()
    signatures = set(
        panel[
            [
                "GENDER",
                "race_4",
                "pid_3",
                "age_5",
                "EDUC4",
                "ideo_3",
                "missing_demographic_fields",
            ]
        ]
        .fillna("__PANEL_NA__")
        .itertuples(index=False, name=None)
    )

    expected = {
        ("Female", "White", "Democrat", "18-29", "College", "Liberal", ""),
        ("Male", "Black", "Republican", "30-44", "HS or less", "__PANEL_NA__", "ideo_3"),
        ("Female", "Asian", "__PANEL_NA__", "45-59", "Some college", "Moderate", "pid_3"),
    }
    assert signatures == expected


def test_missing_study_demographics_are_not_fabricated_when_column_absent():
    analytic = _analytic().drop(columns=["pid_3", "ideo_3"])
    panel, audit = effect_panel_from_analytic_sample(
        analytic,
        study_id="study",
        effect_id="study:outcome:hyp1",
        fields=FIELDS,
    )

    assert "pid_3" not in panel.columns
    assert "ideo_3" not in panel.columns
    assert audit["profile_fields_available"] == "GENDER|race_4|age_5|EDUC4"


def test_deterministic_rerun_gives_identical_effect_panel():
    first, first_audit = _panel()
    second, second_audit = _panel()

    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit


def test_single_effect_panel_is_byte_identical_for_paired_arms():
    panel, audit = _panel()
    control_bytes = panel.to_csv(index=False)
    treatment_bytes = panel.to_csv(index=False)

    assert control_bytes == treatment_bytes
    assert audit["same_panel_control_treatment"] is True


def test_effect_specific_panels_can_differ_when_analytic_samples_differ():
    base, _ = _panel("study:outcome:hyp1")
    changed = pd.concat(
        [
            _analytic(),
            pd.DataFrame(
                [
                    {
                        "GENDER": "Female",
                        "race_4": "White",
                        "pid_3": "Democrat",
                        "age_5": "18-29",
                        "EDUC4": "College",
                        "ideo_3": "Liberal",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    other, _ = effect_panel_from_analytic_sample(
        changed,
        study_id="study",
        effect_id="study:other:hyp1",
        fields=FIELDS,
    )

    assert base[["profile_signature_id", "signature_allocated_n"]].to_csv(index=False) != other[
        ["profile_signature_id", "signature_allocated_n"]
    ].to_csv(index=False)


def test_largest_remainder_allocation_respects_one_slot_share_bound():
    counts = pd.Series({"a": 1, "b": 2, "c": 7})
    allocations = largest_remainder_allocations(counts, n_f=DEFAULT_EXTERNAL_N_F, tie_key="unit")
    empirical = counts / counts.sum()
    panel_share = allocations / DEFAULT_EXTERNAL_N_F

    assert int(allocations.sum()) == DEFAULT_EXTERNAL_N_F
    assert ((panel_share - empirical).abs() <= (1 / DEFAULT_EXTERNAL_N_F)).all()


def test_effect_panel_pathway_does_not_use_multinomial_replacement_sampling():
    source = inspect.getsource(effect_panel_from_analytic_sample) + inspect.getsource(largest_remainder_allocations)

    assert "rng.choice" not in source
    assert "replace=True" not in source


def test_generated_primary_effect_panels_cover_all_136_effects():
    panel = pd.read_csv("data/generated/external_primary_f_panels.csv", low_memory=False)
    audit = pd.read_csv("outputs/calibration/external_f_effect_panels_audit.csv")
    csv_text = panel.to_csv(index=False)

    assert audit["effect_id"].nunique() == 136
    assert len(panel) == 136 * DEFAULT_EXTERNAL_N_F
    assert panel.groupby("effect_id").size().eq(DEFAULT_EXTERNAL_N_F).all()
    assert audit["status"].eq("PASS").all()
    assert not (FORBIDDEN & set(panel.columns))
    assert "condition_id" not in csv_text
