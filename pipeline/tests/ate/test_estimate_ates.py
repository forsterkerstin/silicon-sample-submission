"""estimate_raw_ates(): raw (uncalibrated) treatment effects computed
directly from native-scale responses."""

from __future__ import annotations

import pandas as pd
import pytest

from ate.estimate_ates import ProfilePopulation, StudyDefinition, estimate_raw_ate, estimate_raw_ates
from ate.normalize_effects import to_percent_of_range


def _responses():
    return pd.DataFrame(
        [
            {"condition": "control", "trust_post": 40, "donation_ams": 2},
            {"condition": "control", "trust_post": 50, "donation_ams": 4},
            {"condition": "Consensus", "trust_post": 60, "donation_ams": 5},
            {"condition": "Consensus", "trust_post": 70, "donation_ams": 7},
            {"condition": "Funding", "trust_post": 30, "donation_ams": 1},
        ]
    )


def test_raw_ate_is_treatment_mean_minus_control_mean():
    out = estimate_raw_ates(_responses(), ["trust_post"])
    row = out[out["condition"] == "Consensus"].iloc[0]
    assert row["control_mean"] == pytest.approx(45.0)
    assert row["treatment_mean"] == pytest.approx(65.0)
    assert row["raw_ate"] == pytest.approx(20.0)


def test_control_excluded_from_treatment_rows():
    out = estimate_raw_ates(_responses(), ["trust_post"])
    assert "control" not in set(out["condition"])


def test_multiple_outcomes_produce_separate_rows():
    out = estimate_raw_ates(_responses(), ["trust_post", "donation_ams"])
    assert set(out["outcome"]) == {"trust_post", "donation_ams"}
    assert len(out) == 2 * out["condition"].nunique()


def test_sample_sizes_reported():
    out = estimate_raw_ates(_responses(), ["trust_post"])
    row = out[out["condition"] == "Funding"].iloc[0]
    assert row["n_control"] == 2
    assert row["n_treatment"] == 1


def test_no_control_rows_raises():
    df = pd.DataFrame([{"condition": "Consensus", "trust_post": 60}])
    with pytest.raises(ValueError):
        estimate_raw_ates(df, ["trust_post"])


def test_nan_values_dropped_not_propagated():
    df = pd.DataFrame(
        [
            {"condition": "control", "trust_post": 40},
            {"condition": "control", "trust_post": None},
            {"condition": "Consensus", "trust_post": 60},
        ]
    )
    out = estimate_raw_ates(df, ["trust_post"])
    row = out.iloc[0]
    assert row["control_mean"] == pytest.approx(40.0)
    assert row["n_control"] == 1


def test_population_explicit_weighted_ate_uses_declared_weights():
    responses = pd.DataFrame(
        [
            {"condition": "control", "y": 0, "w": 9},
            {"condition": "control", "y": 10, "w": 1},
            {"condition": "treat", "y": 10, "w": 9},
            {"condition": "treat", "y": 0, "w": 1},
        ]
    )
    study = StudyDefinition("s1", "treat", "control", "y", "general U.S. adults", weight_column="w")
    population = ProfilePopulation(responses, "general U.S. adults", "study_respondent_weighted", weight_column="w")

    out = estimate_raw_ate(study, population, model_config={"model": "fixed"}, repetitions=3)

    assert out["weights_used"] is True
    assert out["raw_ate"] == pytest.approx(8.0)
    assert out["raw_ate"] != pytest.approx(0.0)  # the unweighted arm difference in this constructed case


def test_population_mismatch_fails_before_ate_calculation():
    responses = pd.DataFrame([{"condition": "control", "y": 0}, {"condition": "treat", "y": 1}])
    study = StudyDefinition("s1", "treat", "control", "y", "students only")
    population = ProfilePopulation(responses, "general U.S. adults", "representative_us_fallback")

    with pytest.raises(ValueError, match="target population"):
        estimate_raw_ate(study, population)


def test_declared_weight_column_must_exist():
    responses = pd.DataFrame([{"condition": "control", "y": 0}, {"condition": "treat", "y": 1}])
    study = StudyDefinition("s1", "treat", "control", "y", "general U.S. adults", weight_column="w")
    population = ProfilePopulation(responses, "general U.S. adults", "representative_us_fallback")

    with pytest.raises(ValueError, match="weight column"):
        estimate_raw_ate(study, population)


def test_percent_of_range_normalization_happens_after_native_ate_estimation():
    responses = pd.DataFrame(
        [
            {"condition": "control", "donation": 1, "w": 1},
            {"condition": "control", "donation": 3, "w": 1},
            {"condition": "treat", "donation": 4, "w": 1},
            {"condition": "treat", "donation": 6, "w": 1},
        ]
    )
    study = StudyDefinition("s1", "treat", "control", "donation", "general U.S. adults", weight_column="w")
    population = ProfilePopulation(responses, "general U.S. adults", "study_respondent_weighted", weight_column="w")

    native = estimate_raw_ate(study, population)["raw_ate"]
    effect_pp = to_percent_of_range(native, 0, 10)

    assert native == pytest.approx(3.0)
    assert effect_pp == pytest.approx(30.0)


def test_changing_population_changes_ate_without_changing_protocol():
    responses = pd.DataFrame(
        [
            {"condition": "control", "y": 0, "w_a": 9, "w_b": 1},
            {"condition": "control", "y": 10, "w_a": 1, "w_b": 9},
            {"condition": "treat", "y": 10, "w_a": 9, "w_b": 1},
            {"condition": "treat", "y": 0, "w_a": 1, "w_b": 9},
        ]
    )
    protocol = {"model": "same-model", "temperature": 1.0, "prompt_template": "same"}
    study_a = StudyDefinition("s1", "treat", "control", "y", "population A", weight_column="w_a")
    study_b = StudyDefinition("s1", "treat", "control", "y", "population B", weight_column="w_b")
    pop_a = ProfilePopulation(responses, "population A", "study_respondent_weighted", weight_column="w_a")
    pop_b = ProfilePopulation(responses, "population B", "study_respondent_weighted", weight_column="w_b")

    ate_a = estimate_raw_ate(study_a, pop_a, model_config=protocol, repetitions=5)
    ate_b = estimate_raw_ate(study_b, pop_b, model_config=protocol, repetitions=5)

    assert ate_a["model_config"] == ate_b["model_config"] == protocol
    assert ate_a["repetitions"] == ate_b["repetitions"] == 5
    assert ate_a["raw_ate"] != ate_b["raw_ate"]
