"""fit_ate_calibration() / fit_hierarchical_shrinkage() /
select_shrinkage_specification(): correctness on synthetic archives with a
known true relationship (ported from the pre-refactor test_shrinkage.py --
same math, renamed lambda -> lambda_ate for clarity per the native-response
architecture, where lambda_ate calibrates ATE magnitudes, never a
probability vector)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from ate.calibrate_lambda import (
    fit_calibration_model_comparison,
    fit_ate_calibration,
    fit_hierarchical_shrinkage,
    load_treatment_families,
    load_ate_archive,
    select_shrinkage_specification,
    summarize_calibration_rows,
    validate_archive_eligibility,
)


def make_synthetic_archive(true_lambda, n_studies=12, effects_per_study=4, noise_sd=0.5, seed=0):
    rng = np.random.default_rng(seed)
    model_ate, human_ate, study_id = [], [], []
    for s in range(n_studies):
        x = rng.uniform(2, 20, size=effects_per_study)  # model-predicted effects, pp scale
        y = true_lambda * x + rng.normal(0, noise_sd, size=effects_per_study)  # human effects, shrunk + noise
        model_ate.extend(x.tolist())
        human_ate.extend(y.tolist())
        study_id.extend([f"study_{s}"] * effects_per_study)
    return model_ate, human_ate, study_id


def test_global_shrinkage_recovers_known_lambda_ate():
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=0.4, noise_sd=0.3)
    fit = fit_ate_calibration(model_ate, human_ate, study_id)
    assert abs(fit["lambda_ate"] - 0.4) < 0.1
    assert fit["rmse_loso_shrunk"] < fit["rmse_loso_raw"]


def test_global_shrinkage_prefers_lambda_ate_1_when_no_shrinkage_is_true():
    # true_lambda=1 (no exaggeration): shrinkage should not improve on raw.
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=1.0, noise_sd=0.3)
    fit = fit_ate_calibration(model_ate, human_ate, study_id)
    assert fit["lambda_ate"] == pytest.approx(1.0, abs=0.15)


def test_global_calibration_does_not_clip_lambda_by_default():
    # A pathological anti-correlated archive is not silently forced into [0, 1].
    model_ate = [1.0, 2.0, 3.0, 4.0]
    human_ate = [-1.0, -2.0, -3.0, -4.0]
    study_id = ["s1", "s1", "s2", "s2"]
    fit = fit_ate_calibration(model_ate, human_ate, study_id)
    assert fit["lambda_ate"] == pytest.approx(-1.0)


def test_global_shrinkage_requires_at_least_2_studies():
    with pytest.raises(ValueError):
        fit_ate_calibration([1.0, 2.0], [1.0, 2.0], ["only_one_study", "only_one_study"])


def test_calibration_model_comparison_writes_m0_m1_m2_outputs(tmp_path):
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=0.5, n_studies=5, effects_per_study=3, noise_sd=0.0)
    fit = fit_calibration_model_comparison(model_ate, human_ate, study_id, outputs_dir=tmp_path)

    assert fit["model_name"] == "M1"
    assert fit["calibration_alpha"] == pytest.approx(0.0)
    assert fit["calibration_lambda"] == pytest.approx(0.5)
    assert {row["model_name"] for row in fit["comparison_rows"]} == {"M0", "M1", "M2"}
    assert (tmp_path / "calibration_model_comparison.csv").exists()
    assert (tmp_path / "calibration_loso_predictions.csv").exists()
    assert (tmp_path / "calibration_selected_model.json").exists()


def test_loso_predictions_hold_out_whole_studies(tmp_path):
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=0.7, n_studies=4, effects_per_study=2, noise_sd=0.0)
    fit = fit_calibration_model_comparison(model_ate, human_ate, study_id, outputs_dir=tmp_path)
    for pred in fit["loso_predictions"]:
        assert pred["held_out_study_id"] == study_id[pred["effect_index"]]


def test_hierarchical_shrinkage_with_high_ridge_collapses_toward_global():
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=0.5, noise_sd=0.3, n_studies=16)
    treatment_family = ["fam_a" if i % 2 == 0 else "fam_b" for i in range(len(model_ate))]
    outcome_family = ["attitude" if i % 3 == 0 else "donation" for i in range(len(model_ate))]

    global_fit = fit_ate_calibration(model_ate, human_ate, study_id)
    hier_fit = fit_hierarchical_shrinkage(model_ate, human_ate, study_id, treatment_family, outcome_family, ridge_penalty=1e6)
    family_lambdas = list(hier_fit["lambda_ate_by_family"].values())
    # with an enormous ridge penalty, every family's lambda_ate must collapse near the single global value.
    assert max(family_lambdas) - min(family_lambdas) < 0.05
    assert abs(np.mean(family_lambdas) - global_fit["lambda_ate_unconstrained"]) < 0.1


def test_hierarchical_shrinkage_recovers_a_real_family_difference_with_enough_data():
    rng = np.random.default_rng(1)
    model_ate, human_ate, study_id, treatment_family, outcome_family = [], [], [], [], []
    for s in range(30):
        fam = "fam_hi" if s % 2 == 0 else "fam_lo"
        true_lam = 0.8 if fam == "fam_hi" else 0.2
        x = rng.uniform(5, 15, size=6)
        y = true_lam * x + rng.normal(0, 0.2, size=6)
        model_ate.extend(x.tolist())
        human_ate.extend(y.tolist())
        study_id.extend([f"study_{s}"] * 6)
        treatment_family.extend([fam] * 6)
        outcome_family.extend(["attitude"] * 6)

    hier_fit = fit_hierarchical_shrinkage(model_ate, human_ate, study_id, treatment_family, outcome_family, ridge_penalty=0.5)
    lam_hi = hier_fit["lambda_ate_by_family"]["fam_hi x attitude"]
    lam_lo = hier_fit["lambda_ate_by_family"]["fam_lo x attitude"]
    assert lam_hi > lam_lo  # the real family difference (0.8 vs 0.2) must survive a mild ridge penalty


def test_select_shrinkage_specification_picks_a_real_family_difference():
    rng = np.random.default_rng(2)
    model_ate, human_ate, study_id, treatment_family, outcome_family = [], [], [], [], []
    for s in range(30):
        fam = "fam_hi" if s % 2 == 0 else "fam_lo"
        true_lam = 0.9 if fam == "fam_hi" else 0.1
        x = rng.uniform(5, 15, size=6)
        y = true_lam * x + rng.normal(0, 0.15, size=6)
        model_ate.extend(x.tolist())
        human_ate.extend(y.tolist())
        study_id.extend([f"study_{s}"] * 6)
        treatment_family.extend([fam] * 6)
        outcome_family.extend(["attitude"] * 6)

    selection = select_shrinkage_specification(model_ate, human_ate, study_id, treatment_family, outcome_family)
    assert selection["winner"] in ("none", "global", "hierarchical")
    assert set(selection["rmse_loso"].keys()) == {"none", "global", "hierarchical"}
    # a strong, real per-family difference should make the hierarchical spec win.
    assert selection["winner"] == "hierarchical"


def test_load_treatment_families_covers_all_16_interventions():
    families = load_treatment_families()
    assert len(families) == 16
    assert "control" not in {k.lower() for k in families}
    assert families["Interview Prof. Maraun"] == "Collaboration and peer-review"
    assert families["Social justice"] == "Other"


def _archive_row(**overrides):
    row = {
        "study_id": "s1",
        "effect_id": "s1:e1",
        "outcome": "y",
        "model_ate": "2.0",
        "human_ate": "1.0",
        "treatment_family": "",
        "outcome_family": "attitude",
        "target_population": "general U.S. adults",
        "synthetic_target_population": "general U.S. adults",
        "population_type": "general_us_adult",
        "is_general_us_adult": "True",
        "is_specialized_population": "False",
        "study_weights_available": "False",
        "profile_variables_available": "False",
        "population_matching_method": "representative_us_fallback",
        "weights_used": "False",
        "num_profiles": "100",
        "treatment_type": "survey_experiment",
        "randomized_between_subjects": "True",
        "materials_available": "True",
        "outcome_type": "attitude",
        "outcome_min": "0",
        "outcome_max": "100",
        "outcome_range": "100",
        "finite_range": "True",
        "main_effect_compatible": "True",
        "human_ate_native": "1.0",
        "synthetic_ate_native": "2.0",
        "included_primary_calibration": "True",
        "included_secondary_sensitivity": "False",
        "exclusion_reason": "",
    }
    row.update(overrides)
    return row


def _write_archive(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_general_us_study_is_primary_eligible(tmp_path):
    path = tmp_path / "archive.csv"
    _write_archive(path, [_archive_row(study_id="s1", effect_id="s1:e1"), _archive_row(study_id="s2", effect_id="s2:e1")])

    rows = load_ate_archive(path)

    assert len(rows) == 2
    assert summarize_calibration_rows(rows)["population_types"] == ["general_us_adult"]


def test_specialized_population_without_matching_is_excluded_from_primary(tmp_path):
    path = tmp_path / "archive.csv"
    _write_archive(
        path,
        [
            _archive_row(study_id="s1", effect_id="s1:e1"),
            _archive_row(
                study_id="students",
                effect_id="students:e1",
                target_population="students only",
                synthetic_target_population="general U.S. adults",
                population_type="secondary_population_sensitivity",
                is_general_us_adult="False",
                is_specialized_population="True",
                population_matching_method="not_matched",
                included_primary_calibration="False",
                included_secondary_sensitivity="True",
                exclusion_reason="target population not transport-compatible",
            ),
        ],
    )

    primary = load_ate_archive(path)
    all_rows = load_ate_archive(path, primary_only=False)

    assert [r["study_id"] for r in primary] == ["s1"]
    assert {r["study_id"] for r in all_rows} == {"s1", "students"}


def test_specialized_population_primary_requires_explicit_matching():
    row = _archive_row(
        target_population="clinicians only",
        synthetic_target_population="clinicians only",
        population_type="specialized",
        is_general_us_adult="False",
        is_specialized_population="True",
        population_matching_method="not_matched",
    )

    with pytest.raises(ValueError, match="specialized population"):
        validate_archive_eligibility([row])


def test_study_weights_available_must_be_used_for_primary_rows():
    row = _archive_row(study_weights_available="True", weights_used="False")

    with pytest.raises(ValueError, match="weights"):
        validate_archive_eligibility([row])


def test_target_population_mismatch_fails_for_primary_rows():
    row = _archive_row(synthetic_target_population="students only")

    with pytest.raises(ValueError, match="different populations"):
        validate_archive_eligibility([row])


def test_outcome_range_required_for_normalized_primary_effect():
    row = _archive_row(outcome_range="", finite_range="False")

    with pytest.raises(ValueError, match="outcome_range"):
        validate_archive_eligibility([row])


def test_loso_keeps_all_effects_from_held_out_study_together():
    model_ate, human_ate, study_id = make_synthetic_archive(true_lambda=0.4, n_studies=4, effects_per_study=3, noise_sd=0.0)
    fit = fit_ate_calibration(model_ate, human_ate, study_id)

    assert len(fit["fold_diagnostics"]) == 4
    assert {fold["n_held_out_effects"] for fold in fit["fold_diagnostics"]} == {3}
