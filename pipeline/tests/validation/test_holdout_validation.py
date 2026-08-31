"""Validation split and structural-holdout integrity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from validation.holdout import (  # noqa: E402
    HoldoutIntegrityError,
    apply_frozen_calibration,
    assert_no_holdout_studies,
    build_data_usage_audit,
    build_megastudy_holdout_eligibility,
    classify_megastudy_exclusions,
    climate_holdout_overlap_audit,
    current_secondary_contamination,
    equal_study_summary,
    evaluate_structural_holdout,
    mark_holdout_consumed,
    percent_of_range,
    pooled_metrics,
    raw_vs_calibrated_comparison,
    study_metrics,
)


@pytest.fixture
def repo_root() -> Path:
    return PIPELINE_ROOT


@pytest.fixture
def split_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"study_id": "dev1", "assigned_role": "development_calibration"},
            {"study_id": "hold1", "assigned_role": "structural_holdout"},
            {"study_id": "comp1", "assigned_role": "compromised_holdout"},
        ]
    )


@pytest.fixture
def frozen_manifest() -> dict:
    payload = {"method_hash": "test", "selected_calibration_model": "M2", "final_alpha": 1.5, "final_lambda": 0.25, "N_G": 1000, "N_F": 500}
    return payload


@pytest.fixture
def holdout_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["hold1", "hold1", "hold2", "hold2", "hold2", "hold3"],
            "effect_id": ["a", "b", "c", "d", "e", "f"],
            "intervention": ["t"] * 6,
            "control": ["c"] * 6,
            "outcome": ["o"] * 6,
            "outcome_type": ["attitude"] * 6,
            "outcome_range": [10, 10, 10, 10, 10, 10],
            "human_ate_native": [1, -2, 3, -4, 5, 1],
            "human_se_native": [0.1, 0.2, np.nan, np.nan, 0.3, np.nan],
            "raw_f_ate_native": [2, -1, 2, -8, 4, -1],
        }
    )


def test_no_structural_holdout_study_enters_c_fitting(split_manifest):
    rows = pd.DataFrame({"study_id": ["dev1", "hold1"]})
    with pytest.raises(HoldoutIntegrityError, match="C fitting"):
        assert_no_holdout_studies(rows, split_manifest, context="C fitting")


def test_no_structural_holdout_study_enters_model_selection(split_manifest):
    rows = pd.DataFrame({"study_id": ["dev1", "comp1"]})
    with pytest.raises(HoldoutIntegrityError, match="M0/M1/M2 model selection"):
        assert_no_holdout_studies(rows, split_manifest, context="M0/M1/M2 model selection")


def test_no_structural_holdout_study_enters_f_model_selection(split_manifest):
    rows = pd.DataFrame({"study_id": ["hold1"]})
    with pytest.raises(HoldoutIntegrityError, match="F model selection"):
        assert_no_holdout_studies(rows, split_manifest, context="F model selection")


def test_no_structural_holdout_study_enters_f_convergence_selection(split_manifest):
    rows = pd.DataFrame({"study_id": ["dev1", "hold1"]})
    with pytest.raises(HoldoutIntegrityError, match="F N/R convergence selection"):
        assert_no_holdout_studies(rows, split_manifest, context="F N/R convergence selection")


def test_holdout_evaluation_refuses_without_frozen_manifest(tmp_path, holdout_predictions):
    predictions_path = tmp_path / "holdout.csv"
    holdout_predictions.to_csv(predictions_path, index=False)
    with pytest.raises(HoldoutIntegrityError, match="requires frozen method manifest"):
        evaluate_structural_holdout(predictions_path, manifest_path=tmp_path / "missing.json", outputs_dir=tmp_path)


def test_holdout_uses_frozen_alpha_lambda_exactly(frozen_manifest, holdout_predictions):
    out = apply_frozen_calibration(holdout_predictions, frozen_manifest)
    expected_raw_pp = 100 * holdout_predictions["raw_f_ate_native"] / holdout_predictions["outcome_range"]
    expected_cal = 1.5 + 0.25 * expected_raw_pp
    assert np.allclose(out["calibrated_f_ate_pp"], expected_cal)


def test_holdout_evaluation_cannot_update_calibration_parameters(tmp_path, holdout_predictions):
    calibration_file = tmp_path / "calibration_selected_model.json"
    calibration_file.write_text(json.dumps({"alpha": 99, "lambda": 99}), encoding="utf-8")
    manifest = {"selected_calibration_model": "M1", "final_alpha": 0.0, "final_lambda": 0.5, "N_G": 1000, "N_F": 500}
    from validation.holdout import canonical_hash

    manifest["method_hash"] = canonical_hash(manifest)
    manifest_path = tmp_path / "frozen_method_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions_path = tmp_path / "holdout.csv"
    holdout_predictions.to_csv(predictions_path, index=False)
    evaluate_structural_holdout(predictions_path, manifest_path=manifest_path, outputs_dir=tmp_path)
    assert json.loads(calibration_file.read_text(encoding="utf-8")) == {"alpha": 99, "lambda": 99}


def test_holdout_eligibility_does_not_reference_effect_magnitude_or_prediction_error():
    metadata = pd.DataFrame({"study_id": ["A"], "effect_id": ["A:e1"], "outcome": ["belief"], "condition_name": ["Treatment"], "number_effects_in_archive_row": [2]})
    eligibility = build_megastudy_holdout_eligibility(metadata)
    forbidden = {"human_ate_native", "human_ate_pp", "raw_f_ate_pp", "prediction_error", "sign_agreement"}
    assert forbidden.isdisjoint(set(eligibility.columns))


def test_megastudy_exclusion_classifier_distinguishes_metadata_gap_from_parser_failure():
    metadata = pd.DataFrame(
        {
            "study_id": ["A", "B"],
            "effect_id": ["A:e1", "B:e1"],
            "outcome": ["belief", ""],
            "condition_name": ["Treatment", ""],
            "number_effects_in_archive_row": [2, 2],
        }
    )
    classified = classify_megastudy_exclusions(metadata)
    assert set(classified["eligible"]) == {False}
    assert classified.loc[classified["study_id"] == "A", "primary_exclusion_reason"].item() == "materials_missing"
    assert classified.loc[classified["study_id"] == "B", "primary_exclusion_reason"].item() == "parser_mapping_failure"
    assert set(classified["problem_type"]) == {"data_or_metadata_gap"}


def test_megastudy_exclusion_audit_script_writes_requested_outputs(repo_root):
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "audit_megastudy_exclusions.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    reasons = repo_root / "outputs" / "validation" / "megastudy_exclusion_reasons.csv"
    summary = repo_root / "outputs" / "validation" / "megastudy_exclusion_summary.csv"
    assert reasons.exists()
    assert summary.exists()
    reason_df = pd.read_csv(reasons)
    summary_df = pd.read_csv(summary)
    assert len(reason_df) == 606
    assert reason_df["eligible"].sum() == 0
    assert {"materials_missing", "compromised_development_data"} & set(reason_df["primary_exclusion_reason"])
    assert {"reason_total", "study_reason"} <= set(summary_df["summary_level"])


def test_percentage_of_range_normalization_is_correct():
    assert np.allclose(percent_of_range([2, -1], [4, 2]), [50, -50])


def test_raw_and_calibrated_predictions_are_both_generated(frozen_manifest, holdout_predictions):
    out = apply_frozen_calibration(holdout_predictions, frozen_manifest)
    assert {"raw_f_ate_pp", "calibrated_f_ate_pp", "raw_error_pp", "calibrated_error_pp"} <= set(out.columns)


def test_per_study_and_pooled_metrics_are_produced(frozen_manifest, holdout_predictions):
    out = apply_frozen_calibration(holdout_predictions, frozen_manifest)
    pooled = pooled_metrics(out)
    by_study = study_metrics(out)
    assert set(pooled["prediction"]) == {"raw_F", "calibrated_FC"}
    assert {"hold1", "hold2", "hold3"} <= set(by_study["study_id"])


def test_equal_study_rmse_is_implemented_correctly():
    by_study = pd.DataFrame(
        {
            "study_id": ["a", "b"],
            "prediction": ["raw_F", "raw_F"],
            "rmse": [3.0, 4.0],
            "mae": [2.0, 6.0],
            "sign_accuracy": [0.5, 1.0],
            "pearson": [0.1, 0.3],
            "spearman": [0.2, 0.4],
        }
    )
    summary = equal_study_summary(by_study)
    assert summary.loc[0, "study_equal_rmse"] == pytest.approx(np.sqrt((9 + 16) / 2))
    assert summary.loc[0, "study_equal_mae"] == pytest.approx(4.0)


def test_adjusted_metrics_use_human_se_only_where_available(frozen_manifest, holdout_predictions):
    out = apply_frozen_calibration(holdout_predictions, frozen_manifest)
    with_se = pooled_metrics(out)
    assert "approximation_uses_human_se" in set(with_se["adjusted_metrics_status"])
    out = out.drop(columns=["human_se_pp"])
    without_se = pooled_metrics(out)
    assert set(without_se["adjusted_metrics_status"]) == {"human_se_unavailable"}


def test_climate_overlap_check_detects_duplicate_doi_title_ids():
    metadata = pd.DataFrame({"study_id": ["Voelkel2025", "Other"], "effect_id": ["v1", "o1"]})
    audit = climate_holdout_overlap_audit(metadata)
    assert audit["secondary_archive_match"] is True
    assert audit["matched_study_id"] == "Voelkel2025"
    assert audit["status"] == "contained in structural holdout"


def test_method_changes_after_holdout_opening_mark_consumed(tmp_path):
    status = {
        "frozen_method_hash": "abc",
        "holdout_opened_at": "2026-08-24T00:00:00+00:00",
        "method_changed_after_holdout": False,
        "holdout_still_pristine": True,
        "holdout_consumed": False,
        "notes": "",
    }
    path = tmp_path / "holdout_status.json"
    path.write_text(json.dumps(status), encoding="utf-8")
    updated = mark_holdout_consumed("changed model", path=path)
    assert updated["holdout_consumed"] is True
    assert updated["holdout_still_pristine"] is False


def test_g_validation_and_fc_validation_are_represented_separately(repo_root):
    subprocess.run([sys.executable, str(repo_root / "scripts" / "audit_validation_split.py")], cwd=repo_root, check=True, capture_output=True, text=True)
    assert (repo_root / "outputs" / "validation" / "g_validation" / "g_validation_status.csv").exists()
    assert (repo_root / "outputs" / "validation" / "validation_split_manifest.csv").exists()


def test_doell_vlasceanu_marked_contaminated_given_real_repo_state():
    """data/data63.xlsx and data/validation_vlasceanu_us.csv are present in
    this repo (scripts/build_vlasceanu_validation.py already opened and
    computed real human effect sizes from them) -- Doell (the same
    underlying Vlasceanu et al. 2024 study, matched by condition name in
    megastudies.RDS) must therefore be reported as previously-opened, not a
    pristine holdout candidate."""
    contamination = current_secondary_contamination()
    assert "Doell" in contamination
    assert contamination["Doell"]["human_outcomes_previously_opened"] is True
    assert contamination["Doell"]["used_for_calibration"] is False


def test_voelkel2025_still_marked_contaminated():
    contamination = current_secondary_contamination()
    assert "Voelkel2025" in contamination
    assert contamination["Voelkel2025"]["human_outcomes_previously_opened"] is True


def test_data_usage_audit_reports_doell_as_not_pristine_eligible():
    megastudy_summary = pd.DataFrame({"study_id": ["Doell", "Voelkel2025", "SomeUncompromisedStudy"], "datasets": ["Doell", "Voelkel2025", "SomeUncompromisedStudy"]})
    audit = build_data_usage_audit(pd.DataFrame({"study_id": [], "included_primary_calibration": []}), pd.DataFrame(columns=["study", "study_title"]), megastudy_summary)
    doell_row = audit[audit["study_id"] == "Doell"].iloc[0]
    assert doell_row["human_outcomes_previously_opened"] == True  # noqa: E712
    assert doell_row["eligible_as_pristine_holdout"] == False  # noqa: E712
    uncompromised_row = audit[audit["study_id"] == "SomeUncompromisedStudy"].iloc[0]
    assert uncompromised_row["eligible_as_pristine_holdout"] == True  # noqa: E712 -- a genuinely untouched study is not swept in
