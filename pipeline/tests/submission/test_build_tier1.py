from __future__ import annotations

from numbers import Integral
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import survey_content as sc
from submission import build_tier1 as bt
from submission.final_tier1 import build_final_tier1
from submission.validate_tier1 import validate_tier1


def _synthetic_raw_responses() -> pd.DataFrame:
    schema = yaml.safe_load((sc.REPO_ROOT / "pipeline" / "config" / "benchmark_schema.yaml").read_text())
    rows = []
    for i, condition in enumerate(schema["conditions"]):
        row = {
            "latent_profile_id": f"LP{i:04d}",
            "profile_id": f"LP{i:04d}__{condition.replace(' ', '_')}__R1",
            "condition": condition,
            "gender": "Male",
            "age_band": "18-29",
            "race": "White / Caucasian",
            "education": "Bachelor's degree",
            "income": "$56,000 to $99,999",
            "party": "Independent",
            "state_abbr": "CA",
        }
        for item in sc.load_items():
            if item["scale"] == sc.SCALE_DONATION_0_10:
                value = 5
            elif item["scale"] == sc.SCALE_BINARY_0_1:
                value = 0
            else:
                value = 55 if condition != "control" and item["target_label"] == "trust_post" else 50
            row[item["target_label"]] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_tier1_validates_full_grid_and_preserves_native_support(monkeypatch):
    monkeypatch.setattr(bt, "resolve_lambda_ate", lambda: (1.0, {"specification": "test"}))

    calibrated, report = bt.build_tier1(_synthetic_raw_responses())

    assert report["validation_report"]["n_outcome_ates"] == 16 * 13
    assert len(report["calibration_diagnostics"]) == 16 * len(sc.load_items())
    assert report["calibration_diagnostics"]["absolute_error"].max() == pytest.approx(0.0)
    assert calibrated["donation_ams"].map(lambda x: isinstance(x, Integral)).all()
    assert calibrated["newsletter_signup"].isin([0, 1]).all()
    assert {"gender", "age_band", "race", "education", "income", "party"}.issubset(calibrated.columns)


def test_validate_tier1_rejects_composite_mismatch(monkeypatch):
    monkeypatch.setattr(bt, "resolve_lambda_ate", lambda: (1.0, {"specification": "test"}))
    calibrated, report = bt.build_tier1(_synthetic_raw_responses())
    broken = calibrated.copy()
    broken.loc[0, "trust_multidimensional"] += 10

    with pytest.raises(ValueError, match="trust_multidimensional"):
        validate_tier1(broken, calibration_diagnostics=report["calibration_diagnostics"])


def _row(profile: int, condition: str, *, gender: str = "Male") -> dict:
    row = {
        "latent_profile_id": f"F{profile:04d}",
        "profile_id": f"F{profile:04d}__{condition.replace(' ', '_')}",
        "condition": condition,
        "gender": gender,
        "age_band": "18-29" if profile % 2 == 0 else "60+",
        "race": "White / Caucasian" if profile % 3 else "Black / African American",
        "education": "Bachelor's degree",
        "income": "$56,000 to $99,999",
        "party": "Independent" if profile % 2 == 0 else "Democrat",
        "state_abbr": "CA",
    }
    for item in sc.load_items():
        label = item["target_label"]
        if item["scale"] == sc.SCALE_DONATION_0_10:
            base = 5
            value = base if condition == "control" else 6
        elif item["scale"] == sc.SCALE_BINARY_0_1:
            value = 0 if condition == "control" else int(profile % 2 == 0)
        else:
            base = 50
            effect = 0 if condition == "control" else 2 + (6 if label == "trust_post" and gender == "Female" else 0)
            if label == "funding_5_raw":
                effect *= -1
            value = base + effect
        row[label] = value
    return row


def _synthetic_g_control(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        gender = "Female" if i % 2 else "Male"
        r = _row(i, "control", gender=gender)
        r["latent_profile_id"] = f"G{i:04d}"
        r["profile_id"] = f"G{i:04d}__control"
        rows.append(r)
    return pd.DataFrame(rows)


def _synthetic_g_native(n: int = 6) -> pd.DataFrame:
    schema = yaml.safe_load((sc.REPO_ROOT / "pipeline" / "config" / "benchmark_schema.yaml").read_text())
    rows = []
    for i in range(n):
        gender = "Female" if i % 2 else "Male"
        for condition in schema["conditions"]:
            r = _row(i, condition, gender=gender)
            r["latent_profile_id"] = f"G{i:04d}"
            r["profile_id"] = f"G{i:04d}__{condition.replace(' ', '_')}"
            r["state"] = r["state_abbr"]
            rows.append(r)
    return pd.DataFrame(rows)


def _synthetic_f_responses(n: int = 8) -> pd.DataFrame:
    schema = yaml.safe_load((sc.REPO_ROOT / "pipeline" / "config" / "benchmark_schema.yaml").read_text())
    rows = []
    for i in range(n):
        gender = "Female" if i % 2 else "Male"
        for condition in schema["conditions"]:
            rows.append(_row(i, condition, gender=gender))
    return pd.DataFrame(rows)


def test_primary_final_builder_uses_native_g_treatments_and_f_ate_outputs(tmp_path):
    model = {"model_name": "M2", "calibration_alpha": 1.0, "calibration_lambda": 0.5}

    final, report = build_final_tier1(
        _synthetic_g_native(),
        _synthetic_f_responses(),
        model,
        outputs_dir=tmp_path,
        expected_n_g=None,
        expected_n_f=None,
        require_frozen_f_protocol=False,
    )

    assert len(final) == 6 * 17
    assert "donor_key" not in final.columns
    assert "state" not in final.columns
    assert "state_abbr" not in final.columns
    assert final["profile_id"].is_unique
    assert len(report["raw_g_ates"]) == 16 * 13
    assert len(report["raw_target_ates"]) == 16 * 13
    assert len(report["calibrated_target_ates"]) == 16 * 13
    assert len(report["final_ate_audit"]) == 16 * 13
    assert (tmp_path / "raw_g_ates.csv").exists()
    assert (tmp_path / "raw_target_ates.csv").exists()
    assert (tmp_path / "calibrated_target_ates.csv").exists()
    assert (tmp_path / "final_ate_audit.csv").exists()
    assert (tmp_path / "f_stability_diagnostics.csv").exists()
    assert (tmp_path / "final_hte_interactions.csv").exists()
    assert (tmp_path / "demographic_predictability_full.csv").exists()
    assert final["newsletter_signup"].isin([0, 1]).all()
    assert final["donation_ams"].between(0, 10).all()


def test_common_shift_formula_and_control_rows_are_preserved(tmp_path):
    model = {"model_name": "M1", "calibration_alpha": 0.0, "calibration_lambda": 1.5}
    g_native = _synthetic_g_native()

    final, report = build_final_tier1(
        g_native,
        _synthetic_f_responses(),
        model,
        outputs_dir=tmp_path,
        expected_n_g=None,
        expected_n_f=None,
        require_frozen_f_protocol=False,
    )

    audit = report["final_ate_audit"]
    assert np.allclose(audit["common_shift"], audit["calibrated_target_ate_native"] - audit["raw_g_ate_native"])
    assert np.allclose(audit["pre_projection_ate"], audit["calibrated_target_ate_native"])

    final_control = final[final["condition"] == "control"].sort_values("profile_id").reset_index(drop=True)
    native_control = g_native[g_native["condition"] == "control"].copy()
    native_outcomes = pd.DataFrame([sc.compute_outcomes(row.to_dict()) for _, row in native_control.iterrows()])
    native_control = pd.concat([native_control.reset_index(drop=True).drop(columns=[c for c in native_outcomes.columns if c in native_control]), native_outcomes], axis=1)
    native_control = native_control[final_control.columns].sort_values("profile_id").reset_index(drop=True)
    assert final_control.equals(native_control)


def test_no_active_explicit_heterogeneity_model_terms():
    import inspect
    import submission.build_tier1 as build_module
    import submission.final_tier1 as final_module

    active_source = inspect.getsource(build_module) + inspect.getsource(final_module)
    forbidden = ["fit_" + "hte_" + "ridge", "hte_" + "mode", "lambda * " + "h", "centered " + "demographic", "r" + "ho"]
    for term in forbidden:
        assert term not in active_source


def test_primary_final_builder_refuses_unfrozen_f_protocol_by_default(tmp_path, monkeypatch):
    """outputs/f_reliability/frozen_f_protocol.json is now genuinely
    materialized in this repo (bookkeeping for the already-frozen F*/R_F/v2
    decisions) -- monkeypatch the resolved path to a location that does NOT
    exist, so this test still proves build_final_tier1 refuses by default
    when the frozen F protocol is absent, independent of the real repo's
    current (now-frozen) state."""
    import ate.f_reliability as fr

    monkeypatch.setattr(fr, "frozen_protocol_path", lambda: tmp_path / "nonexistent_frozen_f_protocol.json")
    model = {"model_name": "M0", "calibration_alpha": 0.0, "calibration_lambda": 1.0}

    with pytest.raises(RuntimeError, match="frozen F protocol"):
        build_final_tier1(_synthetic_g_native(), _synthetic_f_responses(), model, outputs_dir=tmp_path / "out", expected_n_g=None, expected_n_f=None)


def test_build_tier1_cli_primary_path_refuses_when_calibration_not_production_ready(tmp_path):
    """The real production CLI entry point (--g-native-responses/--f-responses)
    must refuse rather than silently substitute M0 when the repository's
    actual current archive/calibration state is not production-ready --
    which, before R_F/calibration are frozen, it genuinely is not. This is
    the end-to-end integration proof for BUILD_TIER1_CALIBRATION_FAIL_CLOSED:
    unlike the mocked unit tests in test_build_tier1_calibration_fail_closed.py,
    this exercises the real main() CLI process against the real repo state.
    --allow-unfrozen-f-protocol-for-fixture does NOT bypass this refusal --
    only resolve_calibration_model() (the separate, explicitly-legacy
    --raw-responses fallback) is permissive; the primary path never is."""
    g_path = tmp_path / "g_native.csv"
    f_path = tmp_path / "f_responses.csv"
    out_path = tmp_path / "tier1.csv"
    artifacts = tmp_path / "artifacts"
    _synthetic_g_native().to_csv(g_path, index=False)
    _synthetic_f_responses().to_csv(f_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(sc.REPO_ROOT / "pipeline" / "submission" / "build_tier1.py"),
            "--g-native-responses",
            str(g_path),
            "--f-responses",
            str(f_path),
            "--out",
            str(out_path),
            "--outputs-dir",
            str(artifacts),
            "--expected-n-g",
            "6",
            "--expected-n-f",
            "8",
            "--allow-unfrozen-f-protocol-for-fixture",
        ],
        cwd=sc.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "primary CLI path unexpectedly succeeded despite non-production-ready calibration state"
    assert "CalibrationNotProductionReady" in result.stderr
    assert not out_path.exists()
