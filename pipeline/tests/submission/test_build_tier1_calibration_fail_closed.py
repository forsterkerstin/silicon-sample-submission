"""Adversarial tests proving production Tier-1 construction fails closed on
any calibration provenance problem rather than silently falling back to M0.

Synthetic/monkeypatched only. assert_external_f_predictions_production_ready's
own staleness/synthetic-regeneration/F*/R_F detection is already covered by
tests/calibration/test_external_prediction_provenance.py -- these tests
target resolve_production_calibration_model's own wrapper logic: does it
propagate that failure as a hard refusal, does it require a persisted
artifact, does it require an explicit usable_for_production=True, and does
it catch a calibration fit that has drifted from the current archive.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from calibration.external_prediction_provenance import file_sha256
from submission import build_tier1 as bt


def _write_archive(tmp_path):
    path = tmp_path / "ate_archive.csv"
    pd.DataFrame({"study_id": ["S1"], "effect_id": ["S1:e1"]}).to_csv(path, index=False)
    return path


def _write_selected_model(tmp_path, payload, name="calibration_selected_model.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---- A/B: archive not production-ready (stale / requires_synthetic_regeneration) ----


def test_stale_or_not_production_ready_archive_refuses(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)

    def _raise(archive_df):
        raise RuntimeError("external F predictions are not production-ready:\n- predictions require synthetic regeneration")

    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", _raise)
    with pytest.raises(bt.CalibrationNotProductionReady, match="not production-ready"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=tmp_path / "unused.json")


# ---- C: usable_for_production is False, or simply absent ----


def test_usable_for_production_false_refuses(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = _write_selected_model(tmp_path, {"model_name": "M1", "usable_for_production": False})
    with pytest.raises(bt.CalibrationNotProductionReady, match="usable_for_production"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


def test_usable_for_production_absent_refuses(tmp_path, monkeypatch):
    """Missing the key entirely must fail closed too -- absence is not
    treated as an implicit True."""
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = _write_selected_model(tmp_path, {"model_name": "M1"})
    with pytest.raises(bt.CalibrationNotProductionReady, match="usable_for_production"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


# ---- D: missing calibration artifact ----


def test_missing_calibration_artifact_refuses(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    with pytest.raises(bt.CalibrationNotProductionReady, match="no frozen production calibration artifact"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=tmp_path / "does_not_exist.json")


# ---- E: calibration-load exception refuses rather than M0 fallback ----


def test_malformed_calibration_artifact_refuses(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = tmp_path / "calibration_selected_model.json"
    selected_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(bt.CalibrationNotProductionReady, match="could not be loaded"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


def test_missing_archive_path_refuses(tmp_path):
    with pytest.raises(bt.CalibrationNotProductionReady, match="archive is missing"):
        bt.resolve_production_calibration_model(tmp_path / "no_such_archive.csv", selected_model_path=tmp_path / "unused.json")


# ---- archive-hash provenance: calibration fit against a superseded archive ----


def test_archive_hash_mismatch_refuses(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = _write_selected_model(
        tmp_path, {"model_name": "M1", "usable_for_production": True, "source_ate_archive_sha256": "deadbeef" * 8}
    )
    with pytest.raises(bt.CalibrationNotProductionReady, match="different ate_archive.csv"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


def test_missing_archive_hash_in_artifact_refuses(tmp_path, monkeypatch):
    """An artifact that predates this safety check (no recorded hash at
    all) must also refuse, not be treated as trivially matching."""
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = _write_selected_model(tmp_path, {"model_name": "M1", "usable_for_production": True})
    with pytest.raises(bt.CalibrationNotProductionReady, match="does not record its source"):
        bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


def test_no_failure_path_returns_m0_identity(tmp_path, monkeypatch):
    """The critical regression this whole fix targets: no failure path may
    return an M0-identity dict in place of raising."""
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    broken_artifacts = [
        tmp_path / "missing.json",
        _write_selected_model(tmp_path, {"usable_for_production": False}, name="a.json"),
        _write_selected_model(tmp_path, {}, name="b.json"),
        _write_selected_model(tmp_path, {"usable_for_production": True, "source_ate_archive_sha256": "wrong"}, name="c.json"),
    ]
    for selected_path in broken_artifacts:
        with pytest.raises(bt.CalibrationNotProductionReady):
            bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)


# ---- F: valid frozen production calibration -> accepted ----


def test_valid_frozen_production_calibration_accepted(tmp_path, monkeypatch):
    archive_path = _write_archive(tmp_path)
    monkeypatch.setattr(bt, "assert_external_f_predictions_production_ready", lambda df: None)
    selected_path = _write_selected_model(
        tmp_path,
        {
            "model_name": "M1",
            "calibration_alpha": 0.0,
            "calibration_lambda": 1.2,
            "usable_for_production": True,
            "source_ate_archive_sha256": file_sha256(archive_path),
        },
    )
    model = bt.resolve_production_calibration_model(archive_path, selected_model_path=selected_path)
    assert model["model_name"] == "M1"
    assert model["usable_for_production"] is True
    assert model["calibration_lambda"] == 1.2


# ---- confirms the CLI production branch actually calls the strict resolver ----


def test_production_cli_branch_calls_strict_resolver_not_the_permissive_one():
    import inspect

    source = inspect.getsource(bt.main)
    g_native_branch = source.split("if args.g_native_responses or args.f_responses:")[1].split("elif args.raw_responses:")[0]
    assert "resolve_production_calibration_model()" in g_native_branch
    assert "resolve_calibration_model()" not in g_native_branch
