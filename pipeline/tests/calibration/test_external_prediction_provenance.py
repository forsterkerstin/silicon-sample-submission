from __future__ import annotations

import pandas as pd
import pytest

import calibration.external_prediction_provenance as provenance
from calibration.external_prediction_provenance import (
    EXTERNAL_F_PANEL_VERSION,
    PREDICTION_F_INFERENCE_CONFIG_HASH_COL,
    PREDICTION_F_MODEL_ID_COL,
    PREDICTION_F_PROMPT_PROTOCOL_ID_COL,
    PREDICTION_F_R_F_COL,
    PREDICTION_PANEL_SHA256_COL,
    PREDICTION_PANEL_VERSION_COL,
    PRODUCTION_READY_SYNTHETIC_STATUS,
    assert_external_f_predictions_production_ready,
    current_external_f_panel_provenance,
    f_inference_config_hash,
)


def _frozen_protocol(**overrides) -> dict[str, object]:
    protocol = {
        "selected_f_model": "model-f",
        "n_f": 500,
        "f_num_draws": 1,
        "f_r_f": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_configuration": {"reasoning_effort": "low"},
        "structured_output": "json_schema",
        "prompt_version": "ashokkumar_experiment_forecast_adapted_v1",
        "frozen_at": "2026-08-24T00:00:00+00:00",
    }
    protocol.update(overrides)
    if "f_r_f" in protocol:
        protocol["f_inference_config_hash"] = f_inference_config_hash(protocol)
    return protocol


def _write_protocol(tmp_path, **overrides):
    import json

    path = tmp_path / "frozen_f_protocol.json"
    path.write_text(json.dumps(_frozen_protocol(**overrides), indent=2) + "\n", encoding="utf-8")
    return path


def _write_panel(tmp_path, effect_ids=("s1:y:hyp1",)):
    path = tmp_path / "external_primary_f_panels.csv"
    rows = []
    for effect_id in effect_ids:
        for i in range(2):
            rows.append({"study_id": effect_id.split(":")[0], "effect_id": effect_id, "f_profile_id": f"F{i:03d}"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _production_ready_row(panel_sha: str, *, effect_id: str = "s1:y:hyp1", protocol: dict[str, object] | None = None, **overrides) -> dict[str, object]:
    protocol = protocol or _frozen_protocol()
    row = {
        "study_id": "s1",
        "effect_id": effect_id,
        "model_ate": 2.0,
        "synthetic_ate_native": 0.2,
        "human_ate": 1.0,
        "human_ate_native": 0.1,
        "included_primary_calibration": True,
        "synthetic_prediction_status": PRODUCTION_READY_SYNTHETIC_STATUS,
        "requires_synthetic_regeneration": False,
        PREDICTION_F_MODEL_ID_COL: protocol["selected_f_model"],
        PREDICTION_F_PROMPT_PROTOCOL_ID_COL: protocol["prompt_version"],
        PREDICTION_F_INFERENCE_CONFIG_HASH_COL: f_inference_config_hash(protocol),
        PREDICTION_F_R_F_COL: protocol["f_r_f"],
        PREDICTION_PANEL_VERSION_COL: EXTERNAL_F_PANEL_VERSION,
        PREDICTION_PANEL_SHA256_COL: panel_sha,
        "population_matching_method": "study_effect_analytic_profile_distribution_unweighted_largest_remainder",
        "num_profiles": 500,
    }
    row.update(overrides)
    return row


def test_current_stale_cached_external_f_predictions_are_rejected_for_production():
    archive = pd.read_csv("data/ate_archive.csv")

    with pytest.raises(RuntimeError, match="stale|DEVELOPMENT|regeneration|provenance"):
        assert_external_f_predictions_production_ready(archive)


def test_production_guard_accepts_exact_panel_provenance_when_model_protocol_check_is_disabled(tmp_path):
    panel_path = _write_panel(tmp_path)
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha)])

    provenance = assert_external_f_predictions_production_ready(
        archive,
        panels_path=panel_path,
        require_frozen_model_protocol=False,
        expected_primary_effect_count=1,
    )

    assert provenance[PREDICTION_PANEL_SHA256_COL] == panel_sha


def test_production_guard_rejects_panel_hash_mismatch(tmp_path):
    panel_path = _write_panel(tmp_path)
    archive = pd.DataFrame([_production_ready_row("not-the-current-panel-hash")])

    with pytest.raises(RuntimeError, match="external_f_population_panel_sha256 does not match frozen configuration"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            require_frozen_model_protocol=False,
            expected_primary_effect_count=1,
        )


def test_production_guard_rejects_absent_predictions(tmp_path):
    panel_path = _write_panel(tmp_path)
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, model_ate=None)])

    with pytest.raises(RuntimeError, match="predictions are absent"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            require_frozen_model_protocol=False,
            expected_primary_effect_count=1,
        )


def test_production_guard_requires_frozen_f_model_and_protocol(tmp_path):
    """F* and the frozen F protocol are now genuinely materialized in this
    repo, so a synthetic row using an unrelated placeholder f_model_id
    ("model-f", not the real frozen google/gemma-4-31B-it) now fails on a
    provenance MISMATCH rather than on the prerequisite being entirely
    absent -- still proves the guard requires exact frozen model/protocol
    provenance, just via the (now more informative) mismatch message."""
    panel_path = _write_panel(tmp_path)
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha)])

    with pytest.raises(RuntimeError, match="selected_f_model|frozen F protocol|does not match frozen configuration"):
        assert_external_f_predictions_production_ready(archive, panels_path=panel_path, expected_primary_effect_count=1)


def test_production_guard_accepts_full_frozen_prediction_provenance(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    protocol_path = _write_protocol(tmp_path)
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, protocol=protocol)])
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    out = assert_external_f_predictions_production_ready(
        archive,
        panels_path=panel_path,
        frozen_protocol_path=protocol_path,
        expected_primary_effect_count=1,
    )

    assert out[PREDICTION_F_R_F_COL] == "1"


def test_production_guard_fails_when_r_f_is_unfrozen(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    protocol_path = _write_protocol(tmp_path)
    raw = protocol_path.read_text(encoding="utf-8").replace('  "f_r_f": 1,\n', "").replace('  "f_inference_config_hash": ', '  "unused_hash": ')
    protocol_path.write_text(raw, encoding="utf-8")
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, protocol=protocol)])
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    with pytest.raises(RuntimeError, match="R_F is unfrozen"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            frozen_protocol_path=protocol_path,
            expected_primary_effect_count=1,
        )


def test_production_guard_fails_when_recorded_r_f_differs_from_frozen_r_f(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    protocol_path = _write_protocol(tmp_path)
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, protocol=protocol, **{PREDICTION_F_R_F_COL: 2})])
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    with pytest.raises(RuntimeError, match="f_r_f does not match frozen configuration"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            frozen_protocol_path=protocol_path,
            expected_primary_effect_count=1,
        )


def test_production_guard_fails_when_inference_config_hash_differs(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path)
    protocol_path = _write_protocol(tmp_path)
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, protocol=protocol, **{PREDICTION_F_INFERENCE_CONFIG_HASH_COL: "wrong-hash"})])
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    with pytest.raises(RuntimeError, match="f_inference_config_hash does not match frozen configuration"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            frozen_protocol_path=protocol_path,
            expected_primary_effect_count=1,
        )


def test_production_guard_fails_when_effect_is_missing(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path, effect_ids=("s1:y:hyp1", "s2:y:hyp1"))
    protocol_path = _write_protocol(tmp_path)
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame([_production_ready_row(panel_sha, effect_id="s1:y:hyp1", protocol=protocol)])
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    with pytest.raises(RuntimeError, match="present 1|missing 1 effect"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            frozen_protocol_path=protocol_path,
            expected_primary_effect_count=2,
        )


def test_production_guard_fails_when_effect_is_duplicated(tmp_path, monkeypatch):
    panel_path = _write_panel(tmp_path, effect_ids=("s1:y:hyp1", "s2:y:hyp1"))
    protocol_path = _write_protocol(tmp_path)
    protocol = _frozen_protocol()
    panel_sha = current_external_f_panel_provenance(panel_path)[PREDICTION_PANEL_SHA256_COL]
    archive = pd.DataFrame(
        [
            _production_ready_row(panel_sha, effect_id="s1:y:hyp1", protocol=protocol),
            _production_ready_row(panel_sha, effect_id="s1:y:hyp1", protocol=protocol),
        ]
    )
    monkeypatch.setattr(provenance, "selected_model", lambda role, require_frozen=True: "model-f")

    with pytest.raises(RuntimeError, match="duplicate|missing"):
        assert_external_f_predictions_production_ready(
            archive,
            panels_path=panel_path,
            frozen_protocol_path=protocol_path,
            expected_primary_effect_count=2,
        )
