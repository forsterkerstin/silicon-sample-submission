"""Tests for the target-production guard architecture.

As of the calibration-production freeze, all target-production prerequisites
(F*/G*, R_F/frozen F protocol, a usable_for_production calibration artifact,
the frozen final method manifest) are now genuinely frozen -- so this suite
proves BOTH that the real, now-satisfied prerequisites correctly authorize
prerequisite checks/declaration (in an isolated tmp_path state, never the
real ledger), AND that the guard's internal checks (target-only rows, exact
frozen model, duplicate/off-phase protection, and refusal when prerequisites
are genuinely absent) remain structurally sound via monkeypatching. Nothing
here submits to Together.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inference.target_production_guard import (
    TargetProductionNotAuthorized,
    assert_target_production_prerequisites_frozen,
    declare_target_phase,
    target_production_safety_guard,
)


def test_prerequisites_now_genuinely_frozen_given_real_repo_state():
    """F*/G* (google/gemma-4-31B-it), R_F=1/the frozen F protocol
    (outputs/f_reliability/frozen_f_protocol.json), the calibration
    artifact (outputs/calibration_selected_model.json, usable_for_production=True,
    fit from the real retrieved 136,000-request calibration-production
    batch), and the final method manifest (outputs/validation/frozen_method_manifest.json)
    are ALL now genuinely frozen -- this is the load-bearing proof that the
    real prerequisite checker succeeds against the real repo state today."""
    result = assert_target_production_prerequisites_frozen()
    assert result["selected_f_model"] == "google/gemma-4-31B-it"
    assert result["selected_g_model"] == "google/gemma-4-31B-it"


def test_prerequisites_still_refuse_when_genuinely_unfrozen(monkeypatch):
    """Isolated from real repo state: proves the checker still refuses (and
    names the right reasons) whenever any one prerequisite is genuinely
    missing -- the refusal logic itself, not just today's real outcome."""
    import inference.target_production_guard as tpg

    monkeypatch.setattr(tpg, "CALIBRATION_SELECTED_MODEL_PATH", Path("/nonexistent/calibration_selected_model.json"))
    with pytest.raises(TargetProductionNotAuthorized) as exc_info:
        assert_target_production_prerequisites_frozen()
    assert "calibration is not frozen" in str(exc_info.value)


def test_declare_target_phase_succeeds_with_real_prerequisites_isolated_state(tmp_path):
    """Real prerequisites, but an isolated tmp_path ledger -- never writes
    to the real target-production submission state."""
    result = declare_target_phase("test_isolated_phase", approved_custom_ids={"a", "b"}, cost_cap_usd=1.0, state_path=tmp_path / "state.json")
    assert result["cost_cap_usd"] == 1.0
    assert set(result["approved_custom_ids"]) == {"a", "b"}


def test_declare_target_phase_refuses_without_prerequisites(tmp_path, monkeypatch):
    import inference.target_production_guard as tpg

    monkeypatch.setattr(tpg, "CALIBRATION_SELECTED_MODEL_PATH", Path("/nonexistent/calibration_selected_model.json"))
    with pytest.raises(TargetProductionNotAuthorized):
        declare_target_phase("target_production", approved_custom_ids={"a"}, cost_cap_usd=1.0, state_path=tmp_path / "state.json")


def test_target_guard_refuses_without_declared_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", tmp_path)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    jsonl_path = tmp_path / "batch.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("custom_id,study_id,request_stage\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="has not been declared"):
        target_production_safety_guard(jsonl_path, manifest_path, phase="target_production", state_path=tmp_path / "state.json")


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["custom_id", "study_id", "request_stage"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _body(model="google/gemma-4-31B-it"):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


def test_target_guard_refuses_non_target_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    target_root = tmp_path / "target_production"
    target_root.mkdir()
    jsonl_path = target_root / "batch.jsonl"
    manifest_path = target_root / "manifest.csv"
    _write_jsonl(jsonl_path, [{"custom_id": "c1", "body": _body()}])
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "not_target", "request_stage": "target_production"}])
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", target_root)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    state_path = target_root / "state.json"
    state_path.write_text(
        json.dumps({"schema_version": "target_production_submission_state_v1", "phases": {"target_production": {"approved_custom_ids": ["c1"], "cost_cap_usd": 100.0, "cumulative_requests": 0, "cumulative_worst_case_cost_usd": 0.0, "submissions": []}}, "submitted_custom_ids": [], "successful_custom_ids": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refuses non-target request"):
        target_production_safety_guard(jsonl_path, manifest_path, phase="target_production", state_path=state_path)


def test_target_guard_refuses_unrecognized_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    target_root = tmp_path / "target_production"
    target_root.mkdir()
    jsonl_path = target_root / "batch.jsonl"
    manifest_path = target_root / "manifest.csv"
    _write_jsonl(jsonl_path, [{"custom_id": "c1", "body": _body(model="deepseek-ai/DeepSeek-V4-Pro-0813")}])
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "target", "request_stage": "target_production"}])
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", target_root)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    state_path = target_root / "state.json"
    state_path.write_text(
        json.dumps({"schema_version": "target_production_submission_state_v1", "phases": {"target_production": {"approved_custom_ids": ["c1"], "cost_cap_usd": 100.0, "cumulative_requests": 0, "cumulative_worst_case_cost_usd": 0.0, "submissions": []}}, "submitted_custom_ids": [], "successful_custom_ids": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="only allows the frozen"):
        target_production_safety_guard(jsonl_path, manifest_path, phase="target_production", state_path=state_path)


def test_target_guard_accepts_valid_target_request(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    target_root = tmp_path / "target_production"
    target_root.mkdir()
    jsonl_path = target_root / "batch.jsonl"
    manifest_path = target_root / "manifest.csv"
    _write_jsonl(jsonl_path, [{"custom_id": "c1", "body": _body()}])
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "target", "request_stage": "target_production"}])
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", target_root)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    state_path = target_root / "state.json"
    state_path.write_text(
        json.dumps({"schema_version": "target_production_submission_state_v1", "phases": {"target_production": {"approved_custom_ids": ["c1"], "cost_cap_usd": 100.0, "cumulative_requests": 0, "cumulative_worst_case_cost_usd": 0.0, "submissions": []}}, "submitted_custom_ids": [], "successful_custom_ids": []}),
        encoding="utf-8",
    )
    result = target_production_safety_guard(jsonl_path, manifest_path, phase="target_production", state_path=state_path)
    assert result["submission_allowed"] is True
    assert result["model"] == "google/gemma-4-31B-it"


def test_declare_target_phase_with_explicit_request_stage_differs_from_phase_name(tmp_path):
    result = declare_target_phase("standard_g_v2_smoke", approved_custom_ids={"a", "b"}, cost_cap_usd=1.0, request_stage="standard", state_path=tmp_path / "state.json")
    assert result["request_stage"] == "standard"
    assert result["approved_custom_ids"] == ["a", "b"]


def test_declare_target_phase_without_request_stage_defaults_to_phase_name(tmp_path):
    result = declare_target_phase("standard", approved_custom_ids={"a"}, cost_cap_usd=1.0, state_path=tmp_path / "state.json")
    assert result["request_stage"] == "standard"


def test_redeclaring_same_phase_with_different_request_stage_is_refused(tmp_path):
    state_path = tmp_path / "state.json"
    declare_target_phase("standard_g_v2_smoke", approved_custom_ids={"a"}, cost_cap_usd=1.0, request_stage="standard", state_path=state_path)
    with pytest.raises(RuntimeError, match="request_stage already declared"):
        declare_target_phase("standard_g_v2_smoke", approved_custom_ids={"a"}, cost_cap_usd=1.0, request_stage="something_else", state_path=state_path)


def test_guard_accepts_new_phase_name_authorizing_a_structural_request_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    target_root = tmp_path / "target_production"
    target_root.mkdir()
    jsonl_path = target_root / "batch.jsonl"
    manifest_path = target_root / "manifest.csv"
    _write_jsonl(jsonl_path, [{"custom_id": "c1", "body": _body()}])
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "target", "request_stage": "standard"}])
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", target_root)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    state_path = target_root / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "target_production_submission_state_v1",
                "phases": {"standard_g_v2_smoke": {"approved_custom_ids": ["c1"], "cost_cap_usd": 100.0, "request_stage": "standard", "cumulative_requests": 0, "cumulative_worst_case_cost_usd": 0.0, "submissions": []}},
                "submitted_custom_ids": [],
                "successful_custom_ids": [],
            }
        ),
        encoding="utf-8",
    )
    result = target_production_safety_guard(jsonl_path, manifest_path, phase="standard_g_v2_smoke", state_path=state_path)
    assert result["submission_allowed"] is True


def test_guard_still_refuses_wrong_request_stage_under_new_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "inference.target_production_guard.assert_target_production_prerequisites_frozen",
        lambda: {"selected_f_model": "google/gemma-4-31B-it", "selected_g_model": "google/gemma-4-31B-it"},
    )
    target_root = tmp_path / "target_production"
    target_root.mkdir()
    jsonl_path = target_root / "batch.jsonl"
    manifest_path = target_root / "manifest.csv"
    _write_jsonl(jsonl_path, [{"custom_id": "c1", "body": _body()}])
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "target", "request_stage": "consensus_stage_a"}])  # wrong stage
    monkeypatch.setattr("inference.target_production_guard.TARGET_PRODUCTION_ROOT", target_root)
    monkeypatch.setattr("inference.target_production_guard._under_path", lambda p, root: True)
    state_path = target_root / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "target_production_submission_state_v1",
                "phases": {"standard_g_v2_smoke": {"approved_custom_ids": ["c1"], "cost_cap_usd": 100.0, "request_stage": "standard", "cumulative_requests": 0, "cumulative_worst_case_cost_usd": 0.0, "submissions": []}},
                "submitted_custom_ids": [],
                "successful_custom_ids": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="do not carry request_stage='standard'"):
        target_production_safety_guard(jsonl_path, manifest_path, phase="standard_g_v2_smoke", state_path=state_path)


def test_cli_refuses_unguarded_jsonl_with_target_manifest_rows(tmp_path):
    """The catch-all: an arbitrary path outside every guarded root, but
    whose manifest shows study_id=='target' rows, must be refused by the
    CLI rather than falling through to the unguarded generic submit path."""
    import sys as _sys

    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    import together_batch as cli

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [{"custom_id": "c1", "study_id": "target", "request_stage": "whatever"}])
    assert cli._manifest_contains_target_rows(manifest_path) is True

    non_target_manifest = tmp_path / "manifest2.csv"
    _write_manifest(non_target_manifest, [{"custom_id": "c1", "study_id": "f_model_screen", "request_stage": "f_model_screen"}])
    assert cli._manifest_contains_target_rows(non_target_manifest) is False
    assert cli._manifest_contains_target_rows(None) is False
    assert cli._manifest_contains_target_rows(tmp_path / "does_not_exist.csv") is False
