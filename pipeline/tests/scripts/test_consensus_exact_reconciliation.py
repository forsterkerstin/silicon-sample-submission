"""Tests for scripts/score_consensus_exact_stage.py and
scripts/assemble_consensus_exact_pipeline_state.py, using tiny synthetic
manifests (no real Together submission has happened for the Consensus-exact
pipeline yet)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import score_consensus_exact_stage as scorer  # noqa: E402
import inference.consensus_exact_retry_engine as re_engine  # noqa: E402

SCHEMA = {"type": "object", "properties": {"Q001": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["Q001"]}


def _build_manifest(tmp_path, donors):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "request_manifest.csv"
    jsonl_path = tmp_path / "batch_input.jsonl"
    cids = {}
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=["custom_id", "profile_id", "replicate_id", "request_key"])
        writer.writeheader()
        for i, donor in enumerate(donors):
            cid = f"c-{donor}"
            cids[donor] = cid
            writer.writerow({"custom_id": cid, "profile_id": donor, "replicate_id": "1", "request_key": f"G|{donor}|ConsensusExact|step1|attempt_1"})
            jf.write(json.dumps({"custom_id": cid, "body": {"model": "m", "response_format": {"json_schema": {"schema": SCHEMA}}}}) + "\n")
    return manifest_path, jsonl_path, cids


def test_score_consensus_exact_stage_accounting_and_donor_status(tmp_path):
    donors = ["d1", "d2", "d3", "d4"]
    manifest_path, jsonl_path, cids = _build_manifest(tmp_path, donors)
    out_lines = [json.dumps({"custom_id": cids["d1"], "response": {"body": {"choices": [{"message": {"content": json.dumps({"Q001": 50})}}], "system_fingerprint": "fp"}}})]
    err_lines = [json.dumps({"custom_id": cids["d2"], "error": {"code": "batch_client_error", "message": "Internal server error"}})]
    output_path = tmp_path / "out.jsonl"
    error_path = tmp_path / "err.jsonl"
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    error_path.write_text("\n".join(err_lines) + "\n", encoding="utf-8")
    # d3, d4 absent from both -> missing

    result = scorer.score_stage("step1", manifest_path, jsonl_path, output_path, error_path)
    assert result["expected_donors"] == 4
    assert result["schema_valid"] == 1
    assert result["provider_error"] == 1
    assert result["missing_entirely"] == 2
    assert result["accounting_identity_holds"] is True
    assert result["donor_status"] == {"d1": "SCHEMA_VALID", "d2": "PROVIDER_ERROR", "d3": "NOT_ATTEMPTED", "d4": "NOT_ATTEMPTED"}


def test_pipeline_state_final_eligibility_via_real_reconciliation_reports(tmp_path):
    donors = ["d1", "d2"]
    # step1: both valid.
    m1, j1, c1 = _build_manifest(tmp_path / "s1", donors)
    o1 = (tmp_path / "s1" / "out.jsonl")
    o1.write_text("\n".join(json.dumps({"custom_id": c1[d], "response": {"body": {"choices": [{"message": {"content": json.dumps({"Q001": 1})}}], "system_fingerprint": "fp"}}}) for d in donors) + "\n", encoding="utf-8")
    e1 = tmp_path / "s1" / "err.jsonl"
    e1.write_text("", encoding="utf-8")
    r1 = scorer.score_stage("step1", m1, j1, o1, e1)
    r1_path = tmp_path / "s1_report.json"
    r1_path.write_text(json.dumps(r1), encoding="utf-8")

    # step2: only d1 valid (d2 invalid).
    m2, j2, c2 = _build_manifest(tmp_path / "s2", donors)
    o2 = tmp_path / "s2" / "out.jsonl"
    o2.write_text(
        json.dumps({"custom_id": c2["d1"], "response": {"body": {"choices": [{"message": {"content": json.dumps({"Q001": 1})}}], "system_fingerprint": "fp"}}}) + "\n"
        + json.dumps({"custom_id": c2["d2"], "response": {"body": {"choices": [{"message": {"content": "not json"}}], "system_fingerprint": "fp"}}}) + "\n",
        encoding="utf-8",
    )
    e2 = tmp_path / "s2" / "err.jsonl"
    e2.write_text("", encoding="utf-8")
    r2 = scorer.score_stage("step2", m2, j2, o2, e2)
    r2_path = tmp_path / "s2_report.json"
    r2_path.write_text(json.dumps(r2), encoding="utf-8")

    step1_universe = tmp_path / "u1.json"
    step1_universe.write_text(json.dumps(donors), encoding="utf-8")
    step2_universe = tmp_path / "u2.json"
    step2_universe.write_text(json.dumps(["d1"]), encoding="utf-8")  # only d1 resolved step1 -> eligible for step2

    ledger1 = re_engine.build_stage_ledger(donors, [{"attempt_number": 1, "donor_status": r1["donor_status"]}])
    ledger2 = re_engine.build_stage_ledger(["d1"], [{"attempt_number": 1, "donor_status": {"d1": r2["donor_status"]["d1"]}}])
    assert re_engine.resolved_donors(ledger1) == {"d1", "d2"}
    assert re_engine.resolved_donors(ledger2) == {"d1"}
