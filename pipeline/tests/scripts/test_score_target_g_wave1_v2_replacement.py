"""Tests for the target G Wave-1 v2 replacement scorer, using synthetic
fixtures (never mutating the real retrieved batch files). Confirms the
four-way partition (schema_valid / schema_invalid / provider_error /
missing_entirely) always accounts for exactly the expected count, and that
a provider-level error (present in batch_error.jsonl, no response body) is
never conflated with a schema-invalid response."""

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

import score_target_g_wave1_v2_replacement as scorer  # noqa: E402

REAL_REPORT_PATH = scorer.OUT_PATH


@pytest.fixture(autouse=True)
def _protect_real_report():
    before = REAL_REPORT_PATH.read_bytes() if REAL_REPORT_PATH.exists() else None
    yield
    after = REAL_REPORT_PATH.read_bytes() if REAL_REPORT_PATH.exists() else None
    assert after == before, "test suite must never mutate the real committed validation report"


def _build_part(tmp_path, *, n_valid=2, n_invalid=1, n_error=1, n_missing=1):
    part_dir = tmp_path / "part"
    retrieved_dir = part_dir / "retrieved"
    retrieved_dir.mkdir(parents=True)
    manifest_path = part_dir / "request_manifest.csv"
    jsonl_path = part_dir / "batch_input.jsonl"

    schema = {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 10}}, "required": ["x"]}
    cids = []
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=["custom_id"])
        writer.writeheader()
        idx = 0
        for _ in range(n_valid + n_invalid + n_error + n_missing):
            cid = f"cid-{idx}"
            cids.append(cid)
            writer.writerow({"custom_id": cid})
            body = {"model": "m", "response_format": {"json_schema": {"schema": schema}}}
            jf.write(json.dumps({"custom_id": cid, "body": body}) + "\n")
            idx += 1

    output_lines = []
    for cid in cids[:n_valid]:
        output_lines.append(json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": json.dumps({"x": 1})}}], "system_fingerprint": "healthy-fp"}}}))
    for cid in cids[n_valid : n_valid + n_invalid]:
        output_lines.append(json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": "```json\n" + json.dumps({"x": 1}) + "\n```"}}], "system_fingerprint": "vllm-0.21.0-8326ea74"}}}))
    (retrieved_dir / "batch_output.jsonl").write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    error_lines = []
    for cid in cids[n_valid + n_invalid : n_valid + n_invalid + n_error]:
        error_lines.append(json.dumps({"custom_id": cid, "error": {"code": "batch_client_error", "message": "Internal server error"}}))
    (retrieved_dir / "batch_error.jsonl").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    # remaining n_missing custom_ids appear in neither file.
    return part_dir


def test_four_way_partition_accounts_for_every_expected_id(tmp_path, monkeypatch):
    part_dir = _build_part(tmp_path, n_valid=3, n_invalid=2, n_error=2, n_missing=1)
    # _score_part joins SUBMISSION_ROOT/stage/part; point SUBMISSION_ROOT at
    # part_dir's parent and use stage=part_dir.name, part="" so the join
    # resolves exactly onto the synthetic part_dir.
    monkeypatch.setattr(scorer, "SUBMISSION_ROOT", part_dir.parent)
    r = scorer._score_part(part_dir.name, "")
    assert r["expected"] == 8
    assert r["schema_valid"] == 3
    assert r["schema_invalid"] == 2
    assert r["malformed_json"] == 2
    assert r["provider_error"] == 2
    assert r["missing_entirely"] == 1
    assert r["accounting_closes"] is True


def test_provider_error_never_double_counted_as_schema_invalid(tmp_path, monkeypatch):
    part_dir = _build_part(tmp_path, n_valid=1, n_invalid=0, n_error=3, n_missing=0)
    monkeypatch.setattr(scorer, "SUBMISSION_ROOT", part_dir.parent)
    r = scorer._score_part(part_dir.name, "")
    assert r["provider_error"] == 3
    assert r["schema_invalid"] == 0
    assert r["provider_error_code_counts"] == {"batch_client_error": 3}


def test_real_report_totals_are_internally_consistent():
    if not REAL_REPORT_PATH.exists():
        pytest.skip("real v2 replacement validation report not built in this environment")
    r = json.loads(REAL_REPORT_PATH.read_text(encoding="utf-8"))
    t = r["totals"]
    assert t["schema_valid"] + t["schema_invalid"] + t["provider_error"] + t["missing_entirely"] == t["expected"]
    assert t["accounting_closes"] is True
    for p in r["per_part"]:
        assert p["accounting_closes"] is True
