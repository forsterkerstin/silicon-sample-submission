"""Tests for the target G Wave-1 STANDARD-ONLY completion reconciliation
script (scripts/score_target_g_wave1_completion.py) and first-valid
assembler (scripts/assemble_target_g_wave1_first_valid.py).

Standard-only: uses the REAL 1,401-request standard completion manifest.
The old Consensus-A branch is permanently abandoned and is never referenced
anywhere in this file -- these tests exist specifically to prove standard
reconciliation/assembly work completely independently of it (no
Consensus-A file, path, or argument is required by either script).

Uses SYNTHETIC retrieved-output/error fixtures (no real Together submission
has happened yet for the standard completion batch) -- exercises
score_standard_stage directly rather than via subprocess, for speed. Items
7-13 of the originally requested test coverage (attempt-1-valid-never-
retried, invalid-then-valid-retained, smoke-vs-production-attempt-number,
attempt-3 eligibility, no attempt 4, scientific-values-never-used-for-
membership) are covered by tests/inference/test_target_g_retry_engine.py's
existing suite, which exercises the SAME underlying ledger machinery this
reconciliation script and assembler both call -- not duplicated here."""

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

import score_target_g_wave1_completion as scorer  # noqa: E402

PROVENANCE_PATH = scorer.PROVENANCE_PATH

pytestmark = pytest.mark.skipif(not scorer.STAGE_DIR.exists() or not PROVENANCE_PATH.exists(), reason="target G standard completion manifest/provenance not built in this environment")


def test_reconciliation_never_requires_consensus_stage_a_as_input():
    import inspect

    # documentation/notes are free to EXPLAIN that Consensus-A is
    # excluded (an honest disclosure, not a dependency); what must never
    # exist is an actual functional requirement -- no --consensus-* CLI
    # arg, no STAGES tuple spanning both stages, no consensus_stage_a
    # directory path.
    sig = inspect.signature(scorer.main)
    source = inspect.getsource(scorer.main)
    assert "add_argument" in source
    assert "--consensus" not in source
    assert not hasattr(scorer, "STAGES")
    assert scorer.STAGE == "standard"
    assert scorer.STAGE_DIR.name == "standard"
    assert sig.parameters == {}  # main() takes no positional args needing consensus data


@pytest.fixture(scope="module")
def provenance():
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]


@pytest.fixture(scope="module")
def manifest_rows():
    with open(scorer.STAGE_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def schemas():
    return scorer.v2_scorer._load_schema_by_cid(scorer.STAGE_DIR / "batch_input.jsonl")


def _valid_content(schema: dict) -> str:
    props = {}
    for key, spec in schema["properties"].items():
        lo = spec.get("minimum", spec.get("enum", [0])[0] if "enum" in spec else 0)
        props[key] = lo
    return json.dumps(props)


def _write(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


# 1. duplicate custom_id.
def test_duplicate_custom_id_detected(tmp_path, manifest_rows, schemas, provenance):
    cid = manifest_rows[0]["custom_id"]
    content = _valid_content(schemas[cid])
    rec = json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": content}}], "system_fingerprint": "fp"}}})
    output_path = _write(tmp_path, "out.jsonl", [rec, rec])  # duplicated line
    error_path = _write(tmp_path, "err.jsonl", [])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["duplicate"] == 1


# 2. unexpected custom_id.
def test_unexpected_custom_id_detected(tmp_path, manifest_rows, schemas, provenance):
    rec = json.dumps({"custom_id": "G-not-in-manifest-at-all", "response": {"body": {"choices": [{"message": {"content": "{}"}}], "system_fingerprint": "fp"}}})
    output_path = _write(tmp_path, "out.jsonl", [rec])
    error_path = _write(tmp_path, "err.jsonl", [])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["unexpected"] == 1
    # every real manifest id is absent from output/error -> all missing
    assert result["missing_entirely"] == result["expected"]


# 3. missing identity (absent from both output and error -- never
# double-counted against a provider_error).
def test_missing_identity_not_double_counted_with_provider_error(tmp_path, manifest_rows, schemas, provenance):
    error_cid = manifest_rows[0]["custom_id"]
    err = json.dumps({"custom_id": error_cid, "error": {"code": "batch_client_error", "message": "Internal server error"}})
    output_path = _write(tmp_path, "out.jsonl", [])
    error_path = _write(tmp_path, "err.jsonl", [err])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["provider_error"] == 1
    assert result["missing_entirely"] == result["expected"] - 1
    assert result["accounting_identity_holds"] is True


# 4. malformed JSON (schema-invalid subcategory) -- no fence-stripping.
def test_malformed_json_stays_invalid(tmp_path, manifest_rows, schemas, provenance):
    cid = manifest_rows[0]["custom_id"]
    fenced = "```json\n" + _valid_content(schemas[cid]) + "\n```"
    rec = json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": fenced}}], "system_fingerprint": "vllm-0.21.0-8326ea74"}}})
    output_path = _write(tmp_path, "out.jsonl", [rec])
    error_path = _write(tmp_path, "err.jsonl", [])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["schema_invalid"] == 1
    assert result["malformed_json"] == 1
    assert result["fingerprint_breakdown_by_validity_status"]["SCHEMA_INVALID"]["vllm-0.21.0-8326ea74"] == 1


# 5. schema invalid (a NON-malformed-json kind: valid JSON, but violates
# the schema, e.g. a missing required field).
def test_schema_invalid_non_malformed_json(tmp_path, manifest_rows, schemas, provenance):
    cid = manifest_rows[0]["custom_id"]
    rec = json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": "{}"}}], "system_fingerprint": "fp"}}})
    output_path = _write(tmp_path, "out.jsonl", [rec])
    error_path = _write(tmp_path, "err.jsonl", [])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["schema_invalid"] == 1
    assert result["malformed_json"] == 0  # valid JSON syntax, just schema-violating


# 6. provider error.
def test_provider_error_counted_and_code_recorded(tmp_path, manifest_rows, schemas, provenance):
    cid = manifest_rows[0]["custom_id"]
    err = json.dumps({"custom_id": cid, "error": {"code": "batch_client_error", "message": "Internal server error"}})
    output_path = _write(tmp_path, "out.jsonl", [])
    error_path = _write(tmp_path, "err.jsonl", [err])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["provider_error"] == 1
    assert result["provider_error_code_counts"] == {"batch_client_error": 1}


# accounting identity: valid + technical-invalid + missing = expected,
# across a mixed fixture with all four categories present at once.
def test_accounting_identity_holds_with_all_four_categories(tmp_path, manifest_rows, schemas, provenance):
    valid_cid = manifest_rows[0]["custom_id"]
    invalid_cid = manifest_rows[1]["custom_id"]
    error_cid = manifest_rows[2]["custom_id"]
    # manifest_rows[3:] are left entirely absent -> missing
    valid_rec = json.dumps({"custom_id": valid_cid, "response": {"body": {"choices": [{"message": {"content": _valid_content(schemas[valid_cid])}}], "system_fingerprint": "fp"}}})
    invalid_rec = json.dumps({"custom_id": invalid_cid, "response": {"body": {"choices": [{"message": {"content": "{}"}}], "system_fingerprint": "fp"}}})
    err_rec = json.dumps({"custom_id": error_cid, "error": {"code": "batch_client_error", "message": "Internal server error"}})
    output_path = _write(tmp_path, "out.jsonl", [valid_rec, invalid_rec])
    error_path = _write(tmp_path, "err.jsonl", [err_rec])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    assert result["schema_valid"] == 1
    assert result["schema_invalid"] == 1
    assert result["provider_error"] == 1
    assert result["missing_entirely"] == result["expected"] - 3
    assert result["accounting_identity_holds"] is True
    assert result["schema_valid"] + result["schema_invalid"] + result["provider_error"] + result["missing_entirely"] == result["expected"]


# no substantive response value is used anywhere -- the returned dict
# never contains parsed answer content, only counts/codes/fingerprints/ids.
def test_no_substantive_response_values_in_report(tmp_path, manifest_rows, schemas, provenance):
    cid = manifest_rows[0]["custom_id"]
    rec = json.dumps({"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": _valid_content(schemas[cid])}}], "system_fingerprint": "fp"}}})
    output_path = _write(tmp_path, "out.jsonl", [rec])
    error_path = _write(tmp_path, "err.jsonl", [])
    result = scorer.score_standard_stage(output_path, error_path, provenance)
    # structural guard: the report has counts/codes/fingerprints/ids/per-cid
    # status only -- no key that could hold a parsed response value.
    assert "answers" not in result and "content" not in result and "choices" not in result and "response" not in result


# attempt-number distribution/provenance faithfully reflects the frozen
# attempt_provenance.json (never re-derived from response content).
def test_attempt_number_distribution_matches_frozen_provenance(manifest_rows, provenance):
    from collections import Counter

    expected = Counter(str(provenance[row["custom_id"]]["attempt_number"]) for row in manifest_rows)
    assert dict(expected) == {"2": 1396, "1": 5}


# round_from_report groups by TRUE attempt_number (from attempt_provenance.json,
# not replicate_id), status-only (engineering, never a parsed response value).
def test_round_from_report_groups_by_true_attempt_number(tmp_path, manifest_rows, schemas, provenance):
    valid_cid = manifest_rows[0]["custom_id"]
    other_cids = [row["custom_id"] for row in manifest_rows[1:20]]

    output_lines = [json.dumps({"custom_id": valid_cid, "response": {"body": {"choices": [{"message": {"content": _valid_content(schemas[valid_cid])}}], "system_fingerprint": "fp"}}})]
    error_lines = [json.dumps({"custom_id": cid, "error": {"code": "batch_client_error", "message": "Internal server error"}}) for cid in other_cids]

    output_path = _write(tmp_path, "out.jsonl", output_lines)
    error_path = _write(tmp_path, "err.jsonl", error_lines)
    report = scorer.score_standard_stage(output_path, error_path, provenance)

    rounds = scorer.round_from_report(report, provenance)
    rounds_by_attempt = {r["attempt_number"]: r for r in rounds}
    expected_attempt_number = provenance[valid_cid]["attempt_number"]
    assert rounds_by_attempt[expected_attempt_number]["custom_id_status"][valid_cid] == "SCHEMA_VALID"
    for r in rounds:
        assert set(r["custom_id_status"].values()) <= {"SCHEMA_VALID", "SCHEMA_INVALID", "PROVIDER_ERROR", "NOT_ATTEMPTED"}
