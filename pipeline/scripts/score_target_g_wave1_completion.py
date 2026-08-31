"""Reconciliation and schema/provider-validity accounting for the target G
Wave-1 STANDARD completion batch ONLY (outputs/target_production/
wave1_g_completion/standard/) -- engineering only, exactly like
scripts/score_target_g_wave1_v2_replacement.py, extended to also report
production-attempt-number provenance (from the frozen
attempt_provenance.json) and, from that, the first-valid-assembled count
and how many standard identities are eligible for a future Attempt 3.

STANDARD ONLY. The old Consensus-A completion branch (outputs/
target_production/wave1_g_completion/consensus_stage_a/) is permanently
disabled (see outputs/target_production/consensus_protocol_amendment.json)
and is NEVER required, read, or referenced anywhere in this script --
standard reconciliation must be runnable completely independently of
whatever happens to Consensus. The corrected Consensus pipeline lives
entirely separately in scripts/score_consensus_exact_stage.py.

Never computes, prints, summarizes, ranks, or otherwise exposes any
scientific response value (parsed questionnaire answer). Never fence-strips,
repairs, coerces, or retries. Read-only over already-retrieved files -- this
script never submits, and does not exist to be run before real retrieval.

Frozen spec version: see outputs/target_production/wave1_g_completion/
reconciliation_specification.json (RECONCILIATION_SPEC_VERSION below must
match its "spec_version" field).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PIPELINE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

import score_target_g_wave1_v2_replacement as v2_scorer  # noqa: E402  (reused: _load_schema_by_cid)

COMPLETION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion"
STAGE = "standard"
STAGE_DIR = COMPLETION_ROOT / STAGE
PROVENANCE_PATH = COMPLETION_ROOT / "attempt_provenance.json"
OUT_PATH = COMPLETION_ROOT / "standard_reconciliation_report.json"
RECONCILIATION_SPEC_VERSION = "target_g_wave1_standard_completion_reconciliation_v2"


def _load_manifest_ids(manifest_path: Path) -> set[str]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["custom_id"] for row in csv.DictReader(f)}


def score_standard_stage(retrieved_output_path: Path, retrieved_error_path: Path, provenance: dict) -> dict:
    """provenance: the "per_custom_id" mapping from attempt_provenance.json
    (custom_id -> {intended_identity, request_stage, attempt_number,
    smoke_only}). Only standard-stage custom_ids are read from it."""
    from ate.f_screen_validation import reconciliation_report, validate_response

    manifest_ids = _load_manifest_ids(STAGE_DIR / "request_manifest.csv")
    schema_by_cid = v2_scorer._load_schema_by_cid(STAGE_DIR / "batch_input.jsonl")

    output_records: list[dict] = []
    if retrieved_output_path.exists():
        with open(retrieved_output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    output_records.append(json.loads(line))
    error_records: list[dict] = []
    if retrieved_error_path.exists():
        with open(retrieved_error_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    error_records.append(json.loads(line))

    # unexpected/duplicate are checked against BOTH files (an unexpected or
    # duplicate custom_id can appear in either); "missing entirely" means
    # absent from BOTH -- a provider_error is NOT missing, it is accounted
    # for, so it must never inflate this count (that would break the
    # valid + technical_invalid + missing = expected identity below).
    output_report = reconciliation_report(manifest_ids, output_records)
    error_report = reconciliation_report(manifest_ids, error_records)
    output_by_cid = {r["custom_id"]: r for r in output_records if isinstance(r, dict) and "custom_id" in r}
    error_by_cid = {r["custom_id"]: r for r in error_records if isinstance(r, dict) and "custom_id" in r}

    both_present_ids = set(output_by_cid) & set(error_by_cid)
    unexpected = sorted(set(output_report["unexpected"]) | set(error_report["unexpected"]))
    duplicate = sorted(set(output_report["duplicate"]) | set(error_report["duplicate"]) | both_present_ids)

    schema_valid = schema_invalid = malformed_json = provider_error = missing = 0
    fingerprint_by_status: dict[str, Counter] = {}
    error_code_counts: Counter = Counter()
    attempt_number_counts: Counter = Counter()
    per_custom_id_status: dict[str, str] = {}

    for cid in sorted(manifest_ids):
        prov = provenance.get(cid)
        if prov is None:
            raise RuntimeError(f"custom_id {cid} has no attempt_provenance.json entry -- provenance file is stale relative to the manifest")
        attempt_number_counts[str(prov["attempt_number"])] += 1

        if cid in output_by_cid:
            rec = output_by_cid[cid]
            fp = str(rec.get("response", {}).get("body", {}).get("system_fingerprint"))
            v = validate_response(rec, schema_by_cid.get(cid))
            status = "SCHEMA_VALID" if v["valid"] else "SCHEMA_INVALID"
            fingerprint_by_status.setdefault(status, Counter())[fp] += 1
            per_custom_id_status[cid] = status
            if v["valid"]:
                schema_valid += 1
            else:
                schema_invalid += 1
                if v["reason"].startswith("malformed_json"):
                    malformed_json += 1
        elif cid in error_by_cid:
            provider_error += 1
            per_custom_id_status[cid] = "PROVIDER_ERROR"
            err = error_by_cid[cid].get("error", {})
            error_code_counts[str(err.get("code"))] += 1
        else:
            missing += 1
            per_custom_id_status[cid] = "NOT_ATTEMPTED"

    accounting_closes = schema_valid + schema_invalid + provider_error + missing == len(manifest_ids)

    return {
        "stage": STAGE,
        "expected": len(manifest_ids),
        "returned_output_records": len(output_records),
        "returned_error_records": len(error_records),
        "missing_entirely": missing,
        "unexpected": len(unexpected),
        "duplicate": len(duplicate),
        "malformed_no_custom_id_output": len(output_report["malformed_records"]),
        "malformed_no_custom_id_error": len(error_report["malformed_records"]),
        "schema_valid": schema_valid,
        "schema_invalid": schema_invalid,
        "malformed_json": malformed_json,
        "provider_error": provider_error,
        "provider_error_code_counts": dict(error_code_counts),
        "accounting_identity_holds": accounting_closes,
        "fingerprint_breakdown_by_validity_status": {status: dict(counter) for status, counter in fingerprint_by_status.items()},
        "attempt_number_counts": dict(attempt_number_counts),
        "per_custom_id_status": per_custom_id_status,
    }


def round_from_report(report: dict, provenance: dict) -> list[dict]:
    """Groups this report's own freshly-scored standard custom_ids by their
    TRUE production attempt_number (from attempt_provenance.json), into the
    additional_attempt_rounds shape inference.target_g_retry_engine.build_attempt_ledger
    expects. Standard-only: never references consensus_stage_a."""
    rounds_by_attempt: dict[int, dict] = {}
    for cid, status in report["per_custom_id_status"].items():
        prov = provenance[cid]
        n = prov["attempt_number"]
        r = rounds_by_attempt.setdefault(n, {"attempt_number": n, "custom_id_to_identity": {}, "custom_id_status": {}, "source_label": "wave1_g_completion/standard"})
        r["custom_id_to_identity"][cid] = prov["intended_identity"]
        r["custom_id_status"][cid] = status
    return [rounds_by_attempt[n] for n in sorted(rounds_by_attempt)]


def first_valid_and_attempt3_counts(report: dict, provenance: dict) -> dict:
    """Builds the full target-G Wave-1 ledger (base attempt-1, real,
    already-committed data -- requires no fresh input) with this report's
    round layered on top, then reports ONLY the standard-stage slice.
    Never requires or reads anything about consensus_stage_a as input --
    the ledger computes it internally from already-committed files, but
    nothing here ever asks the caller for it."""
    from inference.target_g_retry_engine import build_attempt_ledger

    rounds = round_from_report(report, provenance)
    ledger = build_attempt_ledger(rounds)
    standard_ledger = {identity: entry for identity, entry in ledger.items() if entry["request_stage"] == STAGE}
    first_valid_assembled = sum(1 for e in standard_ledger.values() if e["resolved"])
    eligible_for_attempt_3 = sum(1 for e in standard_ledger.values() if not e["resolved"] and e.get("next_attempt_number") == 3)
    return {"first_valid_assembled_count": first_valid_assembled, "standard_universe": len(standard_ledger), "eligible_for_attempt_3_count": eligible_for_attempt_3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-output", type=Path, required=True)
    parser.add_argument("--standard-error", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--skip-attempt3-projection", action="store_true", help="skip the ~5-minute full-ledger rebuild (base attempt-1 reclassification); report only raw reconciliation counts")
    args = parser.parse_args()

    if not PROVENANCE_PATH.exists():
        print(json.dumps({"error": f"attempt_provenance.json missing -- run scripts/freeze_target_g_completion_attempt_provenance.py first: {PROVENANCE_PATH}"}, indent=2))
        return 2
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]

    standard = score_standard_stage(args.standard_output, args.standard_error, provenance)
    projection = {} if args.skip_attempt3_projection else first_valid_and_attempt3_counts(standard, provenance)

    result = {
        "reconciliation_spec_version": RECONCILIATION_SPEC_VERSION,
        "note": "engineering/accounting only -- no scientific response value is computed, printed, or exposed anywhere in this report. Standard-only: never reads or requires the abandoned Consensus-A branch.",
        "standard": {k: v for k, v in standard.items() if k != "per_custom_id_status"},
        **projection,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if standard["accounting_identity_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
