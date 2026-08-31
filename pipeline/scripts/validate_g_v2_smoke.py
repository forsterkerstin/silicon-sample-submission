"""Strict, zero-tolerance validator for the 10-request G-v2 engineering
smoke (schema/serving validation ONLY -- never scientific analysis or
model selection). Reuses the frozen, generic
ate.f_screen_validation.validate_response (no fence-stripping, no repair,
no coercion).

Frozen acceptance criteria (before submission, unconditional -- no
invalid-rate tolerance for this smoke):

    EXPECTED_REQUESTS = 10
    MISSING = 0
    UNEXPECTED = 0
    DUPLICATES = 0
    SCHEMA_VALID = 10
    SCHEMA_INVALID = 0
    MALFORMED_JSON = 0

SMOKE_PASS = YES iff all of the above hold exactly. 9/10 is FAIL. A failing
smoke must stop the production path -- this script never retries or
resubmits anything; it only reads already-retrieved files.

Usage:
    python scripts/validate_g_v2_smoke.py \
        --standard-output outputs/target_production/g_v2_engineering_smoke/standard/retrieved/batch_output.jsonl \
        --consensus-a-output outputs/target_production/g_v2_engineering_smoke/consensus_stage_a/retrieved/batch_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
FAILED_FINGERPRINT = "vllm-0.21.0-8326ea74"
EXPECTED_TOTAL = 10


def _load_schema_by_custom_id(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["custom_id"])] = r["body"]["response_format"]["json_schema"]["schema"]
    return out


def _score_stage(stage: str, retrieved_path: Path) -> dict:
    from ate.f_screen_validation import reconciliation_report, validate_response

    manifest_ids = set()
    with open(SMOKE_ROOT / stage / "request_manifest.csv", newline="", encoding="utf-8") as f:
        import csv

        manifest_ids = {row["custom_id"] for row in csv.DictReader(f)}
    schema_by_cid = _load_schema_by_custom_id(SMOKE_ROOT / stage / "batch_input.jsonl")

    raw_records = []
    with open(retrieved_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    report = reconciliation_report(manifest_ids, raw_records)
    raw_by_cid = {r["custom_id"]: r for r in raw_records if isinstance(r, dict) and "custom_id" in r}

    schema_valid = schema_invalid = malformed_json = 0
    fingerprints = []
    for cid in sorted(manifest_ids):
        rec = raw_by_cid.get(cid)
        fp = None
        if rec is not None:
            fp = rec.get("response", {}).get("body", {}).get("system_fingerprint")
        v = validate_response(rec, schema_by_cid.get(cid))
        if v["valid"]:
            schema_valid += 1
        else:
            schema_invalid += 1
            if v["reason"].startswith("malformed_json"):
                malformed_json += 1
        fingerprints.append({"custom_id": cid, "system_fingerprint": fp, "valid": v["valid"], "reason": v["reason"]})

    return {
        "stage": stage,
        "expected": len(manifest_ids),
        "retrieved": len(raw_records),
        "missing": len(report["missing_entirely"]),
        "unexpected": len(report["unexpected"]),
        "duplicates": len(report["duplicate"]),
        "schema_valid": schema_valid,
        "schema_invalid": schema_invalid,
        "malformed_json": malformed_json,
        "per_response": fingerprints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-output", type=Path, required=True)
    parser.add_argument("--consensus-a-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SMOKE_ROOT / "smoke_validation_result.json", help="where to write the result JSON (default: the real committed smoke result path)")
    args = parser.parse_args()

    standard = _score_stage("standard", args.standard_output)
    consensus_a = _score_stage("consensus_stage_a", args.consensus_a_output)

    total_missing = standard["missing"] + consensus_a["missing"]
    total_unexpected = standard["unexpected"] + consensus_a["unexpected"]
    total_duplicates = standard["duplicates"] + consensus_a["duplicates"]
    total_schema_valid = standard["schema_valid"] + consensus_a["schema_valid"]
    total_schema_invalid = standard["schema_invalid"] + consensus_a["schema_invalid"]
    total_malformed = standard["malformed_json"] + consensus_a["malformed_json"]
    total_expected = standard["expected"] + consensus_a["expected"]

    smoke_pass = total_expected == EXPECTED_TOTAL and total_missing == 0 and total_unexpected == 0 and total_duplicates == 0 and total_schema_valid == EXPECTED_TOTAL and total_schema_invalid == 0 and total_malformed == 0

    failed_fp_responses = [r for stage_result in (standard, consensus_a) for r in stage_result["per_response"] if r["system_fingerprint"] == FAILED_FINGERPRINT]

    result = {
        "expected_requests": EXPECTED_TOTAL,
        "total_expected_observed": total_expected,
        "missing": total_missing,
        "unexpected": total_unexpected,
        "duplicates": total_duplicates,
        "schema_valid": total_schema_valid,
        "schema_invalid": total_schema_invalid,
        "malformed_json": total_malformed,
        "smoke_pass": smoke_pass,
        "standard": standard,
        "consensus_stage_a": consensus_a,
        "failed_fingerprint_seen_count": len(failed_fp_responses),
        "failed_fingerprint_responses": failed_fp_responses,
        "failed_fingerprint_any_schema_valid": any(r["valid"] for r in failed_fp_responses) if failed_fp_responses else None,
        "full_replacement_automatically_authorized": False,
    }

    out_path = args.out
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("standard", "consensus_stage_a")}, indent=2))
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
