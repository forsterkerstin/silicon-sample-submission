"""Engineering-only reconciliation for ONE logical stage of the benchmark-
exact Consensus pipeline (STEP_1, STEP_2, STEP_3, or OUTCOMES). Generic
over stage: takes the stage's manifest/jsonl/retrieved-output/retrieved-
error paths explicitly, so the SAME script serves STEP_1 today and
STEP_2/STEP_3/OUTCOMES once their real manifests exist (each is only
buildable after the immediately preceding stage's real, resolved response
-- see inference/consensus_benchmark_exact.py).

Reports, per donor: schema validity (SCHEMA_VALID/SCHEMA_INVALID with
malformed_json breakdown), PROVIDER_ERROR, missing/unexpected/duplicate,
system_fingerprint breakdown by validity status, and production-attempt-
number distribution (attempt_id column, already present in the manifest --
unlike target G Wave-1's completion manifests, Consensus-exact's
attempt_id is never overloaded for collision-avoidance, so no separate
provenance file is needed here).

Never computes, prints, summarizes, or exposes any scientific response
value. No fence-stripping, no repair, no coercion, no retry.
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

import score_target_g_wave1_v2_replacement as v2_scorer  # noqa: E402  (reused: _load_schema_by_cid, _load_jsonl_by_cid)


def score_stage(stage_name: str, manifest_path: Path, jsonl_path: Path, retrieved_output_path: Path, retrieved_error_path: Path) -> dict:
    from ate.f_screen_validation import reconciliation_report, validate_response

    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    manifest_ids = {row["custom_id"] for row in manifest_rows}
    cid_to_attempt_id = {row["custom_id"]: row.get("replicate_id", "1") for row in manifest_rows}
    cid_to_donor = {row["custom_id"]: row["profile_id"] for row in manifest_rows}
    schema_by_cid = v2_scorer._load_schema_by_cid(jsonl_path)

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
    attempt_id_counts: Counter = Counter()
    donor_status: dict[str, str] = {}

    for cid in sorted(manifest_ids):
        attempt_id_counts[str(cid_to_attempt_id[cid])] += 1
        donor = cid_to_donor[cid]
        if cid in output_by_cid:
            rec = output_by_cid[cid]
            fp = str(rec.get("response", {}).get("body", {}).get("system_fingerprint"))
            v = validate_response(rec, schema_by_cid.get(cid))
            status = "SCHEMA_VALID" if v["valid"] else "SCHEMA_INVALID"
            fingerprint_by_status.setdefault(status, Counter())[fp] += 1
            donor_status[donor] = status
            if v["valid"]:
                schema_valid += 1
            else:
                schema_invalid += 1
                if v["reason"].startswith("malformed_json"):
                    malformed_json += 1
        elif cid in error_by_cid:
            provider_error += 1
            donor_status[donor] = "PROVIDER_ERROR"
            err = error_by_cid[cid].get("error", {})
            error_code_counts[str(err.get("code"))] += 1
        else:
            missing += 1
            donor_status[donor] = "NOT_ATTEMPTED"

    accounting_closes = schema_valid + schema_invalid + provider_error + missing == len(manifest_ids)

    return {
        "stage_name": stage_name,
        "expected_donors": len(manifest_ids),
        "returned_output_records": len(output_records),
        "returned_error_records": len(error_records),
        "missing_entirely": missing,
        "unexpected": len(unexpected),
        "duplicate": len(duplicate),
        "schema_valid": schema_valid,
        "schema_invalid": schema_invalid,
        "malformed_json": malformed_json,
        "provider_error": provider_error,
        "provider_error_code_counts": dict(error_code_counts),
        "accounting_identity_holds": accounting_closes,
        "fingerprint_breakdown_by_validity_status": {status: dict(counter) for status, counter in fingerprint_by_status.items()},
        "attempt_id_counts": dict(attempt_id_counts),
        "donor_status": donor_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-name", required=True, choices=["step1", "step2", "step3", "outcomes"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = score_stage(args.stage_name, args.manifest, args.jsonl, args.output, args.error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "donor_status"}
    print(json.dumps(summary, indent=2))
    return 0 if result["accounting_identity_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
