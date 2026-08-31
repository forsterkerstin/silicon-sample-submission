"""Score the target G Wave-1 v2 full-replacement retrieved output for
schema validity AND provider-level delivery -- reconciliation + validity
accounting only, never scientific content (no target-G means/ATEs/
distributions/rankings are computed or exposed anywhere here).

Unlike scripts/score_target_g_wave1_original_run.py (which only read
batch_output.jsonl), this script also reconciles each part's
batch_error.jsonl -- the v2 replacement batches returned explicit
provider-level errors (batch_client_error: Internal server error) for a
nonzero share of requests, a genuinely different failure mode from the v1
run's malformed-JSON problem: these requests never produced ANY response
(valid or invalid) to validate. A custom_id is classified into exactly one
of: SCHEMA_VALID, SCHEMA_INVALID (a response was returned but failed
ate.f_screen_validation.validate_response -- no fence-stripping, no repair),
PROVIDER_ERROR (present in batch_error.jsonl, no response body at all), or
MISSING_ENTIRELY (in neither file). Nothing here retries, repairs, or
resubmits anything.

No LLM calls. Read-only over already-retrieved files.
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

SUBMISSION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "submission"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "g_wave1_v2_replacement_validation_report.json"
FAILED_FINGERPRINT = "vllm-0.21.0-8326ea74"

PARTS = [
    ("standard", "part1"),
    ("standard", "part2"),
    ("standard", "part3"),
    ("standard", "part4"),
    ("standard", "part5"),
    ("consensus_stage_a", "part1"),
]


def _load_manifest_ids(manifest_path: Path) -> set[str]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["custom_id"] for row in csv.DictReader(f)}


def _load_schema_by_cid(jsonl_path: Path) -> dict:
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["custom_id"])] = r["body"]["response_format"]["json_schema"]["schema"]
    return out


def _load_jsonl_by_cid(path: Path) -> tuple[dict, list, list]:
    """Returns (by_cid, unexpected_no_cid, duplicate_cids). Mirrors
    ate.f_screen_validation.reconciliation_report's duplicate handling but
    generalized to also cover error-file records."""
    by_cid: dict = {}
    no_cid: list = []
    duplicates: list = []
    if not path.exists():
        return by_cid, no_cid, duplicates
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")
            if not cid:
                no_cid.append(rec)
                continue
            cid = str(cid)
            if cid in by_cid:
                duplicates.append(cid)
            else:
                by_cid[cid] = rec
    return by_cid, no_cid, duplicates


def _score_part(stage: str, part: str) -> dict:
    from ate.f_screen_validation import validate_response

    part_dir = SUBMISSION_ROOT / stage / part
    retrieved_dir = part_dir / "retrieved"
    manifest_ids = _load_manifest_ids(part_dir / "request_manifest.csv")
    schema_by_cid = _load_schema_by_cid(part_dir / "batch_input.jsonl")

    output_by_cid, output_no_cid, output_duplicates = _load_jsonl_by_cid(retrieved_dir / "batch_output.jsonl")
    error_by_cid, error_no_cid, error_duplicates = _load_jsonl_by_cid(retrieved_dir / "batch_error.jsonl")

    both = set(output_by_cid) & set(error_by_cid)
    unexpected_output = set(output_by_cid) - manifest_ids
    unexpected_error = set(error_by_cid) - manifest_ids

    schema_valid = schema_invalid = malformed_json = provider_error = missing = 0
    fingerprint_counts: Counter = Counter()
    error_code_counts: Counter = Counter()
    invalid_reason_counts: Counter = Counter()

    for cid in sorted(manifest_ids):
        if cid in output_by_cid:
            rec = output_by_cid[cid]
            fp = rec.get("response", {}).get("body", {}).get("system_fingerprint")
            fingerprint_counts[str(fp)] += 1
            v = validate_response(rec, schema_by_cid.get(cid))
            if v["valid"]:
                schema_valid += 1
            else:
                schema_invalid += 1
                reason_key = v["reason"].split(":")[0]
                invalid_reason_counts[reason_key] += 1
                if v["reason"].startswith("malformed_json"):
                    malformed_json += 1
        elif cid in error_by_cid:
            provider_error += 1
            err = error_by_cid[cid].get("error", {})
            error_code_counts[str(err.get("code"))] += 1
        else:
            missing += 1

    return {
        "stage": stage,
        "part": part,
        "expected": len(manifest_ids),
        "schema_valid": schema_valid,
        "schema_invalid": schema_invalid,
        "malformed_json": malformed_json,
        "invalid_reason_counts": dict(invalid_reason_counts),
        "provider_error": provider_error,
        "provider_error_code_counts": dict(error_code_counts),
        "missing_entirely": missing,
        "unexpected_in_output": sorted(unexpected_output)[:20],
        "unexpected_in_error": sorted(unexpected_error)[:20],
        "duplicate_in_output": output_duplicates,
        "duplicate_in_error": error_duplicates,
        "malformed_no_custom_id_in_output": len(output_no_cid),
        "malformed_no_custom_id_in_error": len(error_no_cid),
        "present_in_both_output_and_error": sorted(both),
        "fingerprint_counts": dict(fingerprint_counts),
        "accounting_closes": schema_valid + schema_invalid + provider_error + missing == len(manifest_ids),
    }


def main() -> dict:
    per_part = [_score_part(stage, part) for stage, part in PARTS]

    totals = {
        "expected": sum(p["expected"] for p in per_part),
        "schema_valid": sum(p["schema_valid"] for p in per_part),
        "schema_invalid": sum(p["schema_invalid"] for p in per_part),
        "malformed_json": sum(p["malformed_json"] for p in per_part),
        "provider_error": sum(p["provider_error"] for p in per_part),
        "missing_entirely": sum(p["missing_entirely"] for p in per_part),
    }
    totals["accounting_closes"] = totals["schema_valid"] + totals["schema_invalid"] + totals["provider_error"] + totals["missing_entirely"] == totals["expected"]
    totals["schema_valid_rate_of_expected"] = totals["schema_valid"] / totals["expected"] if totals["expected"] else None
    totals["provider_error_rate_of_expected"] = totals["provider_error"] / totals["expected"] if totals["expected"] else None

    total_fp: Counter = Counter()
    total_error_codes: Counter = Counter()
    for p in per_part:
        for fp, c in p["fingerprint_counts"].items():
            total_fp[fp] += c
        for code, c in p["provider_error_code_counts"].items():
            total_error_codes[code] += c

    failed_fingerprint_count = total_fp.get(FAILED_FINGERPRINT, 0)

    result = {
        "note": "Schema-validity and provider-delivery accounting ONLY on the target G Wave-1 v2 full-replacement retrieved output -- no scientific target-G value is computed or exposed anywhere here. No fence-stripping, no repair, no retry.",
        "per_part": per_part,
        "totals": totals,
        "fingerprint_counts": dict(total_fp),
        "provider_error_code_counts": dict(total_error_codes),
        "failed_fingerprint": FAILED_FINGERPRINT,
        "failed_fingerprint_count": failed_fingerprint_count,
        "failed_fingerprint_recurred": failed_fingerprint_count > 0,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k != "per_part"}, indent=2))
    print(json.dumps({"per_part_summary": [{k: v for k, v in p.items() if k not in ("unexpected_in_output", "unexpected_in_error", "present_in_both_output_and_error")} for p in out["per_part"]]}, indent=2))
