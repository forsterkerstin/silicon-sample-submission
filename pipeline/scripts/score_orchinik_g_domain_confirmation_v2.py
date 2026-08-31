"""Strict, zero-tolerance post-inference validator for the Orchinik G-v2
domain-confirmation manifests (schema/serving validation ONLY -- never
scientific analysis or model selection). Reuses the frozen, generic
ate.f_screen_validation.validate_response / reconciliation_report (no
fence-stripping, no repair, no coercion, no retry).

Scope: 2,545 requests for google/gemma-4-31B-it and 2,545 requests for
deepseek-ai/DeepSeek-V4-Pro-0813 (outputs/domain_validation/
orchinik_g_domain_confirmation_v2/). Reports, per model and combined:
expected/missing/unexpected/duplicate counts, schema_valid/schema_invalid/
malformed_json counts, and a full system_fingerprint breakdown (so a
recurrence of the known-bad vllm-0.21.0-8326ea74 serving-format defect is
immediately visible even if it does not reach the historical 92-98% rate).

This script only reads already-retrieved files. It never submits, retries,
or repairs anything.

Usage:
    python scripts/score_orchinik_g_domain_confirmation_v2.py \
        --gemma-output outputs/domain_validation/orchinik_g_domain_confirmation_v2/google_gemma-4-31B-it/retrieved/batch_output.jsonl \
        --deepseek-output outputs/domain_validation/orchinik_g_domain_confirmation_v2/deepseek-ai_DeepSeek-V4-Pro-0813/retrieved/batch_output.jsonl
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

V2_ROOT = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation_v2"
FAILED_FINGERPRINT = "vllm-0.21.0-8326ea74"
MODEL_DIRS = {
    "google/gemma-4-31B-it": "google_gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V4-Pro-0813": "deepseek-ai_DeepSeek-V4-Pro-0813",
}
EXPECTED_PER_MODEL = 2545


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


def _score_model(model: str, retrieved_path: Path) -> dict:
    from ate.f_screen_validation import reconciliation_report, validate_response

    model_dir = V2_ROOT / MODEL_DIRS[model]
    with open(model_dir / "request_manifest.csv", newline="", encoding="utf-8") as f:
        manifest_ids = {row["custom_id"] for row in csv.DictReader(f)}
    schema_by_cid = _load_schema_by_custom_id(model_dir / "batch_input.jsonl")

    raw_records = []
    with open(retrieved_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    report = reconciliation_report(manifest_ids, raw_records)
    raw_by_cid = {r["custom_id"]: r for r in raw_records if isinstance(r, dict) and "custom_id" in r}

    schema_valid = schema_invalid = malformed_json = 0
    fingerprint_counts: Counter = Counter()
    failed_fingerprint_records = []
    for cid in sorted(manifest_ids):
        rec = raw_by_cid.get(cid)
        fp = None
        if rec is not None:
            fp = rec.get("response", {}).get("body", {}).get("system_fingerprint")
        fingerprint_counts[fp] += 1
        v = validate_response(rec, schema_by_cid.get(cid))
        if v["valid"]:
            schema_valid += 1
        else:
            schema_invalid += 1
            if v["reason"].startswith("malformed_json"):
                malformed_json += 1
        if fp == FAILED_FINGERPRINT:
            failed_fingerprint_records.append({"custom_id": cid, "valid": v["valid"], "reason": v["reason"]})

    return {
        "model": model,
        "expected": len(manifest_ids),
        "retrieved": len(raw_records),
        "missing": len(report["missing_entirely"]),
        "unexpected": len(report["unexpected"]),
        "duplicates": len(report["duplicate"]),
        "schema_valid": schema_valid,
        "schema_invalid": schema_invalid,
        "malformed_json": malformed_json,
        "system_fingerprint_breakdown": {str(k): v for k, v in fingerprint_counts.items()},
        "failed_fingerprint_seen_count": len(failed_fingerprint_records),
        "failed_fingerprint_records": failed_fingerprint_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-output", type=Path, required=True)
    parser.add_argument("--deepseek-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=V2_ROOT / "post_inference_validation_report.json")
    args = parser.parse_args()

    gemma = _score_model("google/gemma-4-31B-it", args.gemma_output)
    deepseek = _score_model("deepseek-ai/DeepSeek-V4-Pro-0813", args.deepseek_output)

    total_expected = gemma["expected"] + deepseek["expected"]
    total_missing = gemma["missing"] + deepseek["missing"]
    total_unexpected = gemma["unexpected"] + deepseek["unexpected"]
    total_duplicates = gemma["duplicates"] + deepseek["duplicates"]
    total_schema_valid = gemma["schema_valid"] + deepseek["schema_valid"]
    total_schema_invalid = gemma["schema_invalid"] + deepseek["schema_invalid"]
    total_malformed = gemma["malformed_json"] + deepseek["malformed_json"]

    all_valid = (
        total_expected == 2 * EXPECTED_PER_MODEL
        and gemma["expected"] == EXPECTED_PER_MODEL
        and deepseek["expected"] == EXPECTED_PER_MODEL
        and total_missing == 0
        and total_unexpected == 0
        and total_duplicates == 0
        and total_schema_invalid == 0
    )

    result = {
        "expected_per_model": EXPECTED_PER_MODEL,
        "total_expected": total_expected,
        "missing": total_missing,
        "unexpected": total_unexpected,
        "duplicates": total_duplicates,
        "schema_valid": total_schema_valid,
        "schema_invalid": total_schema_invalid,
        "malformed_json": total_malformed,
        "all_valid": all_valid,
        "google/gemma-4-31B-it": gemma,
        "deepseek-ai/DeepSeek-V4-Pro-0813": deepseek,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_view = {k: v for k, v in result.items() if k not in ("google/gemma-4-31B-it", "deepseek-ai/DeepSeek-V4-Pro-0813")}
    summary_view["gemma_system_fingerprint_breakdown"] = gemma["system_fingerprint_breakdown"]
    summary_view["deepseek_system_fingerprint_breakdown"] = deepseek["system_fingerprint_breakdown"]
    print(json.dumps(summary_view, indent=2))
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
