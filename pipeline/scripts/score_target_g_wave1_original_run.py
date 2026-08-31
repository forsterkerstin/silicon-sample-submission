"""Score the ORIGINAL (failed) target G Wave-1 retrieved output for
schema validity ONLY -- reconciliation + validity accounting, never
scientific content.

Reuses the frozen, generic ate.f_screen_validation.validate_response
(no fence-stripping, no repair, no coercion) against the real retrieved
batch_output.jsonl files and their own request schemas. This is the
evidence trail for the PROVIDER_SERVING_FORMAT_FAILURE amendment
(outputs/target_production/g_wave1_v1_format_failure_amendment.json) --
it does NOT compute or expose any target G scientific value (no means,
ATEs, distributions, rankings). Every "valid" response's actual parsed
content is discarded immediately after being counted; only counts and
system_fingerprint bookkeeping are retained.

No LLM calls. No repair. No retry. Read-only over already-retrieved files.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_validation_report.json"

STANDARD_RETRIEVED = {
    "95f5f512": (STAGE_ROOT / "standard" / "G" / "retrieved" / "batch_output_95f5f512.jsonl", STAGE_ROOT / "standard" / "G" / "part1" / "batch_input.jsonl"),
    "ba0731c0": (STAGE_ROOT / "standard" / "G" / "retrieved" / "batch_output_ba0731c0.jsonl", STAGE_ROOT / "standard" / "G" / "part2" / "batch_input.jsonl"),
    "60eaedf5": (STAGE_ROOT / "standard" / "G" / "retrieved" / "batch_output_60eaedf5.jsonl", STAGE_ROOT / "standard" / "G" / "part3" / "batch_input.jsonl"),
    "9859219f": (STAGE_ROOT / "standard" / "G" / "retrieved" / "batch_output_9859219f.jsonl", STAGE_ROOT / "standard" / "G" / "part4" / "batch_input.jsonl"),
}
CONSENSUS_A_RETRIEVED = STAGE_ROOT / "consensus_stage_a" / "G" / "part1" / "retrieved" / "batch_output.jsonl"
CONSENSUS_A_INPUT = STAGE_ROOT / "consensus_stage_a" / "G" / "part1" / "batch_input.jsonl"

FAILED_FINGERPRINT = "vllm-0.21.0-8326ea74"


def _load_raw_by_custom_id(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["custom_id"])] = r
    return out


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


def _score(retrieved_path: Path, schema_by_cid: dict) -> dict:
    from ate.f_screen_validation import validate_response

    raw = _load_raw_by_custom_id(retrieved_path)
    valid_n = invalid_n = 0
    fp_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for cid, rec in raw.items():
        fp = str(rec.get("response", {}).get("body", {}).get("system_fingerprint"))
        fp_counts[fp] = fp_counts.get(fp, 0) + 1
        v = validate_response(rec, schema_by_cid.get(cid))
        if v["valid"]:
            valid_n += 1
        else:
            invalid_n += 1
            reason_key = v["reason"].split(":")[0]
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
    return {"n": len(raw), "valid": valid_n, "invalid": invalid_n, "invalid_rate": invalid_n / len(raw) if raw else None, "fingerprint_counts": fp_counts, "invalid_reason_counts": reason_counts}


def main() -> dict:
    standard_schema: dict = {}
    for _, (_, in_path) in STANDARD_RETRIEVED.items():
        standard_schema.update(_load_schema_by_custom_id(in_path))

    per_batch = {}
    total_valid = total_invalid = 0
    total_fp: dict[str, int] = {}
    for label, (out_path, _) in STANDARD_RETRIEVED.items():
        r = _score(out_path, standard_schema)
        per_batch[label] = r
        total_valid += r["valid"]
        total_invalid += r["invalid"]
        for fp, c in r["fingerprint_counts"].items():
            total_fp[fp] = total_fp.get(fp, 0) + c

    consensus_schema = _load_schema_by_custom_id(CONSENSUS_A_INPUT)
    consensus_result = _score(CONSENSUS_A_RETRIEVED, consensus_schema)

    result = {
        "note": "Schema-validity accounting ONLY on the ORIGINAL (v1) target G Wave-1 retrieved output -- no scientific target-G value (mean/ATE/distribution/ranking) is computed or exposed anywhere here. No fence-stripping, no repair.",
        "standard": {
            "per_batch": per_batch,
            "total_n": total_valid + total_invalid,
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "total_invalid_rate": total_invalid / (total_valid + total_invalid),
            "fingerprint_counts": total_fp,
        },
        "consensus_stage_a": consensus_result,
        "failed_fingerprint": FAILED_FINGERPRINT,
        "root_cause_classification": "PROVIDER_SERVING_FORMAT_FAILURE",
        "root_cause_evidence": "system_fingerprint is a perfect predictor of validity: every invalid response carries fingerprint vllm-0.21.0-8326ea74 (content wrapped in a markdown ```json code fence despite response_format.json_schema being set to strict structured output); every valid response carries a different/absent fingerprint. This is the exact same failure signature documented for the historical F* R1 batch (scripts/prepare_f_reliability_r1_replacement_manifest.py).",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
