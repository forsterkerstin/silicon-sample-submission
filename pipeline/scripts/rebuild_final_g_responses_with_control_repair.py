"""Rebuild final_first_valid_native_g_responses.csv and
final_submission_row_provenance.csv incorporating the G control-filler
repair (outputs/target_production/control_replacement_requests_v2/).

scripts/assemble_final_native_g_responses.py and
scripts/build_final_provenance_manifest.py are deliberately left unmodified:
they are the frozen, engineering-only first-valid resolution machinery, and
their own FIRST_VALID_RESPONSE_WINS ledger has no concept of "this
schema-valid response used the wrong (pre-randomization-fix) filler and
must be substituted" -- that is a scientific correction, not a technical
retry, and out of scope for that ledger by design (see
inference/target_g_retry_engine.py's module docstring).

This script instead calls those two modules' own functions unmodified to
get the pre-repair standard-track DataFrame / provenance rows, then
overrides EXACTLY the 666 (profile_id, condition_id="control") rows named
in control_replacement_requests_v2/replacement_request_manifest.csv with
the corrected, already-validated retrieved responses -- reusing the exact
same validate_response/response_key_map relabeling the frozen scripts use,
never inspecting/selecting on substantive values beyond that validity
check. Every other row (all 15 non-control interventions, the 334 retained
necktie control rows, and all 1,000 Consensus rows) passes through byte-
identical to the frozen scripts' own output.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import assemble_final_native_g_responses as base  # noqa: E402
import build_final_provenance_manifest as prov_base  # noqa: E402
import score_target_g_wave1_v2_replacement as v2_scorer  # noqa: E402
from ate.f_screen_validation import validate_response  # noqa: E402

REPAIR_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "control_replacement_requests_v2"
REPAIR_MANIFEST = REPAIR_ROOT / "replacement_request_manifest.csv"
REPAIR_JSONL = REPAIR_ROOT / "replacement_batch_input.jsonl"
REPAIR_OUTPUT = REPAIR_ROOT / "retrieved" / "batch_output.jsonl"
HISTORICAL_STANDARD_MANIFEST = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage" / "standard" / "request_manifest.csv"
EXPECTED_OVERRIDE_COUNT = 666
PROVIDER_BATCH_SOURCE_LABEL = "control_replacement_requests_v2"


def _response_key_map_by_donor() -> dict[str, dict[str, str]]:
    """donor_key -> response_key_map. Item block order (hence the Q00N ->
    target_label mapping) is a pure function of donor_key
    (inference.prompts.g_outcome_block_order), identical across every
    condition for that donor -- verified during Phase 1 read-only checks
    (0/666 mismatches sampled against any historical condition for the same
    donor). Reading it from any one historical standard-track row per donor
    is therefore exact, not an approximation."""
    out: dict[str, dict[str, str]] = {}
    with open(HISTORICAL_STANDARD_MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            donor = row["profile_id"]
            if donor not in out:
                out[donor] = json.loads(row["response_key_map"])
    return out


def _load_jsonl_by_cid(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["custom_id"])] = rec
    return out


def load_control_repair_overrides() -> dict[str, dict[str, Any]]:
    manifest_rows = list(csv.DictReader(open(REPAIR_MANIFEST, newline="", encoding="utf-8")))
    if len(manifest_rows) != EXPECTED_OVERRIDE_COUNT:
        raise RuntimeError(f"control-repair manifest has {len(manifest_rows)} rows, expected {EXPECTED_OVERRIDE_COUNT}")
    schema_by_cid = v2_scorer._load_schema_by_cid(REPAIR_JSONL)
    output_by_cid = _load_jsonl_by_cid(REPAIR_OUTPUT)
    key_map_by_donor = _response_key_map_by_donor()

    overrides: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        donor = row["donor_key"]
        cid = row["custom_id"]
        if row["condition_id"] != "control":
            raise RuntimeError(f"control-repair manifest row for {donor} is not condition_id=control: {row['condition_id']!r}")
        raw_record = output_by_cid.get(cid)
        validation = validate_response(raw_record, schema_by_cid.get(cid))
        if not validation["valid"]:
            raise RuntimeError(f"control-repair response for donor {donor} (custom_id {cid}) is not schema-valid ({validation['reason']}) -- refusing to use it")
        content = raw_record["response"]["body"]["choices"][0]["message"]["content"]
        raw = json.loads(content)
        key_map = key_map_by_donor.get(donor)
        if key_map is None:
            raise RuntimeError(f"no historical response_key_map found for donor {donor}")
        missing = set(key_map) - set(raw)
        if missing:
            raise RuntimeError(f"control-repair response for donor {donor} missing Q-key(s) {sorted(missing)}")
        items = {key_map[q]: raw[q] for q in key_map}
        overrides[donor] = {"profile_id": donor, "condition_id": "control", **items}

    if len(overrides) != EXPECTED_OVERRIDE_COUNT:
        raise RuntimeError(f"built {len(overrides)} unique donor overrides, expected {EXPECTED_OVERRIDE_COUNT}")
    return overrides


def apply_control_repair(standard_df: pd.DataFrame, overrides: dict[str, dict[str, Any]]) -> pd.DataFrame:
    df = standard_df.copy()
    mask = (df["condition_id"] == "control") & (df["profile_id"].isin(overrides))
    if int(mask.sum()) != EXPECTED_OVERRIDE_COUNT:
        raise RuntimeError(f"control repair mask matched {int(mask.sum())} rows, expected exactly {EXPECTED_OVERRIDE_COUNT}")
    item_cols = [c for c in df.columns if c not in ("profile_id", "condition_id")]
    for idx in df.index[mask]:
        donor = df.at[idx, "profile_id"]
        override_row = overrides[donor]
        missing = set(item_cols) - set(override_row)
        if missing:
            raise RuntimeError(f"override row for donor {donor} is missing item column(s) {sorted(missing)}")
        for col in item_cols:
            df.at[idx, col] = override_row[col]
    return df


def build_repaired_provenance_rows(overrides: dict[str, dict[str, Any]], manifest_rows_by_donor: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    provenance = json.loads(prov_base.PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]
    report = prov_base.completion_scorer.score_standard_stage(
        prov_base.COMPLETION_STANDARD_DIR / "retrieved" / "batch_output.jsonl",
        prov_base.COMPLETION_STANDARD_DIR / "retrieved" / "batch_error.jsonl",
        provenance,
    )
    completion_round = prov_base.completion_scorer.round_from_report(report, provenance)
    ledger = prov_base.build_attempt_ledger(additional_attempt_rounds=completion_round)
    first_valid = prov_base.assemble_first_valid(ledger)

    rows = []
    for identity, entry in first_valid.items():
        if entry["request_stage"] != "standard":
            continue
        if not entry["resolved"]:
            raise RuntimeError(f"standard identity {identity} is unresolved -- cannot build provenance manifest")
        donor = ledger[identity]["profile_id"]
        condition = ledger[identity]["condition_id"]
        if condition == "control" and donor in overrides:
            repair_row = manifest_rows_by_donor[donor]
            rows.append(
                {
                    "profile_id": donor,
                    "condition_id": condition,
                    "track": "standard",
                    "source_custom_id": repair_row["custom_id"],
                    "provider_batch_source": PROVIDER_BATCH_SOURCE_LABEL,
                    "selected_attempt": repair_row["replacement_replicate_id"],
                }
            )
        else:
            rows.append(
                {
                    "profile_id": donor,
                    "condition_id": condition,
                    "track": "standard",
                    "source_custom_id": entry["source_custom_id"],
                    "provider_batch_source": entry["provider_batch_source"],
                    "selected_attempt": entry["selected_attempt"],
                }
            )

    with open(prov_base.CONSENSUS_OUTCOMES_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "profile_id": row["profile_id"],
                    "condition_id": row["condition_id"],
                    "track": "consensus_outcomes",
                    "source_custom_id": row["custom_id"],
                    "provider_batch_source": "consensus_exact/outcomes",
                    "selected_attempt": 1,
                }
            )
    return rows


def main() -> dict:
    overrides = load_control_repair_overrides()

    standard_df = base.load_standard_responses()
    repaired_standard_df = apply_control_repair(standard_df, overrides)
    consensus_df = base.load_consensus_responses()
    df = pd.concat([repaired_standard_df, consensus_df], ignore_index=True)

    if len(df) != base.EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"assembled {len(df)} total rows, expected exactly {base.EXPECTED_TOTAL_ROWS}")
    dup = df.duplicated(subset=["profile_id", "condition_id"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicate (profile_id, condition_id) identity pair(s) in assembled universe")
    item_cols = [c for c in df.columns if c not in ("profile_id", "condition_id")]
    if len(item_cols) != base.EXPECTED_ITEMS_PER_ROW:
        raise RuntimeError(f"expected exactly {base.EXPECTED_ITEMS_PER_ROW} raw item columns, found {len(item_cols)}")
    if df[item_cols].isna().any().any():
        raise RuntimeError("assembled response universe contains missing raw item value(s)")
    per_condition_counts = df["condition_id"].value_counts()
    if len(per_condition_counts) != 17 or not (per_condition_counts == 1000).all():
        raise RuntimeError(f"expected exactly 17 conditions with 1,000 rows each, got: {per_condition_counts.to_dict()}")
    n_respondents = df["profile_id"].nunique()
    if n_respondents != 1000:
        raise RuntimeError(f"expected exactly 1,000 distinct respondents, got {n_respondents}")
    master_donors = set(pd.read_csv(base.G_MASTER_PATH)["donor_key"].astype(str))
    unknown_donors = set(df["profile_id"].astype(str)) - master_donors
    if unknown_donors:
        raise RuntimeError(f"assembled universe references unknown donor_key(s): {sorted(unknown_donors)[:5]}")

    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.OUT_CSV, index=False)
    summary = {
        "n_rows": len(df),
        "n_respondents": n_respondents,
        "n_conditions": len(per_condition_counts),
        "rows_per_condition": per_condition_counts.to_dict(),
        "n_item_columns": len(item_cols),
        "standard_rows": len(repaired_standard_df),
        "consensus_rows": len(consensus_df),
        "csv_path": str(base.OUT_CSV.relative_to(PIPELINE_ROOT)),
        "csv_sha256": base._sha256_file(base.OUT_CSV),
        "control_repair_applied": True,
        "control_repair_rows_overridden": EXPECTED_OVERRIDE_COUNT,
        "control_repair_source": str(REPAIR_ROOT.relative_to(PIPELINE_ROOT)),
    }
    base.OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_rows_by_donor = {r["donor_key"]: r for r in csv.DictReader(open(REPAIR_MANIFEST, newline="", encoding="utf-8"))}
    provenance_rows = build_repaired_provenance_rows(overrides, manifest_rows_by_donor)
    if len(provenance_rows) != 17000:
        raise RuntimeError(f"assembled provenance for {len(provenance_rows)} rows, expected exactly 17000")
    seen = {(r["profile_id"], r["condition_id"]) for r in provenance_rows}
    if len(seen) != 17000:
        raise RuntimeError("duplicate (profile_id, condition_id) pairs in provenance")
    prov_base.OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_base.OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prov_base.FIELDS)
        writer.writeheader()
        writer.writerows(provenance_rows)

    summary["provenance_rows"] = len(provenance_rows)
    summary["provenance_path"] = str(prov_base.OUT_PATH.relative_to(PIPELINE_ROOT))
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
