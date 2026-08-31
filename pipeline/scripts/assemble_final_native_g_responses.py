"""Assemble the complete, real, first-valid target-G native response
universe (17,000 rows: 1,000 respondents x 17 conditions) from already-
retrieved production data only. Read-only over already-retrieved files.
No inference, no submission, no target human outcome access.

This is the ONE place in the final-materialization pipeline that reads
actual native G response VALUES (raw per-item answers), because building
the final Tier-1 file requires them. It never inspects target HUMAN
outcomes, never repairs/coerces/retries a response, and never selects
which identities are included on any basis other than the already-
established (engineering-only) first-valid resolution status.

Standard track (16,000 = 1,000 respondents x 16 conditions: control + 15
non-Consensus interventions): resolved via the SAME frozen ledger machinery
already used for engineering reconciliation --
inference.target_g_retry_engine.build_attempt_ledger /
assemble_first_valid, fed with the real completion round via
scripts/score_target_g_wave1_completion.py's own
score_standard_stage/round_from_report (no duplicated reconciliation
logic). For each resolved identity, the actual response is read from
whichever physical retrieved file the ledger says is its first-valid
source (attempt 1 -> wave1_g_v2_replacement/submission/standard/partN/,
attempt 2 -> wave1_g_completion/standard/) and relabeled from its
manifest row's response_key_map (Q00N -> target_label).

Consensus track (1,000 = 1,000 respondents x 1 condition): the real
OUTCOMES-stage response only (outputs/target_production/consensus_exact/
outcomes/) -- STEP_1/STEP_2/STEP_3 are pure chaining scaffolding, never an
additional submitted observation (see
inference/consensus_benchmark_exact.py's module docstring).

Fails closed: raises if any of the 17,000 intended identities is
unresolved, duplicated, or has a response that does not validate against
its own frozen request-time schema. Never imputes, drops, or coerces.
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

import score_target_g_wave1_completion as completion_scorer  # noqa: E402
import score_target_g_wave1_v2_replacement as v2_scorer  # noqa: E402
from ate.f_screen_validation import validate_response  # noqa: E402
from inference.target_g_retry_engine import build_attempt_ledger, assemble_first_valid  # noqa: E402
from inference.together_batch import G_MASTER_PATH  # noqa: E402

WAVE1_V2_SUBMISSION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "submission"
COMPLETION_STANDARD_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "standard"
CONSENSUS_OUTCOMES_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "outcomes"
PROVENANCE_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "attempt_provenance.json"

OUT_DIR = PIPELINE_ROOT / "outputs" / "target_production"
OUT_CSV = OUT_DIR / "final_first_valid_native_g_responses.csv"
OUT_SUMMARY = OUT_DIR / "final_first_valid_native_g_responses_summary.json"

EXPECTED_STANDARD_IDENTITIES = 16000
EXPECTED_CONSENSUS_IDENTITIES = 1000
EXPECTED_TOTAL_ROWS = 17000
EXPECTED_ITEMS_PER_ROW = 44


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl_by_cid(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "custom_id" in rec:
                out[str(rec["custom_id"])] = rec
    return out


def _relabel(raw: dict[str, Any], response_key_map: dict[str, str]) -> dict[str, Any]:
    missing = set(response_key_map) - set(raw)
    if missing:
        raise RuntimeError(f"response missing Q-key(s) {sorted(missing)} required by its own response_key_map")
    return {response_key_map[q]: raw[q] for q in response_key_map}


def _manifest_rows_by_cid(manifest_path: Path) -> dict[str, dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["custom_id"]: row for row in csv.DictReader(f)}


class _StandardPhysicalResolver:
    """Resolves (custom_id, provider_batch_source) -> real relabeled item
    values, loading each physical retrieved file at most once."""

    def __init__(self) -> None:
        self._output_cache: dict[str, dict[str, dict]] = {}
        self._schema_cache: dict[str, dict[str, dict]] = {}
        self._manifest_cache: dict[str, dict[str, dict]] = {}

    def _paths_for_source(self, provider_batch_source: str) -> tuple[Path, Path, Path]:
        if provider_batch_source == "wave1_g_completion/standard":
            base = COMPLETION_STANDARD_DIR
        elif provider_batch_source.startswith("standard/part"):
            part = provider_batch_source.split("/", 1)[1]
            base = WAVE1_V2_SUBMISSION_ROOT / "standard" / part
        else:
            raise RuntimeError(f"unrecognized standard-track provider_batch_source: {provider_batch_source!r}")
        return base / "batch_input.jsonl", base / "retrieved" / "batch_output.jsonl", base / "request_manifest.csv"

    def resolve(self, custom_id: str, provider_batch_source: str) -> dict[str, Any]:
        if provider_batch_source not in self._output_cache:
            input_path, output_path, manifest_path = self._paths_for_source(provider_batch_source)
            self._schema_cache[provider_batch_source] = v2_scorer._load_schema_by_cid(input_path)
            self._output_cache[provider_batch_source] = _load_jsonl_by_cid(output_path)
            self._manifest_cache[provider_batch_source] = _manifest_rows_by_cid(manifest_path)

        output_by_cid = self._output_cache[provider_batch_source]
        schema_by_cid = self._schema_cache[provider_batch_source]
        manifest_by_cid = self._manifest_cache[provider_batch_source]

        if custom_id not in output_by_cid:
            raise RuntimeError(f"custom_id {custom_id} not found in resolved source {provider_batch_source!r} output")
        raw_record = output_by_cid[custom_id]
        validation = validate_response(raw_record, schema_by_cid.get(custom_id))
        if not validation["valid"]:
            raise RuntimeError(f"custom_id {custom_id} (source {provider_batch_source!r}) is not schema-valid ({validation['reason']}) -- contradicts prior reconciliation, refusing to use it")
        content = raw_record["response"]["body"]["choices"][0]["message"]["content"]
        raw = json.loads(content)

        manifest_row = manifest_by_cid[custom_id]
        response_key_map = json.loads(manifest_row["response_key_map"])
        items = _relabel(raw, response_key_map)
        return {"profile_id": manifest_row["profile_id"], "condition_id": manifest_row["condition_id"], **items}


def load_standard_responses() -> pd.DataFrame:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]
    report = completion_scorer.score_standard_stage(
        COMPLETION_STANDARD_DIR / "retrieved" / "batch_output.jsonl",
        COMPLETION_STANDARD_DIR / "retrieved" / "batch_error.jsonl",
        provenance,
    )
    completion_round = completion_scorer.round_from_report(report, provenance)
    ledger = build_attempt_ledger(additional_attempt_rounds=completion_round)
    first_valid = assemble_first_valid(ledger)

    standard_entries = {identity: entry for identity, entry in first_valid.items() if entry["request_stage"] == "standard"}
    if len(standard_entries) != EXPECTED_STANDARD_IDENTITIES:
        raise RuntimeError(f"expected exactly {EXPECTED_STANDARD_IDENTITIES} standard identities, found {len(standard_entries)}")
    unresolved = [identity for identity, entry in standard_entries.items() if not entry["resolved"]]
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} standard identities are unresolved -- cannot assemble final response universe (first unresolved: {sorted(unresolved)[:5]})")

    resolver = _StandardPhysicalResolver()
    rows = []
    for identity, entry in standard_entries.items():
        row = resolver.resolve(entry["source_custom_id"], entry["provider_batch_source"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_STANDARD_IDENTITIES:
        raise RuntimeError(f"assembled {len(df)} standard rows, expected {EXPECTED_STANDARD_IDENTITIES}")
    return df


def load_consensus_responses() -> pd.DataFrame:
    manifest_by_cid = _manifest_rows_by_cid(CONSENSUS_OUTCOMES_DIR / "request_manifest.csv")
    schema_by_cid = v2_scorer._load_schema_by_cid(CONSENSUS_OUTCOMES_DIR / "batch_input.jsonl")
    output_by_cid = _load_jsonl_by_cid(CONSENSUS_OUTCOMES_DIR / "retrieved" / "batch_output.jsonl")

    if len(manifest_by_cid) != EXPECTED_CONSENSUS_IDENTITIES:
        raise RuntimeError(f"Consensus OUTCOMES manifest has {len(manifest_by_cid)} rows, expected {EXPECTED_CONSENSUS_IDENTITIES}")

    rows = []
    for cid, manifest_row in manifest_by_cid.items():
        if cid not in output_by_cid:
            raise RuntimeError(f"Consensus OUTCOMES custom_id {cid} is unresolved -- cannot assemble final response universe")
        raw_record = output_by_cid[cid]
        validation = validate_response(raw_record, schema_by_cid.get(cid))
        if not validation["valid"]:
            raise RuntimeError(f"Consensus OUTCOMES response for custom_id {cid} is not schema-valid ({validation['reason']})")
        content = raw_record["response"]["body"]["choices"][0]["message"]["content"]
        raw = json.loads(content)
        response_key_map = json.loads(manifest_row["response_key_map"])
        items = _relabel(raw, response_key_map)
        rows.append({"profile_id": manifest_row["profile_id"], "condition_id": manifest_row["condition_id"], **items})

    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_CONSENSUS_IDENTITIES:
        raise RuntimeError(f"assembled {len(df)} Consensus rows, expected {EXPECTED_CONSENSUS_IDENTITIES}")
    if not (df["condition_id"] == "Consensus").all():
        raise RuntimeError("Consensus OUTCOMES manifest contains a non-Consensus condition_id")
    return df


def main() -> dict:
    standard_df = load_standard_responses()
    consensus_df = load_consensus_responses()
    df = pd.concat([standard_df, consensus_df], ignore_index=True)

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f"assembled {len(df)} total rows, expected exactly {EXPECTED_TOTAL_ROWS}")
    dup = df.duplicated(subset=["profile_id", "condition_id"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicate (profile_id, condition_id) identity pair(s) in assembled universe")

    item_cols = [c for c in df.columns if c not in ("profile_id", "condition_id")]
    if len(item_cols) != EXPECTED_ITEMS_PER_ROW:
        raise RuntimeError(f"expected exactly {EXPECTED_ITEMS_PER_ROW} raw item columns, found {len(item_cols)}: {sorted(item_cols)}")
    if df[item_cols].isna().any().any():
        raise RuntimeError("assembled response universe contains missing raw item value(s)")

    per_condition_counts = df["condition_id"].value_counts()
    if len(per_condition_counts) != 17 or not (per_condition_counts == 1000).all():
        raise RuntimeError(f"expected exactly 17 conditions with 1,000 rows each, got: {per_condition_counts.to_dict()}")
    n_respondents = df["profile_id"].nunique()
    if n_respondents != 1000:
        raise RuntimeError(f"expected exactly 1,000 distinct respondents, got {n_respondents}")

    master_donors = set(pd.read_csv(G_MASTER_PATH)["donor_key"].astype(str))
    unknown_donors = set(df["profile_id"].astype(str)) - master_donors
    if unknown_donors:
        raise RuntimeError(f"assembled universe references unknown donor_key(s): {sorted(unknown_donors)[:5]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    summary = {
        "n_rows": len(df),
        "n_respondents": n_respondents,
        "n_conditions": len(per_condition_counts),
        "rows_per_condition": per_condition_counts.to_dict(),
        "n_item_columns": len(item_cols),
        "standard_rows": len(standard_df),
        "consensus_rows": len(consensus_df),
        "csv_path": str(OUT_CSV.relative_to(PIPELINE_ROOT)),
        "csv_sha256": _sha256_file(OUT_CSV),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
