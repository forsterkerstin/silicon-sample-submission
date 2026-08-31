"""Companion to scripts/assemble_final_native_g_responses.py: records, for
every one of the 17,000 (profile_id, condition_id) identities, EXACTLY
which physical production response it was locked from -- source custom_id,
provider batch source, and selected attempt number for the standard track;
the OUTCOMES-stage custom_id for the Consensus track. Engineering
provenance only, no scientific response values. This is what makes every
final submitted row traceable back to its real, already-reconciled
production response.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import score_target_g_wave1_completion as completion_scorer  # noqa: E402
from inference.target_g_retry_engine import build_attempt_ledger, assemble_first_valid  # noqa: E402

COMPLETION_STANDARD_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "standard"
CONSENSUS_OUTCOMES_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "outcomes"
PROVENANCE_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "attempt_provenance.json"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "final_submission_row_provenance.csv"

FIELDS = ["profile_id", "condition_id", "track", "source_custom_id", "provider_batch_source", "selected_attempt"]


def main() -> dict:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]
    report = completion_scorer.score_standard_stage(
        COMPLETION_STANDARD_DIR / "retrieved" / "batch_output.jsonl",
        COMPLETION_STANDARD_DIR / "retrieved" / "batch_error.jsonl",
        provenance,
    )
    completion_round = completion_scorer.round_from_report(report, provenance)
    ledger = build_attempt_ledger(additional_attempt_rounds=completion_round)
    first_valid = assemble_first_valid(ledger)

    rows = []
    for identity, entry in first_valid.items():
        if entry["request_stage"] != "standard":
            continue
        if not entry["resolved"]:
            raise RuntimeError(f"standard identity {identity} is unresolved -- cannot build provenance manifest")
        rows.append(
            {
                "profile_id": ledger[identity]["profile_id"],
                "condition_id": ledger[identity]["condition_id"],
                "track": "standard",
                "source_custom_id": entry["source_custom_id"],
                "provider_batch_source": entry["provider_batch_source"],
                "selected_attempt": entry["selected_attempt"],
            }
        )

    with open(CONSENSUS_OUTCOMES_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
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

    if len(rows) != 17000:
        raise RuntimeError(f"assembled provenance for {len(rows)} rows, expected exactly 17000")
    seen = {(r["profile_id"], r["condition_id"]) for r in rows}
    if len(seen) != 17000:
        raise RuntimeError(f"duplicate (profile_id, condition_id) pairs in provenance: {len(rows)} rows but only {len(seen)} distinct pairs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {"n_rows": len(rows), "out_path": str(OUT_PATH.relative_to(PIPELINE_ROOT))}


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
