"""Record (OFFLINE GOVERNANCE ONLY) that the already-submitted target-F
Wave-1 batches were generated under the earlier candidate-primary (M2)
plan and are unused for S2's final predictions -- S2 has zero target-F
dependence.

This writes a NEW, separate provenance note; it does NOT modify
outputs/target_production/target_production_submission_state.json (the
real submission ledger) in any way -- that file's history (including the
target-F Wave-1 submissions) is preserved exactly as-is. No retry, no F
Consensus Stage B, no use of F results in any scientific method-selection
or final-prediction path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "target_production_submission_state.json"
OUT_PATH = PIPELINE_ROOT / "outputs" / "validation" / "target_f_wave1_reclassified_provenance.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    ledger_before = LEDGER_PATH.read_bytes()
    ledger = json.loads(ledger_before)

    f_batch_ids = []
    for phase, info in ledger.get("phases", {}).items():
        for s in info.get("submissions", []):
            batch = s.get("submit_result", {}).get("batch", {})
            # F standard partitions are 13,000 requests; F consensus_stage_a is 500 -- distinguishes F from G rows
            # in this ledger the same way scripts/target_wave1_status_report.py's role inference does.
            request_count = s.get("request_count")
            is_f = (phase == "standard" and request_count == 13000) or (phase == "consensus_stage_a" and request_count == 500)
            if is_f:
                f_batch_ids.append({"phase": phase, "batch_id": batch.get("id"), "request_count": request_count})

    note = {
        "note": "OFFLINE GOVERNANCE RECORD ONLY -- does not modify the real target-production ledger, does not retry/resubmit anything, does not retrieve target F output",
        "final_method": "S2_MCONST_GSHAPE",
        "target_f_dependence_of_final_method": False,
        "target_f_wave1_batches": f_batch_ids,
        "target_f_wave1_batch_count": len(f_batch_ids),
        "status_per_batch": "GENERATED_UNDER_SUPERSEDED_CANDIDATE_METHOD",
        "usage_for_final_s2_predictions": "UNUSED_FOR_FINAL_S2_PREDICTIONS",
        "prohibited_actions": [
            "retry target-F Wave-1 requests",
            "submit F Consensus Stage B",
            "use target-F predictions in S2",
            "use target-F results to modify S2",
            "make S2 contingent on target-F completion",
        ],
        "future_archival_retrieval_note": "if these raw F outputs are later retrieved for archival purposes only, that retrieval must not feed any scientific method-selection or final-prediction path",
        "ledger_path": str(LEDGER_PATH.relative_to(PIPELINE_ROOT)),
        "ledger_sha256_at_time_of_this_note": _sha256_file(LEDGER_PATH),
        "ledger_untouched_by_this_script": True,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")

    ledger_after = LEDGER_PATH.read_bytes()
    if ledger_before != ledger_after:
        raise RuntimeError("target-production ledger was unexpectedly modified while writing this provenance note")

    note["provenance_note_sha256"] = _sha256_file(OUT_PATH)
    return note


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
