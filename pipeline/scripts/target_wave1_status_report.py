"""Target Wave-1 status report (OFFLINE, READ-ONLY).

Builds a status table for the 14 already-submitted target Wave-1 batches
purely from the local submission ledger
(outputs/target_production/target_production_submission_state.json) and the
stage-pure partition summary
(outputs/target_production/wave1/by_stage/summary.json). Makes NO network
calls -- this environment has no TOGETHER_API_KEY, so batch STATUS/
COMPLETED_AT reflect only what was recorded at submission time, not live
state. Also prints the exact read-only `together_batch.py status` commands
needed to refresh status with real credentials.

Never submits, retries, or modifies the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "target_production_submission_state.json"
STAGE_SUMMARY_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "summary.json"


def _role_from_request_count(phase: str, request_count: int) -> str:
    # G standard partitions are 4,000 requests each; G consensus_stage_a is 1,000.
    # F standard partitions are 13,000 requests each; F consensus_stage_a is 500.
    if phase == "consensus_stage_a":
        return "G" if request_count == 1000 else "F"
    return "G" if request_count == 4000 else "F"


def build_status_table() -> dict:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    stage_summary = json.loads(STAGE_SUMMARY_PATH.read_text(encoding="utf-8"))

    rows = []
    # Only the two original Wave-1 phases -- other phases in the same ledger
    # (e.g. the later G-v2 smoke/full-replacement phases) are separate
    # submissions tracked by their own reports, not part of this one.
    for phase in ("standard", "consensus_stage_a"):
        info = ledger["phases"][phase]
        for s in info["submissions"]:
            batch = s["submit_result"]["batch"]
            role = _role_from_request_count(phase, s["request_count"])
            rows.append(
                {
                    "role": role,
                    "stage": phase,
                    "request_count": s["request_count"],
                    "batch_id": batch["id"],
                    "status_last_recorded": batch["status"],
                    "created_at": batch["created_at"],
                    "completed_at": batch["completed_at"],
                    "input_file_id": batch["input_file_id"],
                }
            )
    rows.sort(key=lambda r: (r["role"], r["stage"], r["created_at"]))

    # Part-number labels within a (role, stage) group of equal-sized
    # partitions are NOT recoverable from the ledger alone -- it records
    # batch/file ids and timestamps, not which local by_stage/part<N>
    # jsonl each batch_id was built from. Mark that honestly rather than
    # guessing an order.
    for role in ("G", "F"):
        for stage in ("standard", "consensus_stage_a"):
            group = [r for r in rows if r["role"] == role and r["stage"] == stage]
            n = len(group)
            for r in group:
                r["part_label"] = "part1" if n == 1 else f"ambiguous (1 of {n} same-sized {role} {stage} partitions -- ledger does not record which by_stage/partN jsonl this batch_id came from)"

    totals_by_status: dict[str, int] = {}
    for r in rows:
        totals_by_status[r["status_last_recorded"]] = totals_by_status.get(r["status_last_recorded"], 0) + 1

    consensus_stage_a = {
        "G": next(r for r in rows if r["role"] == "G" and r["stage"] == "consensus_stage_a"),
        "F": next(r for r in rows if r["role"] == "F" and r["stage"] == "consensus_stage_a"),
    }

    status_commands = [f"python scripts/together_batch.py status --batch-id {r['batch_id']}  # {r['role']} {r['stage']} {r['part_label']}, {r['request_count']} requests" for r in rows]

    return {
        "note": "STATUS/COMPLETED_AT below are the LAST RECORDED VALUES AT SUBMISSION TIME ONLY -- this environment has no TOGETHER_API_KEY, so no live status query was made. Run the printed commands with working credentials to refresh.",
        "rows": rows,
        "totals_by_status_last_recorded": totals_by_status,
        "total_partitions": len(rows),
        "total_requests": sum(r["request_count"] for r in rows),
        "consensus_stage_a": consensus_stage_a,
        "status_query_commands": status_commands,
    }


def main() -> dict:
    result = build_status_table()
    out_dir = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_status_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wave1_batch_status_last_recorded.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
