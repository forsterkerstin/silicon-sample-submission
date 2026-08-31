"""Combine the four logical Consensus-exact stages' reconciliation reports
(scripts/score_consensus_exact_stage.py --out files) into overall pipeline
state: per stage, which donors are resolved (first-valid) vs. still
pending vs. exhausted; and which donors are FINAL-ROW ELIGIBLE (resolved on
ALL FOUR stages).

Engineering only -- reads each stage's donor_status dict (SCHEMA_VALID /
SCHEMA_INVALID / PROVIDER_ERROR / NOT_ATTEMPTED per donor) and combines via
inference.consensus_exact_retry_engine.build_stage_ledger/
final_row_eligible_donors. Never reads a scientific response value.

Not runnable to real completion until STEP_1 has been submitted/retrieved
at least once (this script accepts N rounds per stage via repeated
--stepN-report flags, in attempt order).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.consensus_exact_retry_engine import build_stage_ledger, final_row_eligible_donors, pending_donors, resolved_donors  # noqa: E402

STEP_NAMES = ("step1", "step2", "step3", "outcomes")
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "pipeline_state.json"


def _rounds_from_reports(report_paths: list[Path]) -> list[dict]:
    rounds = []
    for i, path in enumerate(report_paths, start=1):
        report = json.loads(path.read_text(encoding="utf-8"))
        rounds.append({"attempt_number": i, "donor_status": report["donor_status"]})
    return rounds


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for step in STEP_NAMES:
        parser.add_argument(f"--{step}-universe", type=Path, help=f"json file: list of donor_keys eligible to attempt {step}")
        parser.add_argument(f"--{step}-report", action="append", default=[], type=Path, help=f"one score_consensus_exact_stage.py --out file per attempt round for {step}, in order")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    ledgers = {}
    for step in STEP_NAMES:
        universe_path = getattr(args, f"{step}_universe")
        report_paths = getattr(args, f"{step}_report")
        if universe_path is None:
            continue
        universe = json.loads(universe_path.read_text(encoding="utf-8"))
        rounds = _rounds_from_reports(report_paths)
        ledgers[step] = build_stage_ledger(universe, rounds)

    result = {"stages": {}, "final_row_eligible_donor_count": 0}
    for step, ledger in ledgers.items():
        result["stages"][step] = {
            "universe": len(ledger),
            "resolved": len(resolved_donors(ledger)),
            "pending": len(pending_donors(ledger)),
            "exhausted": sum(1 for e in ledger.values() if not e["resolved"] and e["attempt_count"] >= 3),
        }
    if all(step in ledgers for step in STEP_NAMES):
        eligible = final_row_eligible_donors(ledgers["step1"], ledgers["step2"], ledgers["step3"], ledgers["outcomes"])
        result["final_row_eligible_donor_count"] = len(eligible)
        result["final_row_eligible_donors"] = sorted(eligible)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k != "final_row_eligible_donors"}, indent=2))
