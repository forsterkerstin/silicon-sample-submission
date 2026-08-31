"""Offline first-valid production-response assembler for target G Wave-1
STANDARD identities ONLY (OFFLINE ONLY, engineering only). Combines the
retained-valid attempt-1 responses (base ledger, already-committed real
data) with the standard completion batch's responses (freshly scored by
scripts/score_target_g_wave1_completion.py's score_standard_stage) using
the already-frozen FIRST_VALID_RESPONSE_WINS rule, via
inference.target_g_retry_engine.build_attempt_ledger/assemble_first_valid.

STANDARD ONLY. Never requires, reads, or references the abandoned
Consensus-A completion branch -- see outputs/target_production/
consensus_protocol_amendment.json. The corrected Consensus pipeline's own
assembly lives in scripts/assemble_consensus_exact_pipeline_state.py.

Reports ONLY engineering/accounting information -- intended identities,
first-valid-assembled count, identities still lacking a valid production
response (and how many are eligible for Attempt 3), attempt-number
provenance. Never summarizes, computes, or exposes any survey answer or ATE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PIPELINE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

from inference.target_g_retry_engine import assemble_first_valid, build_attempt_ledger  # noqa: E402
import score_target_g_wave1_completion as reconciler  # noqa: E402

STAGE = "standard"
PROVENANCE_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "attempt_provenance.json"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "standard_first_valid_assembly_report.json"


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-output", type=Path, required=True)
    parser.add_argument("--standard-error", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))["per_custom_id"]
    standard_report = reconciler.score_standard_stage(args.standard_output, args.standard_error, provenance)
    rounds = reconciler.round_from_report(standard_report, provenance)

    ledger = build_attempt_ledger(rounds)
    standard_ledger = {identity: entry for identity, entry in ledger.items() if entry["request_stage"] == STAGE}
    provenance_report = assemble_first_valid(standard_ledger)

    resolved = [p for p in provenance_report.values() if p["resolved"]]
    unresolved = [p for p in provenance_report.values() if not p["resolved"]]
    result = {
        "stage": STAGE,
        "intended_identities": len(standard_ledger),
        "first_valid_assembled": len(resolved),
        "still_unresolved": len(unresolved),
        "resolved_by_selected_attempt": {str(n): sum(1 for p in resolved if p["selected_attempt"] == n) for n in (1, 2, 3)},
        "unresolved_by_next_attempt_number": {str(n): sum(1 for p in unresolved if p.get("next_attempt_number") == n) for n in (1, 2, 3, None)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
