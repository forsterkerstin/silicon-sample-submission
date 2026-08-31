"""Freeze per-custom_id attempt-number provenance for the current target G
Wave-1 completion manifests (OFFLINE ONLY).

Writes outputs/target_production/wave1_g_completion/attempt_provenance.json:
{custom_id: {"intended_identity", "request_stage", "attempt_number",
"smoke_only"}}. This is pure engineering/accounting metadata -- no
scientific response value is read or written anywhere here.

Exists because the completion manifest's own `replicate_id` column is a
WIRE-level discriminator, not always equal to the true production
attempt_number (see inference.target_g_retry_engine.build_completion_requests's
docstring: a smoke-only identity's first production attempt uses
wire_replicate_id=2 to avoid colliding with the smoke's already-submitted
replicate_id=1 custom_id, while its true attempt_number is 1). Recomputing
the full ledger (~5 minutes: re-validates all 16,990 attempt-1 responses)
is done ONCE here so downstream reconciliation/scoring never has to pay
that cost again -- it just reads this small frozen lookup file.

Does not touch the already-frozen completion manifests/jsonl files (their
SHA256 is pinned in inference/target_g_completion_guard.py and must never
change) or any scientific target-G artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.target_g_retry_engine import STAGES, build_attempt_ledger, identities_pending_next_attempt  # noqa: E402

COMPLETION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion"
OUT_PATH = COMPLETION_ROOT / "attempt_provenance.json"


def main() -> dict:
    ledger = build_attempt_ledger()
    pending = identities_pending_next_attempt(ledger)

    provenance: dict[str, dict] = {}
    counts = {"standard": {}, "consensus_stage_a": {}}
    for stage in STAGES:
        with open(COMPLETION_ROOT / stage / "request_manifest.csv", newline="", encoding="utf-8") as f:
            import csv

            manifest_rows = list(csv.DictReader(f))
        pending_set = set(pending[stage])
        manifest_by_identity = {}
        for row in manifest_rows:
            from inference.target_g_retry_engine import intended_identity_from_request_key

            identity = intended_identity_from_request_key(row["request_key"])
            manifest_by_identity[identity] = row

        if set(manifest_by_identity) != pending_set:
            raise RuntimeError(f"stage {stage}: completion manifest identities do not exactly match the current pending set (manifest has {len(manifest_by_identity)}, pending has {len(pending_set)})")

        for identity, row in manifest_by_identity.items():
            entry = ledger[identity]
            attempt_number = entry["next_attempt_number"]
            smoke_only = entry["attempt_count"] == 0
            provenance[row["custom_id"]] = {
                "intended_identity": identity,
                "request_stage": stage,
                "attempt_number": attempt_number,
                "smoke_only": smoke_only,
            }
            key = str(attempt_number)
            counts[stage][key] = counts[stage].get(key, 0) + 1

    result = {
        "note": "engineering/accounting provenance only -- no scientific response value is present anywhere in this file",
        "per_custom_id": provenance,
        "attempt_number_counts": counts,
        "total_entries": len(provenance),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"total_entries": len(provenance), "attempt_number_counts": counts}


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
