"""OFFLINE ONLY. Declare the two Orchinik domain-confirmation v2 submission
guard phases (orchinik_g_domain_confirmation_v2_gemma /
orchinik_g_domain_confirmation_v2_deepseek).

No API calls, no submission. This only verifies the canonical manifest/
jsonl/serving-amendment files on disk still match their frozen SHA256, then
registers each phase's exact allowlist (the canonical manifest's own
custom_id column -- never caller-supplied) and cost cap in outputs/
domain_validation/orchinik_g_domain_confirmation_v2/
orchinik_domain_confirmation_submission_state.json.

Must be run once (idempotently -- re-running is a no-op if state already
matches) before scripts/together_batch.py submit can use either phase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.orchinik_domain_confirmation_guard import PHASES, declare_orchinik_domain_confirmation_phase  # noqa: E402


def main() -> dict:
    return {phase: declare_orchinik_domain_confirmation_phase(phase) for phase in PHASES}


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
