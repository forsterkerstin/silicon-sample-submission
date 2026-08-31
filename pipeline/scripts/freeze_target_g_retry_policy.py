"""Freeze the bounded target G Wave-1 production-retry policy (OFFLINE
ONLY). Writes outputs/target_production/target_g_retry_policy.json.

Does not touch any frozen scientific artifact (S2, calibration, the v1/v2
Orchinik protocols, the frozen final method manifest). Records the policy
itself plus the current, mechanically-verified failure provenance and
completion-manifest counts -- never any scientific target-G value.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.target_g_retry_engine import EXPECTED_UNIVERSE_SIZE, MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY  # noqa: E402

ATTEMPT1_REPORT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "g_wave1_v2_replacement_validation_report.json"
COMPLETION_SUMMARY_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "summary.json"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "target_g_retry_policy.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    if not ATTEMPT1_REPORT_PATH.exists():
        raise FileNotFoundError(f"attempt-1 validation report missing: {ATTEMPT1_REPORT_PATH}")
    if not COMPLETION_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"completion manifest summary missing -- run scripts/build_target_g_wave1_completion_manifest.py first: {COMPLETION_SUMMARY_PATH}")
    attempt1 = json.loads(ATTEMPT1_REPORT_PATH.read_text(encoding="utf-8"))
    completion = json.loads(COMPLETION_SUMMARY_PATH.read_text(encoding="utf-8"))

    smoke_only = EXPECTED_UNIVERSE_SIZE - attempt1["totals"]["expected"]
    expected_completion_count = attempt1["totals"]["schema_invalid"] + attempt1["totals"]["provider_error"] + smoke_only

    policy = {
        "policy_type": "TARGET_G_WAVE1_BOUNDED_PRODUCTION_RETRY_V1",
        "MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY": MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY,
        "FIRST_VALID_RESPONSE_WINS": True,
        "SMOKE_IS_NOT_PRODUCTION": True,
        "RETRY_MEMBERSHIP_ENGINEERING_ONLY": True,
        "SCIENTIFIC_RESPONSE_VALUES_USED_FOR_RETRY": False,
        "UNLIMITED_RETRIES": False,
        "AUTOMATIC_PAID_RETRY": False,
        "ATTEMPT_4_ALLOWED": False,
        "selection_rule": "First strictly schema-valid production response, in increasing attempt order, wins and is permanently locked -- never regenerated, never re-chosen even if a later attempt also happens to be valid.",
        "retry_eligibility": "explicit provider error (no response) OR schema-invalid response OR genuinely missing provider result -- never conditioned on answer value, outcome, condition effect, demographic subgroup, extremeness, plausibility, target mean, ATE, or any scientific summary.",
        "validator": "ate.f_screen_validation.validate_response (frozen, generic, fail-closed: no fence-stripping, no JSON repair, no coercion, no malformed-value salvage, no outcome-dependent filtering, no retry-on-failure).",
        "intended_production_universe": EXPECTED_UNIVERSE_SIZE,
        "attempt_1_provenance": {
            "source": "16,990-request G-v2 full production replacement (excludes the 10 engineering-smoke-only identities)",
            "valid": attempt1["totals"]["schema_valid"],
            "malformed_json": attempt1["totals"]["malformed_json"],
            "schema_invalid_total": attempt1["totals"]["schema_invalid"],
            "provider_error": attempt1["totals"]["provider_error"],
            "missing_entirely": attempt1["totals"]["missing_entirely"],
        },
        "smoke_only_intended_identities": smoke_only,
        "current_completion": {
            "expected_count": expected_completion_count,
            "standard_count": completion["stages"]["standard"]["requests"],
            "consensus_stage_a_count": completion["stages"]["consensus_stage_a"]["requests"],
            "total_count": completion["total_requests"],
            "standard_manifest_sha256": completion["stages"]["standard"]["manifest_sha256"],
            "consensus_stage_a_manifest_sha256": completion["stages"]["consensus_stage_a"]["manifest_sha256"],
            "standard_cost_cap_usd": completion["stages"]["standard"]["worst_case_cost_usd"],
            "consensus_stage_a_cost_cap_usd": completion["stages"]["consensus_stage_a"]["worst_case_cost_usd"],
            "total_cost_cap_usd": completion["total_worst_case_cost_usd"],
            "attempt_number_counts": {stage: completion["stages"][stage]["attempt_number_counts"] for stage in ("standard", "consensus_stage_a")},
        },
        "note": (
            "Retry membership is determined solely by provider delivery and strict schema validity before target "
            "scientific summaries are computed. The first valid production response is retained and is never "
            "regenerated. At most three production attempts are permitted per intended identity."
        ),
        "consensus_stage_b_authorized": False,
        "attempt_3_authorized": False,
        "target_g_scientific_outputs_accessed": False,
        "target_human_outcomes_used": False,
        "new_paid_inference_performed": False,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(OUT_PATH)
    (OUT_PATH.parent / "target_g_retry_policy.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    policy["policy_artifact_sha256"] = sha
    return policy


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
