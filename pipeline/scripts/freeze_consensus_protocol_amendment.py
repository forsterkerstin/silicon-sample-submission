"""Freeze the Consensus benchmark-exact protocol amendment (OFFLINE ONLY,
prospective correction). Writes outputs/target_production/
consensus_protocol_amendment.json.

Documents the FAIL_MATERIAL_SEQUENCE_MISMATCH finding (from an offline
public-instrument audit, BEFORE any target-G scientific output or target
human outcome was inspected for method selection or calibration), the
correction (inference/consensus_benchmark_exact.py's item-3-always-middle
ordering + true per-item interleaved feedback), and the disposition of
every prior Consensus output (marked SCIENTIFICALLY_UNUSED_FOR_FINAL_SUBMISSION,
never entering the final 17,000-row dataset; the old 82-request completion
manifest is disabled, never submitted).

Does not touch any non-Consensus scientific artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from inference.consensus_benchmark_exact import CONSENSUS_EXACT_PROTOCOL_ID, LEGAL_ORDERS  # noqa: E402
from inference.consensus_exact_guard import PHASES as CONSENSUS_EXACT_PHASES  # noqa: E402

OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_protocol_amendment.json"
LEGACY_MARKER_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "legacy_consensus_outputs_unused.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    step1 = CONSENSUS_EXACT_PHASES["consensus_exact_step1"]
    amendment = {
        "amendment_type": "CONSENSUS_BENCHMARK_EXACT_SEQUENCE_CORRECTION",
        "audit_classification": "FAIL_MATERIAL_SEQUENCE_MISMATCH",
        "audit_method": "offline comparison of the public questionnaire/instrument (survey/questionnaire.txt lines 507-543, corroborated by survey/survey.qsf's FL_137 block structure) against the implementation, performed BEFORE any target-G scientific output or target human outcome was inspected for method selection or calibration",
        "public_spec_requirements": {
            "item_order": "[Randomize with #3 always in the middle] -- item #3 (net_zero_before_2085) always occupies the middle position; the only legal orders are 1-3-2 and 2-3-1",
            "feedback_timing": "Feedback: Given directly after each item -- each item's correction is shown immediately after that item's own estimate and before the next item's estimate",
        },
        "prior_implementation_deviation": {
            "item_order": "consensus_interaction_order (inference/prompts.py) used a full unconstrained hash-permutation across all 3 items -- no enforcement anywhere that item #3 is always the middle position",
            "feedback_timing": "all three items' feedback was batched together in Stage B, after Stage A's three estimates were already fully collected, instead of interleaved per item",
        },
        "correction": {
            "module": "inference/consensus_benchmark_exact.py",
            "legal_orders": list(LEGAL_ORDERS),
            "item_3_always_middle": True,
            "order_assignment": "deterministic, donor_key-only (assign_consensus_exact_order) -- never depends on replicate_id/attempt_id, so order is stable across retries and was never altered after observing any response",
            "sequence": [
                "CONSENSUS_STEP_1: show the first ordered item, obtain only that estimate, no feedback/correct-answer visible -- then append that item's fixed feedback",
                "CONSENSUS_STEP_2: show item #3 (always second), obtain only that estimate, item-1 feedback visible, item-3's own correct feedback NOT visible until after the response -- then append item #3's fixed feedback",
                "CONSENSUS_STEP_3: show the remaining item, obtain only that estimate, feedback from the previous two items visible, this item's own correct feedback NOT visible until after the response -- then append that item's fixed feedback",
                "CONSENSUS_OUTCOMES: after feedback has been supplied for all three items, administer the full post-treatment questionnaire using the same respondent and conversation state",
            ],
            "no_future_correct_answer_leakage": True,
            "retry_bound": "MAX_PRODUCTION_ATTEMPTS_PER_STAGE_IDENTITY = 3 per logical stage (inference/consensus_exact_retry_engine.py), FIRST_VALID_RESPONSE_WINS, retry membership determined solely by provider delivery and strict schema validity, attempt number never changes donor/item order/prompt content/prior valid responses/feedback content/questionnaire content",
        },
        "prior_consensus_outputs_disposition": {
            "status": "SCIENTIFICALLY_UNUSED_FOR_FINAL_SUBMISSION",
            "applies_to": "every previously generated Consensus Stage-A output, including technically valid responses, engineering-smoke outputs, and failed/provider-error responses, across the original v1 batch, the G-v2 full production replacement, and the 82-request completion manifest",
            "archived_not_deleted": True,
            "will_never_enter_final_17000_row_dataset": True,
            "legacy_82_request_completion_manifest": {
                "path": "outputs/target_production/wave1_g_completion/consensus_stage_a/",
                "disabled": True,
                "guard_phase_removed": "target_g_wave1_completion_consensus_a (see inference/target_g_completion_guard.py's DISABLED_LEGACY_CONSENSUS_A_COMPLETION_PHASE)",
                "submitted": False,
            },
        },
        "unaffected_scope": {
            "standard_target_g_data": "UNAFFECTED -- inference/target_g_retry_engine.py, inference/target_g_completion_guard.py's target_g_wave1_completion_standard phase, and the 1,401-request standard completion manifest are untouched by this amendment",
            "external_calibration_estimator": "UNTOUCHED",
            "donor_population": "UNTOUCHED (same 1,000 G donors from data/generated/g_personas_master.csv)",
        },
        "no_target_g_scientific_values_or_human_outcomes_used_in_this_decision": True,
        "consensus_exact_step1_manifest": {
            "requests": step1["expected_request_count"],
            "manifest_sha256": step1["manifest_sha256"],
            "jsonl_sha256": step1["jsonl_sha256"],
            "cost_cap_usd": step1["cost_cap_usd"],
            "model": step1["model"],
        },
        "consensus_exact_protocol_id": CONSENSUS_EXACT_PROTOCOL_ID,
        "target_g_scientific_outputs_accessed": False,
        "target_human_outcomes_used": False,
        "new_paid_inference_performed": False,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(OUT_PATH)
    (OUT_PATH.parent / "consensus_protocol_amendment.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    amendment["amendment_sha256"] = sha

    # Also freeze a compact, standalone marker recording which prior
    # Consensus artifacts are unused, for scripts that only need this fact
    # without loading the full amendment.
    legacy_marker = {
        "status": "SCIENTIFICALLY_UNUSED_FOR_FINAL_SUBMISSION",
        "reason": "FAIL_MATERIAL_SEQUENCE_MISMATCH",
        "amendment_sha256": sha,
        "unused_sources": [
            "outputs/target_production/wave1/by_stage/consensus_stage_a/",
            "outputs/target_production/wave1_g_v2_replacement/by_stage/consensus_stage_a/",
            "outputs/target_production/wave1_g_v2_replacement/submission/consensus_stage_a/",
            "outputs/target_production/g_v2_engineering_smoke/consensus_stage_a/",
            "outputs/target_production/wave1_g_completion/consensus_stage_a/",
        ],
        "may_never_enter_final_17000_row_dataset": True,
    }
    LEGACY_MARKER_PATH.write_text(json.dumps(legacy_marker, indent=2) + "\n", encoding="utf-8")

    return amendment


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
