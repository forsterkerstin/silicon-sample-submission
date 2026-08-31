"""Freeze the target G Wave-1 completion reconciliation specification
(OFFLINE ONLY, prospective -- written before completion-batch outputs
exist). Writes outputs/target_production/wave1_g_completion/
reconciliation_specification.json.

Records the rules implemented by scripts/score_target_g_wave1_completion.py
and scripts/assemble_target_g_wave1_first_valid.py: what is reported, how
"missing entirely" is defined (absent from BOTH output and error, so a
provider_error is never double-counted as missing), and the accounting
identity that must hold. No scientific value is recorded here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import score_target_g_wave1_completion as scorer  # noqa: E402

OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "reconciliation_specification.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    spec = {
        "spec_version": scorer.RECONCILIATION_SPEC_VERSION,
        "scope": "target G Wave-1 CURRENT completion manifest, STANDARD ONLY (1,401 requests); never attempt-3, the abandoned Consensus-A branch, the corrected Consensus-exact pipeline, F, Orchinik, or any other target/development inference.",
        "standard_only": True,
        "consensus_a_never_required_as_input": "the old Consensus-A completion branch is permanently disabled (outputs/target_production/consensus_protocol_amendment.json) -- this reconciliation path never reads, requires, or accepts any Consensus-A file or argument",
        "engineering_only": True,
        "scientific_response_values_used": False,
        "reported": [
            "expected identity count",
            "returned output-record count",
            "returned error-record count",
            "missing-entirely identities (absent from BOTH output and error)",
            "unexpected identities (present in output or error but not in the manifest)",
            "duplicate identities (repeated custom_id in output or error, or present in both)",
            "schema-valid responses",
            "schema-invalid responses",
            "malformed_json count (a schema-invalid subcategory)",
            "provider-error count (explicit provider/error-file record, no response body at all)",
            "provider-error code breakdown",
            "system_fingerprint breakdown, split by validity status (schema_valid vs schema_invalid)",
            "production-attempt-number distribution (from the frozen attempt_provenance.json, never re-derived from response content)",
            "first-valid-assembled count (base attempt-1 + this completion round, standard-only slice of the full ledger)",
            "count of standard identities eligible for a future Attempt 3",
        ],
        "accounting_identity": "schema_valid + schema_invalid + provider_error + missing_entirely == expected -- verified, not assumed; scoring refuses (nonzero exit) if it does not hold",
        "missing_entirely_definition": "a custom_id absent from BOTH the retrieved output file and the retrieved error file. A provider_error (present in the error file) is NEVER counted as missing_entirely -- doing so would double-count it against the accounting identity above.",
        "no_fence_stripping": True,
        "no_repair": True,
        "no_coercion": True,
        "no_outcome_dependent_filtering": True,
        "validator": "ate.f_screen_validation.validate_response (frozen, generic, fail-closed)",
        "first_valid_assembly_rule": "FIRST_VALID_RESPONSE_WINS in increasing attempt order, via inference.target_g_retry_engine.assemble_first_valid -- an identity with a schema-valid production response is locked and never regenerated, even if a later attempt is also valid (that case is itself a fail-closed error in build_attempt_ledger, not silently resolved).",
        "smoke_never_counts_as_production": True,
        "implementation": {
            "reconciliation_script": "scripts/score_target_g_wave1_completion.py",
            "reconciliation_script_sha256": _sha256_file(PIPELINE_ROOT / "scripts" / "score_target_g_wave1_completion.py"),
            "assembler_script": "scripts/assemble_target_g_wave1_first_valid.py",
            "assembler_script_sha256": _sha256_file(PIPELINE_ROOT / "scripts" / "assemble_target_g_wave1_first_valid.py"),
            "attempt_provenance_freezer": "scripts/freeze_target_g_completion_attempt_provenance.py",
            "attempt_provenance_freezer_sha256": _sha256_file(PIPELINE_ROOT / "scripts" / "freeze_target_g_completion_attempt_provenance.py"),
            "retry_engine": "inference/target_g_retry_engine.py (untouched by the Consensus correction)",
        },
        "exact_post_retrieval_reconciliation_command": (
            "python scripts/score_target_g_wave1_completion.py "
            "--standard-output outputs/target_production/wave1_g_completion/standard/retrieved/batch_output.jsonl "
            "--standard-error outputs/target_production/wave1_g_completion/standard/retrieved/batch_error.jsonl"
        ),
        "exact_post_reconciliation_first_valid_assembly_command": (
            "python scripts/assemble_target_g_wave1_first_valid.py "
            "--standard-output outputs/target_production/wave1_g_completion/standard/retrieved/batch_output.jsonl "
            "--standard-error outputs/target_production/wave1_g_completion/standard/retrieved/batch_error.jsonl"
        ),
        "target_g_scientific_values_accessed": False,
        "new_paid_inference_performed": False,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(OUT_PATH)
    (OUT_PATH.parent / "reconciliation_specification.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    spec["spec_artifact_sha256"] = sha
    return spec


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
