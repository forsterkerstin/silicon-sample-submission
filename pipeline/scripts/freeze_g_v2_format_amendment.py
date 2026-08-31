"""Freeze the G-v2 PROVIDER_SERVING_FORMAT_FAILURE amendment (OFFLINE ONLY).

Writes outputs/target_production/g_wave1_v1_format_failure_amendment.json.
Documents the root-cause classification for the original (v1) target G
Wave-1 batch, the format-only fix (G_FORMAT_INSTRUCTION_V2, appended never
replacing the v1 closing sentence), and the fresh-draws-for-the-complete-
universe replacement manifest -- all built and hashed here, none submitted.

Does not touch the original v1 manifests/raw output/ledger, does not
compute or expose any target G scientific value, does not submit anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
V1_VALIDATION_REPORT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_validation_report.json"
V2_REPLACEMENT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement"
V2_STAGE_ROOT = V2_REPLACEMENT_ROOT / "by_stage"
SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
OUT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_format_failure_amendment.json"

V1_STANDARD_MANIFEST_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "standard" / "G" / "request_manifest.csv"
V1_CONSENSUS_A_MANIFEST_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "consensus_stage_a" / "G" / "request_manifest.csv"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_hashes() -> dict:
    import survey_content as sc
    from inference.prompts import build_g_prompt_render, text_hash

    items = sc.load_items()
    profile = {
        "age": 40,
        "gender": "Female",
        "race_ethnicity": "White / Caucasian",
        "education": "Bachelor's degree",
        "household_income": "$56,000 to $99,999",
        "party_id": "Independent",
        "political_ideology": "Moderate",
        "state_abbr": "CA",
        "religion": "Protestant",
    }
    v1 = build_g_prompt_render(profile, "Stimulus.", items, donor_key="D1", condition_id="control")
    v2 = build_g_prompt_render(profile, "Stimulus.", items, donor_key="D1", condition_id="control", response_format_instruction_version="v2")
    if v1.system_prompt != v2.system_prompt:
        raise ValueError("G-v2 must not change the system prompt -- format-only change must be in the closing instruction only")
    if v1.response_schema != v2.response_schema:
        raise ValueError("G-v2 must not change the response schema")
    v1_user_stripped = v1.user_prompt
    v2_user_stripped = v2.user_prompt[: len(v1.user_prompt)]
    if v1_user_stripped != v2_user_stripped:
        raise ValueError("G-v2 user prompt must be the v1 user prompt with ONLY an appended suffix -- everything before it must be byte-identical")
    return {
        "old_prompt_sha256": text_hash(v1.system_prompt + "\n" + v1.user_prompt),
        "new_v2_prompt_sha256": text_hash(v2.system_prompt + "\n" + v2.user_prompt),
        "scientific_prompt_equivalence_verified": True,
    }


def main() -> dict:
    if not V1_VALIDATION_REPORT_PATH.exists():
        raise FileNotFoundError(f"v1 validation report missing: {V1_VALIDATION_REPORT_PATH}")
    v1_report = json.loads(V1_VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))

    stage_summary = json.loads((V2_STAGE_ROOT / "summary.json").read_text(encoding="utf-8"))
    smoke_summary = json.loads((SMOKE_ROOT / "summary.json").read_text(encoding="utf-8"))
    prompt_hashes = _prompt_hashes()

    # collision-safety against the original v1 manifests specifically (spot check;
    # the full cross-outputs collision scan was run separately before partitioning)
    import csv

    def _ids(path):
        with open(path, newline="", encoding="utf-8") as f:
            return {row["custom_id"] for row in csv.DictReader(f)}

    v1_ids = _ids(V1_STANDARD_MANIFEST_PATH) | _ids(V1_CONSENSUS_A_MANIFEST_PATH)
    v2_ids = _ids(V2_STAGE_ROOT / "standard" / "request_manifest.csv") | _ids(V2_STAGE_ROOT / "consensus_stage_a" / "request_manifest.csv")
    overlap = v1_ids & v2_ids
    if overlap:
        raise ValueError(f"G-v2 replacement custom_ids collide with the original v1 run: {sorted(overlap)[:5]}")

    amendment = {
        "amendment_type": "TARGET_G_WAVE1_PROVIDER_SERVING_FORMAT_FAILURE_V2_REPAIR",
        "root_cause_classification": "PROVIDER_SERVING_FORMAT_FAILURE",
        "failed_backend_fingerprint": v1_report["failed_fingerprint"],
        "original_standard_valid": v1_report["standard"]["total_valid"],
        "original_standard_invalid": v1_report["standard"]["total_invalid"],
        "original_standard_total": v1_report["standard"]["total_n"],
        "original_consensus_a_valid": v1_report["consensus_stage_a"]["valid"],
        "original_consensus_a_invalid": v1_report["consensus_stage_a"]["invalid"],
        "original_consensus_a_total": v1_report["consensus_stage_a"]["n"],
        "failure_reason": "malformed_json caused by markdown code-fence wrapping despite response_format.json_schema strict structured output",
        "scientific_target_outputs_inspected": False,
        "scientific_prompt_content_changed": False,
        "replacement_uses_fresh_draws": True,
        "original_failed_run_preserved": True,
        "original_v1_manifest_paths": {
            "standard": str(V1_STANDARD_MANIFEST_PATH.relative_to(PIPELINE_ROOT)),
            "consensus_stage_a": str(V1_CONSENSUS_A_MANIFEST_PATH.relative_to(PIPELINE_ROOT)),
        },
        "original_v1_manifest_sha256": {"standard": _sha256_file(V1_STANDARD_MANIFEST_PATH), "consensus_stage_a": _sha256_file(V1_CONSENSUS_A_MANIFEST_PATH)},
        "g_v2_format_only_change": {
            "description": "Appends G_FORMAT_INSTRUCTION_V2 verbatim after the existing v1 closing instruction sentence -- never replaces or removes it. System prompt, persona fields, stimuli, questionnaire wording, response anchors, item ordering, schema, and model/sampling configuration are all byte-identical to v1.",
            "instruction_text": (
                "Return ONLY the raw JSON object matching the supplied schema. Do not use Markdown. "
                "Do not use ```json or any other code fences. Do not place text before or after the JSON. "
                "The response must begin with { and end with }."
            ),
            **prompt_hashes,
        },
        "no_fence_stripping_in_validation": True,
        "no_repair": True,
        "no_scientific_values_used_from_failed_run": True,
        "no_retry_of_invalid_subset_only": True,
        "smoke": {
            "standard_requests": smoke_summary["per_stage"]["standard"]["requests"],
            "consensus_a_requests": smoke_summary["per_stage"]["consensus_stage_a"]["requests"],
            "total_requests": smoke_summary["total_requests"],
            "manifest_sha256": {"standard": smoke_summary["per_stage"]["standard"]["manifest_sha256"], "consensus_stage_a": smoke_summary["per_stage"]["consensus_stage_a"]["manifest_sha256"]},
            "purpose": "schema/serving validation ONLY -- never used for scientific analysis or model selection; not submitted",
        },
        "full_replacement": {
            "standard_requests": stage_summary["standard"]["requests"],
            "consensus_a_requests": stage_summary["consensus_stage_a"]["requests"],
            "total_requests": stage_summary["standard"]["requests"] + stage_summary["consensus_stage_a"]["requests"],
            "manifest_sha256": {"standard": stage_summary["standard"]["manifest_sha256"], "consensus_stage_a": stage_summary["consensus_stage_a"]["manifest_sha256"]},
            "zero_collision_with_original_v1_run": True,
            "manifest_root": str(V2_REPLACEMENT_ROOT.relative_to(PIPELINE_ROOT)),
        },
        "target_g_scientific_outputs_accessed": False,
        "target_human_outcomes_used": False,
        "new_paid_inference_performed": False,
    }

    OUT_PATH.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(OUT_PATH)
    (OUT_PATH.parent / "g_wave1_v1_format_failure_amendment.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    amendment["amendment_artifact_sha256"] = sha
    return amendment


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
