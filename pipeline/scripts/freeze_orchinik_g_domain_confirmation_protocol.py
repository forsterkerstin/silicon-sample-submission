"""Freeze the Orchinik G-vs-DeepSeek domain-confirmation protocol (OFFLINE
ONLY, before any model prediction is observed).

Writes outputs/domain_validation/frozen_orchinik_g_domain_confirmation.json.
Supersedes (without deleting or rewriting) the earlier ORCHINIK_GAMMA_VALIDATION
role recorded in outputs/domain_validation/frozen_domain_validation_protocol.json
(commit 7ea86ef) -- that artifact, the downloaded source files, and the
previously computed 50-cell human ATE surface are all left untouched.

Makes no model call. Does not touch MU_EXTERNAL, gamma_G, S1/S2, or any
target G/F artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_DATA_DIR = PIPELINE_ROOT / "data" / "domain_validation" / "orchinik"
OUT_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"
MANIFEST_ROOT = OUT_DIR / "orchinik_g_domain_confirmation"
PRIOR_PROTOCOL_PATH = OUT_DIR / "frozen_domain_validation_protocol.json"

CONFIRMATION_MODULE_PATH = PIPELINE_ROOT / "ate" / "orchinik_g_domain_confirmation.py"
MANIFEST_BUILDER_PATH = PIPELINE_ROOT / "scripts" / "build_orchinik_g_domain_confirmation_manifest.py"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_dir: Path = OUT_DIR) -> dict:
    if not PRIOR_PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"prior protocol artifact missing (must not be deleted): {PRIOR_PROTOCOL_PATH}")
    prior_protocol_sha256 = _sha256_file(PRIOR_PROTOCOL_PATH)

    manifest_summary_path = MANIFEST_ROOT / "summary.json"
    if not manifest_summary_path.exists():
        raise FileNotFoundError(f"manifest summary missing -- run build_orchinik_g_domain_confirmation_manifest.py first: {manifest_summary_path}")
    manifest_summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))

    source_files = {
        "Bovitz_qualtrics.docx": _sha256_file(DOMAIN_DATA_DIR / "Bovitz_qualtrics.docx"),
        "Bovitz_qualtrics.qsf": _sha256_file(DOMAIN_DATA_DIR / "Bovitz_qualtrics.qsf"),
        "final_bovitz_raw.csv": _sha256_file(DOMAIN_DATA_DIR / "final_bovitz_raw.csv"),
        "final_clean.csv": _sha256_file(DOMAIN_DATA_DIR / "final_clean.csv"),
        "bovitz_data_clean.R": _sha256_file(DOMAIN_DATA_DIR / "bovitz_data_clean.R"),
        "analysis.Rmd": _sha256_file(DOMAIN_DATA_DIR / "analysis.Rmd"),
    }

    gemma_stats = manifest_summary["models"]["google/gemma-4-31B-it"]
    deepseek_stats = manifest_summary["models"]["deepseek-ai/DeepSeek-V4-Pro-0813"]

    protocol = {
        "status": "PROSPECTIVELY_FROZEN_BEFORE_MODEL_PREDICTIONS_OBSERVED",
        "purpose": "DOMAIN_SPECIFIC_CONFIRMATION_OF_FROZEN_G_MODEL_CHOICE",
        "original_g_selection": "ATP_WASSERSTEIN_SCREEN",
        "selected_g": "google/gemma-4-31B-it",
        "comparator": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "study": "ORCHINIK_2024",
        "study_citation": "Orchinik et al. (2024), 'Learning from and about scientists: Consensus messaging shapes perceptions of climate change and climate scientists', PNAS Nexus, DOI 10.1093/pnasnexus/pgae485",
        "sample": "MAIN_BOVITZ_NATIONALLY_REPRESENTATIVE_US_SAMPLE",
        "design": "ACTUAL_ASSIGNED_CONDITION_ONLY",
        "primary_metric": "EQUAL_WEIGHT_MEAN_W1_OVER_75_ARM_OUTCOME_CONSENSUS_CELLS",
        "scientific_consequence": "CONFIRMATION_ONLY_NO_AUTOMATIC_RESELECTION",
        "target_f_dependence": False,
        "target_g_dependence": False,
        "mu_external_dependence": False,
        "gamma_g_dependence": False,
        "prior_role_relationship": {
            "prior_role": "ORCHINIK_GAMMA_VALIDATION",
            "prior_role_status": "SUPERSEDED_FOR_CURRENT_METHOD_DECISION",
            "prior_protocol_artifact": str(PRIOR_PROTOCOL_PATH.relative_to(PIPELINE_ROOT)),
            "prior_protocol_artifact_sha256": prior_protocol_sha256,
            "prior_artifacts_preserved": True,
            "prior_50_cell_human_ate_surface_retained_for": "provenance and possible later descriptive analysis only -- must not be used to estimate gamma_G, alter gamma_G=1, decide S1 vs S2, alter MU_EXTERNAL, or alter target predictions",
        },
        "source_files": source_files,
        "eligible_donors": manifest_summary["eligible_respondents"],
        "arm_counts": {"control": 847, "skill_aka_History": 837, "trust_aka_Institutions": 861},
        "condition_label_mapping": {"skill": "History", "trust": "Institutions", "control": "control", "verification": "from the actual Qualtrics instrument text (Skill Intervention block = history-of-climate-science passage; Trust Intervention block = institutional-bias-safeguards passage); not inferred from variable names"},
        "focal_outcomes": ["human_caused_climate_change (cc)", "bias_of_pro_consensus_scientists (pro_bias)", "bias_of_anti_consensus_scientists (anti_bias)", "skill_of_pro_consensus_scientists (pro_skill)", "skill_of_anti_consensus_scientists (anti_skill)"],
        "consensus_levels": [50, 75, 90, 97, 99],
        "primary_cells": 75,
        "cell_weighting": "equal (no weighting by human arm sample size, no post-hoc cell selection)",
        "persona_construction": "pretreatment demographics only (age, gender, race, education, income, party identification + lean, political ideology [social and economic], belief in God/Gods intensity) -- no prior climate belief, consensus estimate, trust/bias/skill judgment, institutional-trust rating, affect-thermometer, or any other outcome/mediator/posttreatment field",
        "invalid_response_policy": "fail-closed: no coercion, no clipping, no fence-stripping, no JSON repair, no inference of malformed values, no automatic paid retry -- same philosophy as the original ATP G screen (ate.f_screen_validation.validate_response)",
        "model_configuration": {
            "gemma": {"temperature": 1.0, "top_p": 0.95, "n": 1, "enable_thinking": False, "chain_of_thought": False, "response_format": "constrained structured JSON"},
            "deepseek": "exact previously frozen DeepSeek G-screen serving configuration (temperature=1.0, top_p=0.95, n=1, its own frozen low-reasoning setting, constrained structured JSON) -- applied automatically per-model via inference.together_batch._chat_body/model_engine_config, not hand-duplicated here",
            "scientific_prompt_identical_across_models": True,
        },
        "gemma_manifest_sha256": gemma_stats["manifest_sha256"],
        "gemma_jsonl_sha256": gemma_stats["jsonl_sha256"],
        "gemma_request_count": gemma_stats["requests"],
        "gemma_worst_case_cost_usd": gemma_stats["worst_case_cost_usd"],
        "deepseek_manifest_sha256": deepseek_stats["manifest_sha256"],
        "deepseek_jsonl_sha256": deepseek_stats["jsonl_sha256"],
        "deepseek_request_count": deepseek_stats["requests"],
        "deepseek_worst_case_cost_usd": deepseek_stats["worst_case_cost_usd"],
        "total_new_requests": manifest_summary["total_requests"],
        "total_worst_case_cost_usd": manifest_summary["total_worst_case_cost_usd"],
        "confirmation_rule": {
            "pass": "L_ORCHINIK(Gemma) < L_ORCHINIK(DeepSeek) => DOMAIN_SPECIFIC_G_CONFIRMATION = PASS",
            "fail": "L_ORCHINIK(Gemma) > L_ORCHINIK(DeepSeek) => DOMAIN_SPECIFIC_G_CONFIRMATION = FAIL_MIXED_EVIDENCE",
            "tie": "exact numerical tie => DOMAIN_SPECIFIC_G_CONFIRMATION = TIE",
            "automatic_g_reselection": False,
            "note": "G* remains google/gemma-4-31B-it under all three outcomes unless a later, separate, explicit human-approved methodological decision is made",
        },
        "orchinik_used_to_estimate_mu_external": False,
        "orchinik_used_to_estimate_gamma_g": False,
        "orchinik_used_to_select_s1_vs_s2": False,
        "orchinik_used_to_calibrate_target_ates": False,
        "current_s2_unchanged": {"mu_external": 1.9558595458395387, "gamma_g": 1.0},
        "implementation": {
            "confirmation_module": "ate/orchinik_g_domain_confirmation.py",
            "confirmation_module_sha256": _sha256_file(CONFIRMATION_MODULE_PATH),
            "manifest_builder": "scripts/build_orchinik_g_domain_confirmation_manifest.py",
            "manifest_builder_sha256": _sha256_file(MANIFEST_BUILDER_PATH),
        },
        "manuscript_wording": {
            "rationale": (
                "The respondent simulator was prospectively selected using an external American Trends Panel "
                "distributional-fidelity screen. Because that screen used general survey items rather than "
                "climate-science outcomes, we subsequently conducted a separate domain-specific confirmation "
                "using the nationally representative U.S. climate-science experiment of Orchinik et al. (2024). "
                "Both candidate simulators generated the study's focal 0-100 response battery for each "
                "participant under the experimental condition actually assigned to that participant. The "
                "prespecified confirmation criterion was equal-weight mean Wasserstein-1 distance across the 75 "
                "condition-by-outcome-by-consensus-level response distributions. This analysis was confirmatory "
                "only and did not retroactively redefine the original model-selection rule or calibrate the S2 "
                "treatment-effect surface."
            ),
            "limitation": (
                "The domain-specific confirmation assesses response-distribution fidelity under climate-science "
                "interventions; it does not establish that the simulator's relative treatment-effect surface is "
                "correctly calibrated. S2 preserves that centered surface with unit weight as a prospectively "
                "frozen structural assumption."
            ),
            "results_status": "PENDING -- no inference has been run; do not fill in until real results exist",
        },
        "target_g_scientific_outputs_inspected": False,
        "target_human_outcomes_used": False,
        "orchinik_human_data_used": True,
        "orchinik_identified_post_initial_g_selection": True,
        "post_primary_domain_validation": True,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frozen_orchinik_g_domain_confirmation.json"
    out_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(out_path)
    (out_dir / "frozen_orchinik_g_domain_confirmation.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    protocol["protocol_sha256"] = sha
    return protocol


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
