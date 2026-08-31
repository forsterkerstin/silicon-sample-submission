"""Freeze the S2-only final-submission manifest (OFFLINE GOVERNANCE ONLY).

Writes outputs/validation/frozen_final_submission_manifest_s2.json. Records
the governance decision that S2 (MCONST_GSHAPE) is the SOLE final challenge
submission method -- Primary M2, Secondary-1 MCONST, and Approach 3 are
reclassified as development candidates, not submitted. Does not modify the
historical outputs/validation/frozen_method_manifest.json (the old M2
candidate manifest) or any other primary/secondary artifact -- this is a
NEW, separate, versioned manifest.

Makes no target G/F retrieval, no inference, no ledger modification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = PIPELINE_ROOT / "outputs" / "validation"
SECONDARY_DIAG_DIR = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic"

OLD_M2_MANIFEST_PATH = VALIDATION_DIR / "frozen_method_manifest.json"
PRIMARY_ARTIFACT_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
SECONDARY_1_ARTIFACT_PATH = SECONDARY_DIAG_DIR / "secondary_calibration_selected_model.json"
SECONDARY_2_ARTIFACT_PATH = SECONDARY_DIAG_DIR / "secondary_2_mconst_gshape_method.json"
PROJECTOR_ARTIFACT_PATH = SECONDARY_DIAG_DIR / "target_projection_method.json"
A3_ARTIFACT_PATH = PIPELINE_ROOT / "outputs" / "approach3_prompt_ensemble" / "frozen_a3_method.json"

OUT_PATH = VALIDATION_DIR / "frozen_final_submission_manifest_s2.json"

EXPECTED_OLD_M2_SHA256 = "0fb04d97f7e360e3f8222559eab6b8e25cd9f988eb84f0bda2de11ebe17875b6"
EXPECTED_S2_SHA256 = "07518f45735ec4f702d6a67521cb06cf4a1c306f4e3e206ce823aac262abeaad"

REGISTRATION_RATIONALE = (
    "External human experiments are used to calibrate the overall normalized treatment-effect level. Native "
    "Tier-1 LLM simulation supplies the relative intervention-by-outcome ATE structure and respondent-level "
    "response heterogeneity. The external development analysis found that calibration-in-the-large transported "
    "better across held-out studies than the tested F-dependent calibration models. The G-shape recentering "
    "rule was frozen before inspecting target G outputs. It is therefore a target-blind test of whether silicon "
    "sampling contains useful relative causal-effect structure beyond a generic externally estimated human-"
    "effect prior."
)

REGISTRATION_LIMITATION = (
    "S2 itself was not externally validated as superior to flat MCONST; only the external level calibration was "
    "selected by external whole-study validation. The relative G effect shape is a prospectively frozen "
    "target-blind modeling assumption."
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    for p in (OLD_M2_MANIFEST_PATH, PRIMARY_ARTIFACT_PATH, SECONDARY_1_ARTIFACT_PATH, SECONDARY_2_ARTIFACT_PATH, PROJECTOR_ARTIFACT_PATH, A3_ARTIFACT_PATH):
        if not p.exists():
            raise FileNotFoundError(f"required upstream artifact missing: {p}")

    old_m2_sha = _sha256_file(OLD_M2_MANIFEST_PATH)
    if old_m2_sha != EXPECTED_OLD_M2_SHA256:
        raise ValueError(f"old M2 manifest hash changed unexpectedly: expected {EXPECTED_OLD_M2_SHA256}, got {old_m2_sha} -- it must be preserved untouched")
    s2_sha = _sha256_file(SECONDARY_2_ARTIFACT_PATH)
    if s2_sha != EXPECTED_S2_SHA256:
        raise ValueError(f"S2 artifact hash changed unexpectedly: expected {EXPECTED_S2_SHA256}, got {s2_sha}")
    projector_sha = _sha256_file(PROJECTOR_ARTIFACT_PATH)

    manifest = {
        "final_method": "S2_MCONST_GSHAPE",
        "status": "SOLE_FINAL_SUBMISSION",
        "target_g_output_blind_at_s2_freeze": True,
        "target_human_outcome_blind": True,
        "g_star_model": "google/gemma-4-31B-it",
        "ces_source_sha256": "73cb40367efd0f92b8593cb9555660f68bdc95d8c1aad38f41aa13ab089ca43d",
        "g_n": 1000,
        "conditions": 17,
        "external_mu": 1.9558595458395387,
        "s2_artifact_sha256": s2_sha,
        "s2_artifact_path": str(SECONDARY_2_ARTIFACT_PATH.relative_to(PIPELINE_ROOT)),
        "s2_centering": "unweighted global mean over exactly 208 normalized ATE cells (16 interventions x 13 outcomes)",
        "gamma_g_shape": 1.0,
        "target_f_dependence": False,
        "target_g_dependence": True,
        "common_shift_equation": {
            "tau_G_aj": "mean_i(Y_G_iaj - Y_G_i0j)",
            "g_aj": "100 * tau_G_aj / R_j",
            "g_bar": "(1/208) * sum_aj g_aj",
            "theta_hat_aj": "MU_EXTERNAL + (g_aj - g_bar)",
            "tau_hat_aj": "(R_j / 100) * theta_hat_aj",
            "c_aj": "tau_hat_aj - tau_G_aj",
            "c_aj_closed_form": "(R_j / 100) * (MU_EXTERNAL - g_bar)",
            "invariant": "all interventions for a given outcome receive the same pre-projection additive shift; native-G cross-intervention relative ATE structure is preserved exactly",
        },
        "support_projection": {
            "status": "PROSPECTIVELY_FROZEN",
            "artifact_path": str(PROJECTOR_ARTIFACT_PATH.relative_to(PIPELINE_ROOT)),
            "artifact_sha256": projector_sha,
        },
        "final_materializer_module": "ate/s2_final_materializer.py",
        "final_materializer_sha256": _sha256_file(PIPELINE_ROOT / "ate" / "s2_final_materializer.py"),
        "final_materializer_accepts_target_f_input": False,
        "final_invalid_response_policy": (
            "no repair, coercion, clipping, imputation, or stochastic fallback for any invalid/missing G response "
            "at any stage (schema-invalid responses are excluded, never retried or repaired) -- consistent with "
            "ate/target_projection.py's fail-closed policy and this project's general no-repair rule; the S2 "
            "materializer receives only already-valid native G responses"
        ),
        "final_ces_donor_provenance": "config/population.yaml population_source_amendments (TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN), frozen_roster.sha256 bound above",
        "supersedes_for_submission_purposes": {
            "historical_m2_candidate_manifest_path": str(OLD_M2_MANIFEST_PATH.relative_to(PIPELINE_ROOT)),
            "historical_m2_candidate_manifest_sha256": old_m2_sha,
            "note": "preserved untouched, NOT deleted or rewritten -- reclassified as development provenance, not a submission artifact",
        },
        "reclassification_for_final_submission_purposes": {
            "PRIMARY_M2": "DEVELOPMENT_CANDIDATE_NOT_SUBMITTED",
            "SECONDARY_1_MCONST": "DEVELOPMENT_BASELINE_NOT_SUBMITTED",
            "A3_PROMPT_ENSEMBLE": "DEVELOPMENT_PROPOSAL_NOT_SUBMITTED",
            "S2_MCONST_GSHAPE": "SOLE_FINAL_SUBMISSION",
        },
        "selection_basis": "S2 was selected based only on external-development evidence (whole-study LOSO calibration-in-the-large transporting better than F-dependent models) and scientific interpretability, before target G output inspection",
        "target_f_wave1_status": "GENERATED_UNDER_SUPERSEDED_CANDIDATE_METHOD / UNUSED_FOR_FINAL_S2_PREDICTIONS -- see scripts/reclassify_target_f_wave1_provenance.py output; the target-production ledger itself is left untouched",
        "target_g_operational_graph": {
            "wave1": {"ordinary": 16000, "consensus_stage_a": 1000, "total": 17000, "status": "already submitted, not rebuilt"},
            "wave2": {"consensus_stage_b": 1000, "status": "required once G Consensus Stage A is complete; not yet built"},
            "total": 18000,
        },
        "registration_rationale": REGISTRATION_RATIONALE,
        "registration_limitation": REGISTRATION_LIMITATION,
        "internal_design_search_disclosure_required": [
            "G model comparison",
            "F model comparison",
            "F reliability work",
            "136-effect F calibration",
            "M0/M1/M2 comparison",
            "robust/contextual secondary diagnostics",
            "MCONST selection",
            "S2 prospective freeze",
        ],
        "not_preregistered_from_project_inception": True,
        "upstream_provenance_pointers": {
            "primary_artifact_sha256": _sha256_file(PRIMARY_ARTIFACT_PATH),
            "secondary_1_artifact_sha256": _sha256_file(SECONDARY_1_ARTIFACT_PATH),
            "a3_artifact_sha256": _sha256_file(A3_ARTIFACT_PATH),
        },
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(OUT_PATH)
    (VALIDATION_DIR / "frozen_final_submission_manifest_s2.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    manifest["s2_final_submission_manifest_sha256"] = sha
    return manifest


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
