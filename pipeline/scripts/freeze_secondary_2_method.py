"""Prospectively freeze the Secondary-2 MCONST_GSHAPE method (OFFLINE ONLY).

Writes outputs/secondary_calibration_diagnostic/secondary_2_mconst_gshape_method.json.
Does NOT invoke ate.secondary_2_mconst_gshape on any real target G output --
this only records the frozen equation/constants/provenance, before any
target G model output has been retrieved or inspected. Does not touch the
primary calibration artifact, frozen_method_manifest.json, the Secondary-1
artifact, or any target manifest/ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ate.secondary_2_mconst_gshape import EXPECTED_N_CELLS, EXTERNAL_MU, GAMMA_G_SHAPE

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic"
SECONDARY_1_ARTIFACT_PATH = OUT_DIR / "secondary_calibration_selected_model.json"
POPULATION_CONFIG_PATH = PIPELINE_ROOT / "config" / "population.yaml"
G_STAR = "google/gemma-4-31B-it"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_ces_source_sha256() -> str:
    cfg = yaml.safe_load(POPULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    for amendment in cfg.get("population_source_amendments", []):
        if amendment.get("amendment_type") == "TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN":
            return amendment["frozen_roster"]["sha256"]
    raise ValueError("frozen CES source sha256 not found in config/population.yaml")


def main() -> dict:
    if not SECONDARY_1_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Secondary-1 artifact missing: {SECONDARY_1_ARTIFACT_PATH}")
    secondary_1_sha256 = _sha256_file(SECONDARY_1_ARTIFACT_PATH)
    ces_sha256 = _frozen_ces_source_sha256()

    artifact = {
        "method_name": "MCONST_GSHAPE",
        "status": "PROSPECTIVELY_FROZEN_BEFORE_TARGET_G_OUTPUT_INSPECTION",
        "post_primary_external_development": True,
        "target_human_outcome_blind": True,
        "target_g_outputs_inspected_before_freeze": False,
        "equation": {
            "g_aj": "100 * tau_G_aj / R_j",
            "g_bar": "(1/208) * sum over a=1..16, j=1..13 of g_aj  (equal weight per cell)",
            "theta_hat_s2_aj": "EXTERNAL_MU + GAMMA_G_SHAPE * (g_aj - g_bar)",
            "tau_hat_s2_aj": "(R_j / 100) * theta_hat_s2_aj",
            "common_shift_identity": "c_aj = tau_hat_s2_aj - tau_g_native == (R_j / 100) * (EXTERNAL_MU - g_bar)",
        },
        "external_mu": EXTERNAL_MU,
        "gamma_g_shape": GAMMA_G_SHAPE,
        "centering_scope": "all_208_normalized_target_ATE_cells",
        "cell_weighting": "equal",
        "expected_cells": EXPECTED_N_CELLS,
        "expected_interventions": 16,
        "expected_outcomes": 13,
        "target_f_dependence": False,
        "target_g_dependence": True,
        "no_fitting_or_tuning": True,
        "forbidden_operations": [
            "sign_clipping",
            "winsorization",
            "truncation_of_g_ate_shape",
            "shrinkage_of_centered_g_shape",
            "gamma_tuning",
            "per_outcome_scaling",
            "per_intervention_scaling",
            "nonlinear_transformation",
            "target_f_blending",
            "target_dependent_model_selection",
            "outcome_specific_external_means",
        ],
        "secondary_1_artifact_path": str(SECONDARY_1_ARTIFACT_PATH.relative_to(PIPELINE_ROOT)),
        "secondary_1_artifact_sha256": secondary_1_sha256,
        "ces_source_sha256": ces_sha256,
        "g_star_model": G_STAR,
        "implementation_module": "ate/secondary_2_mconst_gshape.py",
        "relationship_to_secondary_1": {
            "secondary_1_mconst": "theta_hat_S1_aj = MU_EXTERNAL for all 208 cells (identical target)",
            "secondary_2_mconst_gshape": "theta_hat_S2_aj = MU_EXTERNAL + (g_aj - g_bar) -- same global normalized mean, cross-cell shape inherited exactly from native G",
            "secondary_2_claimed_superior_to_secondary_1": False,
        },
        "future_application_pipeline_documented_only": [
            "native G outputs",
            "208 native G ATEs",
            "normalize by outcome range",
            "compute global g_bar",
            "recenter to MU_EXTERNAL",
            "convert back to native scale",
            "common shift native G treatment respondents",
            "existing deterministic support projection",
            "recompute constructs",
            "final Tier-1 validation",
        ],
        "not_executed_in_this_freeze": True,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = OUT_DIR / "secondary_2_mconst_gshape_method.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(artifact_path)
    (OUT_DIR / "secondary_2_mconst_gshape_method.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    artifact["secondary_2_artifact_sha256"] = sha
    return artifact


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
