"""Freeze the ONE-SHOT, non-adaptive, post-freeze external diagnostic
validation of the final S2 (MCONST_GSHAPE) estimator against Orchinik et
al. (2024, PNAS Nexus) -- OFFLINE GOVERNANCE ONLY.

Scientific role: POST_FREEZE_EXTERNAL_DIAGNOSTIC_VALIDATION. This protocol
cannot select, tune, recalibrate, or alter any part of the final method --
MU_EXTERNAL and GAMMA_G_SHAPE are read from the already-frozen
ate.secondary_2_mconst_gshape module and asserted unchanged, never
re-estimated. Whatever this validation shows, predictions/
team_10_T1_primary_v1.csv and the frozen method do not change; a future
change would require a new, separate, explicitly authorized decision.

Writes outputs/domain_validation/frozen_orchinik_final_validation_protocol.json.
Makes no target/Orchinik retrieval, no inference, no ledger modification,
no submission. Every formula/metric/comparator/bootstrap parameter here is
fixed BEFORE any new inference is run and BEFORE
outputs/domain_validation/orchinik_human_ate_surface.json's values are
read by any scoring code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.secondary_2_mconst_gshape import EXTERNAL_MU, GAMMA_G_SHAPE  # noqa: E402
from inference.orchinik_domain_confirmation_guard import PHASES as ORCHINIK_PHASES  # noqa: E402

DOMAIN_VALIDATION_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"
OUT_PATH = DOMAIN_VALIDATION_DIR / "frozen_orchinik_final_validation_protocol.json"
HUMAN_SURFACE_PATH = DOMAIN_VALIDATION_DIR / "orchinik_human_ate_surface.json"
METRICS_MODULE_PATH = PIPELINE_ROOT / "ate" / "domain_validation_metrics.py"

GEMMA_PHASE = "orchinik_g_domain_confirmation_v2_gemma"
EXPECTED_EXTERNAL_MU = 1.9558595458395387
EXPECTED_GAMMA_G_SHAPE = 1.0
EXPECTED_METRICS_MODULE_SHA256 = "03184f0e6a64953ce86eaf2275d6ef97a01886cf2c130b0ca4ef4767259e4b6c"
N_INTERVENTIONS = 2
N_OUTCOMES = 25
N_CELLS = 50
R_J = 100.0
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_N_BOOT = 10000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> dict:
    if EXTERNAL_MU != EXPECTED_EXTERNAL_MU:
        raise ValueError(f"MU_EXTERNAL drifted: expected {EXPECTED_EXTERNAL_MU}, found {EXTERNAL_MU}")
    if GAMMA_G_SHAPE != EXPECTED_GAMMA_G_SHAPE:
        raise ValueError(f"GAMMA_G_SHAPE drifted: expected {EXPECTED_GAMMA_G_SHAPE}, found {GAMMA_G_SHAPE}")

    metrics_sha = _sha256_file(METRICS_MODULE_PATH)
    if metrics_sha != EXPECTED_METRICS_MODULE_SHA256:
        raise ValueError(f"ate/domain_validation_metrics.py hash drifted: expected {EXPECTED_METRICS_MODULE_SHA256}, got {metrics_sha}")

    gemma_spec = ORCHINIK_PHASES[GEMMA_PHASE]
    manifest_sha = _sha256_file(gemma_spec["manifest_path"])
    jsonl_sha = _sha256_file(gemma_spec["jsonl_path"])
    if manifest_sha != gemma_spec["manifest_sha256"]:
        raise ValueError(f"Orchinik Gemma manifest hash drifted: expected {gemma_spec['manifest_sha256']}, got {manifest_sha}")
    if jsonl_sha != gemma_spec["jsonl_sha256"]:
        raise ValueError(f"Orchinik Gemma jsonl hash drifted: expected {gemma_spec['jsonl_sha256']}, got {jsonl_sha}")
    if gemma_spec["expected_request_count"] != 2545:
        raise ValueError("expected_request_count drifted from 2545")
    if gemma_spec["cost_cap_usd"] != 3.699383:
        raise ValueError("cost_cap_usd drifted from 3.699383")

    human_surface_sha = _sha256_file(HUMAN_SURFACE_PATH) if HUMAN_SURFACE_PATH.exists() else None

    protocol = {
        "status": "PROSPECTIVELY_FROZEN_BEFORE_INFERENCE",
        "scientific_role": "POST_FREEZE_EXTERNAL_DIAGNOSTIC_VALIDATION",
        "may_select_tune_recalibrate_or_alter_final_method": False,
        "target_predictions_will_change_regardless_of_result": False,
        "final_tier1_csv": "predictions/team_10_T1_primary_v1.csv",
        "final_tier1_sha256_at_freeze": "a7c8e82e9a8f76a97e7e9e9845103dedb6c5e9c3b71f1f49ddff1db1a5579b54",
        "frozen_method_constants": {
            "mu_external": EXTERNAL_MU,
            "gamma_g_shape": GAMMA_G_SHAPE,
            "source_module": "ate/secondary_2_mconst_gshape.py",
            "re_estimated_here": False,
        },
        "simulation": {
            "manifest_phase": GEMMA_PHASE,
            "model": gemma_spec["model"],
            "manifest_path": str(gemma_spec["manifest_path"].relative_to(PIPELINE_ROOT)),
            "manifest_sha256": gemma_spec["manifest_sha256"],
            "jsonl_path": str(gemma_spec["jsonl_path"].relative_to(PIPELINE_ROOT)),
            "jsonl_sha256": gemma_spec["jsonl_sha256"],
            "expected_request_count": gemma_spec["expected_request_count"],
            "cost_cap_usd": gemma_spec["cost_cap_usd"],
            "deepseek_run_authorized": False,
            "additional_retries_beyond_governed_engineering_policy_authorized": False,
        },
        "grid": {
            "interventions": ["skill", "trust"],
            "n_interventions": N_INTERVENTIONS,
            "n_outcomes": N_OUTCOMES,
            "n_cells": N_CELLS,
            "R_j": R_J,
            "R_j_source": "every Orchinik focal item is a 0-100 integer slider (ate/orchinik_g_domain_confirmation.py); R_j is uniform across all 50 cells",
        },
        "formula": {
            "tau_G_aj": "mean(Y_treatment_aj) - mean(Y_control_j)",
            "g_aj": "100 * tau_G_aj / R_j",
            "g_bar": "mean over all 50 g_aj, computed strictly on Orchinik's own grid -- never mixed with the Silicon Sample target 208-cell grid's own g_bar",
            "theta_hat_aj": "mu_external + (g_aj - g_bar)",
            "no_refitting": True,
            "no_new_gamma_or_mu_estimated": True,
        },
        "primary_metric": {
            "name": "RMSE",
            "definition": "sqrt(mean((theta_hat_aj - theta_ext_aj)^2)) across all 50 normalized ATE cells, where theta_ext_aj is Orchinik's real human ATE normalized the same way (100 * h_e / R_j)",
        },
        "pre_specified_comparators": {
            "A_raw_gemma": "RMSE(g_aj, theta_ext_aj) across all 50 cells",
            "B_flat_mu_external": "RMSE(mu_external for every cell, theta_ext_aj) across all 50 cells",
        },
        "diagnostics": [
            "MAE(theta_hat_aj, theta_ext_aj)",
            "Pearson correlation(theta_hat_aj, theta_ext_aj)",
            "Spearman correlation(theta_hat_aj, theta_ext_aj)",
            "sign_agreement(theta_hat_aj, theta_ext_aj) over cells with both terms nonzero",
            "RMSE restricted to the skill-arm cells (25 cells)",
            "RMSE restricted to the trust-arm cells (25 cells)",
        ],
        "bootstrap": {
            "reused_from": "ate/domain_validation_metrics.py (already frozen, unmodified)",
            "module_sha256": metrics_sha,
            "cluster_unit": "respondent/donor id",
            "seed": BOOTSTRAP_SEED,
            "n_boot": BOOTSTRAP_N_BOOT,
            "interval": "2.5/97.5 percentile, descriptive uncertainty only",
        },
        "no_post_hoc_pass_fail_threshold": True,
        "governance": [
            "human Orchinik effect values may only be opened after all simulation and scoring rules in this document are committed/frozen",
            "no metric in this document may be changed after seeing performance",
            "no target prediction may change regardless of result",
            "mu_external and gamma_g_shape must remain unchanged regardless of result",
            "no result-dependent rerun of this validation",
            "no DeepSeek run under this protocol",
            "no automatic retry beyond the already-governed engineering (schema/delivery) retry policy",
            "no paid submission is authorized by this freeze -- a separate, explicit submission action is required",
        ],
        "human_data_source": {
            "path": str(HUMAN_SURFACE_PATH.relative_to(PIPELINE_ROOT)) if HUMAN_SURFACE_PATH.exists() else None,
            "sha256_at_freeze": human_surface_sha,
            "opened_by_this_freeze_script": False,
        },
        "scoring_script": "scripts/score_orchinik_final_validation.py",
        "frozen_at_git_commit": _git_commit(),
    }

    DOMAIN_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol["protocol_sha256"] = _sha256_file(OUT_PATH)
    return protocol


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))
