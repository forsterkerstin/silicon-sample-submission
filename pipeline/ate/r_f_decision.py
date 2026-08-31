"""Sequential R_F freeze decision, using ONLY the three pre-existing numerical
engineering thresholds from model_selection_r_f_rule_manifest.json:

    max_invalid_response_rate = 0.005
    replicate_pairwise_ate_rmse_pp = 2.0
    max_condition_outcome_replicate_abs_diff_pp = 5.0

These three are the actual pass/fail reliability gate -- nothing else. Nested
N=50/100/250/500 convergence, MAD, Pearson, Spearman, and sign agreement are
computed and reported (via f_reliability.convergence_by_effect /
stochastic_reliability_by_effect) but are diagnostics only: they never gate
the R_F decision here, and no threshold is invented for them.

STAGE R1: two independent single-draw (R_F=1) estimates compared directly.
  Pass -> freeze R_F=1.
STAGE R2 (only reached if R1 fails): a THIRD and FOURTH independent draw are
  required, and average(draw1,draw2) is compared against average(draw3,draw4)
  using the SAME three thresholds. Pass -> freeze R_F=2.
  Fail -> STOP; no R_F is frozen, and no further paid escalation happens
  without a new explicit human decision (this module never auto-continues to
  a hypothetical R_F=3+).

Implemented and tested now, offline; not run against any real data yet.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ate.f_reliability import stochastic_reliability_by_effect

R_F_PASS_THRESHOLDS = {
    "max_invalid_response_rate": 0.005,
    "replicate_pairwise_ate_rmse_pp": 2.0,
    "max_condition_outcome_replicate_abs_diff_pp": 5.0,
}

# Diagnostics that are reported alongside every stage decision but never gate it.
R_F_DIAGNOSTICS = ("nested_n_50_100_250_convergence_vs_500", "mad", "pearson", "spearman", "sign_agreement")


def _passes_thresholds(*, invalid_response_rate: float, replicate_rmse_pp: float, max_abs_diff_pp: float) -> bool:
    return (
        invalid_response_rate <= R_F_PASS_THRESHOLDS["max_invalid_response_rate"]
        and replicate_rmse_pp <= R_F_PASS_THRESHOLDS["replicate_pairwise_ate_rmse_pp"]
        and max_abs_diff_pp <= R_F_PASS_THRESHOLDS["max_condition_outcome_replicate_abs_diff_pp"]
    )


def _compare_two_draws(a: pd.DataFrame, b: pd.DataFrame, *, outputs_dir) -> dict[str, Any]:
    """a, b: per-effect DataFrames with columns study_id, effect_id, z_pp (one
    row per effect -- for stage R1 these are raw single-draw estimates; for
    stage R2 they are already-averaged 2-draw estimates)."""
    wide = pd.DataFrame(
        {
            "study_id": a["study_id"].tolist(),
            "effect_id": a["effect_id"].tolist(),
        }
    )
    long_form = pd.concat(
        [
            wide.assign(replicate="replicate_1", z_native=a["z_pp"].to_numpy(), z_pp=a["z_pp"].to_numpy()),
            wide.assign(replicate="replicate_2", z_native=b["z_pp"].to_numpy(), z_pp=b["z_pp"].to_numpy()),
        ],
        ignore_index=True,
    )
    _by_effect, summary = stochastic_reliability_by_effect(long_form, outputs_dir=outputs_dir)
    return summary


def stage_r1_decision(
    draw1: pd.DataFrame,
    draw2: pd.DataFrame,
    *,
    invalid_response_rate: float,
    outputs_dir,
) -> dict[str, Any]:
    """draw1, draw2: per-effect z_pp DataFrames from two independent R_F=1 runs."""
    summary = _compare_two_draws(draw1, draw2, outputs_dir=outputs_dir)
    passed = _passes_thresholds(
        invalid_response_rate=invalid_response_rate,
        replicate_rmse_pp=summary["rmse"],
        max_abs_diff_pp=summary["max_abs_diff"],
    )
    return {
        "stage": "R1",
        "thresholds": dict(R_F_PASS_THRESHOLDS),
        "observed": {
            "invalid_response_rate": invalid_response_rate,
            "replicate_rmse_pp": summary["rmse"],
            "max_abs_diff_pp": summary["max_abs_diff"],
        },
        "diagnostics": {"mad_mean_abs_diff_pp": summary["mean_abs_diff"], "mad_median_abs_diff_pp": summary["median_abs_diff"], "pearson": summary["pearson"], "spearman": summary["spearman"], "sign_agreement": summary["sign_agreement"]},
        "decision": "FREEZE_R_F" if passed else "ESCALATE_TO_STAGE_R2",
        "r_f": 1 if passed else None,
    }


def stage_r2_decision(
    draw1: pd.DataFrame,
    draw2: pd.DataFrame,
    draw3: pd.DataFrame,
    draw4: pd.DataFrame,
    *,
    invalid_response_rate: float,
    outputs_dir,
) -> dict[str, Any]:
    """Only called after stage_r1_decision returns ESCALATE_TO_STAGE_R2. Two
    independent draw PAIRS, each averaged to represent one R_F=2 estimate,
    then compared against each other with the same three thresholds."""
    for df in (draw1, draw2, draw3, draw4):
        if not {"study_id", "effect_id", "z_pp"}.issubset(df.columns):
            raise ValueError("each draw must have columns study_id, effect_id, z_pp")
    if not (draw1[["study_id", "effect_id"]].reset_index(drop=True).equals(draw2[["study_id", "effect_id"]].reset_index(drop=True))
            and draw1[["study_id", "effect_id"]].reset_index(drop=True).equals(draw3[["study_id", "effect_id"]].reset_index(drop=True))
            and draw1[["study_id", "effect_id"]].reset_index(drop=True).equals(draw4[["study_id", "effect_id"]].reset_index(drop=True))):
        raise ValueError("all four draws must cover the identical study_id/effect_id rows in the same order")

    avg_12 = draw1.copy()
    avg_12["z_pp"] = (draw1["z_pp"].to_numpy() + draw2["z_pp"].to_numpy()) / 2
    avg_34 = draw3.copy()
    avg_34["z_pp"] = (draw3["z_pp"].to_numpy() + draw4["z_pp"].to_numpy()) / 2

    summary = _compare_two_draws(avg_12, avg_34, outputs_dir=outputs_dir)
    passed = _passes_thresholds(
        invalid_response_rate=invalid_response_rate,
        replicate_rmse_pp=summary["rmse"],
        max_abs_diff_pp=summary["max_abs_diff"],
    )
    return {
        "stage": "R2",
        "thresholds": dict(R_F_PASS_THRESHOLDS),
        "observed": {
            "invalid_response_rate": invalid_response_rate,
            "replicate_rmse_pp": summary["rmse"],
            "max_abs_diff_pp": summary["max_abs_diff"],
        },
        "diagnostics": {"mad_mean_abs_diff_pp": summary["mean_abs_diff"], "mad_median_abs_diff_pp": summary["median_abs_diff"], "pearson": summary["pearson"], "spearman": summary["spearman"], "sign_agreement": summary["sign_agreement"]},
        "decision": "FREEZE_R_F" if passed else "STOP_REQUIRE_NEW_EXPLICIT_DECISION",
        "r_f": 2 if passed else None,
    }
