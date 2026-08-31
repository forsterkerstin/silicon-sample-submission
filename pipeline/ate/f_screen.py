"""Mini F-model screen: cheap, small-scale RAW synthetic-ATE-vs-human-ATE
comparison used ONLY to select F* (candidate model identity) before the
expensive full 136-effect R_F reliability/M0-M1-M2 calibration run.

This is deliberately NOT the M0/M1/M2 calibration comparison -- it compares
each candidate's raw (uncalibrated) normalized effect directly against the
frozen human effect, using the same study-equal weighting principle as the
whole-study LOSO calibration comparison in ate/calibrate_lambda.py. Frozen and
tested BEFORE either candidate's screen requests are built.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ate.normalize_effects import to_percent_of_range

REQUIRED_ROW_KEYS = {"study_id", "effect_id", "theta_l_pp", "theta_h_pp"}


def f_screen_theta_l_pp(control_native: Sequence[float], treatment_native: Sequence[float], scale_low: float, scale_high: float) -> float:
    """theta_L = 100 * raw_ate / R for one effect/candidate: the paired-profile
    raw synthetic ATE (mean(treatment) - mean(control)) on its native scale,
    normalized to percent-of-range exactly like the human side (theta_H)."""
    if not control_native or not treatment_native:
        raise ValueError("f_screen_theta_l_pp requires non-empty control and treatment response lists")
    raw_ate = float(np.mean(treatment_native)) - float(np.mean(control_native))
    return to_percent_of_range(raw_ate, scale_low, scale_high)


def _study_equal_rmse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weight each effect 1/n_s within its study (so a study's per-study MSE is
    just the plain mean of its effects' squared diffs), then average per-study
    MSE EQUALLY across unique studies (each study counts once regardless of
    how many effects it contributed), then sqrt. Same study-equal principle as
    calibrate_lambda.py's whole-study LOSO weighting."""
    by_study: dict[str, list[float]] = {}
    for row in rows:
        by_study.setdefault(str(row["study_id"]), []).append((float(row["theta_l_pp"]) - float(row["theta_h_pp"])) ** 2)
    per_study_mse = {study: float(np.mean(sq_diffs)) for study, sq_diffs in by_study.items()}
    rmse = float(math.sqrt(np.mean(list(per_study_mse.values()))))
    return {"rmse": rmse, "per_study_mse": per_study_mse, "n_studies": len(by_study), "n_effects": len(rows)}


def f_screen_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """MAE, Pearson, Spearman, sign agreement -- diagnostics only, never the
    selection criterion, never silently averaged into the primary metric."""
    theta_l = np.array([float(r["theta_l_pp"]) for r in rows])
    theta_h = np.array([float(r["theta_h_pp"]) for r in rows])
    diff = theta_l - theta_h
    mae = float(np.mean(np.abs(diff)))
    if len(rows) >= 3 and len(set(theta_l)) > 1 and len(set(theta_h)) > 1:
        pearson = float(pearsonr(theta_l, theta_h).statistic)
        spearman = float(spearmanr(theta_l, theta_h).correlation)
    else:
        pearson = math.nan
        spearman = math.nan
    sign_agreement = float(np.mean(np.sign(theta_l) == np.sign(theta_h)))
    return {"mae": mae, "pearson": pearson, "spearman": spearman, "sign_agreement": sign_agreement}


def score_f_screen_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: one dict per effect for ONE candidate model, each with
    study_id, effect_id, theta_l_pp, theta_h_pp. Returns the primary
    study-equal RMSE plus diagnostics, kept in clearly separate keys."""
    missing = [set(row) for row in rows if not REQUIRED_ROW_KEYS.issubset(row)]
    if missing:
        raise ValueError(f"f-screen rows missing required keys; expected {sorted(REQUIRED_ROW_KEYS)}")
    primary = _study_equal_rmse(rows)
    diagnostics = f_screen_diagnostics(rows)
    return {"primary_study_equal_rmse_pp": primary["rmse"], "primary_detail": primary, "diagnostics": diagnostics}


def select_f_star(
    candidate_scores: dict[str, dict[str, Any]],
    *,
    invalid_response_rate: dict[str, float],
    realized_cost_usd: dict[str, float],
) -> dict[str, Any]:
    """Lowest primary_study_equal_rmse_pp wins. Tie-break (only on an EXACT
    tie of the primary metric): lower invalid-response rate, then lower
    realized inference cost, then deterministic lexical model id. M0/M1/M2 is
    never fit here -- this only ever picks a candidate model identity."""
    models = sorted(candidate_scores.keys())
    if set(models) != set(invalid_response_rate) or set(models) != set(realized_cost_usd):
        raise ValueError("candidate_scores, invalid_response_rate, and realized_cost_usd must cover the same model set")

    def sort_key(model: str) -> tuple:
        return (
            round(candidate_scores[model]["primary_study_equal_rmse_pp"], 10),
            round(invalid_response_rate[model], 10),
            round(realized_cost_usd[model], 10),
            model,
        )

    winner = min(models, key=sort_key)
    return {
        "f_star": winner,
        "ranked": sorted(models, key=sort_key),
        "primary_metric_values": {m: candidate_scores[m]["primary_study_equal_rmse_pp"] for m in models},
        "tie_break_rule": "lower invalid-response rate, then lower realized inference cost, then deterministic lexical model id",
    }
