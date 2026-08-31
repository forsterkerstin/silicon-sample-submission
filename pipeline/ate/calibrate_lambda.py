"""pipeline/ate/calibrate_lambda.py

External ATE calibration: compare three population-aligned calibration
models using whole-study leave-one-study-out validation:

    M0: y_hat = x
    M1: y_hat = lambda * x
    M2: y_hat = alpha + lambda * x

`x` is the synthetic effect in percentage-of-range units and `y` is the
human effect in the same units. Eligibility and population transportability
are validated before any model fitting; eligibility never depends on model
performance.

This module is entirely about calibrating ATE MAGNITUDES (scalar effect
sizes on the percent-of-range scale from ate/normalize_effects.py) -- it has
never operated on, and does not need, individual response distributions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
ATE_ARCHIVE_PATH = DATA_DIR / "ate_archive.csv"

REQUIRED_ELIGIBILITY_FIELDS = [
    "study_id",
    "effect_id",
    "target_population",
    "synthetic_target_population",
    "population_type",
    "is_general_us_adult",
    "is_specialized_population",
    "study_weights_available",
    "profile_variables_available",
    "population_matching_method",
    "weights_used",
    "treatment_type",
    "randomized_between_subjects",
    "materials_available",
    "outcome_type",
    "outcome_min",
    "outcome_max",
    "outcome_range",
    "finite_range",
    "main_effect_compatible",
    "included_primary_calibration",
    "included_secondary_sensitivity",
    "exclusion_reason",
]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value: object) -> float | None:
    text = str(value).strip()
    return None if text == "" else float(text)


def load_ate_archive(path: Path | str = ATE_ARCHIVE_PATH, *, primary_only: bool = True) -> list[dict[str, object]]:
    """Load population-aligned external effects for lambda_ate calibration.

    The archive CSV must carry explicit eligibility metadata. By default,
    only pre-declared `included_primary_calibration` rows are returned, and
    validation fails loudly if a primary row lacks a target population,
    uses incompatible human/synthetic target populations, ignores declared
    study weights, lacks an outcome range, or otherwise violates the
    transportability contract.
    """
    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    if not raw_rows:
        raise ValueError(f"{path} has no rows -- fill in the archive's model/human ATE pairs")
    validate_archive_eligibility(raw_rows)
    rows = []
    for r in raw_rows:
        if primary_only and not _parse_bool(r.get("included_primary_calibration")):
            continue
        if r.get("model_ate", "") == "" or r.get("human_ate", "") == "":
            continue
        rows.append(
            {
                "study_id": r["study_id"],
                "effect_id": r["effect_id"],
                "outcome": r["outcome"],
                "model_ate": float(r["model_ate"]),
                "human_ate": float(r["human_ate"]),
                "treatment_family": r.get("treatment_family") or None,
                "outcome_family": r.get("outcome_family") or None,
                "target_population": r["target_population"],
                "synthetic_target_population": r["synthetic_target_population"],
                "population_type": r["population_type"],
                "population_matching_method": r["population_matching_method"],
                "weights_used": _parse_bool(r.get("weights_used")),
                "study_weights_available": _parse_bool(r.get("study_weights_available")),
                "included_primary_calibration": _parse_bool(r.get("included_primary_calibration")),
                "included_secondary_sensitivity": _parse_bool(r.get("included_secondary_sensitivity")),
                "outcome_range": float(r["outcome_range"]),
                "outcome_type": r["outcome_type"],
                "human_ate_native": _optional_float(r.get("human_ate_native", "")),
                "synthetic_ate_native": _optional_float(r.get("synthetic_ate_native", "")),
            }
        )
    if primary_only and not rows:
        raise ValueError(f"{path} has no primary calibration rows after population eligibility filtering")
    return rows


def validate_archive_eligibility(rows: Sequence[dict[str, object]]) -> None:
    """Hard-stop validation for population transportability metadata.

    Eligibility is metadata-driven only: this function never looks at model
    error or lambda fit performance.
    """
    problems: list[str] = []
    for i, row in enumerate(rows):
        missing = [field for field in REQUIRED_ELIGIBILITY_FIELDS if field not in row]
        if missing:
            problems.append(f"row {i} missing eligibility field(s): {missing}")
            continue
        effect_id = row.get("effect_id") or f"row {i}"
        included_primary = _parse_bool(row.get("included_primary_calibration"))
        included_secondary = _parse_bool(row.get("included_secondary_sensitivity"))
        target_population = str(row.get("target_population", "")).strip()
        synthetic_target_population = str(row.get("synthetic_target_population", "")).strip()
        population_matching_method = str(row.get("population_matching_method", "")).strip()
        outcome_range = _optional_float(row.get("outcome_range", ""))
        finite_range = _parse_bool(row.get("finite_range"))
        study_weights_available = _parse_bool(row.get("study_weights_available"))
        weights_used = _parse_bool(row.get("weights_used"))
        is_specialized = _parse_bool(row.get("is_specialized_population"))

        if included_primary:
            if not target_population:
                problems.append(f"{effect_id}: primary row missing target_population")
            if not synthetic_target_population:
                problems.append(f"{effect_id}: primary row missing synthetic_target_population")
            if target_population and synthetic_target_population and target_population != synthetic_target_population:
                problems.append(f"{effect_id}: human and synthetic effects target different populations")
            if not population_matching_method or population_matching_method == "not_matched":
                problems.append(f"{effect_id}: population matching is undefined")
            if is_specialized and population_matching_method not in {"exact_population_match", "study_respondent_weighted", "raked_to_study_margins"}:
                problems.append(f"{effect_id}: specialized population included in primary without explicit population matching")
            if study_weights_available and not weights_used:
                problems.append(f"{effect_id}: study weights are available but not used")
            if not finite_range or outcome_range is None or outcome_range <= 0:
                problems.append(f"{effect_id}: outcome_range is missing or invalid for normalized effect")
            for field in ("randomized_between_subjects", "materials_available", "main_effect_compatible"):
                if not _parse_bool(row.get(field)):
                    problems.append(f"{effect_id}: primary row has {field}=False")
        elif not included_secondary and not str(row.get("exclusion_reason", "")).strip():
            problems.append(f"{effect_id}: excluded row missing exclusion_reason")
    if problems:
        raise ValueError("calibration archive eligibility validation failed:\n- " + "\n- ".join(problems))


def summarize_calibration_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    population_types = sorted({str(r.get("population_type", "")) for r in rows if r.get("population_type")})
    return {
        "n_studies": len({r["study_id"] for r in rows}),
        "n_effects": len(rows),
        "population_types": population_types,
        "n_using_survey_weights": sum(1 for r in rows if r.get("weights_used")),
        "n_using_representative_us_fallback": sum(1 for r in rows if r.get("population_matching_method") == "representative_us_fallback"),
        "n_using_study_respondent_profile_distribution_unweighted": sum(
            1 for r in rows if r.get("population_matching_method") == "study_respondent_profile_distribution_unweighted"
        ),
        "n_using_study_effect_analytic_profile_distribution_unweighted_largest_remainder": sum(
            1 for r in rows
            if r.get("population_matching_method") == "study_effect_analytic_profile_distribution_unweighted_largest_remainder"
        ),
        "n_attitude_effects": sum(1 for r in rows if r.get("outcome_type") == "attitude"),
        "n_behavioral_effects": sum(1 for r in rows if r.get("outcome_type") in {"binary_behavior", "behavior"}),
    }


def _study_weights(study_id: Sequence[str]) -> np.ndarray:
    groups = _study_groups(study_id)
    weights = np.zeros(len(study_id), dtype=float)
    for sid, idx in groups.items():
        for i in idx:
            weights[i] = 1.0 / len(idx)
    return weights


def _fit_weighted_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray, *, intercept: bool) -> tuple[float, float]:
    if intercept:
        design = np.column_stack([np.ones(len(x)), x])
    else:
        design = x[:, None]
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    if intercept:
        return float(coef[0]), float(coef[1])
    return 0.0, float(coef[0])


def _fit_candidate(model_name: str, x: np.ndarray, y: np.ndarray, study_id: Sequence[str]) -> dict[str, float | str]:
    if model_name == "M0":
        return {"model_name": "M0", "alpha": 0.0, "lambda": 1.0}
    weights = _study_weights(study_id)
    if model_name == "M1":
        alpha, lam = _fit_weighted_linear(x, y, weights, intercept=False)
    elif model_name == "M2":
        alpha, lam = _fit_weighted_linear(x, y, weights, intercept=True)
    else:
        raise ValueError(f"unknown calibration model {model_name!r}")
    return {"model_name": model_name, "alpha": alpha, "lambda": lam}


def _predict(params: dict[str, float | str], x: np.ndarray) -> np.ndarray:
    return float(params["alpha"]) + float(params["lambda"]) * x


def fit_calibration_model_comparison(
    synthetic_effect_pp: Sequence[float],
    human_effect_pp: Sequence[float],
    study_id: Sequence[str],
    *,
    outputs_dir: Path | str | None = OUTPUTS_DIR,
) -> dict[str, object]:
    """Select M0/M1/M2 by whole-study LOSO RMSE with equal total study
    weighting in training. The selected model is refit on all eligible
    effects and represented by `(calibration_alpha, calibration_lambda)`.
    """
    x = np.asarray(synthetic_effect_pp, dtype=float)
    y = np.asarray(human_effect_pp, dtype=float)
    groups = _study_groups(study_id)
    study_ids = list(groups.keys())
    if len(study_ids) < 2:
        raise ValueError("need at least 2 distinct studies for leave-one-study-out validation")

    candidates = ["M0", "M1", "M2"]
    loso_sq: dict[str, list[float]] = {name: [] for name in candidates}
    loso_predictions: list[dict[str, object]] = []

    for held_out in study_ids:
        held_idx = groups[held_out]
        train_studies = [sid for sid in study_ids if sid != held_out]
        train_idx = [i for sid in train_studies for i in groups[sid]]
        for model_name in candidates:
            params = _fit_candidate(model_name, x[train_idx], y[train_idx], [study_id[i] for i in train_idx])
            pred = _predict(params, x[held_idx])
            mse = float(np.mean((pred - y[held_idx]) ** 2))
            loso_sq[model_name].append(mse)
            for i, y_hat in zip(held_idx, pred):
                loso_predictions.append(
                    {
                        "held_out_study_id": held_out,
                        "effect_index": i,
                        "model_name": model_name,
                        "alpha_train": params["alpha"],
                        "lambda_train": params["lambda"],
                        "synthetic_effect_pp": x[i],
                        "human_effect_pp": y[i],
                        "predicted_human_effect_pp": float(y_hat),
                        "squared_error": float((y_hat - y[i]) ** 2),
                    }
                )

    rmse = {name: float(np.sqrt(np.mean(loso_sq[name]))) for name in candidates}
    # Simpler model wins exact/tiny numerical ties.
    winner = min(candidates, key=lambda name: (round(rmse[name], 15), candidates.index(name)))
    selected = _fit_candidate(winner, x, y, study_id)
    selected_model = {
        "model_name": winner,
        "alpha": float(selected["alpha"]),
        "lambda": float(selected["lambda"]),
        "calibration_alpha": float(selected["alpha"]),
        "calibration_lambda": float(selected["lambda"]),
        "loso_rmse_M0": rmse["M0"],
        "loso_rmse_M1": rmse["M1"],
        "loso_rmse_M2": rmse["M2"],
        "number_studies": len(study_ids),
        "number_effects": len(x),
        "weighting_rule": "training effects weighted 1/n_s so every study has equal total fitting weight; held-out study MSEs averaged equally",
    }

    comparison_rows = [{"model_name": name, "loso_rmse": rmse[name], "selected": name == winner} for name in candidates]
    if outputs_dir is not None:
        out = Path(outputs_dir)
        out.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(comparison_rows).to_csv(out / "calibration_model_comparison.csv", index=False)
        pd.DataFrame(loso_predictions).to_csv(out / "calibration_loso_predictions.csv", index=False)
        (out / "calibration_selected_model.json").write_text(json.dumps(selected_model, indent=2) + "\n", encoding="utf-8")

    return {
        **selected_model,
        "rmse_loso": rmse,
        "comparison_rows": comparison_rows,
        "loso_predictions": loso_predictions,
    }


def load_treatment_families(path=None) -> dict[str, str]:
    """condition title -> treatment family, from survey/condition_codenames.csv's
    own "tag" column (Collaboration and peer-review / Scientific methods and
    results / Applications and impact / Others' endorsement / Values / Other)
    -- read-only, verbatim from the organizer file, not invented here.
    """
    path = path or (Path(__file__).resolve().parent.parent.parent / "survey" / "condition_codenames.csv")
    families: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = row["title"].strip()
            if title.lower() != "control":
                families[title] = row["tag"].strip()
    return families


def _study_groups(study_id: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(study_id):
        groups.setdefault(s, []).append(i)
    return groups


def fit_ate_calibration(model_ate_pp: Sequence[float], human_ate_pp: Sequence[float], study_id: Sequence[str]) -> dict:
    """Compatibility wrapper around the primary M0/M1/M2 calibration
    selector. Unlike the old slope-only routine, this does not constrain
    lambda to [0, 1] and never changes eligibility based on fit quality.
    """
    fit = fit_calibration_model_comparison(model_ate_pp, human_ate_pp, study_id, outputs_dir=None)
    fold_diagnostics = []
    selected_predictions = [p for p in fit["loso_predictions"] if p["model_name"] == fit["model_name"]]
    for held_out in sorted({p["held_out_study_id"] for p in selected_predictions}):
        rows = [p for p in selected_predictions if p["held_out_study_id"] == held_out]
        fold_diagnostics.append(
            {
                "held_out_study_id": held_out,
                "n_held_out_effects": len(rows),
                "alpha_train": rows[0]["alpha_train"],
                "lambda_ate_train": rows[0]["lambda_train"],
                "rmse_selected": float(np.sqrt(np.mean([r["squared_error"] for r in rows]))),
            }
        )
    return {
        **fit,
        "lambda_ate": fit["calibration_lambda"],
        "lambda_ate_unconstrained": fit["calibration_lambda"],
        "alpha_ate": fit["calibration_alpha"],
        "rmse_loso_raw": fit["loso_rmse_M0"],
        "rmse_loso_shrunk": fit[f"loso_rmse_{fit['model_name']}"],
        "n_studies": fit["number_studies"],
        "fold_diagnostics": fold_diagnostics,
    }


def _design_matrix(treatment_family: Sequence[str], outcome_family: Sequence[str]) -> tuple[np.ndarray, list[str], list[str]]:
    t_levels = sorted(set(treatment_family))
    o_levels = sorted(set(outcome_family))
    n = len(treatment_family)
    x_t = np.zeros((n, len(t_levels)))
    z_o = np.zeros((n, len(o_levels)))
    for i in range(n):
        x_t[i, t_levels.index(treatment_family[i])] = 1.0
        z_o[i, o_levels.index(outcome_family[i])] = 1.0
    design = np.hstack([np.ones((n, 1)), x_t, z_o])
    return design, t_levels, o_levels


def fit_hierarchical_shrinkage(
    model_ate_pp: Sequence[float],
    human_ate_pp: Sequence[float],
    study_id: Sequence[str],
    treatment_family: Sequence[str],
    outcome_family: Sequence[str],
    ridge_penalty: float = 4.0,
) -> dict:
    """logit(lambda_ate_to) = beta0 + x_t'beta + z_o'gamma, fit by minimizing
    the same per-study-weighted leave-one-study-out squared-error loss as
    fit_ate_calibration(), with an L2 (ridge) penalty on beta/gamma shrinking
    every family's lambda_ate toward the single global value (setting
    beta=gamma=0 reduces exactly to the global model, sigmoid(beta0)) --
    `ridge_penalty` is the ridge weight; larger values pull family effects
    closer to 0 (i.e., closer to the global-only model).
    """
    x = np.asarray(model_ate_pp, dtype=float)
    y = np.asarray(human_ate_pp, dtype=float)
    design, t_levels, o_levels = _design_matrix(treatment_family, outcome_family)
    n_beta = len(t_levels)
    n_gamma = len(o_levels)
    groups = _study_groups(study_id)
    study_ids = list(groups.keys())
    n_studies = len(study_ids)
    if n_studies < 2:
        raise ValueError("need at least 2 distinct studies for leave-one-study-out validation")

    def lambdas_for(params: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        logit = design[idx] @ params
        return 1.0 / (1.0 + np.exp(-logit))

    def loss(params: np.ndarray, idx: Sequence[int], study_ids_here: Sequence[str]) -> float:
        s = len(study_ids_here)
        idx_to_pos = {ix: p for p, ix in enumerate(idx)}
        total = 0.0
        for sid in study_ids_here:
            members = [i for i in groups[sid] if i in idx_to_pos]
            positions = [idx_to_pos[i] for i in members]
            lam = lambdas_for(params, [idx[p] for p in positions])
            err = np.mean((lam * x[[idx[p] for p in positions]] - y[[idx[p] for p in positions]]) ** 2)
            total += err / s
        ridge = ridge_penalty * float(np.sum(params[1:] ** 2))  # never penalize the intercept beta0
        return total + ridge

    def fit_on(idx: Sequence[int], study_ids_here: Sequence[str]) -> np.ndarray:
        x0 = np.zeros(1 + n_beta + n_gamma)
        result = minimize(loss, x0, args=(idx, study_ids_here), method="L-BFGS-B")
        return result.x

    def study_sq_err(fit_fn) -> list[float]:
        errs = []
        for held_out in study_ids:
            train_studies = [s for s in study_ids if s != held_out]
            train_idx = [i for s in train_studies for i in groups[s]]
            params = fit_fn(train_idx, train_studies)
            held_idx = groups[held_out]
            lam = lambdas_for(params, held_idx)
            errs.append(float(np.mean((lam * x[held_idx] - y[held_idx]) ** 2)))
        return errs

    rmse_loso_hierarchical = float(np.sqrt(np.mean(study_sq_err(fit_on))))
    final_params = fit_on(list(range(len(x))), study_ids)

    lambda_ate_by_family: dict[str, float] = {}
    for ti, t in enumerate(t_levels):
        for oi, o in enumerate(o_levels):
            row = np.zeros(len(final_params))
            row[0] = 1
            row[1 + ti] = 1
            row[1 + n_beta + oi] = 1
            lambda_ate_by_family[f"{t} x {o}"] = float(1.0 / (1.0 + np.exp(-row @ final_params)))

    return {
        "params": final_params.tolist(),
        "treatment_levels": t_levels,
        "outcome_levels": o_levels,
        "lambda_ate_by_family": lambda_ate_by_family,
        "rmse_loso_hierarchical": rmse_loso_hierarchical,
        "ridge_penalty": ridge_penalty,
    }


def select_shrinkage_specification(
    model_ate_pp: Sequence[float],
    human_ate_pp: Sequence[float],
    study_id: Sequence[str],
    treatment_family: Sequence[str],
    outcome_family: Sequence[str],
) -> dict:
    """Compare lambda_ate=1, the single global lambda_ate, and the
    hierarchical treatment/outcome-family lambda_ate on held-out
    (leave-one-study-out) RMSE; return whichever wins, with all three RMSEs
    reported for transparency.
    """
    global_fit = fit_ate_calibration(model_ate_pp, human_ate_pp, study_id)
    hierarchical_fit = fit_hierarchical_shrinkage(model_ate_pp, human_ate_pp, study_id, treatment_family, outcome_family)

    candidates = {
        "none": global_fit["rmse_loso_raw"],
        "global": global_fit["rmse_loso_shrunk"],
        "hierarchical": hierarchical_fit["rmse_loso_hierarchical"],
    }
    winner = min(candidates, key=candidates.get)
    return {
        "winner": winner,
        "rmse_loso": candidates,
        "global_fit": global_fit,
        "hierarchical_fit": hierarchical_fit,
    }
