"""Secondary-submission calibration study (OFFLINE DEVELOPMENT ANALYSIS ONLY).

Investigates the negative global slope in the frozen PRIMARY calibration
(M2: alpha=1.4508167066782651, lambda=-0.2425283492527821) and evaluates a
fixed, prospectively specified family of secondary calibration models
(MCONST, M2, M2R, M3, M3R) against the already-frozen 136-effect/31-study
external calibration table, using whole-study leave-one-study-out (LOSO)
validation with equal total weight per study throughout.

This script:
  - reads outputs/calibration_production/frozen_136_effect_calibration_table.csv
    and outputs/calibration_selected_model.json read-only;
  - reuses ate.calibrate_lambda's frozen, unmodified _fit_candidate /
    _study_groups / _study_weights for every "ordinary" (OLS) fit, so M2 here
    is byte-identical in method to the primary calibration;
  - never writes to any primary frozen artifact, target manifest, target
    production guard/ledger, or the scientific-bakeoff ledger;
  - performs no inference, accesses no target human outcomes;
  - does NOT select or freeze a secondary submission -- it only returns the
    evidence for a later, separate decision;
  - writes every output under outputs/secondary_calibration_diagnostic/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.linear_model import HuberRegressor

from ate.calibrate_lambda import _fit_candidate, _study_groups, _study_weights

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
FROZEN_TABLE_PATH = PIPELINE_ROOT / "outputs" / "calibration_production" / "frozen_136_effect_calibration_table.csv"
PRIMARY_ARTIFACT_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
OUT_DIR = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic"

BOOTSTRAP_SEED = 20260826  # frozen, as specified in the task -- disclosed verbatim, not tuned
N_BOOTSTRAP = 10_000

HUBER_KWARGS = dict(epsilon=1.345, alpha=0.0, fit_intercept=True, max_iter=10_000, tol=1e-10)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_frozen_table() -> pd.DataFrame:
    df = pd.read_csv(FROZEN_TABLE_PATH)
    required = {"study_id", "effect_id", "theta_L_pp", "theta_H_pp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"frozen 136-effect table missing columns: {sorted(missing)}")
    return df


def load_primary_frozen_m2() -> dict:
    d = json.loads(PRIMARY_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if d.get("model_name") != "M2":
        raise ValueError("primary frozen artifact is not M2 -- diagnostic assumptions no longer hold")
    return d


# ---------------------------------------------------------------------------
# Model fitting primitives
# ---------------------------------------------------------------------------

def _study_means(x: np.ndarray, study_id: list[str], studies_subset: list[str] | None = None) -> dict[str, float]:
    groups = _study_groups(study_id)
    keys = studies_subset if studies_subset is not None else list(groups.keys())
    return {s: float(np.mean(x[groups[s]])) for s in keys}


def fit_m2_ols(x: np.ndarray, y: np.ndarray, study_id: list[str]) -> dict:
    params = _fit_candidate("M2", x, y, study_id)
    return {"alpha": float(params["alpha"]), "lambda": float(params["lambda"])}


def predict_m2(params: dict, x: np.ndarray) -> np.ndarray:
    return params["alpha"] + params["lambda"] * x


def fit_m2_huber(x: np.ndarray, y: np.ndarray, study_id: list[str]) -> dict:
    w = _study_weights(study_id)
    huber = HuberRegressor(**HUBER_KWARGS)
    huber.fit(x[:, None], y, sample_weight=w)
    return {"alpha": float(huber.intercept_), "lambda": float(huber.coef_[0])}


def fit_m3_ols(x: np.ndarray, y: np.ndarray, study_id: list[str], xbar_by_study: dict[str, float]) -> dict:
    xbar = np.array([xbar_by_study[s] for s in study_id])
    within = x - xbar
    w = _study_weights(study_id)
    design = np.column_stack([np.ones(len(x)), xbar, within])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    return {"alpha": float(coef[0]), "beta_B": float(coef[1]), "beta_W": float(coef[2])}


def predict_m3(params: dict, x: np.ndarray, xbar: np.ndarray) -> np.ndarray:
    return params["alpha"] + params["beta_B"] * xbar + params["beta_W"] * (x - xbar)


def fit_m3_huber(x: np.ndarray, y: np.ndarray, study_id: list[str], xbar_by_study: dict[str, float]) -> dict:
    xbar = np.array([xbar_by_study[s] for s in study_id])
    within = x - xbar
    w = _study_weights(study_id)
    design = np.column_stack([xbar, within])
    huber = HuberRegressor(**HUBER_KWARGS)
    huber.fit(design, y, sample_weight=w)
    return {"alpha": float(huber.intercept_), "beta_B": float(huber.coef_[0]), "beta_W": float(huber.coef_[1])}


def fit_mconst(y: np.ndarray, study_id: list[str]) -> float:
    w = _study_weights(study_id)
    return float(np.average(y, weights=w))


# ---------------------------------------------------------------------------
# Section 1: global-slope stability / influence
# ---------------------------------------------------------------------------

def section1_influence(x: np.ndarray, y: np.ndarray, study_id: list[str], lambda_full: float) -> tuple[pd.DataFrame, dict]:
    groups = _study_groups(study_id)
    studies = sorted(groups)
    rows = []
    for held in studies:
        train_studies = [s for s in studies if s != held]
        train_idx = [i for s in train_studies for i in groups[s]]
        params = fit_m2_ols(x[train_idx], y[train_idx], [study_id[i] for i in train_idx])
        rows.append({"study_id": held, "alpha_minus_s": params["alpha"], "lambda_minus_s": params["lambda"]})
    df = pd.DataFrame(rows).sort_values("study_id").reset_index(drop=True)

    lam = df["lambda_minus_s"].to_numpy()
    summary = {
        "lambda_loso_delete_min": float(np.min(lam)),
        "lambda_loso_delete_median": float(np.median(lam)),
        "lambda_loso_delete_max": float(np.max(lam)),
        "n_negative": int(np.sum(lam < 0)),
        "n_positive": int(np.sum(lam > 0)),
        "n_sign_changes": int(np.sum(np.sign(lam) != np.sign(lambda_full))),
        "most_influential_study_for_lambda": df.loc[(df["lambda_minus_s"] - lambda_full).abs().idxmax(), "study_id"],
    }
    return df, summary


def section1_bootstrap(x: np.ndarray, y: np.ndarray, study_id: list[str], *, seed: int, n_boot: int) -> tuple[dict, np.ndarray]:
    groups = _study_groups(study_id)
    studies = np.array(sorted(groups))
    rng = np.random.default_rng(seed)
    lambdas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(studies, size=len(studies), replace=True)
        idx_list: list[int] = []
        sid_list: list[str] = []
        for copy_num, s in enumerate(draw):
            idx = groups[s]
            idx_list.extend(idx)
            sid_list.extend([f"{s}__boot{copy_num}"] * len(idx))
        params = fit_m2_ols(x[idx_list], y[idx_list], sid_list)
        lambdas[b] = params["lambda"]
    summary = {
        "n_bootstrap": n_boot,
        "seed": seed,
        "lambda_bootstrap_mean": float(np.mean(lambdas)),
        "lambda_bootstrap_median": float(np.median(lambdas)),
        "lambda_bootstrap_p025": float(np.percentile(lambdas, 2.5)),
        "lambda_bootstrap_p975": float(np.percentile(lambdas, 97.5)),
        "p_lambda_lt_0": float(np.mean(lambdas < 0)),
    }
    return summary, lambdas


# ---------------------------------------------------------------------------
# Section 2: within-study vs between-study decomposition
# ---------------------------------------------------------------------------

def section2_decomposition(df: pd.DataFrame) -> dict:
    study_means = df.groupby("study_id").agg(xbar_s=("theta_L_pp", "mean"), ybar_s=("theta_H_pp", "mean"))
    between_pearson = float(np.corrcoef(study_means["xbar_s"], study_means["ybar_s"])[0, 1])
    between_spearman = float(stats.spearmanr(study_means["xbar_s"], study_means["ybar_s"]).correlation)
    between_slope, *_ = stats.linregress(study_means["xbar_s"], study_means["ybar_s"])[:2]

    merged = df.merge(study_means, left_on="study_id", right_index=True)
    merged["x_within"] = merged["theta_L_pp"] - merged["xbar_s"]
    merged["y_within"] = merged["theta_H_pp"] - merged["ybar_s"]

    within_pearson = float(np.corrcoef(merged["x_within"], merged["y_within"])[0, 1])
    within_spearman = float(stats.spearmanr(merged["x_within"], merged["y_within"]).correlation)

    weights = _study_weights(merged["study_id"].tolist())
    sw = np.sqrt(weights)
    design = merged["x_within"].to_numpy()[:, None]
    coef, *_ = np.linalg.lstsq(design * sw[:, None], merged["y_within"].to_numpy() * sw, rcond=None)
    within_ols_slope = float(coef[0])

    reversal = (np.sign(between_slope) != np.sign(within_ols_slope)) and between_slope != 0 and within_ols_slope != 0

    return {
        "between_study_pearson": between_pearson,
        "between_study_spearman": between_spearman,
        "between_study_ols_slope": float(between_slope),
        "within_study_centered_pearson": within_pearson,
        "within_study_centered_spearman": within_spearman,
        "within_study_centered_ols_slope": within_ols_slope,
        "within_between_sign_reversal": "YES" if reversal else "NO",
    }


# ---------------------------------------------------------------------------
# Section 5+7: whole-study outer LOSO validation for all 5 models,
# capturing M3/M3R per-fold coefficients for the section-7 stability table.
# ---------------------------------------------------------------------------

def run_loso_validation(x: np.ndarray, y: np.ndarray, study_id: list[str]) -> dict:
    groups = _study_groups(study_id)
    studies = sorted(groups)
    models = ["MCONST", "M2", "M2R", "M3", "M3R"]
    per_study_mse = {m: [] for m in models}
    per_study_mae = {m: [] for m in models}
    fold_predictions = []
    m3_fold_coefs = []
    m3r_fold_coefs = []

    for held in studies:
        held_idx = groups[held]
        train_studies = [s for s in studies if s != held]
        train_idx = [i for s in train_studies for i in groups[s]]
        x_train, y_train = x[train_idx], y[train_idx]
        sid_train = [study_id[i] for i in train_idx]

        xbar_train = _study_means(x, study_id, train_studies)
        xbar_held = float(np.mean(x[held_idx]))  # held-out study's own F/synthetic values only

        mconst_pred_val = fit_mconst(y_train, sid_train)
        preds_mconst = np.full(len(held_idx), mconst_pred_val)

        m2_params = fit_m2_ols(x_train, y_train, sid_train)
        preds_m2 = predict_m2(m2_params, x[held_idx])

        m2r_params = fit_m2_huber(x_train, y_train, sid_train)
        preds_m2r = predict_m2(m2r_params, x[held_idx])

        m3_params = fit_m3_ols(x_train, y_train, sid_train, xbar_train)
        preds_m3 = predict_m3(m3_params, x[held_idx], np.full(len(held_idx), xbar_held))
        m3_fold_coefs.append({"study_id": held, "beta_B_minus_s": m3_params["beta_B"], "beta_W_minus_s": m3_params["beta_W"]})

        m3r_params = fit_m3_huber(x_train, y_train, sid_train, xbar_train)
        preds_m3r = predict_m3(m3r_params, x[held_idx], np.full(len(held_idx), xbar_held))
        m3r_fold_coefs.append({"study_id": held, "beta_B_minus_s": m3r_params["beta_B"], "beta_W_minus_s": m3r_params["beta_W"]})

        y_true = y[held_idx]
        preds_by_model = {"MCONST": preds_mconst, "M2": preds_m2, "M2R": preds_m2r, "M3": preds_m3, "M3R": preds_m3r}
        for m, pred in preds_by_model.items():
            mse = float(np.mean((pred - y_true) ** 2))
            mae = float(np.mean(np.abs(pred - y_true)))
            per_study_mse[m].append({"study_id": held, "mse": mse})
            per_study_mae[m].append(mae)
            for i, eff_idx in enumerate(held_idx):
                fold_predictions.append(
                    {
                        "model": m,
                        "held_out_study_id": held,
                        "effect_id": None,  # filled by caller using original df
                        "effect_index": eff_idx,
                        "synthetic_theta_L_pp": float(x[eff_idx]),
                        "human_theta_H_pp": float(y_true[i]),
                        "predicted_pp": float(pred[i]),
                        "squared_error": float((pred[i] - y_true[i]) ** 2),
                    }
                )

    loso_table = []
    for m in models:
        mses = [r["mse"] for r in per_study_mse[m]]
        loso_mse = float(np.mean(mses))
        loso_table.append(
            {
                "model": m,
                "loso_mse": loso_mse,
                "loso_rmse": float(np.sqrt(loso_mse)),
                "loso_mae": float(np.mean(per_study_mae[m])),
            }
        )

    return {
        "loso_table": loso_table,
        "per_study_mse": per_study_mse,
        "fold_predictions": fold_predictions,
        "m3_fold_coefs": m3_fold_coefs,
        "m3r_fold_coefs": m3r_fold_coefs,
    }


# ---------------------------------------------------------------------------
# Section 6: full-data fits for interpretation
# ---------------------------------------------------------------------------

def section6_full_fits(x: np.ndarray, y: np.ndarray, study_id: list[str]) -> dict:
    xbar_all = _study_means(x, study_id)
    out = {}

    m2 = fit_m2_ols(x, y, study_id)
    fitted = predict_m2(m2, x)
    out["M2"] = {
        **m2,
        "fitted_min": float(np.min(fitted)),
        "fitted_max": float(np.max(fitted)),
        "largest_abs_residual": float(np.max(np.abs(fitted - y))),
        "pearson_fitted_vs_human": float(np.corrcoef(fitted, y)[0, 1]),
        "spearman_fitted_vs_human": float(stats.spearmanr(fitted, y).correlation),
    }

    m2r = fit_m2_huber(x, y, study_id)
    fitted = predict_m2(m2r, x)
    out["M2R"] = {
        **m2r,
        "fitted_min": float(np.min(fitted)),
        "fitted_max": float(np.max(fitted)),
        "largest_abs_residual": float(np.max(np.abs(fitted - y))),
        "pearson_fitted_vs_human": float(np.corrcoef(fitted, y)[0, 1]),
        "spearman_fitted_vs_human": float(stats.spearmanr(fitted, y).correlation),
    }

    xbar_arr = np.array([xbar_all[s] for s in study_id])
    m3 = fit_m3_ols(x, y, study_id, xbar_all)
    fitted = predict_m3(m3, x, xbar_arr)
    out["M3"] = {
        **m3,
        "fitted_min": float(np.min(fitted)),
        "fitted_max": float(np.max(fitted)),
        "largest_abs_residual": float(np.max(np.abs(fitted - y))),
        "pearson_fitted_vs_human": float(np.corrcoef(fitted, y)[0, 1]),
        "spearman_fitted_vs_human": float(stats.spearmanr(fitted, y).correlation),
    }

    m3r = fit_m3_huber(x, y, study_id, xbar_all)
    fitted = predict_m3(m3r, x, xbar_arr)
    out["M3R"] = {
        **m3r,
        "fitted_min": float(np.min(fitted)),
        "fitted_max": float(np.max(fitted)),
        "largest_abs_residual": float(np.max(np.abs(fitted - y))),
        "pearson_fitted_vs_human": float(np.corrcoef(fitted, y)[0, 1]),
        "spearman_fitted_vs_human": float(stats.spearmanr(fitted, y).correlation),
    }
    return out


# ---------------------------------------------------------------------------
# Section 7: coefficient stability of M3 / M3R across the 31 LOSO folds
# ---------------------------------------------------------------------------

def section7_stability(fold_coefs: list[dict]) -> dict:
    beta_w = np.array([r["beta_W_minus_s"] for r in fold_coefs])
    beta_b = np.array([r["beta_B_minus_s"] for r in fold_coefs])
    return {
        "beta_w_min": float(np.min(beta_w)),
        "beta_w_median": float(np.median(beta_w)),
        "beta_w_max": float(np.max(beta_w)),
        "n_beta_w_positive": int(np.sum(beta_w > 0)),
        "n_beta_w_negative": int(np.sum(beta_w < 0)),
        "beta_b_min": float(np.min(beta_b)),
        "beta_b_median": float(np.median(beta_b)),
        "beta_b_max": float(np.max(beta_b)),
    }


def main() -> dict:
    df = load_frozen_table()
    x = df["theta_L_pp"].to_numpy(dtype=float)
    y = df["theta_H_pp"].to_numpy(dtype=float)
    study_id = df["study_id"].tolist()

    primary = load_primary_frozen_m2()
    lambda_full = float(primary["lambda"])
    primary_hash_before = _sha256_file(PRIMARY_ARTIFACT_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Section 1 ---
    influence_df, influence_summary = section1_influence(x, y, study_id, lambda_full)
    influence_df.to_csv(OUT_DIR / "loso_delete_alpha_lambda.csv", index=False)

    bootstrap_summary, boot_draws = section1_bootstrap(x, y, study_id, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
    pd.DataFrame({"lambda_bootstrap_draw": boot_draws}).to_csv(OUT_DIR / "lambda_bootstrap_draws.csv", index=False)

    # --- Section 2 ---
    section2 = section2_decomposition(df)

    # --- Sections 5 & 7 (LOSO run, shared) ---
    loso = run_loso_validation(x, y, study_id)
    loso_table = loso["loso_table"]
    per_study_mse_rows = []
    for m, rows in loso["per_study_mse"].items():
        for r in rows:
            per_study_mse_rows.append({"model": m, **r})
    pd.DataFrame(per_study_mse_rows).to_csv(OUT_DIR / "loso_per_study_mse.csv", index=False)

    id_by_index = df["effect_id"].to_dict()
    for row in loso["fold_predictions"]:
        row["effect_id"] = id_by_index[row["effect_index"]]
    pd.DataFrame(loso["fold_predictions"]).to_csv(OUT_DIR / "loso_fold_predictions.csv", index=False)

    pd.DataFrame(loso["m3_fold_coefs"]).to_csv(OUT_DIR / "m3_loso_fold_coefficients.csv", index=False)
    pd.DataFrame(loso["m3r_fold_coefs"]).to_csv(OUT_DIR / "m3r_loso_fold_coefficients.csv", index=False)

    rmse_by_model = {r["model"]: r["loso_rmse"] for r in loso_table}

    # --- Section 6 ---
    full_fits = section6_full_fits(x, y, study_id)

    # --- Section 7 ---
    m3_stability = section7_stability(loso["m3_fold_coefs"])
    m3r_stability = section7_stability(loso["m3r_fold_coefs"])

    # --- Section 8: interpretation only ---
    global_negative_slope_stable = bool(np.max(influence_df["lambda_minus_s"]) < 0)
    robust_m2_slope_sign = "negative" if full_fits["M2R"]["lambda"] < 0 else ("positive" if full_fits["M2R"]["lambda"] > 0 else "zero")
    within_sign = "positive" if section2["within_study_centered_ols_slope"] > 0 else ("negative" if section2["within_study_centered_ols_slope"] < 0 else "zero")
    between_sign = "positive" if section2["between_study_ols_slope"] > 0 else ("negative" if section2["between_study_ols_slope"] < 0 else "zero")

    best_model = min(rmse_by_model, key=rmse_by_model.get)
    beta_w_m3r = np.array([r["beta_W_minus_s"] for r in loso["m3r_fold_coefs"]])
    beta_w_m3r_sign_stable = bool(np.all(beta_w_m3r > 0) or np.all(beta_w_m3r < 0))

    section8 = {
        "global_negative_slope_stable": "YES" if global_negative_slope_stable else "NO",
        "robust_m2_slope_sign": robust_m2_slope_sign,
        "within_study_slope_sign": within_sign,
        "between_study_slope_sign": between_sign,
        "within_between_sign_reversal": section2["within_between_sign_reversal"],
        "m2r_improves_m2": "YES" if rmse_by_model["M2R"] < rmse_by_model["M2"] else "NO",
        "m3_improves_m2": "YES" if rmse_by_model["M3"] < rmse_by_model["M2"] else "NO",
        "m3r_improves_m2": "YES" if rmse_by_model["M3R"] < rmse_by_model["M2"] else "NO",
        "m3r_improves_m3": "YES" if rmse_by_model["M3R"] < rmse_by_model["M3"] else "NO",
        "m3r_improves_m2r": "YES" if rmse_by_model["M3R"] < rmse_by_model["M2R"] else "NO",
        "best_loso_model": best_model,
        "best_loso_rmse": rmse_by_model[best_model],
        "m3r_beta_b": full_fits["M3R"]["beta_B"],
        "m3r_beta_w": full_fits["M3R"]["beta_W"],
        "m3r_beta_w_sign_stable": "YES" if beta_w_m3r_sign_stable else "NO",
    }

    primary_hash_after = _sha256_file(PRIMARY_ARTIFACT_PATH)

    result = {
        "note": "OFFLINE DEVELOPMENT ANALYSIS ONLY -- no secondary submission is chosen or frozen here; primary calibration/target artifacts are untouched",
        "input_table_path": str(FROZEN_TABLE_PATH.relative_to(PIPELINE_ROOT)),
        "input_table_sha256": _sha256_file(FROZEN_TABLE_PATH),
        "primary_frozen_m2": {"alpha": float(primary["alpha"]), "lambda": lambda_full},
        "primary_artifact_sha256_before": primary_hash_before,
        "primary_artifact_sha256_after": primary_hash_after,
        "primary_artifacts_unchanged": primary_hash_before == primary_hash_after,
        "huber_kwargs": HUBER_KWARGS,
        "section1_influence": {**influence_summary},
        "section1_bootstrap": bootstrap_summary,
        "section2_decomposition": section2,
        "section5_loso_table": loso_table,
        "section6_full_data_fits": full_fits,
        "section7_m3_stability": m3_stability,
        "section7_m3r_stability": m3r_stability,
        "section8_interpretation": section8,
        "package_versions": {"sklearn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "secondary_diagnostic_complete": True,
    }

    summary_path = OUT_DIR / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["secondary_diagnostic_sha256"] = _sha256_file(summary_path)
    (OUT_DIR / "diagnostic_summary.sha256.txt").write_text(result["secondary_diagnostic_sha256"] + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
