"""Final secondary-calibration decision study (OFFLINE ONLY).

Evaluates exactly one additional, prospectively fixed candidate against the
already-established MCONST baseline: MSHRINK, a convex shrinkage of the
ordinary M2 prediction toward the study-equal constant mean,

    yhat_SHRINK = mu_train + w * (yhat_M2 - mu_train),   w in {0.0, 0.1, ..., 1.0}

selected by strictly nested whole-study leave-one-study-out CV (inner loop
picks w from the 30 training studies only; outer loop scores the held-out
study). Reuses the frozen M2 fitting primitives from
ate.calibrate_lambda / scripts.secondary_calibration_diagnostic unmodified.

Applies one prospective rule (MSHRINK wins iff its nested LOSO MSE is
strictly less than MCONST's outer-fold MSE on the identical 31 folds; ties
go to MCONST) to select and freeze ONE secondary calibration artifact. Does
not reopen the model family, does not touch the primary calibration
artifact / frozen method manifest / target manifests / target-production
guard state, performs no inference, and uses no target human outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
for _p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ate.calibrate_lambda import _study_groups, _study_weights  # noqa: E402
from secondary_calibration_diagnostic import (  # noqa: E402
    FROZEN_TABLE_PATH,
    PRIMARY_ARTIFACT_PATH,
    _sha256_file,
    fit_m2_ols,
    load_frozen_table,
    load_primary_frozen_m2,
    predict_m2,
)

OUT_DIR = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic"
W_GRID = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0 -- fixed, not tuned beyond this grid


def mu_study_equal(y: np.ndarray, study_id: list[str]) -> float:
    w = _study_weights(study_id)
    return float(np.average(y, weights=w))


def _select_w_by_loso(x: np.ndarray, y: np.ndarray, study_id: list[str], groups: dict[str, list[int]], candidate_studies: list[str]) -> tuple[float, dict[float, float]]:
    """Leave-one-study-out CV over exactly `candidate_studies`, picking w
    from W_GRID by lowest mean (equal-study-weighted) held-out MSE.
    Exact ties go to the smaller w."""
    scores = {w: [] for w in W_GRID}
    for held in candidate_studies:
        train_studies = [s for s in candidate_studies if s != held]
        train_idx = [i for s in train_studies for i in groups[s]]
        held_idx = groups[held]
        x_tr, y_tr = x[train_idx], y[train_idx]
        sid_tr = [study_id[i] for i in train_idx]
        mu = mu_study_equal(y_tr, sid_tr)
        m2 = fit_m2_ols(x_tr, y_tr, sid_tr)
        yhat_m2 = predict_m2(m2, x[held_idx])
        y_true = y[held_idx]
        for w in W_GRID:
            yhat = mu + w * (yhat_m2 - mu)
            scores[w].append(float(np.mean((yhat - y_true) ** 2)))
    mean_scores = {w: float(np.mean(v)) for w, v in scores.items()}
    best_w = min(W_GRID, key=lambda w: (round(mean_scores[w], 12), w))
    return best_w, mean_scores


def nested_outer_loop(x: np.ndarray, y: np.ndarray, study_id: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _study_groups(study_id)
    studies = sorted(groups)
    shrink_rows = []
    mconst_rows = []

    for outer_held in studies:
        train_studies = [s for s in studies if s != outer_held]
        held_idx = groups[outer_held]
        y_true = y[held_idx]

        # --- inner LOSO over the 30 training studies selects w ---
        best_w, _inner_scores = _select_w_by_loso(x, y, study_id, groups, train_studies)

        # --- refit on the FULL 30 training studies, predict the held-out study's F values ---
        train_idx = [i for s in train_studies for i in groups[s]]
        x_tr, y_tr = x[train_idx], y[train_idx]
        sid_tr = [study_id[i] for i in train_idx]
        mu_train = mu_study_equal(y_tr, sid_tr)
        m2_train = fit_m2_ols(x_tr, y_tr, sid_tr)
        yhat_m2 = predict_m2(m2_train, x[held_idx])
        yhat_shrink = mu_train + best_w * (yhat_m2 - mu_train)

        outer_mse = float(np.mean((yhat_shrink - y_true) ** 2))
        outer_mae = float(np.mean(np.abs(yhat_shrink - y_true)))
        shrink_rows.append({"outer_study": outer_held, "inner_selected_w": best_w, "outer_mse": outer_mse, "outer_mae": outer_mae})

        mconst_mse = float(np.mean((mu_train - y_true) ** 2))
        mconst_mae = float(np.mean(np.abs(mu_train - y_true)))
        mconst_rows.append({"outer_study": outer_held, "mconst_pred": mu_train, "outer_mse": mconst_mse, "outer_mae": mconst_mae})

    return pd.DataFrame(shrink_rows), pd.DataFrame(mconst_rows)


def final_all_development_fit(x: np.ndarray, y: np.ndarray, study_id: list[str], *, secondary_selected: str) -> dict:
    groups = _study_groups(study_id)
    studies = sorted(groups)
    mu_all = mu_study_equal(y, study_id)

    if secondary_selected == "MCONST":
        return {"secondary_mu": mu_all, "secondary_w": None, "secondary_alpha": None, "secondary_lambda": None}

    final_w, _ = _select_w_by_loso(x, y, study_id, groups, studies)
    m2_all = fit_m2_ols(x, y, study_id)
    return {
        "secondary_mu": mu_all,
        "secondary_w": final_w,
        "secondary_alpha": m2_all["alpha"],
        "secondary_lambda": m2_all["lambda"],
    }


def main() -> dict:
    df = load_frozen_table()
    x = df["theta_L_pp"].to_numpy(dtype=float)
    y = df["theta_H_pp"].to_numpy(dtype=float)
    study_id = df["study_id"].tolist()

    primary = load_primary_frozen_m2()
    primary_hash_before = _sha256_file(PRIMARY_ARTIFACT_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shrink_df, mconst_df = nested_outer_loop(x, y, study_id)
    shrink_df.to_csv(OUT_DIR / "mshrink_nested_outer_fold_results.csv", index=False)
    mconst_df.to_csv(OUT_DIR / "mconst_outer_fold_results.csv", index=False)

    mshrink_mse = float(shrink_df["outer_mse"].mean())
    mshrink_mae = float(shrink_df["outer_mae"].mean())
    mconst_mse = float(mconst_df["outer_mse"].mean())
    mconst_mae = float(mconst_df["outer_mae"].mean())

    w_vals = shrink_df["inner_selected_w"].to_numpy()
    w_summary = {
        "w_min": float(np.min(w_vals)),
        "w_median": float(np.median(w_vals)),
        "w_max": float(np.max(w_vals)),
        "n_w_eq_0": int(np.sum(w_vals == 0.0)),
        "n_w_le_0_2": int(np.sum(w_vals <= 0.2)),
        "n_w_ge_0_8": int(np.sum(w_vals >= 0.8)),
    }

    # --- prospective selection rule (ties go to MCONST) ---
    secondary_selected = "MSHRINK" if mshrink_mse < mconst_mse else "MCONST"

    final_fit = final_all_development_fit(x, y, study_id, secondary_selected=secondary_selected)

    effective_f_dependence = 0.0 if secondary_selected == "MCONST" else final_fit["secondary_w"]

    secondary_artifact = {
        "post_primary_external_development": True,
        "target_human_outcome_blind": True,
        "secondary_calibration_selected": secondary_selected,
        "secondary_mu": final_fit["secondary_mu"],
        "secondary_w": final_fit["secondary_w"],
        "secondary_alpha": final_fit["secondary_alpha"],
        "secondary_lambda": final_fit["secondary_lambda"],
        "effective_secondary_f_dependence": effective_f_dependence,
        "selection_rule": "MSHRINK selected iff MSHRINK_NESTED_LOSO_MSE < MCONST_OUTER_MSE on identical 31 outer whole-study folds; ties (including exact) go to MCONST",
        "mshrink_nested_loso_mse": mshrink_mse,
        "mshrink_nested_loso_rmse": float(np.sqrt(mshrink_mse)),
        "mshrink_nested_loso_mae": mshrink_mae,
        "mconst_outer_mse": mconst_mse,
        "mconst_outer_rmse": float(np.sqrt(mconst_mse)),
        "mconst_outer_mae": mconst_mae,
        "w_grid": W_GRID,
        "source_table": str(FROZEN_TABLE_PATH.relative_to(PIPELINE_ROOT)),
        "source_table_sha256": _sha256_file(FROZEN_TABLE_PATH),
        "primary_frozen_m2_for_reference": {"alpha": float(primary["alpha"]), "lambda": float(primary["lambda"])},
    }
    artifact_path = OUT_DIR / "secondary_calibration_selected_model.json"
    artifact_path.write_text(json.dumps(secondary_artifact, indent=2) + "\n", encoding="utf-8")

    primary_hash_after = _sha256_file(PRIMARY_ARTIFACT_PATH)

    result = {
        "note": "OFFLINE ONLY -- freezes a SECONDARY calibration artifact; primary calibration/target artifacts untouched",
        "mconst_outer_rmse": secondary_artifact["mconst_outer_rmse"],
        "mconst_outer_mse": mconst_mse,
        "mconst_outer_mae": mconst_mae,
        "mshrink_nested_loso_rmse": secondary_artifact["mshrink_nested_loso_rmse"],
        "mshrink_nested_loso_mse": mshrink_mse,
        "mshrink_nested_loso_mae": mshrink_mae,
        "w_summary": w_summary,
        "secondary_calibration_selected": secondary_selected,
        "secondary_mu": final_fit["secondary_mu"],
        "secondary_w": final_fit["secondary_w"],
        "secondary_alpha": final_fit["secondary_alpha"],
        "secondary_lambda": final_fit["secondary_lambda"],
        "effective_secondary_f_dependence": effective_f_dependence,
        "primary_artifact_sha256_before": primary_hash_before,
        "primary_artifact_sha256_after": primary_hash_after,
        "primary_artifacts_unchanged": primary_hash_before == primary_hash_after,
        "target_human_outcomes_used": False,
        "new_paid_inference_performed": False,
        "secondary_artifact_path": str(artifact_path.relative_to(PIPELINE_ROOT)),
    }

    summary_path = OUT_DIR / "secondary_calibration_decision_summary.json"
    result_for_disk = {**result, **secondary_artifact}
    summary_path.write_text(json.dumps(result_for_disk, indent=2) + "\n", encoding="utf-8")

    result["secondary_artifact_sha256"] = _sha256_file(artifact_path)
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
