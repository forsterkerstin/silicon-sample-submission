"""Reusable external validation diagnostics for the frozen native G protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from ate.normalize_effects import OUTCOME_SCALE_BOUNDS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "validation_g"
MODERATORS = ["gender", "age_band", "race", "education", "income", "party"]


def validate_g_against_human(
    g_responses: pd.DataFrame,
    human_responses: pd.DataFrame,
    *,
    outcomes: Sequence[str] | None = None,
    moderators: Sequence[str] = MODERATORS,
    outputs_dir: Path | str = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Compare native G and external human respondent-level distributions.

    This is diagnostic-only. It never mutates G responses and never tunes the
    submission toward hidden benchmark outcomes.
    """
    outcomes = list(outcomes or OUTCOME_SCALE_BOUNDS.keys())
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    distribution_rows = []
    subgroup_rows = []
    for outcome in outcomes:
        if outcome not in g_responses.columns or outcome not in human_responses.columns:
            continue
        g = g_responses[outcome].dropna().astype(float)
        h = human_responses[outcome].dropna().astype(float)
        if g.empty or h.empty:
            continue
        h_var = float(h.var(ddof=0))
        hist_range = (min(float(g.min()), float(h.min())), max(float(g.max()), float(h.max())))
        if hist_range[0] == hist_range[1]:
            overlap = 1.0
        else:
            g_hist, _ = np.histogram(g, bins=20, range=hist_range, density=True)
            h_hist, edges = np.histogram(h, bins=20, range=hist_range, density=True)
            overlap = float(np.minimum(g_hist, h_hist).sum() * (edges[1] - edges[0]))
        distribution_rows.append(
            {
                "outcome": outcome,
                "g_mean": float(g.mean()),
                "human_mean": float(h.mean()),
                "mean_error": float(g.mean() - h.mean()),
                "variance_ratio": float(g.var(ddof=0) / h_var) if h_var > 0 else np.nan,
                "ks": float(ks_2samp(g, h).statistic),
                "wasserstein_1": float(wasserstein_distance(g, h)),
                "overlap": overlap,
            }
        )
        for moderator in moderators:
            if moderator not in g_responses.columns or moderator not in human_responses.columns:
                continue
            common_levels = sorted(set(g_responses[moderator].dropna().astype(str)) & set(human_responses[moderator].dropna().astype(str)))
            for level in common_levels:
                gv = g_responses.loc[g_responses[moderator].astype(str) == level, outcome].dropna().astype(float)
                hv = human_responses.loc[human_responses[moderator].astype(str) == level, outcome].dropna().astype(float)
                if gv.empty or hv.empty:
                    continue
                subgroup_rows.append(
                    {
                        "outcome": outcome,
                        "moderator": moderator,
                        "level": level,
                        "g_mean": float(gv.mean()),
                        "human_mean": float(hv.mean()),
                        "mean_error": float(gv.mean() - hv.mean()),
                        "n_g": len(gv),
                        "n_human": len(hv),
                    }
                )
    distribution = pd.DataFrame(distribution_rows)
    subgroup = pd.DataFrame(subgroup_rows)
    if not subgroup.empty:
        subgroup["subgroup_mean_rmse"] = subgroup.groupby(["outcome", "moderator"])["mean_error"].transform(lambda x: float(np.sqrt(np.mean(x**2))))
    distribution.to_csv(outputs / "distribution_metrics.csv", index=False)
    subgroup.to_csv(outputs / "subgroup_mean_errors.csv", index=False)
    return {"distribution_metrics": distribution, "subgroup_mean_errors": subgroup}
