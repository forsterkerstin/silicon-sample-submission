"""Generic statistical primitives shared by the two domain-validation
exercises (Howe 2019 domain-specific G confirmation; Orchinik 2024 S2
relative-ATE-shape validation).

Pure functions of caller-supplied data -- no dependency on target G/F
output, no dependency on real Howe/Orchinik data (that lives in
scripts/compute_orchinik_human_ate_surface.py and the frozen protocol).
Tested exclusively against synthetic fixtures (tests/ate/
test_domain_validation_metrics.py) per the domain-validation freeze spec.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

# ---------------------------------------------------------------------------
# Howe: arm-equal normalized Wasserstein-1 loss
# ---------------------------------------------------------------------------


def wasserstein1(human_samples: Sequence[float], synthetic_samples: Sequence[float]) -> float:
    """1-Wasserstein (earth mover's) distance between two empirical
    distributions on the real line: the classic closed form for 1D
    distributions -- integral of |CDF_1 - CDF_2}| -- computed via sorted
    order statistics (mean absolute difference of the two sorted samples,
    resampled to a common size via order-statistic interpolation when
    sample sizes differ)."""
    if not human_samples or not synthetic_samples:
        raise ValueError("both sample sets must be non-empty")
    h = sorted(human_samples)
    s = sorted(synthetic_samples)
    nh, ns = len(h), len(s)
    if nh == ns:
        return sum(abs(a - b) for a, b in zip(h, s)) / nh
    # order-statistic interpolation onto a common [0,1] quantile grid of size lcm-free
    # (use nh*ns evaluation points via each sample's own quantile function -- exact for
    # piecewise-constant empirical CDFs)
    n = nh * ns
    total = 0.0
    for i in range(1, n + 1):
        q = (i - 0.5) / n
        h_idx = min(nh - 1, int(q * nh))
        s_idx = min(ns - 1, int(q * ns))
        total += abs(h[h_idx] - s[s_idx])
    return total / n


def arm_equal_wasserstein_loss(human_by_arm: Mapping[str, Sequence[float]], synthetic_by_arm: Mapping[str, Sequence[float]], *, scale: float = 100.0) -> dict:
    """L = scale * (1/|arms|) * sum_arm W1(human_arm, synthetic_arm) --
    every arm gets equal total weight regardless of its N (Howe Section 3C /
    the ATP G-screen philosophy)."""
    if set(human_by_arm.keys()) != set(synthetic_by_arm.keys()):
        raise ValueError(f"arm sets differ: human={sorted(human_by_arm)} synthetic={sorted(synthetic_by_arm)}")
    arms = sorted(human_by_arm)
    per_arm = {arm: scale * wasserstein1(human_by_arm[arm], synthetic_by_arm[arm]) for arm in arms}
    loss = sum(per_arm.values()) / len(arms)
    return {"loss": loss, "per_arm": per_arm, "n_arms": len(arms)}


def compare_model_losses(loss_a: float, loss_b: float, *, tol: float = 1e-12) -> str:
    """Returns "A", "B", or "TIE" -- exact-tie-safe comparison for the
    Howe DOMAIN_G_CONFIRMATION_PASS decision."""
    if abs(loss_a - loss_b) <= tol:
        return "TIE"
    return "A" if loss_a < loss_b else "B"


# ---------------------------------------------------------------------------
# Orchinik: centering, RMSE_FLAT / RMSE_GAMMA1, gamma_hat diagnostic
# ---------------------------------------------------------------------------


def center(values: Sequence[float]) -> list[float]:
    m = sum(values) / len(values)
    return [v - m for v in values]


def rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def orchinik_shape_test(h: Sequence[float], g: Sequence[float]) -> dict:
    """h, g: aligned sequences of raw (uncentered) human/native-G ATEs over
    the same E cells, in the same order. Returns centered surfaces plus the
    Section 4D/4E primary comparison and diagnostics -- gamma is NEVER fit
    or altered here, only diagnosed."""
    if len(h) != len(g):
        raise ValueError(f"h and g must be the same length, got {len(h)} vs {len(g)}")
    if len(h) == 0:
        raise ValueError("h/g must be non-empty")
    h_c = center(h)
    g_c = center(g)

    rmse_flat = rmse(h_c)  # gamma=0 prediction: hhat_e^c = 0
    residual_gamma1 = [g_c[i] - h_c[i] for i in range(len(h_c))]  # gamma=1 prediction: hhat_e^c = g_e^c
    rmse_gamma1 = rmse(residual_gamma1)
    delta_rmse = rmse_gamma1 - rmse_flat

    denom = sum(x * x for x in g_c)
    gamma_hat = (sum(g_c[i] * h_c[i] for i in range(len(h_c))) / denom) if denom != 0 else float("nan")

    n = len(h_c)
    mean_h, mean_g = sum(h_c) / n, sum(g_c) / n
    cov = sum((h_c[i] - mean_h) * (g_c[i] - mean_g) for i in range(n)) / n
    std_h = math.sqrt(sum((x - mean_h) ** 2 for x in h_c) / n)
    std_g = math.sqrt(sum((x - mean_g) ** 2 for x in g_c) / n)
    pearson = cov / (std_h * std_g) if std_h > 0 and std_g > 0 else float("nan")

    def _rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rank_h, rank_g = _rank(h_c), _rank(g_c)
    mean_rh, mean_rg = sum(rank_h) / n, sum(rank_g) / n
    cov_r = sum((rank_h[i] - mean_rh) * (rank_g[i] - mean_rg) for i in range(n)) / n
    std_rh = math.sqrt(sum((x - mean_rh) ** 2 for x in rank_h) / n)
    std_rg = math.sqrt(sum((x - mean_rg) ** 2 for x in rank_g) / n)
    spearman = cov_r / (std_rh * std_rg) if std_rh > 0 and std_rg > 0 else float("nan")

    nonzero_pairs = [(a, b) for a, b in zip(h_c, g_c) if a != 0 and b != 0]
    sign_agreement = (sum(1 for a, b in nonzero_pairs if (a > 0) == (b > 0)) / len(nonzero_pairs)) if nonzero_pairs else float("nan")

    if delta_rmse < 0:
        support = "POSITIVE"
    elif delta_rmse > 0:
        support = "NEGATIVE"
    else:
        support = "TIE"

    return {
        "h_centered": h_c,
        "g_centered": g_c,
        "rmse_flat": rmse_flat,
        "rmse_gamma1": rmse_gamma1,
        "delta_rmse": delta_rmse,
        "gamma_hat": gamma_hat,
        "centered_pearson": pearson,
        "centered_spearman": spearman,
        "centered_sign_agreement": sign_agreement,
        "external_g_shape_support": support,
    }


# ---------------------------------------------------------------------------
# Respondent/donor-cluster bootstrap (shared shape for both studies)
# ---------------------------------------------------------------------------


def cluster_bootstrap_indices(cluster_ids: Sequence, *, seed: int, n_boot: int):
    """Yields n_boot resamples of the DISTINCT cluster ids (participants or
    donors), sampled with replacement -- every occurrence of a resampled
    cluster carries all of that cluster's repeated measures together, never
    resampling individual rows independently. A deterministic
    numpy-free LCG-based sampler would be an unnecessary reimplementation
    risk; use numpy's Generator, seeded exactly as specified."""
    import numpy as np

    unique = sorted(set(cluster_ids))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.choice(np.array(unique, dtype=object), size=len(unique), replace=True)
        yield list(draw)


def percentile_interval(values: Sequence[float], *, lower: float = 2.5, upper: float = 97.5) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, lower)), float(np.percentile(arr, upper))
