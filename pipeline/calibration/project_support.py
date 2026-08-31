"""Minimum-distortion projection onto benchmark response support.

The final constructor computes ideal treatment outcomes from control values,
calibrated ATEs, and optional HTE deviations. These helpers turn those ideal
values into legal submitted responses while matching the requested treatment
arm total as closely as finite integer support permits.
"""

from __future__ import annotations

import heapq

import numpy as np


def bounded_integer_total(n: int, low: int, high: int, requested_total: float) -> int:
    return int(np.clip(round(float(requested_total)), n * low, n * high))


def _continuous_shift_to_total(ideal: np.ndarray, *, low: int, high: int, target_total: int) -> np.ndarray:
    if target_total <= len(ideal) * low:
        return np.full(len(ideal), low, dtype=float)
    if target_total >= len(ideal) * high:
        return np.full(len(ideal), high, dtype=float)
    lo = low - float(np.max(ideal)) - 1.0
    hi = high - float(np.min(ideal)) + 1.0
    for _ in range(100):
        eta = (lo + hi) / 2.0
        shifted = np.clip(ideal + eta, low, high)
        if float(shifted.sum()) < target_total:
            lo = eta
        else:
            hi = eta
    return np.clip(ideal + (lo + hi) / 2.0, low, high)


def project_integer_to_total(
    values: np.ndarray,
    *,
    low: int,
    high: int,
    target_mean: float | None = None,
    target_total: float | None = None,
) -> np.ndarray:
    """Integer vector in [low, high] with exact nearest attainable total.

    The projection center is `values`. We first solve the continuous bounded
    shift projection, then round using marginal cost in original squared
    error from `values`.
    """
    ideal = np.asarray(values, dtype=float)
    if ideal.ndim != 1:
        raise ValueError("project_integer_to_total expects a 1-D vector")
    if (target_mean is None) == (target_total is None):
        raise ValueError("supply exactly one of target_mean or target_total")
    requested = float(target_total) if target_total is not None else float(target_mean) * len(ideal)
    target = bounded_integer_total(len(ideal), low, high, requested)
    continuous = _continuous_shift_to_total(ideal, low=low, high=high, target_total=target)
    lower = np.floor(continuous).astype(int)
    lower = np.clip(lower, low, high)
    remainder = int(target - int(lower.sum()))
    if remainder < 0:
        raise ValueError("continuous projection rounding produced an infeasible lower total")
    costs: list[tuple[float, int]] = []
    for i, current in enumerate(lower):
        if current >= high:
            continue
        cost = (current + 1 - ideal[i]) ** 2 - (current - ideal[i]) ** 2
        heapq.heappush(costs, (float(cost), i))
    out = lower.copy()
    for _ in range(remainder):
        if not costs:
            raise ValueError("target total is unreachable after continuous projection")
        _, i = heapq.heappop(costs)
        out[i] += 1
    if int(out.sum()) != target:
        raise ValueError(f"projected total {int(out.sum())} != target total {target}")
    return out


def project_binary_to_count(values: np.ndarray, *, target_mean: float | None = None, target_count: float | None = None) -> np.ndarray:
    """Binary vector with count closest to target_mean * n."""
    ideal = np.asarray(values, dtype=float)
    if ideal.ndim != 1:
        raise ValueError("project_binary_to_count expects a 1-D vector")
    if (target_mean is None) == (target_count is None):
        raise ValueError("supply exactly one of target_mean or target_count")
    requested = float(target_count) if target_count is not None else float(target_mean) * len(ideal)
    target_count = bounded_integer_total(len(ideal), 0, 1, requested)
    order = sorted(range(len(ideal)), key=lambda i: (-ideal[i], i))
    out = np.zeros(len(ideal), dtype=int)
    out[order[:target_count]] = 1
    return out


def project_matrix_to_composite_total(
    values: np.ndarray,
    *,
    low: int,
    high: int,
    target_composite_mean: float | None = None,
    target_total: float | None = None,
) -> np.ndarray:
    """Project a raw item matrix so its row-mean composite has target mean."""
    ideal = np.asarray(values, dtype=float)
    if ideal.ndim != 2:
        raise ValueError("project_matrix_to_composite_total expects a 2-D matrix")
    if target_total is None:
        if target_composite_mean is None:
            raise ValueError("supply target_composite_mean or target_total")
        target_total = float(target_composite_mean) * ideal.shape[0] * ideal.shape[1]
    flat = project_integer_to_total(
        ideal.reshape(-1),
        low=low,
        high=high,
        target_total=target_total,
    )
    return flat.reshape(ideal.shape)
