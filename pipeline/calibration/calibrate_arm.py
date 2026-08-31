"""pipeline/calibration/calibrate_arm.py

Legacy single-arm calibration helper: nudge one treatment arm's raw
native-response integers by the minimum possible total amount so the arm's
mean effect (vs. control) matches its lambda_ate-calibrated target ATE,
without ever touching control responses or moving any individual further
than necessary.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence


def calibrate_arm_to_target_ate(
    control_responses: Sequence[float],
    raw_treatment_responses: Sequence[int],
    target_ate_pp: float,
    low: float,
    high: float,
) -> list[int]:
    """Adjust `raw_treatment_responses` (integers in [low, high]) so that
    mean(adjusted) - mean(control_responses) matches `target_ate_pp`
    (percent-of-range, e.g. from an externally selected calibration model)
    as closely as integer arithmetic allows.

    Exact when arithmetically feasible: computes the integer total that
    hits target_ate_pp (rounded to the nearest achievable total, clamped to
    [n*low, n*high]), then distributes the required unit nudges one at a
    time across the respondents with the most remaining headroom toward the
    bound being approached. Every feasible integer solution matching that
    target total has the SAME minimum sum(|y_new - y_raw|) -- each unit of
    required change costs exactly one unit of L1 distortion no matter who
    absorbs it -- so the headroom-first order only avoids early saturation,
    it does not change the achieved distortion. Ties are broken by
    respondent order (the input list's own order), so results are
    deterministic and reproducible.

    Donation (0-10) and slider (0-100) items use this directly; newsletter
    (0/1) is the same algorithm degenerating to flipping the minimum number
    of 0<->1 values -- no special-casing needed.

    If the target is infeasible (every respondent already at the bound
    being approached), returns the closest achievable allocation rather
    than silently pretending the target was hit -- compare the returned
    list's own mean to target_ate_pp if a caller needs to detect this.
    """
    if high <= low:
        raise ValueError(f"scale bounds must have high > low, got low={low}, high={high}")
    lo, hi = round(low), round(high)
    if low != lo or high != hi:
        raise ValueError(f"discrete calibration requires integer bounds, got low={low}, high={high}")
    _validate_integer_responses("control_responses", control_responses, lo, hi)
    _validate_integer_responses("raw_treatment_responses", raw_treatment_responses, lo, hi)

    n = len(raw_treatment_responses)
    if n == 0:
        return []
    if len(control_responses) == 0:
        raise ValueError("control_responses must not be empty")

    control_mean = sum(int(v) for v in control_responses) / len(control_responses)
    target_mean = control_mean + target_ate_pp / 100 * (high - low)
    target_sum = round(target_mean * n)
    target_sum = max(n * lo, min(n * hi, target_sum))

    y = [int(v) for v in raw_treatment_responses]
    delta = target_sum - sum(y)
    if delta == 0:
        return y
    direction = 1 if delta > 0 else -1
    remaining = abs(delta)

    while remaining > 0:
        candidates = [i for i in range(n) if (y[i] < hi if direction > 0 else y[i] > lo)]
        if not candidates:
            break  # saturated: target is infeasible, closest achievable allocation returned
        candidates.sort(key=lambda i: (hi - y[i]) if direction > 0 else (y[i] - lo), reverse=True)
        step = min(len(candidates), remaining)
        for i in candidates[:step]:
            y[i] += direction
        remaining -= step

    return y


def _validate_integer_responses(name: str, values: Sequence[float], lo: int, hi: int) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} contains non-integer value {value!r}")
        if not (lo <= int(value) <= hi):
            raise ValueError(f"{name} contains out-of-bounds value {value!r}; expected [{lo}, {hi}]")
