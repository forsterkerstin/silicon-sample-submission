"""pipeline/ate/normalize_effects.py

Converts a raw-scale ATE (native units: 0-100 slider, $0-10 donation, 0/1
newsletter) to percentage-of-range ("pp") and back, so effects on
heterogeneous scales are comparable on one 0-100pp axis before
lambda_ate calibration (pipeline/ate/calibrate_lambda.py).
"""

from __future__ import annotations

import survey_content as sc


def to_unit_scale(value: float, low: float, high: float) -> float:
    """x = (y - a) / (b - a): map a raw outcome value onto its known [0, 1]
    scale range. Never uses the model's own predicted standard deviation --
    synthetic variance can itself be miscalibrated, so an SD-based
    standardization would let that error contaminate effect-size
    calibration; the *known* scale bounds (e.g. a slider's 0-100, the
    donation's $0-10) are used instead.
    """
    if high <= low:
        raise ValueError(f"scale bounds must have high > low, got low={low}, high={high}")
    return (value - low) / (high - low)


def from_unit_scale(u: float, low: float, high: float) -> float:
    """Inverse of to_unit_scale(): map a [0, 1]-scale value back to the
    outcome's original units."""
    if high <= low:
        raise ValueError(f"scale bounds must have high > low, got low={low}, high={high}")
    return low + u * (high - low)


def to_percent_of_range(raw_ate: float, low: float, high: float) -> float:
    """A raw-scale ATE (native units) to percentage-of-range: 100 * the
    unit-scale difference a raw_ate of this size represents. Equivalent to
    `100 * to_unit_scale(raw_ate, 0, high - low)`, expressed directly since
    an ATE is already a difference (no need to subtract `low` twice).
    """
    if high <= low:
        raise ValueError(f"scale bounds must have high > low, got low={low}, high={high}")
    return 100 * raw_ate / (high - low)


def from_percent_of_range(ate_pp: float, low: float, high: float) -> float:
    """Inverse of to_percent_of_range(): a percentage-of-range ATE back to
    the outcome's native units."""
    if high <= low:
        raise ValueError(f"scale bounds must have high > low, got low={low}, high={high}")
    return ate_pp / 100 * (high - low)


#: known (low, high) scale bounds per outcome. 11 attitudinal outcomes are
#: 0-100 sliders; donation_ams is $0-10; newsletter_signup is already 0/1.
OUTCOME_SCALE_BOUNDS: dict[str, tuple[float, float]] = {
    "trust_multidimensional": (0, 100), "trust_post": (0, 100), "distrust_post": (0, 100),
    "funding_perceptions": (0, 100), "policy_role_mean": (0, 100), "inst_trust_mean": (0, 100),
    "belief_post": (0, 100), "concern_mean": (0, 100), "policy_general": (0, 100),
    "policy_specific_mean": (0, 100), "behavior_mean": (0, 100),
    "donation_ams": (0, 10), "newsletter_signup": (0, 1),
}

#: which of the 3 outcome families each of the 13 outcomes belongs to (used
#: by calibrate_lambda.py's hierarchical, treatment/outcome-family model).
OUTCOME_FAMILY: dict[str, str] = {
    **{k: "attitude" for k in OUTCOME_SCALE_BOUNDS if k not in ("donation_ams", "newsletter_signup")},
    "donation_ams": "donation",
    "newsletter_signup": "binary_behavior",
}

#: (low, high) bounds by RAW ITEM scale type (survey_content.SCALE_*), for
#: calibrating individual raw items (pipeline/calibration/calibrate_arm.py)
#: rather than the 13 named outcome composites above -- every one of the 44
#: raw items is one of these 3 scale types, so this covers all of them
#: without a per-item name lookup.
RAW_ITEM_SCALE_BOUNDS: dict[str, tuple[float, float]] = {
    sc.SCALE_SLIDER_0_100: (0, 100),
    sc.SCALE_DONATION_0_10: (0, 10),
    sc.SCALE_BINARY_0_1: (0, 1),
}
