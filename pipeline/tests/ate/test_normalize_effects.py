"""to_unit_scale()/from_unit_scale()/to_percent_of_range(): bounded-scale
ATE normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from ate.normalize_effects import (
    OUTCOME_FAMILY,
    OUTCOME_SCALE_BOUNDS,
    RAW_ITEM_SCALE_BOUNDS,
    from_percent_of_range,
    from_unit_scale,
    to_percent_of_range,
    to_unit_scale,
)
from ate.target_effects import apply_calibration_to_target_ates


@pytest.mark.parametrize("low,high,value,expected", [(0, 100, 0, 0.0), (0, 100, 100, 1.0), (0, 100, 50, 0.5), (0, 10, 5, 0.5)])
def test_to_unit_scale(low, high, value, expected):
    assert to_unit_scale(value, low, high) == pytest.approx(expected)


def test_round_trip():
    for low, high, value in [(0, 100, 37), (0, 10, 3), (0, 1, 1)]:
        u = to_unit_scale(value, low, high)
        assert from_unit_scale(u, low, high) == pytest.approx(value)


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        to_unit_scale(5, 10, 10)
    with pytest.raises(ValueError):
        from_unit_scale(0.5, 10, 5)


def test_all_13_outcomes_have_scale_bounds():
    from survey_content import OUTCOME_COMPOSITES

    assert set(OUTCOME_COMPOSITES.keys()) == set(OUTCOME_SCALE_BOUNDS.keys())
    assert set(OUTCOME_COMPOSITES.keys()) == set(OUTCOME_FAMILY.keys())


def test_outcome_families_match_the_three_scale_types():
    assert OUTCOME_FAMILY["donation_ams"] == "donation"
    assert OUTCOME_FAMILY["newsletter_signup"] == "binary_behavior"
    assert OUTCOME_FAMILY["trust_multidimensional"] == "attitude"
    assert sum(1 for v in OUTCOME_FAMILY.values() if v == "attitude") == 11


@pytest.mark.parametrize("raw_ate,low,high,expected_pp", [(10, 0, 100, 10.0), (50, 0, 100, 50.0), (5, 0, 10, 50.0), (1, 0, 1, 100.0), (-20, 0, 100, -20.0)])
def test_to_percent_of_range(raw_ate, low, high, expected_pp):
    assert to_percent_of_range(raw_ate, low, high) == pytest.approx(expected_pp)


def test_percent_of_range_round_trip():
    for raw_ate, low, high in [(7, 0, 100), (2, 0, 10), (0.3, 0, 1)]:
        pp = to_percent_of_range(raw_ate, low, high)
        assert from_percent_of_range(pp, low, high) == pytest.approx(raw_ate)


def test_raw_item_scale_bounds_cover_all_three_item_scale_types():
    import survey_content as sc

    assert RAW_ITEM_SCALE_BOUNDS[sc.SCALE_SLIDER_0_100] == (0, 100)
    assert RAW_ITEM_SCALE_BOUNDS[sc.SCALE_DONATION_0_10] == (0, 10)
    assert RAW_ITEM_SCALE_BOUNDS[sc.SCALE_BINARY_0_1] == (0, 1)


def test_m2_native_conversion_uses_outcome_range():
    raw = pd.DataFrame(
        [
            {"condition": "a", "outcome": "trust_post", "raw_f_ate_native": 10.0, "raw_f_ate_pp": 10.0, "synthetic_ate_native": 10.0, "synthetic_effect_pp": 10.0, "outcome_min": 0, "outcome_max": 100, "outcome_range": 100},
            {"condition": "a", "outcome": "donation_ams", "raw_f_ate_native": 1.0, "raw_f_ate_pp": 10.0, "synthetic_ate_native": 1.0, "synthetic_effect_pp": 10.0, "outcome_min": 0, "outcome_max": 10, "outcome_range": 10},
            {"condition": "a", "outcome": "newsletter_signup", "raw_f_ate_native": 0.1, "raw_f_ate_pp": 10.0, "synthetic_ate_native": 0.1, "synthetic_effect_pp": 10.0, "outcome_min": 0, "outcome_max": 1, "outcome_range": 1},
        ]
    )

    out = apply_calibration_to_target_ates(raw, {"model_name": "M2", "calibration_alpha": 5.0, "calibration_lambda": 2.0}, outputs_dir=None)

    assert out.loc[out["outcome"] == "trust_post", "calibrated_ate_native"].item() == pytest.approx(25.0)
    assert out.loc[out["outcome"] == "donation_ams", "calibrated_ate_native"].item() == pytest.approx(2.5)
    assert out.loc[out["outcome"] == "newsletter_signup", "calibrated_ate_native"].item() == pytest.approx(0.25)


def test_stale_development_calibration_is_not_production_usable():
    raw = pd.DataFrame(
        [
            {
                "condition": "a",
                "outcome": "trust_post",
                "synthetic_ate_native": 10.0,
                "synthetic_effect_pp": 10.0,
                "outcome_min": 0,
                "outcome_max": 100,
                "outcome_range": 100,
            }
        ]
    )

    with pytest.raises(ValueError, match="unusable for production"):
        apply_calibration_to_target_ates(
            raw,
            {
                "model_name": "M1",
                "calibration_lambda": -0.41,
                "usable_for_production": False,
                "calibration_status": "DEVELOPMENT_STALE_PENDING_STUDY_SPECIFIC_EXTERNAL_F_REGENERATION",
            },
            outputs_dir=None,
        )
