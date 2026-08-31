"""Regression tests for the R_F sequential freeze decision, implemented and
tested BEFORE any real pilot results are run against it."""

from __future__ import annotations

import pandas as pd

from ate.r_f_decision import R_F_PASS_THRESHOLDS, stage_r1_decision, stage_r2_decision

EFFECTS = pd.DataFrame(
    {
        "study_id": ["A", "A", "B", "C"],
        "effect_id": ["A:e1", "A:e2", "B:e1", "C:e1"],
    }
)


def _draw(z_values: list[float]) -> pd.DataFrame:
    return EFFECTS.assign(z_pp=z_values)


def test_stage_r1_freezes_r_f_1_when_replicates_agree(tmp_path):
    draw1 = _draw([10.0, 20.0, -5.0, 3.0])
    draw2 = _draw([10.5, 19.8, -4.9, 3.1])  # tiny, within-threshold jitter
    result = stage_r1_decision(draw1, draw2, invalid_response_rate=0.0, outputs_dir=tmp_path)
    assert result["decision"] == "FREEZE_R_F"
    assert result["r_f"] == 1
    assert result["stage"] == "R1"
    assert result["thresholds"] == R_F_PASS_THRESHOLDS


def test_stage_r1_escalates_when_replicates_disagree(tmp_path):
    draw1 = _draw([10.0, 20.0, -5.0, 3.0])
    draw2 = _draw([30.0, -10.0, 15.0, -20.0])  # wildly different
    result = stage_r1_decision(draw1, draw2, invalid_response_rate=0.0, outputs_dir=tmp_path)
    assert result["decision"] == "ESCALATE_TO_STAGE_R2"
    assert result["r_f"] is None


def test_stage_r1_fails_on_invalid_response_rate_alone():
    # replicates agree perfectly, but invalid rate exceeds threshold
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        draw1 = _draw([10.0, 20.0, -5.0, 3.0])
        draw2 = _draw([10.0, 20.0, -5.0, 3.0])
        result = stage_r1_decision(draw1, draw2, invalid_response_rate=0.01, outputs_dir=tmp)
        assert result["decision"] == "ESCALATE_TO_STAGE_R2"


def test_stage_r2_freezes_r_f_2_when_averaged_pairs_agree(tmp_path):
    draw1 = _draw([8.0, 22.0, -6.0, 4.0])
    draw2 = _draw([12.0, 18.0, -4.0, 2.0])  # avg(1,2) = [10, 20, -5, 3]
    draw3 = _draw([9.5, 20.2, -4.8, 3.2])
    draw4 = _draw([10.4, 19.6, -5.1, 2.9])  # avg(3,4) close to avg(1,2)
    result = stage_r2_decision(draw1, draw2, draw3, draw4, invalid_response_rate=0.0, outputs_dir=tmp_path)
    assert result["decision"] == "FREEZE_R_F"
    assert result["r_f"] == 2
    assert result["stage"] == "R2"


def test_stage_r2_stops_when_averaged_pairs_still_disagree(tmp_path):
    draw1 = _draw([8.0, 22.0, -6.0, 4.0])
    draw2 = _draw([12.0, 18.0, -4.0, 2.0])
    draw3 = _draw([40.0, -30.0, 25.0, -15.0])
    draw4 = _draw([-20.0, 50.0, -10.0, 30.0])
    result = stage_r2_decision(draw1, draw2, draw3, draw4, invalid_response_rate=0.0, outputs_dir=tmp_path)
    assert result["decision"] == "STOP_REQUIRE_NEW_EXPLICIT_DECISION"
    assert result["r_f"] is None


def test_stage_r2_requires_identical_effect_rows_across_draws(tmp_path):
    import pytest

    mismatched = pd.DataFrame({"study_id": ["A"], "effect_id": ["A:eX"], "z_pp": [1.0]})
    draw1 = _draw([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="identical study_id/effect_id"):
        stage_r2_decision(draw1, draw1, draw1, mismatched, invalid_response_rate=0.0, outputs_dir=tmp_path)


def test_thresholds_are_exactly_the_three_predeclared_values():
    assert R_F_PASS_THRESHOLDS == {
        "max_invalid_response_rate": 0.005,
        "replicate_pairwise_ate_rmse_pp": 2.0,
        "max_condition_outcome_replicate_abs_diff_pp": 5.0,
    }
