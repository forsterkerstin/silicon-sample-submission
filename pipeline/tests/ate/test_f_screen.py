"""Regression tests freezing the mini F-model screen scoring, BEFORE any
candidate screen requests are built/submitted."""

from __future__ import annotations

import math

import pytest

from ate.f_screen import f_screen_diagnostics, f_screen_theta_l_pp, score_f_screen_candidate, select_f_star


def test_theta_l_matches_percent_of_range_convention():
    # 10 control points at 2.0, 10 treatment points at 5.0 -> raw_ate=3.0, scale 1-5 (R=4)
    theta_l = f_screen_theta_l_pp([2.0] * 10, [5.0] * 10, 1, 5)
    assert theta_l == pytest.approx(100 * 3.0 / 4)


def test_theta_l_requires_nonempty_arms():
    with pytest.raises(ValueError):
        f_screen_theta_l_pp([], [1.0], 1, 5)


def test_study_equal_rmse_weights_studies_not_effects():
    # Study A contributes 3 effects all with diff=0 (perfect); study B contributes
    # 1 effect with diff=10. A naive per-effect RMSE would be dominated by A's 3
    # zero-diff effects; the study-equal RMSE must instead average A's MSE (0)
    # and B's MSE (100) equally -> rmse = sqrt((0+100)/2) = sqrt(50).
    rows = [
        {"study_id": "A", "effect_id": "A:e1", "theta_l_pp": 10.0, "theta_h_pp": 10.0},
        {"study_id": "A", "effect_id": "A:e2", "theta_l_pp": 20.0, "theta_h_pp": 20.0},
        {"study_id": "A", "effect_id": "A:e3", "theta_l_pp": 30.0, "theta_h_pp": 30.0},
        {"study_id": "B", "effect_id": "B:e1", "theta_l_pp": 10.0, "theta_h_pp": 0.0},
    ]
    result = score_f_screen_candidate(rows)
    assert result["primary_study_equal_rmse_pp"] == pytest.approx(math.sqrt(50.0))
    assert result["primary_detail"]["n_studies"] == 2
    assert result["primary_detail"]["n_effects"] == 4


def test_score_requires_all_keys():
    with pytest.raises(ValueError):
        score_f_screen_candidate([{"study_id": "A", "effect_id": "A:e1", "theta_l_pp": 1.0}])


def test_diagnostics_are_separate_from_primary_metric():
    rows = [
        {"study_id": "A", "effect_id": "A:e1", "theta_l_pp": 10.0, "theta_h_pp": 5.0},
        {"study_id": "B", "effect_id": "B:e1", "theta_l_pp": -5.0, "theta_h_pp": -5.0},
        {"study_id": "C", "effect_id": "C:e1", "theta_l_pp": 8.0, "theta_h_pp": 12.0},
    ]
    diag = f_screen_diagnostics(rows)
    assert set(diag) == {"mae", "pearson", "spearman", "sign_agreement"}
    assert diag["mae"] == pytest.approx((5.0 + 0.0 + 4.0) / 3)
    assert diag["sign_agreement"] == pytest.approx(1.0)


def test_select_f_star_lowest_primary_rmse_wins():
    scores = {
        "modelA": {"primary_study_equal_rmse_pp": 5.0},
        "modelB": {"primary_study_equal_rmse_pp": 3.0},
    }
    result = select_f_star(scores, invalid_response_rate={"modelA": 0.0, "modelB": 0.0}, realized_cost_usd={"modelA": 1.0, "modelB": 100.0})
    assert result["f_star"] == "modelB"


def test_select_f_star_tie_break_invalid_rate_then_cost_then_lexical():
    scores = {
        "google/gemma-4-31B-it": {"primary_study_equal_rmse_pp": 5.0},
        "deepseek-ai/DeepSeek-V4-Pro-0813": {"primary_study_equal_rmse_pp": 5.0},
    }
    # exact tie on primary; deepseek has lower invalid rate -> wins despite higher cost
    result = select_f_star(
        scores,
        invalid_response_rate={"google/gemma-4-31B-it": 0.01, "deepseek-ai/DeepSeek-V4-Pro-0813": 0.0},
        realized_cost_usd={"google/gemma-4-31B-it": 1.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 100.0},
    )
    assert result["f_star"] == "deepseek-ai/DeepSeek-V4-Pro-0813"

    # now tie on primary AND invalid rate -> lower cost wins
    result2 = select_f_star(
        scores,
        invalid_response_rate={"google/gemma-4-31B-it": 0.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 0.0},
        realized_cost_usd={"google/gemma-4-31B-it": 1.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 100.0},
    )
    assert result2["f_star"] == "google/gemma-4-31B-it"

    # now tie on everything -> lexical order wins (deepseek < google alphabetically)
    result3 = select_f_star(
        scores,
        invalid_response_rate={"google/gemma-4-31B-it": 0.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 0.0},
        realized_cost_usd={"google/gemma-4-31B-it": 1.0, "deepseek-ai/DeepSeek-V4-Pro-0813": 1.0},
    )
    assert result3["f_star"] == "deepseek-ai/DeepSeek-V4-Pro-0813"
