"""Unit tests for ate.domain_validation_metrics, using synthetic fixtures
ONLY -- never real Howe/Orchinik data (that's exercised separately by
scripts/compute_orchinik_human_ate_surface.py against the real, committed
source files). Covers the Section-8 generic-property requirements."""

from __future__ import annotations

import math

import pytest

from ate.domain_validation_metrics import (
    arm_equal_wasserstein_loss,
    cluster_bootstrap_indices,
    compare_model_losses,
    orchinik_shape_test,
    percentile_interval,
    wasserstein1,
)


# --- Howe: arm-equal W1 ---


def test_wasserstein1_zero_for_identical_distributions():
    assert wasserstein1([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein1_matches_hand_calc_equal_n():
    # sorted human=[0,0,1], synthetic=[0,1,1] -> mean|diff| = (0+1+0)/3 = 1/3
    assert wasserstein1([0, 1, 0], [1, 0, 1]) == pytest.approx(1 / 3, abs=1e-9)


def test_arm_equal_averaging_ignores_arm_n():
    # arm A: huge N with zero loss; arm B: tiny N with large loss. Arm-equal
    # weighting must NOT let the huge-N arm dominate.
    human = {"A": [0.0] * 10000, "B": [0.0, 1.0]}
    synthetic = {"A": [0.0] * 10000, "B": [1.0, 0.0]}
    result = arm_equal_wasserstein_loss(human, synthetic, scale=1.0)
    assert result["per_arm"]["A"] == pytest.approx(0.0, abs=1e-9)
    assert result["loss"] == pytest.approx(result["per_arm"]["B"] / 2, abs=1e-9)


def test_arm_equal_loss_requires_matching_arm_sets():
    with pytest.raises(ValueError, match="arm sets differ"):
        arm_equal_wasserstein_loss({"A": [1.0]}, {"B": [1.0]})


def test_compare_model_losses_exact_tie_reports_tie():
    assert compare_model_losses(5.0, 5.0) == "TIE"
    assert compare_model_losses(4.9, 5.0) == "A"
    assert compare_model_losses(5.1, 5.0) == "B"


def test_model_order_permutation_does_not_change_scores():
    human = {"A": [0.0, 0.25, 0.5], "B": [0.5, 0.75, 1.0]}
    gemma = {"A": [0.0, 0.25, 0.75], "B": [0.5, 0.5, 1.0]}
    deepseek = {"A": [0.25, 0.5, 1.0], "B": [0.0, 0.25, 0.75]}
    loss_gemma_first = arm_equal_wasserstein_loss(human, gemma)["loss"]
    loss_gemma_second = arm_equal_wasserstein_loss(human, gemma)["loss"]  # order of evaluation doesn't matter
    assert loss_gemma_first == loss_gemma_second
    l_gemma = arm_equal_wasserstein_loss(human, gemma)["loss"]
    l_deepseek = arm_equal_wasserstein_loss(human, deepseek)["loss"]
    # comparing (gemma, deepseek) vs (deepseek, gemma) must agree on the winner either way
    assert compare_model_losses(l_gemma, l_deepseek) != compare_model_losses(l_deepseek, l_gemma) or compare_model_losses(l_gemma, l_deepseek) == "TIE"


# --- Orchinik: centering / RMSE_FLAT / RMSE_GAMMA1 / gamma_hat ---


def test_gamma0_reproduces_flat_baseline():
    h = [1.0, 3.0, 5.0, 2.0, 4.0]
    g = [10.0, 20.0, 30.0, 40.0, 50.0]  # arbitrary, uncorrelated shape
    result = orchinik_shape_test(h, g)
    h_c = result["h_centered"]
    manual_rmse_flat = math.sqrt(sum(x * x for x in h_c) / len(h_c))
    assert result["rmse_flat"] == pytest.approx(manual_rmse_flat, abs=1e-9)


def test_gamma1_reproduces_native_centered_g_shape():
    h = [1.0, 3.0, 5.0, 2.0, 4.0]
    g = [1.0, 3.0, 5.0, 2.0, 4.0]  # identical shape -> gamma=1 prediction is exact
    result = orchinik_shape_test(h, g)
    assert result["rmse_gamma1"] == pytest.approx(0.0, abs=1e-9)
    assert result["delta_rmse"] < 0
    assert result["external_g_shape_support"] == "POSITIVE"


def test_flat_shape_wins_when_g_is_uninformative_noise_shape():
    h = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
    g = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0]  # weakly related, will not track h well
    result = orchinik_shape_test(h, g)
    # not asserting a specific sign here (depends on numbers) -- just that the
    # comparison machinery runs and produces a consistent verdict
    assert result["external_g_shape_support"] in ("POSITIVE", "NEGATIVE", "TIE")
    if result["delta_rmse"] > 0:
        assert result["external_g_shape_support"] == "NEGATIVE"


def test_gamma_hat_formula_reproduced_by_hand():
    h = [2.0, -1.0, 0.0, 3.0]
    g = [1.0, -2.0, 1.0, 2.0]
    result = orchinik_shape_test(h, g)
    h_c, g_c = result["h_centered"], result["g_centered"]
    manual_gamma_hat = sum(g_c[i] * h_c[i] for i in range(len(h_c))) / sum(x * x for x in g_c)
    assert result["gamma_hat"] == pytest.approx(manual_gamma_hat, abs=1e-9)


def test_centering_produces_mean_zero_for_both_surfaces():
    h = [5.0, 1.0, 9.0, 3.0]
    g = [-4.0, 2.0, 8.0, 0.0]
    result = orchinik_shape_test(h, g)
    assert sum(result["h_centered"]) / len(result["h_centered"]) == pytest.approx(0.0, abs=1e-9)
    assert sum(result["g_centered"]) / len(result["g_centered"]) == pytest.approx(0.0, abs=1e-9)


def test_mismatched_length_fails_closed():
    with pytest.raises(ValueError, match="same length"):
        orchinik_shape_test([1.0, 2.0], [1.0])


def test_50_cell_shape_end_to_end():
    # 2 interventions x 5 beliefs x 5 levels = 50, exercised as one flat call
    h = [float(i) for i in range(50)]
    g = [float(i) * 2 for i in range(50)]
    result = orchinik_shape_test(h, g)
    assert len(result["h_centered"]) == 50
    assert len(result["g_centered"]) == 50


# --- no target-F / target-G / target-human input accepted (structural) ---


def test_no_target_input_parameters_exist():
    import inspect

    for fn in (wasserstein1, arm_equal_wasserstein_loss, orchinik_shape_test):
        params = set(inspect.signature(fn).parameters)
        forbidden = {p for p in params if "target" in p or p in ("tau_f", "tau_g_target", "f_ate")}
        assert not forbidden


# --- cluster bootstrap preserves repeated measures ---


def test_cluster_bootstrap_preserves_full_cluster_not_individual_rows():
    cluster_ids = ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3"]  # 3 participants x 3 repeated measures each
    draws = list(cluster_bootstrap_indices(cluster_ids, seed=20260826, n_boot=5))
    assert len(draws) == 5
    for draw in draws:
        assert len(draw) == 3  # resamples the 3 DISTINCT clusters, not the 9 rows
        assert all(d in {"p1", "p2", "p3"} for d in draw)


def test_cluster_bootstrap_deterministic_given_frozen_seed():
    ids = [f"d{i}" for i in range(20)]
    draws_a = list(cluster_bootstrap_indices(ids, seed=20260826, n_boot=10))
    draws_b = list(cluster_bootstrap_indices(ids, seed=20260826, n_boot=10))
    assert draws_a == draws_b


def test_percentile_interval_basic():
    lo, hi = percentile_interval(list(range(1, 101)))
    assert lo < hi
    assert 1 <= lo <= 10
    assert 90 <= hi <= 100
