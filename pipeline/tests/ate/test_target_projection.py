"""Unit tests for the shared common-shift + support-projection module
(ate.target_projection), using synthetic fixtures ONLY -- never real target
G output. Covers required fixtures A-Q plus brute-force optimality
verification, method-independence, and Section-9 fail-closed behavior."""

from __future__ import annotations

import itertools
import math

import pytest

from ate.target_projection import (
    REAL_BENCHMARK_SHARED_RAW_ITEMS,
    assert_no_conflicting_shared_item_values,
    audit_shared_raw_items,
    build_donor_map,
    compute_common_shift,
    project_binary_k,
    project_bounded_integer,
    project_cell,
    project_composite_cell,
    project_finite_discrete,
    project_target_ate_table,
    verify_common_shift_identities,
)


def _ids(n, prefix="d"):
    return [f"{prefix}{i:03d}" for i in range(n)]


# A. no correction required.
def test_a_no_correction_required():
    ids = _ids(20)
    control = {i: 40 for i in ids}
    treat = {i: 45 for i in ids}
    tau_hat = 5.0  # exactly matches native G ATE -> c_aj == 0
    r = project_cell("a1", "trust_post", control, treat, tau_hat, support_kind="bounded_integer", low=0, high=100)
    assert r["ideal_shift_c"] == pytest.approx(0.0, abs=1e-9)
    assert r["achieved_postprojection_ate"] == pytest.approx(5.0, abs=1e-9)
    assert r["n_responses_changed_by_projection"] == 0


# B. positive common shift.
def test_b_positive_common_shift():
    ids = _ids(20)
    control = {i: 40 for i in ids}
    treat = {i: 45 for i in ids}
    r = project_cell("a1", "trust_post", control, treat, tau_hat_aj=8.0, support_kind="bounded_integer", low=0, high=100)
    assert r["ideal_shift_c"] == pytest.approx(3.0, abs=1e-9)
    assert r["achieved_postprojection_ate"] == pytest.approx(8.0, abs=1e-9)


# C. negative common shift.
def test_c_negative_common_shift():
    ids = _ids(20)
    control = {i: 40 for i in ids}
    treat = {i: 45 for i in ids}
    r = project_cell("a1", "trust_post", control, treat, tau_hat_aj=2.0, support_kind="bounded_integer", low=0, high=100)
    assert r["ideal_shift_c"] == pytest.approx(-3.0, abs=1e-9)
    assert r["achieved_postprojection_ate"] == pytest.approx(2.0, abs=1e-9)


# D. lower-bound saturation.
def test_d_lower_bound_saturation():
    ids = _ids(10)
    control = {i: 5 for i in ids}
    treat = {i: 3 for i in ids}
    r = project_cell("a1", "trust_post", control, treat, tau_hat_aj=-50.0, support_kind="bounded_integer", low=0, high=100)
    assert r["lower_bound_count"] == 10
    assert all(v == 0 for v in r["achieved_treat"].values())
    assert r["achieved_postprojection_ate"] == -5.0  # nearest attainable: all at floor


# E. upper-bound saturation.
def test_e_upper_bound_saturation():
    ids = _ids(10)
    control = {i: 5 for i in ids}
    treat = {i: 8 for i in ids}
    r = project_cell("a1", "trust_post", control, treat, tau_hat_aj=500.0, support_kind="bounded_integer", low=0, high=100)
    assert r["upper_bound_count"] == 10
    assert all(v == 100 for v in r["achieved_treat"].values())


# F. integer balanced rounding.
def test_f_integer_balanced_rounding():
    ideal = {"d0": 10.2, "d1": 10.4, "d2": 10.6, "d3": 10.8}
    r = project_bounded_integer(ideal, low=0, high=100)
    total_ideal = sum(ideal.values())  # 42.0 -> t_star = 42
    assert r["target_total_used"] == 42
    assert sum(r["achieved"].values()) == 42


# G. exact rounding tie.
def test_g_exact_rounding_tie():
    ideal = {"d0": 10.5, "d1": 20.5}
    r = project_bounded_integer(ideal, low=0, high=100, target_total=sum(ideal.values()))
    # documented tie convention: exact .5 ties round down at the base step
    assert r["achieved"]["d0"] in (10, 11)
    assert r["achieved"]["d1"] in (20, 21)
    assert sum(r["achieved"].values()) == 31  # base rounding-down both ties sums to 30, then +1 adjustment to hit t_star=31


# H. binary K allocation.
def test_h_binary_k_allocation():
    ideal = {f"d{i}": v for i, v in enumerate([0.1, 0.9, 0.5, 0.7, 0.3, 0.6, 0.2, 0.8])}
    r = project_binary_k(ideal, target_total=3)
    assert sum(r["achieved"].values()) == 3
    ones = {i for i, v in r["achieved"].items() if v == 1}
    top3_by_value = sorted(ideal, key=lambda i: (-ideal[i], i))[:3]
    assert ones == set(top3_by_value)


# I. requested binary mean below 0 / above 1.
def test_i_binary_target_out_of_range_clips_to_0_or_n():
    ideal = {f"d{i}": 0.5 for i in range(5)}
    r_low = project_binary_k(ideal, target_total=-10)
    assert sum(r_low["achieved"].values()) == 0
    r_high = project_binary_k(ideal, target_total=10)
    assert sum(r_high["achieved"].values()) == 5


# J. composite constituent adjustment.
def test_j_composite_constituent_adjustment():
    ids = _ids(6)
    items = ["behavior_meat", "behavior_transport", "behavior_solar", "behavior_fly", "behavior_talk", "behavior_donate"]
    item_control = {lab: {i: 40 + k for k, i in enumerate(ids)} for lab in items}
    item_treat = {lab: {i: 45 + k for k, i in enumerate(ids)} for lab in items}
    bounds = {lab: (0, 100) for lab in items}
    r = project_composite_cell("a1", "behavior_mean", item_control, item_treat, tau_hat_aj=6.0, item_bounds=bounds)
    assert r["achieved_postprojection_ate"] == pytest.approx(6.0, abs=1e-6)
    for lab in items:
        for i in ids:
            v = r["projected_items"][lab][i]
            assert 0 <= v <= 100
            assert float(v).is_integer()
    # recomputation is mechanical: composite == mean of the projected items
    for i in ids:
        expected = sum(r["projected_items"][lab][i] for lab in items) / len(items)
        assert r["achieved_treat"][i] == pytest.approx(expected, abs=1e-9)


# funding_perceptions: reverse_100 composite, sign-flipped per-item shift.
def test_j2_reverse_100_composite_sign_flipped_shift():
    ids = _ids(10)
    item_control = {"funding_5_raw": {i: 40 for i in ids}}
    item_treat = {"funding_5_raw": {i: 35 for i in ids}}  # raw item DOWN -> funding_perceptions (100-raw) UP
    bounds = {"funding_5_raw": (0, 100)}
    # native composite ATE = (100-35) - (100-40) = 65-60 = 5
    r = project_composite_cell("a1", "funding_perceptions", item_control, item_treat, tau_hat_aj=8.0, item_bounds=bounds)
    assert r["native_g_ate"] == pytest.approx(5.0, abs=1e-9)
    assert r["ideal_shift_c"] == pytest.approx(3.0, abs=1e-9)
    # composite must go UP by c_aj=3, so raw item must go DOWN by 3 (since composite=100-raw)
    for i in ids:
        assert r["projected_items"]["funding_5_raw"][i] == pytest.approx(35 - 3, abs=1)


# K. shared raw item consistency.
def test_k_shared_raw_item_audit_and_conflict_detection():
    synthetic_composites = {
        "outcome_x": ("item", "shared_item"),
        "outcome_y": ("mean", ["shared_item", "other_item"]),
    }
    shared = audit_shared_raw_items(synthetic_composites)
    assert shared == {"shared_item": ["outcome_x", "outcome_y"]}

    # consistent case: both outcomes independently arrive at the same value -> no raise
    consistent = {
        "outcome_x": {"shared_item": {"d0": 5, "d1": 6}},
        "outcome_y": {"shared_item": {"d0": 5, "d1": 6}, "other_item": {"d0": 1, "d1": 2}},
    }
    assert_no_conflicting_shared_item_values(consistent, shared)  # must not raise

    # conflicting case: same donors, different final values -> STOP
    conflicting = {
        "outcome_x": {"shared_item": {"d0": 5, "d1": 6}},
        "outcome_y": {"shared_item": {"d0": 5, "d1": 7}, "other_item": {"d0": 1, "d1": 2}},
    }
    with pytest.raises(ValueError, match="STOP: unresolved joint constraint"):
        assert_no_conflicting_shared_item_values(conflicting, shared)


def test_k2_real_benchmark_has_zero_shared_raw_items():
    assert REAL_BENCHMARK_SHARED_RAW_ITEMS == {}


# L. multiple identical ideal values requiring deterministic tie-break.
def test_l_identical_ideal_values_deterministic_tiebreak():
    ideal = {"d_c": 10.5, "d_a": 10.5, "d_b": 10.5, "d_z": 10.5}
    r1 = project_bounded_integer(ideal, low=0, high=100, target_total=42.0)  # forces +2 from base-rounding-down (40)
    r2 = project_bounded_integer(dict(ideal), low=0, high=100, target_total=42.0)
    assert r1["achieved"] == r2["achieved"]  # deterministic, reproducible
    bumped = sorted(i for i, v in r1["achieved"].items() if v == 11)
    assert bumped == sorted(bumped)  # ascending profile_id order among the tied candidates
    assert bumped == ["d_a", "d_b"]  # lexically smallest ids win the tie


# M. permutation invariance of input row order.
def test_m_input_order_invariance():
    pairs = [("d3", 5.4), ("d1", 2.2), ("d2", 9.9), ("d0", 0.1)]
    natural = dict(pairs)
    shuffled = dict(reversed(pairs))
    r1 = project_bounded_integer(natural, low=0, high=10)
    r2 = project_bounded_integer(shuffled, low=0, high=10)
    assert r1["achieved"] == r2["achieved"]
    assert r1["achieved_mean"] == r2["achieved_mean"]


# N. target mean exactly attainable.
def test_n_target_mean_exactly_attainable():
    ideal = {"d0": 10.0, "d1": 20.0, "d2": 30.0}
    r = project_bounded_integer(ideal, low=0, high=100)
    assert r["mean_error"] == pytest.approx(0.0, abs=1e-12)
    assert r["achieved"] == {"d0": 10, "d1": 20, "d2": 30}


# O. target mean unattainable, nearest feasible mean selected.
def test_o_target_mean_unattainable_nearest_feasible():
    ideal = {"d0": 10.3, "d1": 20.3, "d2": 30.3}  # sum=60.9 -> t_star=61, mean=61/3 not exactly 20.3
    r = project_bounded_integer(ideal, low=0, high=100)
    assert r["target_total_used"] == 61
    assert sum(r["achieved"].values()) == 61
    assert r["mean_error"] != 0.0


# P. centered HTE preserved exactly before projection.
def test_p_centered_hte_preserved_before_projection():
    ids = _ids(8)
    control = {i: 30 + k for k, i in enumerate(ids)}
    treat = {i: 35 + 2 * k for k, i in enumerate(ids)}
    shift = compute_common_shift(control, treat, tau_hat_aj=7.5)
    verify_common_shift_identities(control, treat, 7.5, shift)  # must not raise


# Q. controls byte/value-identical.
def test_q_controls_unchanged():
    ids = _ids(5)
    control = {i: 33 for i in ids}
    treat = {i: 40 for i in ids}
    r = project_cell("a1", "trust_post", control, treat, tau_hat_aj=5.0, support_kind="bounded_integer", low=0, high=100)
    assert r["control"] == control
    assert r["control"] is not control or r["control"] == control  # value-identical regardless of object identity
    for k, v in control.items():
        assert r["control"][k] == v


# --- brute-force optimality verification for small N ---


def _brute_force_min_sq_distance(ideal: dict, low: int, high: int, target_total: int):
    ids = list(ideal.keys())
    best = None
    best_cost = None
    for combo in itertools.product(range(low, high + 1), repeat=len(ids)):
        if sum(combo) != target_total:
            continue
        cost = sum((y - ideal[i]) ** 2 for y, i in zip(combo, ids))
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best = dict(zip(ids, combo))
    return best_cost, best


@pytest.mark.parametrize("seed", range(6))
def test_brute_force_optimality_small_n(seed):
    import random

    rng = random.Random(seed)
    ids = _ids(4)
    ideal = {i: round(rng.uniform(-1, 4), 2) for i in ids}
    low, high = 0, 3
    r = project_bounded_integer(ideal, low=low, high=high)
    t_star = r["target_total_used"]
    achieved_cost = sum((r["achieved"][i] - ideal[i]) ** 2 for i in ids)
    best_cost, _ = _brute_force_min_sq_distance(ideal, low, high, t_star)
    assert achieved_cost == pytest.approx(best_cost, abs=1e-9)


# --- method-independence (Section 11) ---


def test_method_independence_same_code_path_for_any_tau_hat():
    ids = _ids(15)
    control = {i: 40 for i in ids}
    treat = {i: 44 for i in ids}
    # three arbitrary tau_hat values, standing in for Primary M2 / Secondary-1
    # MCONST / Secondary-2 MCONST_GSHAPE -- the function signature has no
    # method parameter, so it cannot branch on provenance by construction.
    import inspect

    sig = inspect.signature(project_cell)
    assert "method" not in sig.parameters and "model" not in sig.parameters and "calibration" not in sig.parameters

    for tau_hat in (1.9558595458395387, 6.7, -2.3):
        r = project_cell("a1", "trust_post", control, treat, tau_hat, support_kind="bounded_integer", low=0, high=100)
        assert r["achieved_postprojection_ate"] == pytest.approx(tau_hat, abs=0.5 / len(ids) + 1e-9)


# --- Section 9 fail-closed behavior ---


def test_fails_closed_on_missing_control_treatment_pair():
    with pytest.raises(ValueError, match="donor id set"):
        compute_common_shift({"d0": 1, "d1": 2}, {"d0": 1, "d2": 2}, tau_hat_aj=1.0)


def test_fails_closed_on_duplicate_donor():
    # a dict's keys can never literally duplicate, so the meaningful place
    # to fail closed on a duplicate donor id is when building the map from
    # a flat record list (e.g. raw retrieved-batch rows) in the first place.
    with pytest.raises(ValueError, match="duplicate donor id"):
        build_donor_map([("d0", 1.0), ("d1", 2.0), ("d0", 3.0)])
    # sanity: no duplicates is fine
    assert build_donor_map([("d0", 1.0), ("d1", 2.0)]) == {"d0": 1.0, "d1": 2.0}


def test_fails_closed_on_nonfinite_target_ate():
    with pytest.raises(ValueError, match="finite"):
        compute_common_shift({"d0": 1}, {"d0": 2}, tau_hat_aj=math.nan)
    with pytest.raises(ValueError, match="finite"):
        compute_common_shift({"d0": 1}, {"d0": 2}, tau_hat_aj=math.inf)


def test_fails_closed_on_unknown_support_kind():
    with pytest.raises(ValueError, match="unknown support_kind"):
        project_cell("a1", "o1", {"d0": 1}, {"d0": 2}, tau_hat_aj=1.0, support_kind="something_else")


def test_finite_discrete_stub_fails_closed():
    with pytest.raises(NotImplementedError, match="STOP"):
        project_finite_discrete({"d0": 1.0}, support_levels=[0, 0.5, 10])


def test_fails_closed_on_missing_constituent_item_metadata():
    with pytest.raises(ValueError, match="missing constituent-item metadata"):
        project_composite_cell(
            "a1",
            "behavior_mean",
            item_control={"behavior_meat": {"d0": 1}},  # missing 5 of 6 items
            item_treat={"behavior_meat": {"d0": 2}},
            tau_hat_aj=1.0,
            item_bounds={"behavior_meat": (0, 100)},
        )


def test_fails_closed_on_inconsistent_donor_sets_across_composite_items():
    items = ["behavior_meat", "behavior_transport", "behavior_solar", "behavior_fly", "behavior_talk", "behavior_donate"]
    item_control = {lab: {"d0": 1, "d1": 2} for lab in items}
    item_treat = {lab: {"d0": 1, "d1": 2} for lab in items}
    item_treat["behavior_meat"] = {"d0": 1, "d2": 2}  # different donor set
    bounds = {lab: (0, 100) for lab in items}
    with pytest.raises(ValueError, match="inconsistent donor id set"):
        project_composite_cell("a1", "behavior_mean", item_control, item_treat, tau_hat_aj=1.0, item_bounds=bounds)


def test_fails_closed_on_duplicate_intervention_outcome_cell_in_batch():
    ids = _ids(5)
    cell = {
        "intervention_id": "a1",
        "outcome_id": "trust_post",
        "control": {i: 40 for i in ids},
        "treat": {i: 44 for i in ids},
        "tau_hat_aj": 5.0,
        "support_kind": "bounded_integer",
        "low": 0,
        "high": 100,
    }
    with pytest.raises(ValueError, match="duplicate intervention/outcome cell"):
        project_target_ate_table([cell, dict(cell)])


def test_batch_runs_multiple_cells_through_one_shared_function():
    ids = _ids(5)
    cells = [
        {
            "intervention_id": f"a{k}",
            "outcome_id": "trust_post",
            "control": {i: 40 for i in ids},
            "treat": {i: 44 for i in ids},
            "tau_hat_aj": tau,
            "support_kind": "bounded_integer",
            "low": 0,
            "high": 100,
        }
        for k, tau in enumerate([1.9558595458395387, 5.0, -3.2])
    ]
    results = project_target_ate_table(cells)
    assert len(results) == 3
    assert {r["intervention_id"] for r in results} == {"a0", "a1", "a2"}
