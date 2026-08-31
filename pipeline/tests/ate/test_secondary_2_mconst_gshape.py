"""Unit tests for the Secondary-2 MCONST_GSHAPE transformation
(ate.secondary_2_mconst_gshape), using synthetic fixture matrices ONLY.

Never invoked on real target G output in this suite -- the 208-cell
fixtures below are hand-built synthetic numbers, not retrieved data.
"""

from __future__ import annotations

import math

import pytest

from ate.secondary_2_mconst_gshape import (
    EXPECTED_N_CELLS,
    EXTERNAL_MU,
    GAMMA_G_SHAPE,
    compute_secondary_2_target_ates,
)

INTERVENTIONS = [f"a{i}" for i in range(1, 17)]  # 16
OUTCOMES = [f"j{j}" for j in range(1, 14)]  # 13
R_BY_OUTCOME = {j: 10.0 + 5.0 * idx for idx, j in enumerate(OUTCOMES)}  # distinct positive ranges


def make_fixture_cells(*, seed_scale: float = 1.0, order="natural"):
    cells = []
    for ai, a in enumerate(INTERVENTIONS):
        for ji, j in enumerate(OUTCOMES):
            # deterministic synthetic native G ATE, varies by cell
            tau = seed_scale * (0.3 * ai - 0.15 * ji + 0.05 * ai * ji % 3)
            cells.append({"intervention_id": a, "outcome_id": j, "tau_g_native": tau, "R_j": R_BY_OUTCOME[j]})
    if order == "reversed":
        cells = list(reversed(cells))
    elif order == "shuffled":
        # deterministic non-natural order without relying on random
        cells = cells[1::2] + cells[0::2]
    return cells


# 1. 208-cell output count preserved.
def test_208_cell_output_count_preserved():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    assert result["n_cells"] == EXPECTED_N_CELLS == 208
    assert len(result["cells"]) == 208


# 2. mean normalized output equals MU_EXTERNAL.
def test_mean_normalized_output_equals_external_mu():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    thetas = [c["theta_hat_s2_aj"] for c in result["cells"]]
    mean_theta = sum(thetas) / len(thetas)
    assert mean_theta == pytest.approx(EXTERNAL_MU, abs=1e-9)


# 3. all pairwise/cell-centered G differences are preserved.
def test_pairwise_g_differences_preserved_exactly():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    g_by_key = {(c["intervention_id"], c["outcome_id"]): c["g_aj"] for c in result["cells"]}
    theta_by_key = {(c["intervention_id"], c["outcome_id"]): c["theta_hat_s2_aj"] for c in result["cells"]}
    keys = list(g_by_key)
    for i in range(0, len(keys), 37):  # sample pairs across the grid
        for k in range(i + 1, min(i + 5, len(keys))):
            a, b = keys[i], keys[k]
            assert (theta_by_key[a] - theta_by_key[b]) == pytest.approx(g_by_key[a] - g_by_key[b], abs=1e-9)


# 4. gamma is exactly 1.
def test_gamma_is_exactly_one_and_rejects_other_values():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    assert result["gamma_g_shape"] == 1.0 == GAMMA_G_SHAPE
    with pytest.raises(ValueError, match="frozen at"):
        compute_secondary_2_target_ates(make_fixture_cells(), gamma=0.9)


# 5. native-scale conversion uses the correct R_j.
def test_native_scale_conversion_uses_correct_r_j():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    for c in result["cells"]:
        expected_tau = (c["R_j"] / 100.0) * c["theta_hat_s2_aj"]
        assert c["tau_hat_s2_aj"] == pytest.approx(expected_tau, abs=1e-12)
        expected_g = 100.0 * c["tau_g_native"] / c["R_j"]
        assert c["g_aj"] == pytest.approx(expected_g, abs=1e-12)


# 6. common-shift algebra reproduces the requested target ATE.
def test_common_shift_algebra_reproduces_target_ate():
    result = compute_secondary_2_target_ates(make_fixture_cells())
    g_bar = result["g_bar"]
    for c in result["cells"]:
        # c_aj = tau_hat - tau_g_native, by definition
        assert c["c_aj"] == pytest.approx(c["tau_hat_s2_aj"] - c["tau_g_native"], abs=1e-12)
        # closed-form simplification from the spec
        closed_form = (c["R_j"] / 100.0) * (EXTERNAL_MU - g_bar)
        assert c["c_aj"] == pytest.approx(closed_form, abs=1e-9)
        # applying the shift back to native G reproduces the target exactly
        assert (c["tau_g_native"] + c["c_aj"]) == pytest.approx(c["tau_hat_s2_aj"], abs=1e-12)
    # every cell within the same outcome gets an identical shift, regardless of intervention
    by_outcome: dict = {}
    for c in result["cells"]:
        by_outcome.setdefault(c["outcome_id"], set()).add(round(c["c_aj"], 9))
    for oid, shifts in by_outcome.items():
        assert len(shifts) == 1, f"outcome {oid} has non-uniform common shift across interventions: {shifts}"


# 7. controls remain unchanged.
def test_controls_remain_unchanged_no_control_input_touched():
    # The function operates purely on already-differenced ATEs (tau_g_native
    # = treatment minus control); it never receives or mutates a control-arm
    # value, so native G controls are structurally unaffected by construction.
    cells = make_fixture_cells()
    control_keys = {"Y_G_i0j", "control", "control_arm", "y_control"}
    for c in cells:
        assert control_keys.isdisjoint(c.keys())
    compute_secondary_2_target_ates(cells)  # runs without needing/consuming any control field
    assert control_keys.isdisjoint(set().union(*[set(c.keys()) for c in cells]))


# 8. centered respondent HTE is preserved before projection.
def test_centered_respondent_hte_preserved_by_constant_shift():
    # Algebraic property of v_iaj = Y_G_iaj + c_aj (the not-yet-implemented
    # downstream common-shift application): adding the same constant c_aj to
    # every respondent in a cell leaves within-cell centered deviations
    # (heterogeneity) exactly unchanged. Demonstrated on a synthetic
    # per-respondent fixture using our computed c_aj.
    result = compute_secondary_2_target_ates(make_fixture_cells())
    c_aj = result["cells"][0]["c_aj"]
    synthetic_respondents = [1.2, 3.4, -0.5, 2.2, 0.1]
    mean_before = sum(synthetic_respondents) / len(synthetic_respondents)
    centered_before = [y - mean_before for y in synthetic_respondents]
    shifted = [y + c_aj for y in synthetic_respondents]
    mean_after = sum(shifted) / len(shifted)
    centered_after = [y - mean_after for y in shifted]
    for before, after in zip(centered_before, centered_after):
        assert before == pytest.approx(after, abs=1e-12)


# 9. no target-F input is accepted or required.
def test_no_target_f_input_accepted_or_required():
    cells = make_fixture_cells()
    # baseline succeeds without any F-related key
    compute_secondary_2_target_ates(cells)
    poisoned = [dict(c) for c in cells]
    poisoned[0]["raw_f_ate_pp"] = 5.0
    with pytest.raises(ValueError, match="target-F input"):
        compute_secondary_2_target_ates(poisoned)


# 10. permutation/order invariance of the global 208-cell mean.
def test_permutation_invariance_of_global_mean():
    natural = compute_secondary_2_target_ates(make_fixture_cells(order="natural"))
    reversed_ = compute_secondary_2_target_ates(make_fixture_cells(order="reversed"))
    shuffled = compute_secondary_2_target_ates(make_fixture_cells(order="shuffled"))
    assert natural["g_bar"] == reversed_["g_bar"] == shuffled["g_bar"]


# 11. failure if cell universe != exact 16x13 expected target universe.
def test_failure_on_wrong_cell_universe_size():
    cells = make_fixture_cells()[:-1]  # drop one cell -> 207
    with pytest.raises(ValueError, match="expected exactly 208 cells"):
        compute_secondary_2_target_ates(cells)

    cells17 = make_fixture_cells() + [{"intervention_id": "a17", "outcome_id": "j1", "tau_g_native": 1.0, "R_j": R_BY_OUTCOME["j1"]}]
    with pytest.raises(ValueError):
        compute_secondary_2_target_ates(cells17)


# 12. failure on missing/duplicate intervention-outcome cells.
def test_failure_on_missing_or_duplicate_cells():
    cells = make_fixture_cells()

    dup = cells[:-1] + [dict(cells[0])]  # replace the last cell with a duplicate of the first
    with pytest.raises(ValueError, match="duplicate"):
        compute_secondary_2_target_ates(dup)

    # drop (a1, j1) entirely and replace it with a duplicate of (a1, j2):
    # still 208 rows, but a1/j1 is genuinely missing from the universe.
    truly_missing = [c for c in cells if not (c["intervention_id"] == "a1" and c["outcome_id"] == "j1")]
    truly_missing.append({"intervention_id": "a1", "outcome_id": "j2", "tau_g_native": 0.0, "R_j": R_BY_OUTCOME["j2"]})
    with pytest.raises(ValueError):
        compute_secondary_2_target_ates(truly_missing)


# 13. failure on nonpositive or missing R_j.
def test_failure_on_nonpositive_or_missing_r_j():
    cells = make_fixture_cells()
    zero_r = [dict(c) for c in cells]
    zero_r[0]["R_j"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        compute_secondary_2_target_ates(zero_r)

    negative_r = [dict(c) for c in cells]
    negative_r[0]["R_j"] = -5.0
    with pytest.raises(ValueError, match="positive"):
        compute_secondary_2_target_ates(negative_r)

    missing_r = [dict(c) for c in cells]
    del missing_r[0]["R_j"]
    with pytest.raises(ValueError, match="positive"):
        compute_secondary_2_target_ates(missing_r)

    inconsistent_r = [dict(c) for c in cells]
    inconsistent_r[0]["R_j"] = inconsistent_r[0]["R_j"] + 1.0  # same outcome, different R than its sibling cells
    with pytest.raises(ValueError, match="inconsistent"):
        compute_secondary_2_target_ates(inconsistent_r)


# 14. existing support projection can consume the resulting target ATE table
#     without a new scientific rule.
def test_output_schema_is_projection_ready_without_new_rule():
    # No common-shift / support-projection implementation exists yet in this
    # repo (target G output hasn't been retrieved), so this is a schema
    # readiness check, not an integration test against real projection code:
    # the output must expose exactly the fields a downstream common-shift +
    # support-projection step needs (native ATE to preserve differences
    # against, the target ATE to project toward, the per-cell native-scale
    # shift, and the outcome range) with no extra embedded transformation.
    result = compute_secondary_2_target_ates(make_fixture_cells())
    required_fields = {"intervention_id", "outcome_id", "tau_g_native", "R_j", "g_aj", "theta_hat_s2_aj", "tau_hat_s2_aj", "c_aj"}
    for c in result["cells"]:
        assert required_fields == set(c.keys())
        # no rule beyond additive common-shift is embedded: tau_g_native + c_aj must land exactly on tau_hat_s2_aj
        assert c["tau_g_native"] + c["c_aj"] == pytest.approx(c["tau_hat_s2_aj"], abs=1e-12)


def test_frozen_constants_match_the_specified_values():
    assert EXTERNAL_MU == 1.9558595458395387
    assert GAMMA_G_SHAPE == 1.0
    assert EXPECTED_N_CELLS == 208
