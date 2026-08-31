"""§25 tests 9-14: IPF raking and controlled integerization."""

from __future__ import annotations

import numpy as np
import pytest

from population import raking


# 9. IPF convergence on a test matrix.
def test_ipf_converges_on_test_matrix():
    seed = np.array([[10.0, 20.0, 5.0], [15.0, 25.0, 10.0]])
    row_targets = np.array([50.0, 35.0])
    col_targets = np.array([30.0, 40.0, 15.0])
    fitted, n_iter, max_err = raking.ipf_2d(seed, row_targets, col_targets)
    assert n_iter > 0
    assert max_err < 1e-10


# 10 & 11. Exact row/column margins.
def test_ipf_reproduces_row_and_column_margins_exactly():
    seed = np.array([[10.0, 20.0, 5.0], [15.0, 25.0, 10.0]])
    row_targets = np.array([50.0, 35.0])
    col_targets = np.array([30.0, 40.0, 15.0])
    fitted, _, _ = raking.ipf_2d(seed, row_targets, col_targets)
    assert np.allclose(fitted.sum(axis=1), row_targets, atol=1e-6)
    assert np.allclose(fitted.sum(axis=0), col_targets, atol=1e-6)


def test_ipf_mismatched_totals_raises():
    seed = np.ones((2, 2))
    with pytest.raises(raking.RakingError):
        raking.ipf_2d(seed, np.array([10.0, 10.0]), np.array([5.0, 5.0]))


# 12. Exactly 40 joint cells.
def test_joint_cells_table_has_exactly_40_rows(quota_age, quota_race, synthetic_pums_factory):
    synthetic = synthetic_pums_factory(seed=5)
    table = raking.build_joint_cells_table(synthetic, quota_age, quota_race)
    assert len(table) == 40
    assert int(table["integer_target_n"].sum()) == 1000


def test_joint_cells_table_matches_quota_files_exactly(quota_age, quota_race, synthetic_pums_factory):
    synthetic = synthetic_pums_factory(seed=6)
    table = raking.build_joint_cells_table(synthetic, quota_age, quota_race)

    achieved_age = table.groupby(["gender", "age_band"])["integer_target_n"].sum().reset_index()
    merged_age = quota_age.merge(achieved_age, on=["gender", "age_band"])
    assert (merged_age["integer_target_n"] == merged_age["target_n"]).all()

    achieved_race = table.groupby(["gender", "race"])["integer_target_n"].sum().reset_index()
    merged_race = quota_race.merge(achieved_race, on=["gender", "race"])
    assert (merged_race["integer_target_n"] == merged_race["target_n"]).all()


# 13. Deterministic controlled rounding.
def test_controlled_integerize_is_deterministic():
    expected = np.array([[1.4, 2.3, 1.3], [2.6, 1.7, 1.7]])
    row_targets = np.array([5, 6])
    col_targets = np.array([4, 4, 3])
    result_a, _ = raking.controlled_integerize(expected, row_targets, col_targets)
    result_b, _ = raking.controlled_integerize(expected, row_targets, col_targets)
    assert np.array_equal(result_a, result_b)
    assert np.array_equal(result_a.sum(axis=1), row_targets)
    assert np.array_equal(result_a.sum(axis=0), col_targets)


def test_controlled_integerize_infeasible_shortfall_raises():
    expected = np.array([[3.9, 3.9], [3.9, 3.9]])  # floors to [[3,3],[3,3]] = row sums 6 each
    with pytest.raises(raking.RakingError):
        raking.controlled_integerize(expected, row_targets=np.array([5, 5]), col_targets=np.array([5, 5]))


# 14. Clear failure for a structural-zero cell that prevents fitting.
def test_structural_zero_cell_fails_clearly():
    seed = np.array([[10.0, 0.0], [15.0, 25.0]])
    with pytest.raises(raking.RakingError, match="structural-zero"):
        raking.ipf_2d(seed, np.array([10.0, 40.0]), np.array([25.0, 25.0]))


def test_joint_cells_table_fails_clearly_on_missing_donor_cell(quota_age, quota_race, synthetic_pums_factory):
    synthetic = synthetic_pums_factory(seed=7)
    # remove every donor from one specific (gender, age_band, race) cell.
    mask = (synthetic["gender"] == "Male") & (synthetic["age_band"] == "18-29") & (synthetic["race"] == "Other")
    starved = synthetic.loc[~mask]
    with pytest.raises(raking.RakingError):
        raking.build_joint_cells_table(starved, quota_age, quota_race)
