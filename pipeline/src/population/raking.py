"""src/population/raking.py

Iterative proportional fitting (raking) of an ACS-PUMS-weighted seed matrix to
the benchmark's published gender x age and gender x race quota margins, and
deterministic controlled integerization of the resulting noninteger expected
counts into the exact 40 gender x age_band x race integer cell targets.

This module is pure numerical logic -- it never reads a raw data file -- so
it is fully testable on synthetic seed matrices (see tests/population/test_raking.py)
independent of whether PUMS ingestion itself can proceed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .constants import AGE_BAND_ORDER, RACE_ORDER
from .io import get_logger

logger = get_logger("raking")


class RakingError(Exception):
    """Raised for any raking/integerization failure: a structural-zero seed
    cell, non-convergent IPF, infeasible or unsolved controlled rounding, or
    a result that violates the margins it was supposed to satisfy exactly.
    """


def ipf_2d(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int, float]:
    """Rake a 2D seed matrix so its row sums match `row_targets` and its
    column sums match `col_targets`.

    Raises RakingError if any seed cell is not strictly positive (a
    structural zero -- no donor weight to redistribute -- makes the margins
    unreachable for that cell), if the two margins' totals disagree, or if
    convergence is not reached within `max_iterations`.

    Returns (fitted_matrix, n_iterations, max_relative_error).
    """
    seed = np.asarray(seed, dtype=float)
    row_targets = np.asarray(row_targets, dtype=float)
    col_targets = np.asarray(col_targets, dtype=float)

    if np.any(seed <= 0):
        bad = np.argwhere(seed <= 0)
        raise RakingError(f"seed matrix has non-positive (structural-zero) cell(s) at {bad.tolist()}; cannot rake")
    if not np.isclose(row_targets.sum(), col_targets.sum()):
        raise RakingError(f"row target total ({row_targets.sum()}) != column target total ({col_targets.sum()})")

    mat = seed.copy()
    max_err = float("inf")
    for iteration in range(1, max_iterations + 1):
        mat *= (row_targets / mat.sum(axis=1))[:, None]
        mat *= (col_targets / mat.sum(axis=0))[None, :]
        row_err = np.max(np.abs(mat.sum(axis=1) - row_targets) / row_targets)
        col_err = np.max(np.abs(mat.sum(axis=0) - col_targets) / col_targets)
        max_err = float(max(row_err, col_err))
        if max_err < tolerance:
            return mat, iteration, max_err

    raise RakingError(f"IPF did not converge within {max_iterations} iterations (max residual error {max_err})")


def controlled_integerize(
    expected: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    tie_epsilon: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert IPF's noninteger expected cell counts into exact integers that
    preserve every row and column margin exactly, via deterministic
    controlled rounding (scipy.optimize.milp):

      1. floor every cell;
      2. compute each row/column's remaining shortfall against its target;
      3. one binary decision variable per cell (does it receive +1?);
      4. constrain every row's and column's shortfall exactly;
      5. objective prefers giving +1 to the largest fractional remainders,
         with a tiny deterministic tie-break in canonical (row, col) order
         (row/col index order, i.e. the caller's AGE_BAND_ORDER/RACE_ORDER)
         so ties never depend on solver-internal nondeterminism.

    Returns (integer_matrix, solve_info) and raises RakingError if the
    shortfalls are infeasible (e.g. flooring already overshoots a target) or
    the MILP does not report success, or if the result somehow fails to
    reproduce the exact margins.
    """
    expected = np.asarray(expected, dtype=float)
    row_targets = np.asarray(row_targets, dtype=int)
    col_targets = np.asarray(col_targets, dtype=int)
    n_rows, n_cols = expected.shape

    floor_mat = np.floor(expected).astype(int)
    frac = expected - floor_mat
    row_shortfall = row_targets - floor_mat.sum(axis=1)
    col_shortfall = col_targets - floor_mat.sum(axis=0)

    if np.any(row_shortfall < 0) or np.any(col_shortfall < 0):
        raise RakingError(
            f"controlled rounding infeasible: flooring already exceeds a target "
            f"(row shortfalls {row_shortfall.tolist()}, col shortfalls {col_shortfall.tolist()})"
        )
    if row_shortfall.sum() != col_shortfall.sum():
        raise RakingError(f"row shortfall total ({row_shortfall.sum()}) != column shortfall total ({col_shortfall.sum()})")

    n = n_rows * n_cols

    def idx(i: int, j: int) -> int:
        return i * n_cols + j

    cost = np.array([-frac[i, j] + tie_epsilon * idx(i, j) for i in range(n_rows) for j in range(n_cols)])

    a_rows = np.zeros((n_rows, n))
    for i in range(n_rows):
        for j in range(n_cols):
            a_rows[i, idx(i, j)] = 1
    a_cols = np.zeros((n_cols, n))
    for j in range(n_cols):
        for i in range(n_rows):
            a_cols[j, idx(i, j)] = 1

    a = np.vstack([a_rows, a_cols])
    b = np.concatenate([row_shortfall, col_shortfall]).astype(float)
    constraints = LinearConstraint(a, lb=b, ub=b)
    bounds = Bounds(lb=0, ub=1)

    result = milp(c=cost, constraints=constraints, bounds=bounds, integrality=np.ones(n))
    if not result.success:
        raise RakingError(f"controlled-rounding MILP did not succeed: status={result.status} message={result.message}")

    x = np.round(result.x).astype(int).reshape(n_rows, n_cols)
    integer_matrix = floor_mat + x

    if not np.array_equal(integer_matrix.sum(axis=1), row_targets):
        raise RakingError("controlled rounding result violates row margins")
    if not np.array_equal(integer_matrix.sum(axis=0), col_targets):
        raise RakingError("controlled rounding result violates column margins")

    solve_info = {"objective": float(result.fun), "status": int(result.status), "message": str(result.message)}
    return integer_matrix, solve_info


def build_seed_matrix(recoded_pums: pd.DataFrame, gender: str) -> np.ndarray:
    """Sum pums_person_weight over age_band x race for one gender, in
    canonical AGE_BAND_ORDER x RACE_ORDER order (§10 step 1). Missing cells
    (no donor at all in that age_band x race combination for this gender)
    are 0, which ipf_2d will correctly reject as a structural zero.
    """
    subset = recoded_pums.loc[recoded_pums["gender"] == gender]
    pivot = subset.pivot_table(index="age_band", columns="race", values="pums_person_weight", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=AGE_BAND_ORDER, columns=RACE_ORDER, fill_value=0)
    return pivot.to_numpy(dtype=float)


def build_joint_cells_table(
    recoded_pums: pd.DataFrame,
    quota_age: pd.DataFrame,
    quota_race: pd.DataFrame,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> pd.DataFrame:
    """Run §10 end-to-end for both genders and assemble the 40-row joint
    cell table with every column the population report needs. `quota_age`
    has columns gender/age_band/target_n; `quota_race` has
    gender/race/target_n (config/quota_gender_age_1000.csv and
    config/quota_gender_race_1000.csv, loaded as DataFrames).
    """
    rows: list[dict[str, Any]] = []
    for gender in ("Male", "Female"):
        age_targets = quota_age.loc[quota_age["gender"] == gender].set_index("age_band").reindex(AGE_BAND_ORDER)["target_n"].to_numpy()
        race_targets = quota_race.loc[quota_race["gender"] == gender].set_index("race").reindex(RACE_ORDER)["target_n"].to_numpy()

        seed = build_seed_matrix(recoded_pums, gender)
        expected, n_iter, max_err = ipf_2d(seed, age_targets, race_targets, tolerance, max_iterations)
        integer_targets, solve_info = controlled_integerize(expected, age_targets, race_targets)

        seed_total = seed.sum()
        for i, age_band in enumerate(AGE_BAND_ORDER):
            for j, race in enumerate(RACE_ORDER):
                rows.append(
                    {
                        "gender": gender,
                        "age_band": age_band,
                        "race": race,
                        "pums_weighted_seed": float(seed[i, j]),
                        "pums_seed_share_within_gender": float(seed[i, j] / seed_total),
                        "ipf_expected_n": float(expected[i, j]),
                        "fractional_remainder": float(expected[i, j] - np.floor(expected[i, j])),
                        "integer_target_n": int(integer_targets[i, j]),
                        "age_margin_target": int(age_targets[i]),
                        "race_margin_target": int(race_targets[j]),
                        "ipf_iterations": n_iter,
                        "ipf_max_error": max_err,
                    }
                )
        logger.info(
            "%s: IPF converged in %d iterations (max error %.2e); controlled rounding status=%s",
            gender, n_iter, max_err, solve_info["status"],
        )

    table = pd.DataFrame(rows)
    n_profiles = int(quota_age["target_n"].sum())
    if len(table) != 40:
        raise RakingError(f"expected exactly 40 joint cells, got {len(table)}")
    if int(table["integer_target_n"].sum()) != n_profiles:
        raise RakingError(f"integer_target_n must sum to {n_profiles} (quota total), got {int(table['integer_target_n'].sum())}")
    if (table["integer_target_n"] < 0).any():
        raise RakingError("controlled rounding produced a negative cell target")
    return table
