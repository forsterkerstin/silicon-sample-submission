"""Secondary-2 prospective method: MCONST_GSHAPE.

Prospectively frozen BEFORE any target G model output has been retrieved or
inspected. Combines Secondary-1's externally-estimated global calibration
level (MU_EXTERNAL, from the frozen 136-effect human archive) with the
target-specific *relative* ATE shape inherited exactly from native target-G
ATEs:

    g_aj = 100 * tau_G_aj / R_j                        (native G ATE, normalized)
    g_bar = unweighted mean of g_aj over all 208 cells  (16 interventions x 13 outcomes)
    theta_hat_S2_aj = MU_EXTERNAL + GAMMA_G_SHAPE * (g_aj - g_bar)
    tau_hat_S2_aj = (R_j / 100) * theta_hat_S2_aj

GAMMA_G_SHAPE is frozen at exactly 1.0 -- never fit or tuned.

This module is a pure function of a caller-supplied native-G ATE-cell table.
It never queries, retrieves, or reads any target model output itself, and it
does not accept or require any target-F input. The resulting per-cell target
ATE table is intended to feed the same (not-yet-implemented, described only
in scientific-plan documentation as of this freeze) common-shift /
deterministic support-projection step already used for the primary method --
no Secondary-2-specific projection rule is introduced here.
"""

from __future__ import annotations

from typing import Mapping, Sequence

EXPECTED_N_INTERVENTIONS = 16
EXPECTED_N_OUTCOMES = 13
EXPECTED_N_CELLS = EXPECTED_N_INTERVENTIONS * EXPECTED_N_OUTCOMES  # 208

EXTERNAL_MU = 1.9558595458395387
GAMMA_G_SHAPE = 1.0

_FORBIDDEN_TARGET_F_KEYS = {"tau_f_native", "target_f", "raw_f_ate_pp", "raw_f_ate_native", "synthetic_effect_pp", "z_se_native"}


def _validate_no_target_f_input(cells: Sequence[Mapping]) -> None:
    for c in cells:
        present = _FORBIDDEN_TARGET_F_KEYS & set(c.keys())
        if present:
            raise ValueError(f"Secondary-2 does not accept target-F input; found forbidden key(s) {sorted(present)} on cell {c.get('intervention_id')!r}/{c.get('outcome_id')!r}")


def _validate_cell_universe(cells: Sequence[Mapping]) -> None:
    if not cells:
        raise ValueError("cells must be non-empty")
    seen = set()
    interventions = set()
    outcomes = set()
    for c in cells:
        key = (c["intervention_id"], c["outcome_id"])
        if key in seen:
            raise ValueError(f"duplicate intervention-outcome cell: {key}")
        seen.add(key)
        interventions.add(c["intervention_id"])
        outcomes.add(c["outcome_id"])
    if len(cells) != EXPECTED_N_CELLS:
        raise ValueError(f"expected exactly {EXPECTED_N_CELLS} cells (16 interventions x 13 outcomes), got {len(cells)}")
    if len(interventions) != EXPECTED_N_INTERVENTIONS:
        raise ValueError(f"expected exactly {EXPECTED_N_INTERVENTIONS} distinct interventions, got {len(interventions)}")
    if len(outcomes) != EXPECTED_N_OUTCOMES:
        raise ValueError(f"expected exactly {EXPECTED_N_OUTCOMES} distinct outcomes, got {len(outcomes)}")
    expected_full = {(a, j) for a in interventions for j in outcomes}
    missing = expected_full - seen
    if missing:
        raise ValueError(f"missing intervention-outcome cells: {sorted(missing)}")


def _validate_ranges(cells: Sequence[Mapping]) -> None:
    r_by_outcome: dict = {}
    for c in cells:
        r = c.get("R_j")
        if r is None or isinstance(r, bool) or not isinstance(r, (int, float)) or not (r > 0):
            raise ValueError(f"R_j must be a positive number; got {r!r} for outcome {c.get('outcome_id')!r}")
        oid = c["outcome_id"]
        if oid in r_by_outcome and r_by_outcome[oid] != r:
            raise ValueError(f"inconsistent R_j for outcome {oid!r}: {r_by_outcome[oid]} vs {r}")
        r_by_outcome[oid] = r


def compute_secondary_2_target_ates(
    cells: Sequence[Mapping],
    *,
    external_mu: float = EXTERNAL_MU,
    gamma: float = GAMMA_G_SHAPE,
) -> dict:
    """cells: iterable of mappings, one per (intervention, outcome), each with
    keys `intervention_id`, `outcome_id`, `tau_g_native` (native target-G ATE:
    treatment minus control), `R_j` (positive native outcome range). Exactly
    16 x 13 = 208 cells required, no duplicates, no missing combinations.

    Returns a dict with the per-cell target-ATE table (`cells`, each augmented
    with `g_aj`, `theta_hat_s2_aj`, `tau_hat_s2_aj`, `c_aj`) plus `g_bar` and
    the frozen constants used.
    """
    if gamma != GAMMA_G_SHAPE:
        raise ValueError(f"gamma_g_shape is frozen at {GAMMA_G_SHAPE} -- no tuning permitted")
    cells = list(cells)
    _validate_no_target_f_input(cells)
    _validate_cell_universe(cells)
    _validate_ranges(cells)

    g_by_key = {(c["intervention_id"], c["outcome_id"]): 100.0 * float(c["tau_g_native"]) / float(c["R_j"]) for c in cells}

    # Sum in a fixed key order so the global mean is exactly permutation-invariant
    # in the caller's input ordering (floating-point summation is order-sensitive).
    ordered_keys = sorted(g_by_key)
    g_bar = sum(g_by_key[k] for k in ordered_keys) / len(ordered_keys)

    results = []
    for c in cells:
        key = (c["intervention_id"], c["outcome_id"])
        g_aj = g_by_key[key]
        r = float(c["R_j"])
        tau_g_native = float(c["tau_g_native"])
        theta_hat = external_mu + gamma * (g_aj - g_bar)
        tau_hat = (r / 100.0) * theta_hat
        c_aj = tau_hat - tau_g_native
        results.append(
            {
                "intervention_id": c["intervention_id"],
                "outcome_id": c["outcome_id"],
                "tau_g_native": tau_g_native,
                "R_j": r,
                "g_aj": g_aj,
                "theta_hat_s2_aj": theta_hat,
                "tau_hat_s2_aj": tau_hat,
                "c_aj": c_aj,
            }
        )

    return {
        "cells": results,
        "g_bar": g_bar,
        "external_mu": external_mu,
        "gamma_g_shape": gamma,
        "n_cells": len(results),
    }
