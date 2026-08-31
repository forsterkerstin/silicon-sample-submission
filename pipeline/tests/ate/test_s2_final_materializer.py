"""Tests for the S2 final target-materialization code path
(ate.s2_final_materializer), using synthetic fixtures ONLY -- never real
target G output. Proves it correctly composes the already-frozen
Secondary-2 shape estimator and shared projector, and rejects any
target-F-shaped input."""

from __future__ import annotations

import pytest

from ate.s2_final_materializer import materialize_s2_target_predictions

INTERVENTIONS = [f"a{i}" for i in range(1, 17)]
OUTCOMES = [f"j{j}" for j in range(1, 14)]
R_BY_OUTCOME = {j: 10.0 + 5.0 * idx for idx, j in enumerate(OUTCOMES)}
IDS = [f"d{i}" for i in range(10)]


def make_single_item_cells():
    cells = []
    for ai, a in enumerate(INTERVENTIONS):
        for ji, j in enumerate(OUTCOMES):
            control = {i: 40 for i in IDS}
            treat = {i: 42 + ai * 0.1 - ji * 0.05 for i in IDS}
            tau_native = sum(treat.values()) / len(treat) - sum(control.values()) / len(control)
            cells.append(
                {
                    "intervention_id": a,
                    "outcome_id": j,
                    "R_j": R_BY_OUTCOME[j],
                    "tau_g_native": tau_native,
                    "control": control,
                    "treat": treat,
                    "support_kind": "bounded_integer",
                    "low": 0,
                    "high": 100,
                }
            )
    return cells


def test_materializes_exactly_208_cells():
    results = materialize_s2_target_predictions(make_single_item_cells())
    assert len(results) == 208
    assert {(r["intervention_id"], r["outcome_id"]) for r in results} == {(a, j) for a in INTERVENTIONS for j in OUTCOMES}


def test_every_cell_carries_full_diagnostics():
    results = materialize_s2_target_predictions(make_single_item_cells())
    required = {"n", "control", "achieved_treat", "native_g_ate", "requested_calibrated_ate", "ideal_shift_c", "preprojection_ate", "achieved_postprojection_ate", "projection_ate_error", "n_responses_changed_by_projection", "fraction_changed"}
    for r in results:
        assert required.issubset(r.keys())


def test_controls_pass_through_unchanged():
    cells = make_single_item_cells()
    results = materialize_s2_target_predictions(cells)
    by_key = {(r["intervention_id"], r["outcome_id"]): r for r in results}
    for c in cells:
        r = by_key[(c["intervention_id"], c["outcome_id"])]
        assert r["control"] == c["control"]


def test_composite_cell_supported_via_item_control_treat():
    cells = [c for c in make_single_item_cells() if c["outcome_id"] != "j1"]  # replace the whole j1 outcome column
    items = ["behavior_meat", "behavior_transport", "behavior_solar", "behavior_fly", "behavior_talk", "behavior_donate"]
    r_j1 = R_BY_OUTCOME["j1"]
    for ai, a in enumerate(INTERVENTIONS):
        item_control = {lab: {i: 40 + k for k, i in enumerate(IDS)} for lab in items}
        item_treat = {lab: {i: 45 + k + ai * 0.01 for k, i in enumerate(IDS)} for lab in items}
        control_composite = {i: sum(item_control[lab][i] for lab in items) / len(items) for i in IDS}
        treat_composite = {i: sum(item_treat[lab][i] for lab in items) / len(items) for i in IDS}
        tau_native = sum(treat_composite.values()) / len(IDS) - sum(control_composite.values()) / len(IDS)
        cells.append(
            {
                "intervention_id": a,
                "outcome": "behavior_mean",
                "R_j": r_j1,
                "tau_g_native": tau_native,
                "item_control": item_control,
                "item_treat": item_treat,
                "item_bounds": {lab: (0, 100) for lab in items},
            }
        )
    results = materialize_s2_target_predictions(cells)
    composite_result = next(r for r in results if r["intervention_id"] == "a1" and r["outcome_id"] == "behavior_mean")
    assert "projected_items" in composite_result
    assert len(results) == 208


@pytest.mark.parametrize("forbidden_key", ["tau_f_native", "target_f", "raw_f_ate_pp", "raw_f_ate_native", "synthetic_effect_pp", "z_se_native"])
def test_rejects_any_target_f_shaped_input(forbidden_key):
    cells = make_single_item_cells()
    cells[0][forbidden_key] = 1.0
    with pytest.raises(ValueError, match="does not accept target-F input"):
        materialize_s2_target_predictions(cells)


def test_fails_closed_on_wrong_cell_universe():
    cells = make_single_item_cells()[:-1]
    with pytest.raises(ValueError):
        materialize_s2_target_predictions(cells)
