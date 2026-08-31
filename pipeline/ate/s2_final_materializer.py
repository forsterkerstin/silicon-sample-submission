"""S2 (MCONST_GSHAPE) final target-materialization code path.

Composes two already-frozen, already-tested modules -- no new scientific
rule is introduced here:

    ate.secondary_2_mconst_gshape.compute_secondary_2_target_ates
        native G 208-cell ATEs -> normalize -> global g_bar -> recenter to
        MU_EXTERNAL -> per-cell theta_hat_s2_aj / tau_hat_s2_aj

    ate.target_projection.project_cell / project_composite_cell
        tau_hat_aj + native G control/treatment responses -> common shift
        -> minimum-distortion support projection -> final per-respondent
        microdata (single-item and composite outcomes respectively)

Prepared as the final production code path (per the S2-promotion
governance decision) but NOT executed on real target G output in this
freeze -- see scripts/freeze_final_submission_manifest_s2.py. Accepts NO
target-F input anywhere in its signature and fails closed if any F-shaped
key is present in the caller-supplied cell data.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ate.secondary_2_mconst_gshape import compute_secondary_2_target_ates
from ate.target_projection import project_cell, project_composite_cell

_FORBIDDEN_TARGET_F_KEYS = {"tau_f_native", "target_f", "raw_f_ate_pp", "raw_f_ate_native", "synthetic_effect_pp", "z_se_native"}


def _assert_no_target_f_input(cell_specs: Sequence[Mapping]) -> None:
    for spec in cell_specs:
        present = _FORBIDDEN_TARGET_F_KEYS & set(spec.keys())
        if present:
            raise ValueError(f"S2 final materializer does not accept target-F input; found forbidden key(s) {sorted(present)} on cell {spec.get('intervention_id')!r}/{spec.get('outcome_id', spec.get('outcome'))!r}")


def materialize_s2_target_predictions(cell_specs: Sequence[Mapping]) -> list[dict]:
    """cell_specs: one dict per (intervention, outcome) cell, exactly 208
    total (16 interventions x 13 outcomes), each with:

      - intervention_id, outcome_id (or "outcome" for composites), R_j
      - tau_g_native: native P1 G ATE for this cell (treatment - control),
        used ONLY to build the S2 208-cell shape input
      - EITHER (single-item outcome): control, treat (donor_id -> native
        value), support_kind ("bounded_integer"/"binary"), low/high
      - OR (composite outcome): item_control, item_treat (item_label ->
        {donor_id: value}), item_bounds (item_label -> (low, high))

    Returns one project_cell/project_composite_cell result dict per cell,
    each carrying the full Section-10 diagnostics. Never reads or requires
    any target-F value -- fails closed if one is supplied.
    """
    cell_specs = list(cell_specs)
    _assert_no_target_f_input(cell_specs)

    shape_cells = [{"intervention_id": c["intervention_id"], "outcome_id": c.get("outcome_id", c.get("outcome")), "R_j": c["R_j"], "tau_g_native": c["tau_g_native"]} for c in cell_specs]
    s2 = compute_secondary_2_target_ates(shape_cells)
    tau_hat_by_key = {(c["intervention_id"], c["outcome_id"]): c["tau_hat_s2_aj"] for c in s2["cells"]}

    results = []
    for spec in cell_specs:
        outcome_id = spec.get("outcome_id", spec.get("outcome"))
        key = (spec["intervention_id"], outcome_id)
        tau_hat = tau_hat_by_key[key]
        if "item_control" in spec:
            r = project_composite_cell(spec["intervention_id"], outcome_id, spec["item_control"], spec["item_treat"], tau_hat, spec["item_bounds"])
        else:
            r = project_cell(
                spec["intervention_id"],
                outcome_id,
                spec["control"],
                spec["treat"],
                tau_hat,
                support_kind=spec["support_kind"],
                low=spec.get("low"),
                high=spec.get("high"),
                support_levels=spec.get("support_levels"),
            )
        results.append(r)
    return results
