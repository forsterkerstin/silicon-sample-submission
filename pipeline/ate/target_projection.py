"""Shared deterministic common-shift + support-projection post-processor.

Prospectively frozen BEFORE any target G model output has been retrieved or
inspected (see scripts/freeze_target_projection_method.py). Shared, without
modification, by every calibration method that produces a target ATE table
(Primary M2, Secondary-1 MCONST, Secondary-2 MCONST_GSHAPE): this module has
no knowledge of which method produced `tau_hat_aj` -- it only consumes a
target ATE number and native G control/treatment responses.

Real-benchmark applicability (verified against survey_content.py before
writing this module, not assumed):

  - Every one of the 44 raw survey items is elicited on a CONSECUTIVE
    integer support: sliders [0,100], donation_ams [0,10], newsletter_signup
    {0,1} (see inference/prompts.py::item_json_schema / _bounds_for_item,
    which requests "type": "integer" for every item, never "number"). No
    raw item in this benchmark has a non-consecutive finite discrete
    support, so `project_finite_discrete` below is implemented as an
    explicit fail-closed stub, not a general solver -- see its docstring.
  - No raw item is referenced by more than one of the 13 outcome composites
    (independently re-verified here: 44 raw items, 44 distinct
    (item -> outcome) mappings, zero items shared across outcomes). So the
    Section-6 "shared raw item" joint-constraint scenario does not currently
    arise in this benchmark; `audit_shared_raw_items` / the joint-constraint
    check are implemented and unit-tested against synthetic fixtures so the
    machinery exists if metadata ever changes, but they are not exercised on
    the real OUTCOME_COMPOSITES today.

Core algorithm (bounded-integer / binary support): each respondent's ideal
shifted value v_i is rounded to its nearest support level (ties go to the
lower level); the resulting total is then adjusted, one +/-1 step at a time,
toward the nearest attainable integer total to sum(v_i), always picking the
respondent(s) with the smallest marginal increase in squared distance
(equivalently: largest residual fractional part for +1 steps, smallest for
-1 steps), with ties broken by ascending profile_id by default, or by a
frozen SHA-256 tie-break hash of (seed, condition, raw item name,
profile_id) when a `tie_break_context` is supplied (see
FROZEN_PROJECTION_TIE_BREAK_SEED below) -- introduced after a diagnostic
found that ascending-profile_id tie-breaking, combined with this
benchmark's demographically-blocked donor id assignment, produced a
materially uneven demographic adjustment pattern; project_composite_cell
below always supplies this context for the real production path. Either
way, the tie-break decides only *which* respondent(s) receive an
already-determined +/-1 step among those tied for the smallest marginal
cost increase -- it never changes which step sizes are taken or the
target total. This greedy adjustment is optimal for minimizing
sum((y_i - v_i)^2) subject to sum(y_i) fixed and y_i in [low, high]
integers, because the per-respondent marginal cost of each additional
unit step is strictly increasing (quadratic cost is convex) -- a textbook
separable-convex-resource-allocation argument; the test suite additionally
brute-force-verifies this against small fixtures.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import survey_content as sc

#: Frozen tie-break seed (see project_bounded_integer's tie_break_context
#: parameter). Fixed once here; never re-derived from data, demographics,
#: or target human outcomes. Changing this string would change which
#: donor is picked among exactly-tied projection candidates but nothing
#: about the projection objective, constraints, or targets.
FROZEN_PROJECTION_TIE_BREAK_SEED = "s2_target_projection_tie_break_v1"


def _tie_break_key(condition: str, raw_item_name: str, donor_id: str) -> str:
    """Stable SHA-256 hash over seed|condition|raw_item_name|donor_id, used
    only to break exact ties in marginal projection cost -- never to select
    which respondents are adjusted in the first place."""
    payload = f"{FROZEN_PROJECTION_TIE_BREAK_SEED}|{condition}|{raw_item_name}|{donor_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Core bounded-integer / binary projection primitive
# ---------------------------------------------------------------------------


def build_donor_map(records: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Build a {donor_id: value} map from a list of (donor_id, value) pairs
    (e.g. raw retrieved-batch rows), failing closed on any duplicate donor
    id rather than silently keeping the last one the way dict(records)
    would. Every *_control/*_treat mapping fed into this module should be
    built through this function whenever the source is a flat record list."""
    seen: dict[str, float] = {}
    for donor_id, value in records:
        if donor_id in seen:
            raise ValueError(f"duplicate donor id: {donor_id!r}")
        seen[donor_id] = value
    return seen


def project_bounded_integer(
    ideal_values: Mapping[str, float],
    *,
    low: int,
    high: int,
    target_total: float | None = None,
    tie_break_context: tuple[str, str] | None = None,
) -> dict:
    """ideal_values: {donor_id: ideal shifted value v_i}. Support is the
    consecutive integer range [low, high] inclusive (binary is low=0,
    high=1). target_total defaults to sum(ideal_values.values()) -- the
    nearest-attainable aggregate is derived from it, never from a separate
    "requested mean" the caller must pre-round.

    tie_break_context: optional (condition, raw_item_name) pair. When
    given, ties among candidates with the exact same marginal cost (the
    ordinary case for this benchmark, since the common shift c_aj is
    identical for every respondent in a cell) are broken by the frozen
    SHA-256 hash in _tie_break_key(*tie_break_context, donor_id) rather
    than by ascending donor_id. Every other aspect of the objective,
    constraints, and target total is unaffected by this parameter -- it
    only reorders which already-tied candidate is picked first. When
    omitted (the default), ties are broken by ascending donor_id exactly
    as before, for backward compatibility with any caller that does not
    supply one.

    Returns {achieved: {id: int}, requested_mean, achieved_mean, mean_error,
    n_changed_from_base, lower_bound_count, upper_bound_count,
    target_total_used}. "changed" diagnostics that compare against the
    original native response are the caller's responsibility (this function
    only knows about v_i and the achieved integer, not the pre-shift native
    value).
    """
    if high < low:
        raise ValueError(f"invalid support: high ({high}) < low ({low})")
    ids = list(ideal_values.keys())
    n = len(ids)
    if n == 0:
        raise ValueError("ideal_values must be non-empty")
    # duplicate ids cannot occur here: ideal_values is a Mapping, whose keys
    # are unique by construction -- use build_donor_map() when constructing
    # from a flat (id, value) record list to fail closed on true duplicates.

    values = {i: float(ideal_values[i]) for i in ids}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("ideal_values must all be finite")

    # base rounding: nearest support level, ties go to the lower integer
    base: dict[str, int] = {}
    for i in ids:
        v = values[i]
        if v <= low:
            base[i] = low
        elif v >= high:
            base[i] = high
        else:
            floor_v = math.floor(v)
            frac = v - floor_v
            base[i] = int(floor_v) + 1 if frac > 0.5 else int(floor_v)

    base_total = sum(base.values())
    min_total = n * low
    max_total = n * high

    requested_total = sum(values.values()) if target_total is None else float(target_total)
    if not math.isfinite(requested_total):
        raise ValueError("target_total must be finite")
    t_star = round(requested_total)
    t_star = max(min_total, min(max_total, t_star))

    if tie_break_context is not None:
        condition, raw_item_name = tie_break_context
        tie_key = {i: _tie_break_key(condition, raw_item_name, i) for i in ids}
    else:
        tie_key = {i: i for i in ids}

    achieved = dict(base)
    residual = {i: values[i] - base[i] for i in ids}  # v_i - base_i
    diff = t_star - base_total

    if diff > 0:
        candidates = [i for i in ids if achieved[i] < high]
        candidates.sort(key=lambda i: (-residual[i], tie_key[i]))
        if len(candidates) < diff:
            raise ValueError("target total not attainable: insufficient headroom under the support upper bound")
        for i in candidates[:diff]:
            achieved[i] += 1
    elif diff < 0:
        need = -diff
        candidates = [i for i in ids if achieved[i] > low]
        candidates.sort(key=lambda i: (residual[i], tie_key[i]))
        if len(candidates) < need:
            raise ValueError("target total not attainable: insufficient headroom under the support lower bound")
        for i in candidates[:need]:
            achieved[i] -= 1

    achieved_total = sum(achieved.values())
    n_changed_from_base = sum(1 for i in ids if achieved[i] != base[i])

    return {
        "achieved": achieved,
        "requested_mean": requested_total / n,
        "achieved_mean": achieved_total / n,
        "mean_error": achieved_total / n - requested_total / n,
        "target_total_used": t_star,
        "n_changed_from_base_rounding": n_changed_from_base,
        "lower_bound_count": sum(1 for v in achieved.values() if v == low),
        "upper_bound_count": sum(1 for v in achieved.values() if v == high),
    }


def project_binary_k(ideal_values: Mapping[str, float], *, target_total: float | None = None) -> dict:
    """Binary {0,1} outcomes: identical engine as project_bounded_integer
    with low=0, high=1 -- assigning K=achieved_mean*N ones to the K highest
    ideal values is exactly what the shared greedy adjustment reduces to for
    a 2-level support (see module docstring)."""
    result = project_bounded_integer(ideal_values, low=0, high=1, target_total=target_total)
    result["k_achieved"] = result["target_total_used"]
    return result


def project_finite_discrete(ideal_values: Mapping[str, float], *, support_levels: Sequence[float], target_total: float | None = None) -> dict:
    """Non-consecutive finite discrete support (e.g. {0, 0.5, 10}).

    NOT implemented as a general solver: exact achievability of an arbitrary
    target total under a non-consecutive support is a combinatorial
    (subset-sum-like) problem that is not guaranteed solvable by simple
    greedy adjustment, and no outcome in this benchmark currently has such a
    support (every one of the 44 raw items has a consecutive-integer or
    binary support -- verified against survey_content.py / inference/prompts.py
    before writing this module). Implementing a speculative general solver
    for a case that does not exist would itself be an undisclosed scientific
    choice made without seeing what such a case would actually require.
    Fails closed per Section 4/9 of the freeze specification.
    """
    raise NotImplementedError(
        "finite_discrete non-consecutive support is not implemented -- STOP: no outcome in the current "
        "benchmark metadata (survey_content.OUTCOME_COMPOSITES / RAW_ITEM_SCALE_BOUNDS) has a non-consecutive "
        "finite discrete support; implementing a general solver now would require a new scientific choice "
        "made without a real case to validate against"
    )


# ---------------------------------------------------------------------------
# Cell-level common-shift construction + identity checks (Section 1)
# ---------------------------------------------------------------------------


def _mean(d: Mapping[str, float]) -> float:
    return sum(d.values()) / len(d)


def compute_common_shift(control: Mapping[str, float], treat: Mapping[str, float], tau_hat_aj: float) -> dict:
    """control/treat: {donor_id: native value}, same donor id set (paired).
    Returns tau_G_aj (native G ATE), c_aj (common shift), and v (ideal
    shifted treatment values, one per donor)."""
    if set(control.keys()) != set(treat.keys()):
        raise ValueError("control and treat must share exactly the same donor id set (missing control/treatment pair)")
    if not math.isfinite(tau_hat_aj):
        raise ValueError("tau_hat_aj must be finite")
    tau_g_aj = _mean(treat) - _mean(control)
    c_aj = tau_hat_aj - tau_g_aj
    v = {i: treat[i] + c_aj for i in treat}
    return {"tau_g_aj": tau_g_aj, "c_aj": c_aj, "v": v}


def verify_common_shift_identities(control: Mapping[str, float], treat: Mapping[str, float], tau_hat_aj: float, shift: dict, *, tol: float = 1e-9) -> None:
    """Raises AssertionError if either Section-1 identity fails."""
    v = shift["v"]
    tau_g_aj = shift["tau_g_aj"]
    mean_v_minus_control = _mean({i: v[i] - control[i] for i in v})
    if abs(mean_v_minus_control - tau_hat_aj) > tol:
        raise AssertionError(f"common-shift mean identity violated: {mean_v_minus_control} != {tau_hat_aj}")
    for i in v:
        lhs = (v[i] - control[i]) - tau_hat_aj
        rhs = (treat[i] - control[i]) - tau_g_aj
        if abs(lhs - rhs) > tol:
            raise AssertionError(f"centered-HTE identity violated for donor {i!r}: {lhs} != {rhs}")


# ---------------------------------------------------------------------------
# Single-item cell projection (Sections 2/3), reusing the shared engine
# ---------------------------------------------------------------------------

SUPPORT_BOUNDED_INTEGER = "bounded_integer"
SUPPORT_BINARY = "binary"
SUPPORT_FINITE_DISCRETE = "finite_discrete"


def project_cell(
    intervention_id: str,
    outcome_id: str,
    control: Mapping[str, float],
    treat: Mapping[str, float],
    tau_hat_aj: float,
    *,
    support_kind: str,
    low: int | None = None,
    high: int | None = None,
    support_levels: Sequence[float] | None = None,
) -> dict:
    """Single-item (non-composite) outcome cell: common-shift + integer
    projection + full diagnostics (Section 10). Controls are returned
    byte/value-identical to the input -- never modified."""
    if len(set(control.keys())) != len(control):
        raise ValueError("duplicate donor id in control")
    shift = compute_common_shift(control, treat, tau_hat_aj)
    verify_common_shift_identities(control, treat, tau_hat_aj, shift)

    if support_kind == SUPPORT_BOUNDED_INTEGER:
        if low is None or high is None:
            raise ValueError("bounded_integer support requires low/high")
        proj = project_bounded_integer(shift["v"], low=low, high=high)
    elif support_kind == SUPPORT_BINARY:
        proj = project_binary_k(shift["v"])
        low, high = 0, 1
    elif support_kind == SUPPORT_FINITE_DISCRETE:
        if support_levels is None:
            raise ValueError("finite_discrete support requires support_levels")
        proj = project_finite_discrete(shift["v"], support_levels=support_levels)
    else:
        raise ValueError(f"unknown support_kind {support_kind!r}")

    achieved = proj["achieved"]
    n = len(achieved)
    achieved_mean = _mean(achieved)
    achieved_ate = achieved_mean - _mean(control)
    n_changed_from_native = sum(1 for i in achieved if achieved[i] != treat[i])

    return {
        "intervention_id": intervention_id,
        "outcome_id": outcome_id,
        "n": n,
        "control": dict(control),  # unchanged, returned for downstream convenience
        "achieved_treat": achieved,
        "native_g_ate": shift["tau_g_aj"],
        "requested_calibrated_ate": tau_hat_aj,
        "ideal_shift_c": shift["c_aj"],
        "preprojection_ate": _mean(shift["v"]) - _mean(control),
        "achieved_postprojection_ate": achieved_ate,
        "projection_ate_error": achieved_ate - tau_hat_aj,
        "n_responses_changed_by_projection": n_changed_from_native,
        "fraction_changed": n_changed_from_native / n,
        "lower_bound_count": proj["lower_bound_count"],
        "upper_bound_count": proj["upper_bound_count"],
    }


# ---------------------------------------------------------------------------
# Composite / multi-item construct handling (Section 5)
# ---------------------------------------------------------------------------


def composite_item_coefficients(outcome: str) -> tuple[float, list[str], list[float]]:
    """Returns (intercept, item_labels, coefficients) for outcome's linear
    recombination rule, read directly from the frozen
    survey_content.OUTCOME_COMPOSITES -- no new scoring rule invented here."""
    kind, spec = sc.OUTCOME_COMPOSITES[outcome]
    if kind == "item":
        return 0.0, [spec], [1.0]
    if kind == "mean":
        k = len(spec)
        return 0.0, list(spec), [1.0 / k for _ in spec]
    if kind == "reverse_100":
        return 100.0, [spec], [-1.0]
    raise ValueError(f"unknown OUTCOME_COMPOSITES kind {kind!r} for outcome {outcome!r}")


def recombine_composite(outcome: str, item_values: Mapping[str, float]) -> float:
    intercept, labels, coefs = composite_item_coefficients(outcome)
    return intercept + sum(c * item_values[label] for label, c in zip(labels, coefs))


def _per_item_ideal_shift(outcome: str, c_aj: float) -> dict[str, float]:
    """Equal-contribution rule: each constituent item's coefficient*(shift)
    contributes an equal 1/K share of the composite-level required shift
    c_aj. For this benchmark's actual composite forms (uniform coef=1/K per
    item for "mean", or a single item with coef=+1/-1), this simplifies to:
    mean/item composites -> shift each item by +c_aj; reverse_100 -> shift
    the one raw item by -c_aj (so the recomputed composite still moves by
    +c_aj). This is a deterministic, symmetric, no-favoritism default with
    no free parameter -- not a per-item weighting choice."""
    intercept, labels, coefs = composite_item_coefficients(outcome)
    k = len(labels)
    return {label: (c_aj / k) / coef for label, coef in zip(labels, coefs)}


def project_composite_cell(
    intervention_id: str,
    outcome: str,
    item_control: Mapping[str, Mapping[str, float]],
    item_treat: Mapping[str, Mapping[str, float]],
    tau_hat_aj: float,
    item_bounds: Mapping[str, tuple[int, int]],
) -> dict:
    """item_control/item_treat: {item_label: {donor_id: native value}}.
    Applies the module-level equal-contribution shift to every constituent
    item, projects each item independently onto its own integer support via
    the shared engine, then mechanically recomputes the composite from the
    frozen OUTCOME_COMPOSITES rule. Controls are never touched."""
    intercept, labels, coefs = composite_item_coefficients(outcome)
    missing = set(labels) - set(item_control.keys()) | set(labels) - set(item_treat.keys()) | set(labels) - set(item_bounds.keys())
    if missing:
        raise ValueError(f"missing constituent-item metadata for outcome {outcome!r}: {sorted(missing)}")

    donor_ids = set(next(iter(item_control.values())).keys())
    for label in labels:
        if set(item_control[label].keys()) != donor_ids or set(item_treat[label].keys()) != donor_ids:
            raise ValueError(f"inconsistent donor id set across constituent items of {outcome!r}")

    control_composite = {i: recombine_composite(outcome, {label: item_control[label][i] for label in labels}) for i in donor_ids}
    treat_composite = {i: recombine_composite(outcome, {label: item_treat[label][i] for label in labels}) for i in donor_ids}
    shift = compute_common_shift(control_composite, treat_composite, tau_hat_aj)
    verify_common_shift_identities(control_composite, treat_composite, tau_hat_aj, shift)
    c_aj = shift["c_aj"]

    per_item_shift = _per_item_ideal_shift(outcome, c_aj)
    item_diagnostics = {}
    projected_items: dict[str, dict[str, float]] = {}
    for label in labels:
        low, high = item_bounds[label]
        ideal = {i: item_treat[label][i] + per_item_shift[label] for i in donor_ids}
        proj = project_bounded_integer(ideal, low=low, high=high, tie_break_context=(intervention_id, label))
        projected_items[label] = proj["achieved"]
        item_diagnostics[label] = {
            "n_changed_from_native": sum(1 for i in donor_ids if proj["achieved"][i] != item_treat[label][i]),
            "lower_bound_count": proj["lower_bound_count"],
            "upper_bound_count": proj["upper_bound_count"],
        }

    achieved_composite = {i: recombine_composite(outcome, {label: projected_items[label][i] for label in labels}) for i in donor_ids}
    achieved_mean = _mean(achieved_composite)
    achieved_ate = achieved_mean - _mean(control_composite)
    n_changed_composite = sum(1 for i in donor_ids if achieved_composite[i] != treat_composite[i])
    n = len(donor_ids)

    return {
        "intervention_id": intervention_id,
        "outcome_id": outcome,
        "n": n,
        "control": dict(control_composite),
        "achieved_treat": achieved_composite,
        "projected_items": projected_items,
        "native_g_ate": shift["tau_g_aj"],
        "requested_calibrated_ate": tau_hat_aj,
        "ideal_shift_c": c_aj,
        "preprojection_ate": _mean(shift["v"]) - _mean(control_composite),
        "achieved_postprojection_ate": achieved_ate,
        "projection_ate_error": achieved_ate - tau_hat_aj,
        "n_responses_changed_by_projection": n_changed_composite,
        "fraction_changed": n_changed_composite / n,
        "constituent_item_diagnostics": item_diagnostics,
    }


# ---------------------------------------------------------------------------
# Shared raw item joint-consistency audit (Section 6) -- not exercised by
# the real OUTCOME_COMPOSITES today (verified zero sharing), implemented and
# tested generically in case metadata ever changes.
# ---------------------------------------------------------------------------


def audit_shared_raw_items(outcome_composites: Mapping[str, tuple[str, object]]) -> dict[str, list[str]]:
    item_to_outcomes: dict[str, list[str]] = {}
    for outcome, (kind, spec) in outcome_composites.items():
        labels = spec if kind == "mean" else [spec]
        for label in labels:
            item_to_outcomes.setdefault(label, []).append(outcome)
    return {item: outcomes for item, outcomes in item_to_outcomes.items() if len(outcomes) > 1}


def assert_no_conflicting_shared_item_values(projected_items_by_outcome: Mapping[str, Mapping[str, Mapping[str, float]]], shared: Mapping[str, list[str]]) -> None:
    """projected_items_by_outcome: {outcome: {item_label: {donor_id: value}}}.
    If a raw item is shared across outcomes being jointly processed, every
    outcome's projection must agree on that item's final value for every
    donor; otherwise this is an unresolved joint constraint and must STOP."""
    for item, outcomes in shared.items():
        outcomes_with_item = [o for o in outcomes if o in projected_items_by_outcome and item in projected_items_by_outcome[o]]
        if len(outcomes_with_item) < 2:
            continue
        reference_outcome = outcomes_with_item[0]
        reference_values = projected_items_by_outcome[reference_outcome][item]
        for other in outcomes_with_item[1:]:
            other_values = projected_items_by_outcome[other][item]
            if reference_values != other_values:
                raise ValueError(
                    f"STOP: unresolved joint constraint -- raw item {item!r} is shared by outcomes "
                    f"{reference_outcome!r} and {other!r} but their independently projected values disagree; "
                    "existing benchmark metadata does not specify how to reconcile this"
                )


REAL_BENCHMARK_SHARED_RAW_ITEMS = audit_shared_raw_items(sc.OUTCOME_COMPOSITES)


# ---------------------------------------------------------------------------
# Batch orchestration across cells (Sections 6/9/11): fails closed on
# duplicate (intervention, outcome) cells and, if any raw item is shared
# across outcomes in the batch, enforces the joint-consistency check.
# ---------------------------------------------------------------------------


def project_target_ate_table(cell_specs: Sequence[Mapping], *, outcome_composites: Mapping[str, tuple[str, object]] = sc.OUTCOME_COMPOSITES) -> list[dict]:
    """cell_specs: list of dicts, each either a single-item cell (keys:
    intervention_id, outcome_id, control, treat, tau_hat_aj, support_kind,
    low/high/support_levels) or a composite cell (keys: intervention_id,
    outcome, item_control, item_treat, tau_hat_aj, item_bounds). Has no
    knowledge of, and takes no parameter identifying, which calibration
    method produced tau_hat_aj -- Primary M2, Secondary-1 MCONST, and
    Secondary-2 MCONST_GSHAPE all flow through this exact same function."""
    seen = set()
    for spec in cell_specs:
        key = (spec["intervention_id"], spec.get("outcome_id", spec.get("outcome")))
        if key in seen:
            raise ValueError(f"duplicate intervention/outcome cell: {key}")
        seen.add(key)

    results = []
    projected_items_by_outcome: dict[str, dict[str, dict[str, float]]] = {}
    for spec in cell_specs:
        if "outcome" in spec:
            r = project_composite_cell(spec["intervention_id"], spec["outcome"], spec["item_control"], spec["item_treat"], spec["tau_hat_aj"], spec["item_bounds"])
            projected_items_by_outcome.setdefault(spec["outcome"], {})
            for label, values in r["projected_items"].items():
                projected_items_by_outcome[spec["outcome"]][label] = values
        else:
            r = project_cell(
                spec["intervention_id"],
                spec["outcome_id"],
                spec["control"],
                spec["treat"],
                spec["tau_hat_aj"],
                support_kind=spec["support_kind"],
                low=spec.get("low"),
                high=spec.get("high"),
                support_levels=spec.get("support_levels"),
            )
        results.append(r)

    shared = audit_shared_raw_items(outcome_composites)
    if shared and projected_items_by_outcome:
        assert_no_conflicting_shared_item_values(projected_items_by_outcome, shared)

    return results
