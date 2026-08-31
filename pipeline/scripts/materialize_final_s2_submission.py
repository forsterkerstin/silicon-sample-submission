"""Run the frozen S2 (MCONST_GSHAPE) external calibration and native-support
projection on the real, first-valid target-G response universe, and
assemble the final 17,000-row Tier-1 submission file.

Reads outputs/target_production/final_first_valid_native_g_responses.csv
(built by scripts/assemble_final_native_g_responses.py from real,
already-retrieved production responses only). Builds the 208
(intervention, outcome) cell_specs required by
ate.s2_final_materializer.materialize_s2_target_predictions -- the sole
final-submission code path per outputs/validation/
frozen_final_submission_manifest_s2.json -- and reassembles its per-cell
results into the official predictions/ CSV shape.

Every outcome (including the four single-raw-item outcomes and the one
reverse-scored outcome) is routed through project_composite_cell with a
one-item item_control/item_treat/item_bounds, so a single code path
handles all 13 outcomes uniformly; ate.target_projection.
composite_item_coefficients (frozen, unmodified) already handles "item"/
"mean"/"reverse_100" composite kinds without special-casing here.

Control responses are carried through completely unmodified (never
touched by calibration or projection). No target human outcome is read,
inferred, or referenced anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts", PIPELINE_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import survey_content as sc  # noqa: E402
from ate.normalize_effects import OUTCOME_SCALE_BOUNDS, RAW_ITEM_SCALE_BOUNDS  # noqa: E402
from ate.s2_final_materializer import materialize_s2_target_predictions  # noqa: E402
from ate.target_projection import composite_item_coefficients, recombine_composite  # noqa: E402
from inference.together_batch import G_MASTER_PATH  # noqa: E402
from population.pums import recode_age_band  # noqa: E402

#: g_personas_master.csv's own "age"/"age_band" columns are CES-2024-based
#: (src.population.ces.CES_SURVEY_YEAR = 2024: age = 2024 - birthyr) and are
#: preserved completely unmodified here -- they are the historical record of
#: what Gemma's rendered prompts actually contained (see inference.prompts.
#: PROFILE_FIELD_ORDER, which renders "age", never "age_band", into the
#: persona description) and must not be changed to avoid misrepresenting
#: what the frozen production inference actually saw. The benchmark's own
#: codebook.csv requires the *submitted* age_band to be cut from
#: age = 2026 - year_birth. Since master["age"] == 2024 - year_birth exactly,
#: year_birth == 2024 - master["age"], so the benchmark-submission age is
#: master["age"] + (2026 - 2024) = master["age"] + 2 -- a pure arithmetic
#: correction requiring no new data source and no change to any
#: historically-frozen persona or prompt input.
CES_PERSONA_CONSTRUCTION_YEAR = 2024
BENCHMARK_SUBMISSION_YEAR = 2026


def _benchmark_age_band(historical_ces_age: int) -> str:
    benchmark_age = int(historical_ces_age) + (BENCHMARK_SUBMISSION_YEAR - CES_PERSONA_CONSTRUCTION_YEAR)
    return recode_age_band(benchmark_age)

NATIVE_CSV = PIPELINE_ROOT / "outputs" / "target_production" / "final_first_valid_native_g_responses.csv"
OUT_DIR = REPO_ROOT / "predictions"
OUT_CSV = OUT_DIR / "team_10_T1_primary_v1.csv"
OUT_DIAGNOSTICS = PIPELINE_ROOT / "outputs" / "target_production" / "final_s2_calibration_diagnostics.csv"
OUT_MANIFEST = PIPELINE_ROOT / "outputs" / "target_production" / "final_s2_submission_manifest.json"

MODERATOR_COLS = ["gender", "age_band", "race", "education", "income", "party"]
TIER1_REQUIRED_ORDER = [
    "profile_id", "condition", *MODERATOR_COLS,
    "trust_multidimensional",
    "trust_competence_1", "trust_competence_2", "trust_competence_3",
    "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
    "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
    "trust_openness_1", "trust_openness_2", "trust_openness_3",
    "trust_post", "distrust_post", "funding_perceptions",
    "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general",
    "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item_bounds() -> dict[str, tuple[int, int]]:
    bounds = {}
    for item in sc.load_items():
        bounds[item["target_label"]] = RAW_ITEM_SCALE_BOUNDS[item["scale"]]
    return bounds


def build_cell_specs(native: pd.DataFrame) -> list[dict]:
    conditions = sorted(native["condition_id"].unique())
    interventions = [c for c in conditions if c != "control"]
    if len(interventions) != 16:
        raise RuntimeError(f"expected exactly 16 interventions, found {len(interventions)}: {interventions}")

    control = native[native["condition_id"] == "control"].set_index("profile_id")
    bounds = _item_bounds()

    cells = []
    for intervention in interventions:
        treat = native[native["condition_id"] == intervention].set_index("profile_id")
        if set(treat.index) != set(control.index):
            raise RuntimeError(f"respondent mismatch between control and {intervention!r}")
        for outcome in sc.OUTCOME_COMPOSITES:
            _, labels, _ = composite_item_coefficients(outcome)
            item_control = {label: control[label].to_dict() for label in labels}
            item_treat = {label: treat[label].to_dict() for label in labels}
            item_bounds = {label: bounds[label] for label in labels}
            control_composite = {i: recombine_composite(outcome, {label: item_control[label][i] for label in labels}) for i in control.index}
            treat_composite = {i: recombine_composite(outcome, {label: item_treat[label][i] for label in labels}) for i in treat.index}
            tau_g_native = sum(treat_composite.values()) / len(treat_composite) - sum(control_composite.values()) / len(control_composite)
            low, high = OUTCOME_SCALE_BOUNDS[outcome]
            cells.append(
                {
                    "intervention_id": intervention,
                    "outcome": outcome,
                    "R_j": high - low,
                    "tau_g_native": tau_g_native,
                    "item_control": item_control,
                    "item_treat": item_treat,
                    "item_bounds": item_bounds,
                }
            )
    if len(cells) != 208:
        raise RuntimeError(f"expected exactly 208 cell_specs, built {len(cells)}")
    return cells


def assemble_final_rows(native: pd.DataFrame, results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(G_MASTER_PATH).set_index("donor_key")
    control = native[native["condition_id"] == "control"].set_index("profile_id")
    item_cols = [c for c in native.columns if c not in ("profile_id", "condition_id")]

    by_intervention: dict[str, dict[str, dict[str, float]]] = {}
    diagnostics_rows = []
    for r in results:
        by_intervention.setdefault(r["intervention_id"], {})
        by_intervention[r["intervention_id"]].update(r["projected_items"])
        diagnostics_rows.append(
            {
                "condition": r["intervention_id"],
                "outcome": r["outcome_id"],
                "n": r["n"],
                "native_g_ate": r["native_g_ate"],
                "target_ate": r["requested_calibrated_ate"],
                "preprojection_ate": r["preprojection_ate"],
                "achieved_ate": r["achieved_postprojection_ate"],
                "absolute_error": abs(r["achieved_postprojection_ate"] - r["requested_calibrated_ate"]),
                "n_responses_changed_by_projection": r["n_responses_changed_by_projection"],
                "fraction_changed": r["fraction_changed"],
            }
        )

    rows = []

    def _row_for(donor_key: str, condition: str, items: dict[str, float]) -> dict:
        outcomes = sc.compute_outcomes(items)
        m = master.loc[donor_key]
        row = {"donor_key": donor_key, "condition": condition}
        for col in MODERATOR_COLS:
            row[col] = _benchmark_age_band(m["age"]) if col == "age_band" else m[col]
        row.update(items)
        row.update(outcomes)
        return row

    for donor_key in control.index:
        items = {label: control.loc[donor_key, label] for label in item_cols}
        rows.append(_row_for(donor_key, "control", items))

    for intervention, item_map in by_intervention.items():
        by_donor: dict[str, dict[str, float]] = {}
        for label, donor_values in item_map.items():
            for donor_key, value in donor_values.items():
                by_donor.setdefault(donor_key, {})[label] = value
        for donor_key, items in by_donor.items():
            missing = set(item_cols) - set(items)
            if missing:
                raise RuntimeError(f"donor {donor_key} intervention {intervention} missing projected item(s): {sorted(missing)}")
            rows.append(_row_for(donor_key, intervention, items))

    df = pd.DataFrame(rows)
    df = df.sort_values(["condition", "donor_key"]).reset_index(drop=True)
    df.insert(0, "profile_id", [f"p{i + 1:05d}" for i in range(len(df))])
    diagnostics = pd.DataFrame(diagnostics_rows)
    return df, diagnostics


def run_integrity_checks(df: pd.DataFrame, native: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    problems = []

    if len(df) != 17000:
        problems.append(f"expected 17000 rows, got {len(df)}")

    counts = df["condition"].value_counts()
    if len(counts) != 17 or not (counts == 1000).all():
        problems.append(f"expected 1000 rows in each of 17 conditions, got {counts.to_dict()}")

    for col in TIER1_REQUIRED_ORDER:
        if col not in df.columns:
            problems.append(f"missing required column: {col}")

    if df[TIER1_REQUIRED_ORDER].isna().any().any():
        problems.append("required columns contain missing value(s)")

    dup = df.duplicated(subset=["donor_key", "condition"])
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate (donor_key, condition) identity pair(s)")
    if df["profile_id"].duplicated().any():
        problems.append("duplicate profile_id value(s) in official submission column")

    for item in sc.load_items():
        label = item["target_label"]
        values = df[label]
        if not values.map(lambda x: float(x).is_integer()).all():
            problems.append(f"{label} contains non-integer raw response(s)")
        low, high = RAW_ITEM_SCALE_BOUNDS[item["scale"]]
        if not values.between(low, high).all():
            problems.append(f"{label} has value(s) outside [{low}, {high}]")

    max_composite_error = 0.0
    for _, row in df.iterrows():
        expected = sc.compute_outcomes(row.to_dict())
        for outcome, value in expected.items():
            err = abs(float(row[outcome]) - float(value))
            max_composite_error = max(max_composite_error, err)
            if err > 1e-6:
                problems.append(f"row {row['profile_id']} {outcome} differs from recomputed composite by {err}")

    from ate.estimate_ates import estimate_raw_ates

    ates = estimate_raw_ates(df, list(sc.OUTCOME_COMPOSITES.keys()))
    if len(ates) != 208:
        problems.append(f"expected 208 condition-outcome ATE cells, got {len(ates)}")
    if not ates[["raw_ate", "control_mean", "treatment_mean"]].notna().all().all():
        problems.append("one or more ATE means is missing")

    control_native = native[native["condition_id"] == "control"].set_index("profile_id")
    control_final = df[df["condition"] == "control"].set_index("donor_key")
    item_cols = [c for c in native.columns if c not in ("profile_id", "condition_id")]
    for label in item_cols:
        if not (control_final.loc[control_native.index, label].astype(float) == control_native[label].astype(float)).all():
            problems.append(f"control-arm item {label} was altered by calibration/projection")

    if diagnostics["absolute_error"].isna().any():
        problems.append("calibration diagnostics contain missing absolute_error")
    if len(diagnostics) != 208:
        problems.append(f"expected 208 calibration diagnostic rows, got {len(diagnostics)}")

    smoke_markers = ("smoke", "dev", "test_", "engineering_smoke")
    if any(m in str(v).lower() for v in df["condition"].unique() for m in smoke_markers):
        problems.append("a condition label looks like a smoke/development marker")

    return {
        "ok": not problems,
        "problems": problems,
        "n_rows": len(df),
        "n_conditions": int(df["condition"].nunique()),
        "rows_per_condition": counts.to_dict(),
        "max_composite_error": max_composite_error,
        "n_ate_cells": len(ates),
        "max_calibration_absolute_error": float(diagnostics["absolute_error"].max()),
    }


def main() -> dict:
    native = pd.read_csv(NATIVE_CSV, dtype={"profile_id": str})
    cell_specs = build_cell_specs(native)
    results = materialize_s2_target_predictions(cell_specs)

    df, diagnostics = assemble_final_rows(native, results)
    report = run_integrity_checks(df, native, diagnostics)
    if not report["ok"]:
        raise RuntimeError("Final Tier-1 integrity checks failed:\n- " + "\n- ".join(report["problems"]))

    #: donor_key is this pipeline's own internal persona-linkage identifier
    #: (used above only for joining/sorting) -- never part of the organizer's
    #: Tier-1 schema (scripts/lib/submission_spec.R's tier1_required has no
    #: such column). pipeline/submission/final_tier1.py's own official-column
    #: construction explicitly raises if it leaks into a final submission;
    #: this materializer previously lacked that same guard. Excluded here so
    #: rematerializing from the frozen native responses is reproducible
    #: without a separate manual post-hoc column strip.
    official_cols = TIER1_REQUIRED_ORDER + [c for c in df.columns if c not in TIER1_REQUIRED_ORDER and c != "donor_key"]
    official_df = df[official_cols]
    if "donor_key" in official_df.columns:
        raise RuntimeError("internal donor_key metadata leaked into final submission")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_df.to_csv(OUT_CSV, index=False)
    diagnostics.to_csv(OUT_DIAGNOSTICS, index=False)

    manifest = {
        "final_method": "S2_MCONST_GSHAPE",
        "source_manifest": "outputs/validation/frozen_final_submission_manifest_s2.json",
        "native_input_csv": str(NATIVE_CSV.relative_to(PIPELINE_ROOT)),
        "native_input_sha256": _sha256_file(NATIVE_CSV),
        "predictions_csv": str(OUT_CSV.relative_to(REPO_ROOT)),
        "predictions_sha256": _sha256_file(OUT_CSV),
        "diagnostics_csv": str(OUT_DIAGNOSTICS.relative_to(PIPELINE_ROOT)),
        "diagnostics_sha256": _sha256_file(OUT_DIAGNOSTICS),
        "integrity_report": report,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k != "integrity_report"} | {"integrity_ok": out["integrity_report"]["ok"], "n_rows": out["integrity_report"]["n_rows"]}, indent=2))
