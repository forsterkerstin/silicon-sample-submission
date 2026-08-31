#!/usr/bin/env python3
"""scripts/build_advocacy_validation.py

A genuine held-out validation of our elicitation pipeline's predicted
treatment effects, using the climate-advocacy megastudy's own real
treatment arms (data/climate_advocacy_megastudy/data/advocacy_data.csv has
`cond`/`condName` columns for 17 real interventions + control -- previously
only its control rows were used, for baseline_references.json's moment
calibration). Unlike that baseline-realism check, this one is NOT circular:
it compares our model's own predicted effect (elicited using the real
intervention text, never seen by the calibration step) against a REAL human
effect computed from a respondent split DISJOINT from the one
build_baseline_reference.py used -- see advocacy_split.py. Same idea as
scripts/build_ate_archive.py's 70-study archive, but domain-matched
(climate advocacy, not a mixed TESS bag) and built from data we already had.

Real per-outcome caveats, not glossed over:
  - Only 4/13 benchmark outcomes have a same-scale item in this megastudy
    (belief_post, policy_general, donation_ams, newsletter_signup -- see
    build_baseline_reference.py's OUTCOME_TO_COLUMN).
  - Only 10/17 conditions have a fully reconstructable real stimulus text
    (scripts/build_advocacy_stimuli.py excluded 7 -- piped-field or
    video-dependent; see data/advocacy_intervention_stimuli.json). The other
    7 conditions still get a real human_ate (no model-side data needed for
    that), just no model_ate to compare it against.
  - The elicitation reuses this benchmark's OWN item wording for
    belief_post/policy_general/donation_ams/newsletter_signup (same
    same-scale correspondence baseline_calibration.py already documents),
    not advocacy_data.csv's exact Qualtrics wording -- a real, disclosed
    wording mismatch, not a hidden one.

model_ate_pp here is RAW (uncalibrated) -- this script measures what the
native-response pipeline actually produces before lambda_ate calibration,
the same quantity ate/calibrate_lambda.py's external archive-based lambda_ate
is meant to correct.

Writes data/validation_advocacy_ate.csv:
  condition, outcome, model_ate_pp (blank if condition unusable),
  human_ate_pp, n_human_control, n_human_treatment, usable, exclusion_reason

Usage:
    python scripts/build_advocacy_validation.py --profiles 10 --backend hf
    CUDA_VISIBLE_DEVICES=0 python scripts/build_advocacy_validation.py --backend vllm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

import survey_content as sc  # noqa: E402
from advocacy_split import VALID, add_split_column  # noqa: E402
from ate.normalize_effects import OUTCOME_SCALE_BOUNDS, RAW_ITEM_SCALE_BOUNDS, to_unit_scale  # noqa: E402
from build_baseline_reference import DATA_PATH as ADVOCACY_DATA_PATH  # noqa: E402
from build_baseline_reference import OUTCOME_TO_COLUMN  # noqa: E402
from inference.client import make_native_client  # noqa: E402
from inference.prompts import ROSTER_PROFILE_COLUMNS  # noqa: E402
from inference.simulate_response import simulate_response  # noqa: E402

STIMULI_PATH = PIPELINE_ROOT / "data" / "advocacy_intervention_stimuli.json"
OUT_PATH = PIPELINE_ROOT / "data" / "validation_advocacy_ate.csv"


def load_human_ates(valid_df: pd.DataFrame) -> list[dict]:
    """Real human ATE (pp-scale) per (condition, outcome), computed only
    from the "valid" respondent split -- disjoint from the "calib" split
    build_baseline_reference.py used, so this is never comparing our
    model against a baseline it was itself anchored to. See advocacy_split.py.
    """
    control = valid_df[valid_df["condName"] == "Control"]
    rows = []
    for outcome, column in OUTCOME_TO_COLUMN.items():
        low, high = OUTCOME_SCALE_BOUNDS[outcome]
        control_series = control[column].dropna()
        control_unit = to_unit_scale(control_series.mean(), low, high)
        for condition, group in valid_df[valid_df["condName"] != "Control"].groupby("condName"):
            series = group[column].dropna()
            if len(series) == 0:
                continue
            treat_unit = to_unit_scale(series.mean(), low, high)
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "human_ate_pp": 100 * (treat_unit - control_unit),
                    "n_human_control": len(control_series),
                    "n_human_treatment": len(series),
                }
            )
    return rows


def elicit_model_ates(usable_stimuli: dict[str, dict], client, n_profiles: int, seed: int) -> list[dict]:
    """Model-side ATE (pp-scale, RAW/uncalibrated), elicited using the REAL
    cleaned stimulus text for each usable condition, on a sample of
    `n_profiles` real demographic profiles from our own roster -- the same
    native-response call (inference.simulate_response) the primary Phase-A
    pipeline uses, so this is a genuine test of the same predictions the
    submission relies on, not a separately-tuned mock. One call per
    (profile, condition) answers all 4 mapped outcomes together.
    """
    roster_candidates = sorted((PIPELINE_ROOT / "data/processed/population").glob("simulation_roster_*.csv"))
    if not roster_candidates:
        raise SystemExit("no data/processed/population/simulation_roster_*.csv found -- run scripts/build_population.py first")
    roster = pd.read_csv(roster_candidates[-1])
    profiles = roster.drop_duplicates("latent_profile_id").sample(n=n_profiles, random_state=seed)

    outcome_items = [it for it in sc.load_items() if it["target_label"] in OUTCOME_TO_COLUMN]

    pp_diffs: dict[tuple[str, str], list[float]] = {(cond, item["target_label"]): [] for cond in usable_stimuli for item in outcome_items}
    for _, row in profiles.iterrows():
        profile = {col: row[col] for col in ROSTER_PROFILE_COLUMNS}
        control_response = simulate_response(profile, "", outcome_items, client)

        for condition, stim in usable_stimuli.items():
            treat_response = simulate_response(profile, stim["stimulus_text"], outcome_items, client)
            for item in outcome_items:
                label = item["target_label"]
                low, high = RAW_ITEM_SCALE_BOUNDS[item["scale"]]
                p0_unit = to_unit_scale(control_response[label], low, high)
                pt_unit = to_unit_scale(treat_response[label], low, high)
                pp_diffs[(condition, label)].append(100 * (pt_unit - p0_unit))

    rows = []
    for (condition, outcome_label), diffs in pp_diffs.items():
        rows.append({"condition": condition, "outcome": outcome_label, "model_ate_pp": sum(diffs) / len(diffs)})
        print(f"  {outcome_label} / {condition}: model_ate_pp={rows[-1]['model_ate_pp']:.2f} (n_profiles={len(diffs)})")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--profiles", type=int, default=10, help="how many roster profiles to average model_ate over (default 10)")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    df = pd.read_csv(ADVOCACY_DATA_PATH, usecols=["ResponseId", "condName", *OUTCOME_TO_COLUMN.values()], low_memory=False)
    df = add_split_column(df)
    valid_df = df[df["_split"] == VALID]
    print(f"loaded {len(df)} respondents; {len(valid_df)} in the held-out validation half (disjoint from build_baseline_reference.py's calibration half)")

    human_rows = load_human_ates(valid_df)
    human_by_key = {(r["condition"], r["outcome"]): r for r in human_rows}

    stimuli = json.loads(STIMULI_PATH.read_text(encoding="utf-8"))
    usable_stimuli = {cond: v for cond, v in stimuli.items() if v["usable"]}
    print(f"{len(usable_stimuli)}/{len(stimuli)} conditions have a usable real stimulus text (see data/advocacy_intervention_stimuli.json)")

    print(f"\nloading native-response client backend={args.backend} ...")
    client = make_native_client(args.backend, model_name=args.model_name)
    print("eliciting model_ate for usable conditions...")
    model_rows = elicit_model_ates(usable_stimuli, client, args.profiles, args.seed)
    model_by_key = {(r["condition"], r["outcome"]): r["model_ate_pp"] for r in model_rows}

    out_rows = []
    for cond, meta in stimuli.items():
        for outcome in OUTCOME_TO_COLUMN:
            human = human_by_key.get((cond, outcome))
            if human is None:
                continue
            out_rows.append(
                {
                    "condition": cond,
                    "outcome": outcome,
                    "model_ate_pp": model_by_key.get((cond, outcome)),
                    "human_ate_pp": human["human_ate_pp"],
                    "n_human_control": human["n_human_control"],
                    "n_human_treatment": human["n_human_treatment"],
                    "usable": meta["usable"],
                    "exclusion_reason": meta["exclusion_reason"],
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(out_df)} rows to {OUT_PATH}")

    compared = out_df.dropna(subset=["model_ate_pp"])
    if len(compared) >= 2:
        errors = compared["model_ate_pp"] - compared["human_ate_pp"]
        rmse = (errors**2).mean() ** 0.5
        bias = errors.mean()
        corr = compared["model_ate_pp"].corr(compared["human_ate_pp"])
        print(f"\nheld-out comparison over {len(compared)} (condition, outcome) pairs with a real model_ate:")
        print(f"  RMSE={rmse:.2f}pp  bias={bias:.2f}pp  corr={corr:.3f}")
    else:
        print("\nfewer than 2 pairs with a real model_ate -- skipping RMSE/correlation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
