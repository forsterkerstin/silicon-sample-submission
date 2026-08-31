#!/usr/bin/env python3
"""scripts/build_vlasceanu_validation.py

A second, fully independent real-data validation source: Vlasceanu et al.
2024 ("Addressing climate change with behavioral science: A global
intervention tournament in 63 countries"), supplied by the user as
data/data63.xlsx. Restricted to the US sample (n=8,253 -- comfortably large;
see the printed count), which is the closest match to this benchmark's own
target population among the 63 countries.

No leakage risk with the climate-advocacy megastudy: this is a completely
separate study, platform, and respondent pool -- unlike advocacy_data.csv,
nothing here was used anywhere else in this pipeline (no baseline
calibration, no shrinkage fitting), so there is no overlapping-sample
concern to split around (contrast advocacy_split.py, which exists precisely
because advocacy_data.csv *is* reused elsewhere).

What this can and can't do, honestly: data63.xlsx has no materials folder in
this repo -- only response data, no intervention stimulus text -- so unlike
build_advocacy_validation.py there is no way to elicit a matching model_ate
for these specific 11 interventions. What it DOES give us:
  1. A real, large-sample, empirically observed distribution of climate
     belief/policy-support effect sizes from a comparable megastudy -- an
     honest plausibility envelope to check our own predicted effect
     magnitudes against (are they wildly outside what real interventions in
     this domain actually produce?), not a pairwise ground truth.
  2. One genuine conceptual match worth calling out explicitly: Vlasceanu's
     "SciConsens" (scientific-consensus messaging) is the same real-world
     mechanism as this benchmark's own "Consensus" condition -- different
     instrument and wording, so not a strict apples-to-apples test, but a
     meaningful, disclosed spot-check once a real run of this benchmark's
     own pipeline produces a "Consensus" belief-outcome effect to compare
     against.

Writes data/validation_vlasceanu_us.csv:
  condition, outcome, human_ate_pp, n_control, n_treatment
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

import ate.normalize_effects as dc  # noqa: E402

DATA_PATH = PIPELINE_ROOT / "data" / "data63.xlsx"
OUT_PATH = PIPELINE_ROOT / "data" / "validation_vlasceanu_us.csv"

#: both belief and policy items are 0-100 sliders in the original study
#: (verified against the real min/max of the loaded columns, not assumed).
OUTCOME_COLUMNS = {
    "belief_mean": ["Belief1", "Belief2", "Belief3", "Belief4"],
    "policy_mean": ["Policy1", "Policy2", "Policy3", "Policy4", "Policy5", "Policy6", "Policy7", "Policy8", "Policy9"],
}

#: the one condition with a genuine, disclosed conceptual match to this
#: benchmark's own condition set -- see module docstring.
CONCEPTUAL_MATCH = {"SciConsens": "Consensus"}


def main() -> int:
    df = pd.read_excel(DATA_PATH, sheet_name="data4joe (1)")
    usa = df[df["Country"] == "Usa"].copy()
    print(f"loaded {len(df)} respondents across {df['Country'].nunique()} countries; {len(usa)} in the US sample")
    if len(usa) < 500:
        print(f"WARNING: US sample (n={len(usa)}) may be too small for a stable per-condition ATE")

    for cols in OUTCOME_COLUMNS.values():
        observed_min = usa[cols].min().min()
        observed_max = usa[cols].max().max()
        assert 0 <= observed_min and observed_max <= 100, f"unexpected item range [{observed_min}, {observed_max}], not 0-100 as assumed"

    usa["belief_mean"] = usa[OUTCOME_COLUMNS["belief_mean"]].mean(axis=1)
    usa["policy_mean"] = usa[OUTCOME_COLUMNS["policy_mean"]].mean(axis=1)

    control = usa[usa["condName"] == "Control"]
    rows = []
    for outcome in ("belief_mean", "policy_mean"):
        control_series = control[outcome].dropna()
        control_unit = dc.to_unit_scale(control_series.mean(), 0, 100)
        for condition, group in usa[usa["condName"] != "Control"].groupby("condName"):
            series = group[outcome].dropna()
            treat_unit = dc.to_unit_scale(series.mean(), 0, 100)
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "human_ate_pp": 100 * (treat_unit - control_unit),
                    "n_control": len(control_series),
                    "n_treatment": len(series),
                    "conceptual_match_to_benchmark_condition": CONCEPTUAL_MATCH.get(condition),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["outcome", "condition"])
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(out_df)} rows to {OUT_PATH}\n")
    print(out_df.to_string(index=False))

    print("\nempirically observed |effect| envelope for this real, comparable megastudy (both outcomes pooled):")
    abs_ate = out_df["human_ate_pp"].abs()
    print(f"  min={abs_ate.min():.2f}pp  median={abs_ate.median():.2f}pp  max={abs_ate.max():.2f}pp")
    print("  (use this as a plausibility bound on our own model's predicted effect magnitudes -- effects far outside")
    print("   this range in either direction, across a comparable number of real interventions, would be a red flag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
