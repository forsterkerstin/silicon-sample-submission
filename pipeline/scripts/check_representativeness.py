#!/usr/bin/env python3
"""scripts/check_representativeness.py

Compares the 1,000-profile panel (data/processed/population/profiles_core_1000.csv)
against two references:

  1. The operative quota margins (config/quota_gender_age_1000.csv,
     config/quota_gender_race_1000.csv) -- exact by construction (raking +
     controlled integerization), reported here as a sanity check, not a new
     finding.
  2. The full PWGTP-weighted eligible ACS population (every donor who passed
     §8's inclusion filters, before the 1,000 were sampled) -- NOT targeted
     by raking, so a close match on education/income/state/age is evidence
     that PPS-within-cell sampling preserved real-world associations, not a
     guaranteed outcome.

Also compares party shares: CES weighted (the training population), the
panel's mean predicted probability, and its realized (sampled) shares.

Re-reads data/csv_pus.zip + data/csv_hus.zip to build the weighted reference
(~1-1.5 minutes) -- this script does not re-run raking/sampling/roster, only
the ingestion+filter+recode steps needed for the comparison population.

Usage: python scripts/check_representativeness.py --config config/population.yaml
Writes reports/population/representativeness_check.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from population import pums  # noqa: E402
from population.io import configure_logging, ensure_dir, get_logger  # noqa: E402

logger = get_logger("check_representativeness")


def weighted_shares(df: pd.DataFrame, col: str, weight_col: str = "pums_person_weight") -> pd.Series:
    return df.groupby(col)[weight_col].sum() / df[weight_col].sum()


def unweighted_shares(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts(normalize=True)


def comparison_table(panel: pd.DataFrame, population: pd.DataFrame, col: str) -> pd.DataFrame:
    pop_share = weighted_shares(population, col) * 100
    panel_share = unweighted_shares(panel, col) * 100
    table = pd.DataFrame({"population_pct": pop_share, "panel_pct": panel_share}).fillna(0.0)
    table["diff_pp"] = table["panel_pct"] - table["population_pct"]
    return table.sort_values("population_pct", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    configure_logging()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    panel = pd.read_csv(REPO_ROOT / paths["processed_dir"] / "profiles_core_1000.csv")

    logger.info("re-ingesting PUMS person+housing to build the weighted eligible-population reference (~1-1.5 min)...")
    pums_raw, _ = pums.read_pums(REPO_ROOT / paths["pums_zip"], REPO_ROOT / paths["pums_housing_zip"])
    filtered, _ = pums.apply_inclusion_filters(pums_raw)
    population = pums.recode_pums(filtered, reference_year=cfg["population"]["reference_year"])
    logger.info("eligible population: %d donors, %.0f people (PWGTP-weighted)", len(population), population["pums_person_weight"].sum())

    lines: list[str] = []
    lines.append("# Representativeness check\n")
    lines.append(
        f"Panel: {len(panel)} selected profiles. Reference population: {len(population):,} PUMS donors "
        f"passing every §8 inclusion filter, representing {population['pums_person_weight'].sum():,.0f} people "
        "when weighted by PWGTP. The reference population is NOT the same as the panel's own sampling frame "
        "restricted to it being re-derivable from raw data -- it is the full weighted eligible population.\n"
    )

    lines.append("## 1. Quota margins (raked -- exact match expected)\n")
    quota_age = pd.read_csv(REPO_ROOT / "config" / "quota_gender_age_1000.csv")
    quota_race = pd.read_csv(REPO_ROOT / "config" / "quota_gender_race_1000.csv")
    achieved_age = panel.groupby(["gender", "age_band"]).size().reset_index(name="achieved_n")
    age_check = quota_age.merge(achieved_age, on=["gender", "age_band"])
    age_check["exact_match"] = age_check["achieved_n"] == age_check["target_n"]
    achieved_race = panel.groupby(["gender", "race"]).size().reset_index(name="achieved_n")
    race_check = quota_race.merge(achieved_race, on=["gender", "race"])
    race_check["exact_match"] = race_check["achieved_n"] == race_check["target_n"]
    lines.append(f"Gender x age_band: {age_check['exact_match'].sum()}/{len(age_check)} cells exact.")
    lines.append(f"Gender x race: {race_check['exact_match'].sum()}/{len(race_check)} cells exact.\n")

    print("\n=== 1. QUOTA MARGINS (raked, exact match expected) ===")
    print(f"Gender x age_band: {age_check['exact_match'].sum()}/{len(age_check)} cells exact")
    print(f"Gender x race:     {race_check['exact_match'].sum()}/{len(race_check)} cells exact")

    for label, col in [("Education", "education"), ("Income", "income"), ("State (top 10)", "state_abbr")]:
        table = comparison_table(panel, population, col)
        if label.startswith("State"):
            table = table.head(10)
        lines.append(f"## {label}: PWGTP-weighted population % vs panel %\n")
        lines.append(table.round(1).to_markdown())
        lines.append("")
        print(f"\n=== {label.upper()}: population % (PWGTP-weighted) vs panel % ===")
        print(table.round(1).to_string())

    pop_age_mean = (population["age"] * population["pums_person_weight"]).sum() / population["pums_person_weight"].sum()
    panel_age_mean = panel["age"].mean()
    lines.append(f"## Age: population weighted mean {pop_age_mean:.1f}, panel mean {panel_age_mean:.1f}\n")
    print(f"\n=== AGE (within raked bands): population weighted mean {pop_age_mean:.1f}, panel mean {panel_age_mean:.1f} ===")

    party_diagnostics_path = REPO_ROOT / paths["reports_dir"] / "party_model_diagnostics.json"
    if party_diagnostics_path.exists():
        import json

        diag = json.load(open(party_diagnostics_path))
        ces_shares = {k: v * 100 for k, v in diag["observed_weighted_party_shares_all_ces"].items()}
        expected = (panel[["party_prob_democrat", "party_prob_republican", "party_prob_independent", "party_prob_other"]].mean() * 100)
        expected.index = [c.replace("party_prob_", "").capitalize() for c in expected.index]
        realized = panel["party"].value_counts(normalize=True) * 100
        party_table = pd.DataFrame({"ces_weighted_pct": pd.Series(ces_shares), "panel_expected_pct": expected, "panel_realized_pct": realized}).round(1)
        lines.append("## Party: CES weighted vs panel expected (mean predicted prob) vs panel realized (sampled)\n")
        lines.append(party_table.to_markdown())
        print("\n=== PARTY: CES weighted vs panel expected vs panel realized ===")
        print(party_table.to_string())

    reports_dir = ensure_dir(REPO_ROOT / paths["reports_dir"])
    with open(reports_dir / "representativeness_check.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("wrote %s", reports_dir / "representativeness_check.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
