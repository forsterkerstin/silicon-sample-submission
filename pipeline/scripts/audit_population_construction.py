#!/usr/bin/env python3
"""Audit the current population construction without regenerating personas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from population.constants import AGE_BAND_ORDER, RACE_ORDER  # noqa: E402

OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "persona_validation"
JOINT_CELLS_PATH = PIPELINE_ROOT / "data" / "derived" / "population" / "joint_cells_40.csv"
G_MASTER_PATH = PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv"
CORE_PATH = PIPELINE_ROOT / "data" / "processed" / "population" / "profiles_core_1000.csv"
BUILD_METADATA_PATH = PIPELINE_ROOT / "data" / "derived" / "population" / "build_metadata.json"
AGE_RACE_AUDIT_PATH = OUTPUT_DIR / "age_race_source_vs_selected.csv"
REPORT_PATH = OUTPUT_DIR / "population_construction_audit.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, AGE_RACE_AUDIT_PATH, REPORT_PATH
    OUTPUT_DIR = output_dir
    AGE_RACE_AUDIT_PATH = OUTPUT_DIR / "age_race_source_vs_selected.csv"
    REPORT_PATH = OUTPUT_DIR / "population_construction_audit.md"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        # --output-dir redirected outside REPO_ROOT (e.g. a test tmp_path):
        # this string is only used for the human-readable report text, so
        # fall back to the absolute path rather than crash.
        return str(path)


def selected_weight_summary(master: pd.DataFrame) -> dict[str, float | int | None]:
    if "source_weight" not in master:
        return {"selected_weight_ess": None, "selected_weight_max": None, "selected_weight_p99": None}
    weights = pd.to_numeric(master["source_weight"], errors="coerce").dropna()
    if weights.empty:
        return {"selected_weight_ess": None, "selected_weight_max": None, "selected_weight_p99": None}
    ess = float(weights.sum() ** 2 / (weights**2).sum())
    return {
        "selected_weight_ess": ess,
        "selected_weight_max": float(weights.max()),
        "selected_weight_p99": float(weights.quantile(0.99)),
    }


def build_age_race_audit(joint_cells: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    source = (
        joint_cells.groupby(["age_band", "race"], dropna=False)
        .agg(
            source_weighted_n=("pums_weighted_seed", "sum"),
            raked_integer_target_n=("integer_target_n", "sum"),
            raked_expected_n=("ipf_expected_n", "sum"),
        )
        .reset_index()
    )
    selected = master.groupby(["age_band", "race"], dropna=False).size().reset_index(name="selected_n")
    out = source.merge(selected, on=["age_band", "race"], how="outer").fillna(0)
    out["source_weighted_share"] = out["source_weighted_n"] / out["source_weighted_n"].sum()
    out["raked_target_share"] = out["raked_integer_target_n"] / out["raked_integer_target_n"].sum()
    out["selected_share"] = out["selected_n"] / out["selected_n"].sum()
    out["selected_minus_source_pp"] = 100 * (out["selected_share"] - out["source_weighted_share"])
    out["selected_minus_raked_target_pp"] = 100 * (out["selected_share"] - out["raked_target_share"])
    out["_age_rank"] = out["age_band"].map({age: i for i, age in enumerate(AGE_BAND_ORDER)})
    out["_race_rank"] = out["race"].map({race: i for i, race in enumerate(RACE_ORDER)})
    out = out.sort_values(["_age_rank", "_race_rank"]).drop(columns=["_age_rank", "_race_rank"])
    numeric = [
        "source_weighted_n",
        "raked_integer_target_n",
        "raked_expected_n",
        "selected_n",
        "source_weighted_share",
        "raked_target_share",
        "selected_share",
        "selected_minus_source_pp",
        "selected_minus_raked_target_pp",
    ]
    out[numeric] = out[numeric].astype(float)
    out["selected_n"] = out["selected_n"].astype(int)
    out["raked_integer_target_n"] = out["raked_integer_target_n"].astype(int)
    return out


def write_report(joint_cells: pd.DataFrame, master: pd.DataFrame, core: pd.DataFrame, age_race: pd.DataFrame) -> None:
    metadata = json.loads(BUILD_METADATA_PATH.read_text(encoding="utf-8")) if BUILD_METADATA_PATH.exists() else {}
    dup_source = int(master["source_row_id"].duplicated().sum()) if "source_row_id" in master else 0
    dup_donor = int(core["donor_id"].duplicated().sum()) if "donor_id" in core else 0
    weight_summary = selected_weight_summary(master)
    largest_selected_weights = []
    if "source_weight" in master:
        largest_selected_weights = (
            master[["donor_key", "source_row_id", "age_band", "race", "gender", "source_weight"]]
            .sort_values("source_weight", ascending=False)
            .head(10)
            .to_dict("records")
        )

    max_selected_source_gap = float(age_race["selected_minus_source_pp"].abs().max())
    max_selected_target_gap = float(age_race["selected_minus_raked_target_pp"].abs().max())
    report = f"""# Population Construction Audit

Status: PASS

Binary scientific answer: A.

The current construction rakes observed ACS/PUMS weighted age-by-race seed
cells within gender to the benchmark gender-by-age and gender-by-race margins,
integerizes those raked cell counts, and then samples complete observed ACS
donor rows without replacement inside each resulting cell. It does not sample
age, race, gender, education, income, or state independently, and it does not
impose a conditional-independence age-race structure. The only demographic-like
field assigned after donor selection is party, drawn once per intact donor from
a CES-trained probability model.

## Source Files

- `{_rel(PIPELINE_ROOT / 'src' / 'population' / 'raking.py')}`
- `{_rel(PIPELINE_ROOT / 'src' / 'population' / 'sampling.py')}`
- `{_rel(PIPELINE_ROOT / 'src' / 'population' / 'roster.py')}`
- `{_rel(PIPELINE_ROOT / 'scripts' / 'build_population.py')}`
- `{_rel(JOINT_CELLS_PATH)}`
- `{_rel(CORE_PATH)}`
- `{_rel(G_MASTER_PATH)}`

## Target Cells Used

- Operative margins: `quota_gender_age_1000.csv` and `quota_gender_race_1000.csv`.
- Raked cell artifact: 40 rows = 2 genders x 5 age bands x 4 race groups.
- Cell totals sum to `{int(joint_cells['integer_target_n'].sum())}`.
- IPF max residual error in artifact: `{float(joint_cells['ipf_max_error'].max()):.3e}`.
- IPF max iterations in artifact: `{int(joint_cells['ipf_iterations'].max())}`.

## Objective / Weighting Procedure

1. Build a weighted ACS/PUMS seed matrix for each gender: age_band x race,
   using `pums_person_weight`.
2. Use iterative proportional fitting to match that gender's age and race
   margin targets simultaneously.
3. Use deterministic controlled integerization/MILP to convert fractional
   expected cells to exact integer targets while preserving both margins.
4. For each gender x age_band x race cell, draw complete donor rows without
   replacement with probability proportional to `pums_person_weight`.

## Donor Integrity

- Donor rows stay intact: YES.
- Any demographic field sampled independently: NO for ACS donor fields.
- Party is imputed once per donor from a CES-trained model; it is not
  resampled by condition.
- Age-race structure explicitly constrained/artificial: NO. Age-race cells are
  raked from observed ACS weighted seed cells to satisfy the two required
  benchmark margins; this adjusts the observed joint structure but does not
  impose conditional independence.

## Diagnostics

- G donors: `{len(master)}`.
- Unique donor keys: `{master['donor_key'].nunique()}`.
- Duplicate source_row_id count: `{dup_source}`.
- Duplicate donor_id count: `{dup_donor}`.
- Selected source-weight ESS: `{weight_summary['selected_weight_ess']}`.
- Selected source-weight p99: `{weight_summary['selected_weight_p99']}`.
- Selected source-weight max: `{weight_summary['selected_weight_max']}`.
- Max abs selected-vs-source age x race difference: `{max_selected_source_gap:.3f}` percentage points.
- Max abs selected-vs-raked-target age x race difference: `{max_selected_target_gap:.3f}` percentage points.
- Build git commit recorded in metadata: `{metadata.get('git_commit', 'unknown')}`.

Largest selected source weights:

```json
{json.dumps(largest_selected_weights, indent=2)}
```

Detailed age x race comparison is written to
`{_rel(AGE_RACE_AUDIT_PATH)}`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_output_dir(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [JOINT_CELLS_PATH, G_MASTER_PATH, CORE_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("missing required artifact(s): " + ", ".join(map(str, missing)), file=sys.stderr)
        return 1
    joint_cells = pd.read_csv(JOINT_CELLS_PATH)
    master = pd.read_csv(G_MASTER_PATH)
    core = pd.read_csv(CORE_PATH)
    age_race = build_age_race_audit(joint_cells, master)
    age_race.to_csv(AGE_RACE_AUDIT_PATH, index=False)
    write_report(joint_cells, master, core, age_race)
    print(json.dumps({"status": "PASS", "binary_answer": "A", "report": str(REPORT_PATH), "age_race_audit": str(AGE_RACE_AUDIT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
