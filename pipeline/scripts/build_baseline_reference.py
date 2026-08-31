#!/usr/bin/env python3
"""scripts/build_baseline_reference.py

Approach §1's external baseline-calibration source: a real, closely
domain-matched megastudy ("a megastudy of behavioral interventions to
catalyze public, political, and financial climate advocacy") supplied by the
user as data/wv7c3-osfstorage-archive.zip and extracted (read-only) into
data/climate_advocacy_megastudy/. Not literally "trust in scientists" (§1's
first-choice example), but a genuine climate-attitudes/behavior study with
several items on the *exact same scales* this benchmark uses.

Four of this megastudy's control-condition (condName == "Control") items
correspond directly to one of this benchmark's own 13 outcomes, same scale:

  belief_1     (0-100 slider) -> belief_post
  policy_1     (0-100 slider) -> policy_general
  donation     ($0-10)        -> donation_ams
  newsletter1  (0/1 binary)   -> newsletter_signup

No other of this benchmark's 9 remaining outcomes (the trust subscales,
institutional trust, concern, specific policy items, behavior_mean) has a
clearly corresponding item in this megastudy -- left absent from the
reference, not guessed.

Writes data/baseline_references.json: {outcome: {mean, variance, n, source}},
loaded by baseline_calibration.py's consumers (generate_responses.py) as
BaselineReference objects, instead of re-parsing the 23MB source CSV per run.

No-leakage note: this megastudy's real treatment arms are ALSO used as a
held-out validation set (scripts/build_advocacy_validation.py compares our
model's predicted effects against real condition-vs-control effects computed
from this same file). Using the SAME respondents for both would mean part of
any apparent validation "success" is just the model's control baseline
having been anchored to those exact rows, not genuine agreement -- so this
script only ever touches the deterministic "calib" half of respondents
(advocacy_split.split_for()); build_advocacy_validation.py is restricted to
the disjoint "valid" half. See tests/elicitation/test_advocacy_split.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from advocacy_split import CALIB, add_split_column  # noqa: E402

DATA_PATH = PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "data" / "advocacy_data.csv"
OUT_PATH = PIPELINE_ROOT / "data" / "baseline_references.json"

#: benchmark outcome -> (megastudy column, same-scale correspondence documented above)
OUTCOME_TO_COLUMN: dict[str, str] = {
    "belief_post": "belief_1",
    "policy_general": "policy_1",
    "donation_ams": "donation",
    "newsletter_signup": "newsletter1",
}

SOURCE_CITATION = (
    "Control-condition respondents (condName == 'Control') from the CALIBRATION half "
    "(see advocacy_split.py) of the climate-advocacy megastudy supplied as "
    "data/wv7c3-osfstorage-archive.zip (see data/climate_advocacy_megastudy/data/codebook_advocacy.pdf)"
)


def main() -> int:
    columns = ["ResponseId", "condName", *OUTCOME_TO_COLUMN.values()]
    df = pd.read_csv(DATA_PATH, usecols=columns, low_memory=False)
    df = add_split_column(df)
    calib = df[df["_split"] == CALIB]
    control = calib[calib["condName"] == "Control"]
    print(f"loaded {len(df)} respondents ({len(calib)} in the calibration half); {len(control)} in the control condition")

    references = {}
    for outcome, column in OUTCOME_TO_COLUMN.items():
        series = control[column].dropna()
        references[outcome] = {
            "mean": float(series.mean()),
            "variance": float(series.var()),
            "n": int(series.count()),
            "source": f"{SOURCE_CITATION}; column '{column}'",
        }
        print(f"{outcome} (<- {column}): mean={references[outcome]['mean']:.2f} variance={references[outcome]['variance']:.2f} n={references[outcome]['n']}")

    OUT_PATH.write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
