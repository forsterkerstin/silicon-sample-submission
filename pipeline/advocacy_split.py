"""pipeline/advocacy_split.py

A single, deterministic respondent-level split of the climate-advocacy
megastudy (data/climate_advocacy_megastudy/data/advocacy_data.csv), so that
the same real respondents are never used both to CALIBRATE our pipeline
(baseline_references.json, via scripts/build_baseline_reference.py) and to
VALIDATE it (scripts/build_advocacy_validation.py's real treatment-effect
comparison) -- using the same rows for both would let the validation
"pass" partly because the model's own baseline was anchored to those exact
rows, not because the model is actually right.

Both build scripts import `split_for(response_id)` from here, so there is
exactly one definition of the split, not two independently-written ones
that could silently drift apart. tests/elicitation/test_advocacy_split.py
asserts the two halves are disjoint and stable.
"""

from __future__ import annotations

import zlib

import pandas as pd

CALIB = "calib"  # -> scripts/build_baseline_reference.py (control-moment calibration)
VALID = "valid"  # -> scripts/build_advocacy_validation.py (held-out real ATEs)


def split_for(response_id: str) -> str:
    """Deterministic across runs/machines (zlib.crc32, not Python's
    randomized-per-process hash() on strings -- same convention as
    scripts/generate_responses.py's stable_seed()).
    """
    return CALIB if zlib.crc32(str(response_id).encode("utf-8")) % 2 == 0 else VALID


def add_split_column(df: pd.DataFrame, id_column: str = "ResponseId") -> pd.DataFrame:
    """Returns a copy of `df` with a `_split` column ("calib"/"valid")."""
    out = df.copy()
    out["_split"] = out[id_column].map(split_for)
    return out
