"""Structural tests for the real, already-assembled final first-valid
native G response universe. Never inspects scientific meaning of values --
only engineering shape: counts, uniqueness, native-support ranges."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import survey_content as sc  # noqa: E402
import assemble_final_native_g_responses as builder  # noqa: E402
from ate.normalize_effects import RAW_ITEM_SCALE_BOUNDS  # noqa: E402

pytestmark = pytest.mark.skipif(not builder.OUT_CSV.exists(), reason="final native G response universe not assembled in this environment")


@pytest.fixture(scope="module")
def native() -> pd.DataFrame:
    return pd.read_csv(builder.OUT_CSV, dtype={"profile_id": str})


def test_exactly_17000_rows_1000_respondents_17_conditions(native):
    assert len(native) == 17000
    assert native["profile_id"].nunique() == 1000
    counts = native["condition_id"].value_counts()
    assert len(counts) == 17
    assert (counts == 1000).all()


def test_no_duplicate_identity_pairs(native):
    assert not native.duplicated(subset=["profile_id", "condition_id"]).any()


def test_exactly_44_raw_item_columns_no_missing(native):
    item_cols = [c for c in native.columns if c not in ("profile_id", "condition_id")]
    assert len(item_cols) == 44
    assert not native[item_cols].isna().any().any()


def test_every_raw_item_within_its_native_support(native):
    for item in sc.load_items():
        label = item["target_label"]
        low, high = RAW_ITEM_SCALE_BOUNDS[item["scale"]]
        values = native[label]
        assert values.between(low, high).all(), f"{label} has out-of-range values"
        assert (values == values.round()).all(), f"{label} has non-integer values"


def test_consensus_rows_sourced_only_from_outcomes_stage(native):
    consensus = native[native["condition_id"] == "Consensus"]
    assert len(consensus) == 1000
    assert set(consensus["profile_id"]) == set(native["profile_id"].unique())


def test_summary_hash_matches_csv_on_disk():
    import hashlib
    import json

    summary = json.loads(builder.OUT_SUMMARY.read_text(encoding="utf-8"))
    assert summary["csv_sha256"] == hashlib.sha256(builder.OUT_CSV.read_bytes()).hexdigest()
    assert summary["n_rows"] == 17000
