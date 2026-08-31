"""advocacy_split.py: the deterministic calib/valid split that keeps
scripts/build_baseline_reference.py (calibration) and
scripts/build_advocacy_validation.py (held-out validation) from ever using
the same climate-advocacy-megastudy respondents -- the whole point of the
split is that using the same rows for both would let the validation "pass"
partly because the model's baseline was anchored to those exact rows."""

from __future__ import annotations

from advocacy_split import CALIB, VALID, add_split_column, split_for


def test_split_is_binary():
    assert {split_for(f"R_{i}") for i in range(200)} <= {CALIB, VALID}


def test_split_is_deterministic_across_calls():
    ids = [f"R_{i}" for i in range(500)]
    first = [split_for(i) for i in ids]
    second = [split_for(i) for i in ids]
    assert first == second


def test_split_is_roughly_balanced():
    ids = [f"R_{i}" for i in range(10_000)]
    n_calib = sum(split_for(i) == CALIB for i in ids)
    assert 0.4 < n_calib / len(ids) < 0.6


def test_add_split_column_partitions_disjointly(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"ResponseId": [f"R_{i}" for i in range(300)], "value": range(300)})
    out = add_split_column(df)
    calib_ids = set(out.loc[out["_split"] == CALIB, "ResponseId"])
    valid_ids = set(out.loc[out["_split"] == VALID, "ResponseId"])
    assert calib_ids.isdisjoint(valid_ids)
    assert calib_ids | valid_ids == set(df["ResponseId"])


def test_baseline_reference_builder_only_reads_calib_half():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "scripts" / "build_baseline_reference.py").read_text(encoding="utf-8")
    assert "CALIB" in source and "_split" in source, "build_baseline_reference.py's builder must filter to the calibration split"
