"""Structural tests for the real, already-materialized final Tier-1
submission file: schema/shape, control-arm invariance under calibration,
composite consistency, and 208-cell ATE coverage. Never asserts anything
about target human outcomes."""

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
import materialize_final_s2_submission as materializer  # noqa: E402

pytestmark = pytest.mark.skipif(not materializer.OUT_CSV.exists(), reason="final Tier-1 submission not materialized in this environment")


@pytest.fixture(scope="module")
def final_df() -> pd.DataFrame:
    return pd.read_csv(materializer.OUT_CSV, dtype={"profile_id": str})


@pytest.fixture(scope="module")
def native_df() -> pd.DataFrame:
    return pd.read_csv(materializer.NATIVE_CSV, dtype={"profile_id": str})


def test_required_columns_present_and_no_missing(final_df):
    for col in materializer.TIER1_REQUIRED_ORDER:
        assert col in final_df.columns, f"missing required column {col}"
    assert not final_df[materializer.TIER1_REQUIRED_ORDER].isna().any().any()


def test_17000_rows_1000_per_condition_unique_profile_id(final_df):
    assert len(final_df) == 17000
    counts = final_df["condition"].value_counts()
    assert len(counts) == 17
    assert (counts == 1000).all()
    assert not final_df["profile_id"].duplicated().any()


def test_control_arm_unchanged_by_calibration(final_df, native_df):
    native_control = native_df[native_df["condition_id"] == "control"].set_index("profile_id")
    final_control = final_df[final_df["condition"] == "control"].set_index("donor_key")
    item_cols = [c for c in native_df.columns if c not in ("profile_id", "condition_id")]
    for label in item_cols:
        assert (final_control.loc[native_control.index, label].astype(float) == native_control[label].astype(float)).all(), f"control item {label} altered"


def test_composites_recompute_exactly(final_df):
    for _, row in final_df.sample(min(200, len(final_df)), random_state=0).iterrows():
        expected = sc.compute_outcomes(row.to_dict())
        for outcome, value in expected.items():
            assert abs(float(row[outcome]) - float(value)) < 1e-6


def test_208_ate_cells_present(final_df):
    from ate.estimate_ates import estimate_raw_ates

    ates = estimate_raw_ates(final_df, list(sc.OUTCOME_COMPOSITES.keys()))
    assert len(ates) == 208
    assert ates[["raw_ate", "control_mean", "treatment_mean"]].notna().all().all()


def test_diagnostics_have_208_rows_and_finite_projection_error():
    diagnostics = pd.read_csv(materializer.OUT_DIAGNOSTICS)
    assert len(diagnostics) == 208
    assert diagnostics["absolute_error"].notna().all()
    assert (diagnostics["absolute_error"] < 5.0).all()


def test_manifest_hashes_match_files_on_disk():
    import hashlib
    import json

    manifest = json.loads(materializer.OUT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["predictions_sha256"] == hashlib.sha256(materializer.OUT_CSV.read_bytes()).hexdigest()
    assert manifest["integrity_report"]["ok"] is True
