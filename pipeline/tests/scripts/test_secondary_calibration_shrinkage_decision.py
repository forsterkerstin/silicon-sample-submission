"""Structural tests for the final secondary-calibration shrinkage decision
study (MSHRINK vs MCONST, nested whole-study LOSO, prospective selection
rule). Confirms the nested procedure runs against the real frozen table,
never touches the primary artifact, reproduces the earlier MCONST outer-fold
MSE exactly (same 31 folds as the prior diagnostic), and freezes exactly one
secondary artifact under outputs/secondary_calibration_diagnostic/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from secondary_calibration_diagnostic import PRIMARY_ARTIFACT_PATH  # noqa: E402
from secondary_calibration_shrinkage_decision import OUT_DIR, W_GRID, main  # noqa: E402


def test_w_grid_is_fixed_eleven_point_grid():
    assert W_GRID == [round(i * 0.1, 1) for i in range(11)]
    assert len(W_GRID) == 11


def test_primary_artifact_untouched():
    before = PRIMARY_ARTIFACT_PATH.read_bytes()
    result = main()
    after = PRIMARY_ARTIFACT_PATH.read_bytes()
    assert before == after
    assert result["primary_artifacts_unchanged"] is True


def test_mconst_outer_mse_matches_prior_diagnostic_loso_mse():
    result = main()
    assert abs(result["mconst_outer_mse"] - 88.42975638795151) < 1e-6


def test_nested_outer_fold_csvs_have_31_rows():
    main()
    shrink_df = pd.read_csv(OUT_DIR / "mshrink_nested_outer_fold_results.csv")
    mconst_df = pd.read_csv(OUT_DIR / "mconst_outer_fold_results.csv")
    assert len(shrink_df) == 31
    assert len(mconst_df) == 31
    assert set(shrink_df["outer_study"]) == set(mconst_df["outer_study"])
    assert shrink_df["inner_selected_w"].isin(W_GRID).all()


def test_selection_rule_is_prospective_and_deterministic():
    r1 = main()
    r2 = main()
    assert r1["secondary_calibration_selected"] == r2["secondary_calibration_selected"]
    assert r1["mshrink_nested_loso_mse"] == r2["mshrink_nested_loso_mse"]
    if r1["mshrink_nested_loso_mse"] < r1["mconst_outer_mse"]:
        assert r1["secondary_calibration_selected"] == "MSHRINK"
    else:
        assert r1["secondary_calibration_selected"] == "MCONST"


def test_secondary_artifact_is_frozen_with_required_disclosure_fields():
    result = main()
    artifact = json.loads((OUT_DIR / "secondary_calibration_selected_model.json").read_text(encoding="utf-8"))
    assert artifact["post_primary_external_development"] is True
    assert artifact["target_human_outcome_blind"] is True
    assert artifact["secondary_calibration_selected"] in {"MCONST", "MSHRINK"}
    if artifact["secondary_calibration_selected"] == "MCONST":
        assert artifact["secondary_w"] is None
        assert artifact["secondary_alpha"] is None
        assert artifact["secondary_lambda"] is None
        assert result["effective_secondary_f_dependence"] == 0.0
    else:
        assert artifact["secondary_w"] in W_GRID
        assert artifact["secondary_alpha"] is not None
        assert artifact["secondary_lambda"] is not None


def test_secondary_artifact_written_only_under_secondary_diagnostic_dir():
    main()
    assert OUT_DIR.name == "secondary_calibration_diagnostic"
    assert (OUT_DIR / "secondary_calibration_selected_model.json").exists()
    assert (OUT_DIR / "secondary_calibration_decision_summary.json").exists()
