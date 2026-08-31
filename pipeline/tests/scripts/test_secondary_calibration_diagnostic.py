"""Structural tests for the offline secondary-calibration study (MCONST,
M2, M2R robust-Huber, M3, M3R robust-Huber; whole-study LOSO; influence and
bootstrap on M2's lambda; within/between decomposition). Confirms it runs
against the real frozen 136-effect table, reproduces the frozen primary
M2 alpha/lambda in its own full-data refit to within float tolerance
(never overwrites the primary artifact), and writes only under
outputs/secondary_calibration_diagnostic/.
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

from secondary_calibration_diagnostic import (  # noqa: E402
    HUBER_KWARGS,
    OUT_DIR,
    PRIMARY_ARTIFACT_PATH,
    load_frozen_table,
    load_primary_frozen_m2,
    main,
)


def test_frozen_table_loads_136_effects_31_studies():
    df = load_frozen_table()
    assert len(df) == 136
    assert df["study_id"].nunique() == 31


def test_primary_artifact_untouched_hash_before_and_after():
    before = PRIMARY_ARTIFACT_PATH.read_bytes()
    result = main()
    after = PRIMARY_ARTIFACT_PATH.read_bytes()
    assert before == after
    assert result["primary_artifacts_unchanged"] is True


def test_full_data_m2_matches_primary_frozen_artifact_closely():
    primary = load_primary_frozen_m2()
    result = main()
    m2 = result["section6_full_data_fits"]["M2"]
    assert m2["alpha"] == primary["alpha"]
    assert abs(m2["lambda"] - primary["lambda"]) < 1e-6


def test_loso_table_has_all_five_models_equal_study_weighting():
    result = main()
    table = {r["model"]: r for r in result["section5_loso_table"]}
    assert set(table) == {"MCONST", "M2", "M2R", "M3", "M3R"}
    primary = load_primary_frozen_m2()
    assert table["M2"]["loso_rmse"] == primary["loso_rmse_M2"]
    for r in table.values():
        assert r["loso_mse"] > 0
        assert r["loso_rmse"] > 0
        assert r["loso_mae"] > 0


def test_bootstrap_uses_frozen_seed_and_is_deterministic():
    r1 = main()
    r2 = main()
    assert r1["section1_bootstrap"]["seed"] == 20260826
    assert r1["section1_bootstrap"]["n_bootstrap"] == 10_000
    assert r1["section1_bootstrap"]["lambda_bootstrap_mean"] == r2["section1_bootstrap"]["lambda_bootstrap_mean"]
    assert r1["section1_bootstrap"]["lambda_bootstrap_p025"] == r2["section1_bootstrap"]["lambda_bootstrap_p025"]


def test_huber_hyperparameters_are_exactly_as_specified_and_fixed():
    assert HUBER_KWARGS == dict(epsilon=1.345, alpha=0.0, fit_intercept=True, max_iter=10_000, tol=1e-10)


def test_influence_table_has_31_rows_and_written_to_disk():
    result = main()
    influence = result["section1_influence"]
    assert influence["n_negative"] + influence["n_positive"] == 31
    csv_path = OUT_DIR / "loso_delete_alpha_lambda.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 31
    assert df["study_id"].nunique() == 31


def test_m3_and_m3r_fold_coefficient_files_have_31_rows_each():
    main()
    for name in ("m3_loso_fold_coefficients.csv", "m3r_loso_fold_coefficients.csv"):
        df = pd.read_csv(OUT_DIR / name)
        assert len(df) == 31
        assert {"beta_B_minus_s", "beta_W_minus_s"}.issubset(df.columns)


def test_loso_fold_predictions_cover_all_effects_for_every_model():
    main()
    df = pd.read_csv(OUT_DIR / "loso_fold_predictions.csv")
    assert set(df["model"].unique()) == {"MCONST", "M2", "M2R", "M3", "M3R"}
    for model, group in df.groupby("model"):
        assert len(group) == 136


def test_section8_interpretation_fields_present_and_valid():
    result = main()
    s8 = result["section8_interpretation"]
    yes_no_fields = (
        "global_negative_slope_stable",
        "within_between_sign_reversal",
        "m2r_improves_m2",
        "m3_improves_m2",
        "m3r_improves_m2",
        "m3r_improves_m3",
        "m3r_improves_m2r",
        "m3r_beta_w_sign_stable",
    )
    for f in yes_no_fields:
        assert s8[f] in ("YES", "NO")
    assert s8["best_loso_model"] in {"MCONST", "M2", "M2R", "M3", "M3R"}


def test_summary_self_hash_matches_written_file_and_is_deterministic():
    result = main()
    on_disk = json.loads((OUT_DIR / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert on_disk["secondary_diagnostic_complete"] is True
    sha_file = (OUT_DIR / "diagnostic_summary.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["secondary_diagnostic_sha256"]
    assert len(sha_file) == 64


def test_output_written_only_under_secondary_calibration_diagnostic_dir():
    assert OUT_DIR.name == "secondary_calibration_diagnostic"
    assert OUT_DIR.parent == PIPELINE_ROOT / "outputs"
