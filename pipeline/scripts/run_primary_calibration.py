#!/usr/bin/env python3
"""Fit/finalize C on development-calibration rows only."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from ate.calibrate_lambda import fit_calibration_model_comparison, load_ate_archive, summarize_calibration_rows  # noqa: E402
from calibration.external_prediction_provenance import assert_external_f_predictions_production_ready, file_sha256  # noqa: E402
from validation.holdout import (  # noqa: E402
    ATE_ARCHIVE_PATH,
    PRIMARY_STUDY_FEATURES_PATH,
    VALIDATION_DIR,
    assert_no_holdout_in_calibration_archive,
    build_validation_split_manifest,
    extract_megastudy_effect_metadata,
    megastudy_study_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit primary C calibration.")
    parser.add_argument(
        "--production",
        action="store_true",
        help="require production-ready external F predictions, frozen panel provenance, and frozen F model/protocol before fitting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ate_archive_df = pd.read_csv(ATE_ARCHIVE_PATH)
    if args.production:
        assert_external_f_predictions_production_ready(ate_archive_df)
    primary_features = pd.read_csv(PRIMARY_STUDY_FEATURES_PATH)
    effect_metadata = extract_megastudy_effect_metadata(out_path=VALIDATION_DIR / "megastudy_effect_metadata.csv")
    split = build_validation_split_manifest(primary_features, ate_archive_df, megastudy_study_summary(effect_metadata))
    assert_no_holdout_in_calibration_archive(ate_archive_df, split)

    rows = load_ate_archive(ATE_ARCHIVE_PATH, primary_only=True)
    if "requires_synthetic_regeneration" in ate_archive_df.columns:
        primary_mask = ate_archive_df["included_primary_calibration"].astype(str).str.lower().isin({"true", "1", "yes"})
        stale_population_predictions = bool(
            ate_archive_df.loc[primary_mask, "requires_synthetic_regeneration"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        )
    else:
        stale_population_predictions = False
    model_ate_pp = [r["model_ate"] for r in rows]
    human_ate_pp = [r["human_ate"] for r in rows]
    study_id = [r["study_id"] for r in rows]
    fit = fit_calibration_model_comparison(model_ate_pp, human_ate_pp, study_id, outputs_dir=PIPELINE_ROOT / "outputs")
    if stale_population_predictions:
        fit["calibration_status"] = "DEVELOPMENT_STALE_PENDING_STUDY_SPECIFIC_EXTERNAL_F_REGENERATION"
        fit["usable_for_production"] = False
        fit["stale_reason"] = "Primary external synthetic effects were not regenerated from study-specific empirical respondent F panels."
        selected_path = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
        selected_payload = {k: v for k, v in fit.items() if k not in {"loso_predictions", "comparison_rows", "rmse_loso"}}
        selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    elif args.production:
        # Only a --production run that (a) passed assert_external_f_predictions_production_ready
        # above and (b) is not stale may mark the persisted artifact usable_for_production=True --
        # this is the ONLY place that ever sets that flag True, so a production Tier-1 build can
        # trust a bare usable_for_production=True read from disk without re-deriving it.
        fit["calibration_status"] = "PRODUCTION_READY"
        fit["usable_for_production"] = True
        fit["source_ate_archive_sha256"] = file_sha256(ATE_ARCHIVE_PATH)
        selected_path = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
        selected_payload = {k: v for k, v in fit.items() if k not in {"loso_predictions", "comparison_rows", "rmse_loso"}}
        selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    summary = summarize_calibration_rows(rows)
    (VALIDATION_DIR / "primary_calibration_fit_summary.json").write_text(
        json.dumps({"fit": {k: v for k, v in fit.items() if k not in {"loso_predictions", "comparison_rows"}}, "calibration_set_summary": summary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_model": fit["model_name"], "alpha": fit["calibration_alpha"], "lambda": fit["calibration_lambda"], "production": bool(args.production), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
