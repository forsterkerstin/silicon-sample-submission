#!/usr/bin/env python3
"""Create validation split, usage, eligibility, and overlap audit artifacts.

This command does not evaluate structural-holdout performance and does not
estimate or modify calibration parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from validation.holdout import (  # noqa: E402
    ATE_ARCHIVE_PATH,
    PRIMARY_STUDY_FEATURES_PATH,
    VALIDATION_DIR,
    assert_no_holdout_in_calibration_archive,
    build_data_usage_audit,
    build_megastudy_holdout_eligibility,
    build_validation_split_manifest,
    climate_holdout_overlap_audit,
    extract_megastudy_effect_metadata,
    megastudy_study_summary,
    write_initial_holdout_status,
    write_placeholder_g_validation_status,
)


def main() -> int:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ate_archive = pd.read_csv(ATE_ARCHIVE_PATH)
    primary_features = pd.read_csv(PRIMARY_STUDY_FEATURES_PATH)
    effect_metadata = extract_megastudy_effect_metadata(out_path=VALIDATION_DIR / "megastudy_effect_metadata.csv")
    megastudy_summary = megastudy_study_summary(effect_metadata)
    megastudy_summary.to_csv(VALIDATION_DIR / "megastudy_study_summary.csv", index=False)

    usage = build_data_usage_audit(ate_archive, primary_features, megastudy_summary)
    usage.to_csv(VALIDATION_DIR / "data_usage_audit.csv", index=False)

    split = build_validation_split_manifest(primary_features, ate_archive, megastudy_summary)
    split.to_csv(VALIDATION_DIR / "validation_split_manifest.csv", index=False)
    assert_no_holdout_in_calibration_archive(ate_archive, split)

    eligibility = build_megastudy_holdout_eligibility(effect_metadata)
    eligibility.to_csv(VALIDATION_DIR / "megastudy_holdout_eligibility.csv", index=False)

    overlap = climate_holdout_overlap_audit(effect_metadata)
    (VALIDATION_DIR / "climate_holdout_overlap_audit.json").write_text(json.dumps(overlap, indent=2) + "\n", encoding="utf-8")
    write_initial_holdout_status(notes="split audit created; structural holdout performance not opened")
    write_placeholder_g_validation_status()

    summary = {
        "primary_archive_studies": int(len(primary_features)),
        "primary_archive_effect_rows_current": int(len(ate_archive)),
        "primary_development_effects_current": int((ate_archive["included_primary_calibration"] == True).sum()),
        "secondary_megastudies": int(megastudy_summary["study_id"].nunique()),
        "secondary_effect_rows": int(len(effect_metadata)),
        "secondary_eligible_effect_rows_current": int((eligibility["eligible"] == True).sum()),
        "compromised_holdout_studies": sorted(split.loc[split["assigned_role"] == "compromised_holdout", "study_id"].astype(str).unique()),
        "holdout_opened": False,
    }
    (VALIDATION_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
