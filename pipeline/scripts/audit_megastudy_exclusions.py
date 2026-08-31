#!/usr/bin/env python3
"""Write metadata-only exclusion audits for the secondary megastudy holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from validation.holdout import (  # noqa: E402
    VALIDATION_DIR,
    classify_megastudy_exclusions,
    extract_megastudy_effect_metadata,
    summarize_megastudy_exclusions,
)


def main() -> int:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = VALIDATION_DIR / "megastudy_effect_metadata.csv"
    metadata = extract_megastudy_effect_metadata(out_path=metadata_path)
    classified = classify_megastudy_exclusions(metadata)
    summary = summarize_megastudy_exclusions(classified)
    reasons_path = VALIDATION_DIR / "megastudy_exclusion_reasons.csv"
    summary_path = VALIDATION_DIR / "megastudy_exclusion_summary.csv"
    classified.to_csv(reasons_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "effects": int(len(classified)),
        "eligible_effects": int(classified["eligible"].sum()) if not classified.empty else 0,
        "studies": int(classified["study_id"].nunique()) if not classified.empty else 0,
        "reason_counts": classified["primary_exclusion_reason"].value_counts().to_dict() if not classified.empty else {},
        "outputs": [str(reasons_path), str(summary_path)],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
