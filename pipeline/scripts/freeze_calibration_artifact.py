"""Assemble the final frozen calibration artifact (outputs/calibration_selected_model.json)
from the real LOSO fit already computed by scripts/score_f_calibration_production.py.
Preserves the prior (stale, development-archive) artifact at
outputs/calibration_production/superseded/. No new fit, no new decision --
pure provenance assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

CALIB_OUT = PIPELINE_ROOT / "outputs" / "calibration_production"
LIVE_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
CALIB_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "f_calibration_production"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    selected = json.loads((CALIB_OUT / "calibration_selected_model.json").read_text(encoding="utf-8"))
    table = pd.read_csv(CALIB_OUT / "frozen_136_effect_calibration_table.csv")
    scoring_result = json.loads((CALIB_OUT / "calibration_scoring_result.json").read_text(encoding="utf-8"))

    batch_ids = {part: scoring_result["per_partition"][part]["batch_id"] for part in scoring_result["per_partition"]}
    raw_output_sha256 = {part: scoring_result["per_partition"][part]["batch_output_sha256"] for part in scoring_result["per_partition"]}

    payload = {
        **selected,
        "usable_for_production": True,
        "effect_universe": {
            "n_effects": int(len(table)),
            "n_studies": int(table["study_id"].nunique()),
            "effect_ids_sha256": hashlib.sha256("\n".join(sorted(table["effect_id"])).encode("utf-8")).hexdigest(),
            "study_ids": sorted(table["study_id"].unique().tolist()),
        },
        "raw_effect_table": {
            "path": "outputs/calibration_production/frozen_136_effect_calibration_table.csv",
            "sha256": sha256_file(CALIB_OUT / "frozen_136_effect_calibration_table.csv"),
        },
        "loso_folds": {
            "rule": "whole-study leave-one-study-out; all effects from a held-out study held out together",
            "n_folds": int(table["study_id"].nunique()),
            "predictions_path": "outputs/calibration_production/calibration_loso_predictions.csv",
            "predictions_sha256": sha256_file(CALIB_OUT / "calibration_loso_predictions.csv"),
        },
        "reconciliation": scoring_result["reconciliation"],
        "invalid_rate": scoring_result["invalid_rate"],
        "provenance": {
            "phase": "f_calibration_production",
            "model": "google/gemma-4-31B-it",
            "r_f": 1,
            "response_format_instruction_version": "v2",
            "replicate_id": 7,
            "calibration_manifest_freeze_commit": "43f9778",
            "batch_ids": batch_ids,
            "raw_output_sha256_by_partition": raw_output_sha256,
            "scoring_script": "scripts/score_f_calibration_production.py",
            "scorer_module": "ate.f_calibration_validation.score_f_calibration_production_from_raw (synthetic-tested before submission)",
            "loso_fit_module": "ate.calibrate_lambda.fit_calibration_model_comparison (unmodified, pre-existing)",
            "git_commit": git_commit(),
        },
        "superseded_stale_artifact": {
            "path": "outputs/calibration_production/superseded/stale_development_calibration_selected_model.json",
            "reason": "pre-existing artifact fit against the development/stale synthetic-prediction archive (synthetic_prediction_status=DEVELOPMENT_STALE_REQUIRES_REGENERATION...), before any real F-model calibration-production inference existed",
        },
    }
    LIVE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"FINAL_CALIBRATION_ARTIFACT_SHA256 = {sha256_file(LIVE_PATH)}")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("loso_folds",)}, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
