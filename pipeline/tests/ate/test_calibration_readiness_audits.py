from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[2]


def test_calibration_readiness_audit_writes_population_and_sign_outputs():
    result = subprocess.run(
        [sys.executable, str(PIPELINE_ROOT / "scripts" / "audit_calibration_readiness.py")],
        cwd=PIPELINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = PIPELINE_ROOT / "outputs" / "calibration"
    availability_path = out / "population_alignment_availability.csv"
    sign_path = out / "sign_alignment_audit.csv"
    summary_path = out / "sign_alignment_summary.json"
    plot_path = out / "current_raw_f_vs_human.png"
    assert availability_path.exists()
    assert sign_path.exists()
    assert summary_path.exists()
    assert plot_path.exists()

    availability = pd.read_csv(availability_path)
    assert len(availability) == 31
    assert {
        "study_id",
        "respondent_demographics_available",
        "survey_weights_available",
        "currently_used_method",
        "preferred_method_under_protocol",
        "action_needed",
    } <= set(availability.columns)

    sign = pd.read_csv(sign_path)
    assert len(sign) == 136
    assert {
        "human_ate_native",
        "synthetic_ate_native",
        "human_ate_pp",
        "synthetic_ate_pp",
        "treatment_label",
        "control_label",
        "alignment_status",
    } <= set(sign.columns)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_effects"] == 136
    assert "slope_through_origin" in summary
