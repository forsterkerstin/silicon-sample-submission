"""Structural verification of the real, already-retrieved and scored
136,000-request calibration-production batch and the resulting frozen
calibration artifact / final method manifest / re-partitioned target
Wave-1 guard state. Reads real outputs on disk; skipped if not present in
this environment. No submission, no new construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CALIB_OUT = PIPELINE_ROOT / "outputs" / "calibration_production"
CALIBRATION_MODEL_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
FROZEN_METHOD_MANIFEST_PATH = PIPELINE_ROOT / "outputs" / "validation" / "frozen_method_manifest.json"
STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage"

pytestmark = pytest.mark.skipif(not CALIB_OUT.exists(), reason="calibration-production not scored in this environment")


def test_reconciliation_clean_except_one_known_missing():
    result = json.loads((CALIB_OUT / "calibration_scoring_result.json").read_text(encoding="utf-8"))
    rec = result["reconciliation"]
    assert rec["integrity_ok"] is True
    assert rec["unexpected"] == []
    assert rec["duplicate"] == []
    assert len(rec["missing_entirely"]) == 1


def test_136_effects_scored_with_expected_paired_n():
    table = pd.read_csv(CALIB_OUT / "frozen_136_effect_calibration_table.csv")
    assert len(table) == 136
    assert table["study_id"].nunique() == 31
    assert (table["paired_n"] <= 500).all()
    assert (table["paired_n"] >= 499).all()
    assert table["theta_H_pp"].notna().all()


def test_calibration_selected_model_usable_for_production():
    model = json.loads(CALIBRATION_MODEL_PATH.read_text(encoding="utf-8"))
    assert model["usable_for_production"] is True
    assert model["model_name"] in {"M0", "M1", "M2"}
    assert model["number_effects"] == 136
    assert model["number_studies"] == 31
    assert "provenance" in model
    assert model["provenance"]["calibration_manifest_freeze_commit"] == "43f9778"


def test_calibration_tie_break_order_respected():
    model = json.loads(CALIBRATION_MODEL_PATH.read_text(encoding="utf-8"))
    rmse = {"M0": model["loso_rmse_M0"], "M1": model["loso_rmse_M1"], "M2": model["loso_rmse_M2"]}
    best = min(rmse.values())
    winners = [name for name, v in rmse.items() if round(v, 15) == round(best, 15)]
    order = ["M0", "M1", "M2"]
    expected = min(winners, key=order.index)
    assert model["model_name"] == expected


def test_final_method_manifest_binds_calibration_and_frozen_decisions():
    manifest = json.loads(FROZEN_METHOD_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["selected_G_model"] == "google/gemma-4-31B-it"
    assert manifest["selected_F_model"] == "google/gemma-4-31B-it"
    assert manifest["F_stochastic_draw_count"] == 1
    assert manifest["N_G"] == 1000
    assert manifest["N_F"] == 500
    model = json.loads(CALIBRATION_MODEL_PATH.read_text(encoding="utf-8"))
    assert manifest["selected_calibration_model"] == model["model_name"]
    assert manifest["final_alpha"] == model["alpha"]
    assert manifest["final_lambda"] == model["lambda"]


def test_target_production_prerequisites_now_frozen():
    from inference.target_production_guard import assert_target_production_prerequisites_frozen

    result = assert_target_production_prerequisites_frozen()
    assert result["selected_f_model"] == "google/gemma-4-31B-it"
    assert result["selected_g_model"] == "google/gemma-4-31B-it"


@pytest.mark.skipif(not STAGE_ROOT.exists(), reason="stage-repartitioned Wave-1 not built")
def test_stage_repartitioned_wave1_counts_match():
    summary = json.loads((STAGE_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["G_standard"]["requests"] == 16000
    assert summary["G_consensus_stage_a"]["requests"] == 1000
    assert summary["F_standard"]["requests"] == 104000
    assert summary["F_consensus_stage_a"]["requests"] == 500


@pytest.mark.skipif(not STAGE_ROOT.exists(), reason="stage-repartitioned Wave-1 not built")
def test_stage_repartitioned_wave1_every_partition_stage_pure():
    for role in ("G", "F"):
        for stage in ("standard", "consensus_stage_a"):
            manifest = pd.read_csv(STAGE_ROOT / stage / role / "request_manifest.csv")
            assert set(manifest["request_stage"]) == {stage}
            assert set(manifest["study_id"]) == {"target"}
