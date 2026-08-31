"""Structural verification of the already-built target Wave-1 manifests
(scripts/prepare_target_wave1_manifest.py) -- reads the real materialized
CSVs/JSONLs under outputs/target_production/wave1/. No submission, no new
construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
WAVE1_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1"

pytestmark = pytest.mark.skipif(not WAVE1_ROOT.exists(), reason="target Wave-1 manifest not built in this environment")


def _manifest(role: str) -> pd.DataFrame:
    return pd.read_csv(WAVE1_ROOT / role / "google_gemma-4-31B-it" / "request_manifest.csv")


def test_g_wave1_is_exactly_17000_standard_plus_consensus_stage_a():
    g = _manifest("G")
    assert len(g) == 17000
    assert g["custom_id"].is_unique
    assert set(g["request_stage"]) == {"standard", "consensus_stage_a"}
    assert int((g["request_stage"] == "consensus_stage_a").sum()) == 1000
    assert int((g["request_stage"] == "standard").sum()) == 16000
    assert set(g["study_id"]) == {"target"}


def test_f_wave1_is_exactly_104500_standard_plus_consensus_stage_a():
    f = _manifest("F")
    assert len(f) == 104500
    assert f["custom_id"].is_unique
    assert set(f["request_stage"]) == {"standard", "consensus_stage_a"}
    assert int((f["request_stage"] == "consensus_stage_a").sum()) == 500
    assert int((f["request_stage"] == "standard").sum()) == 104000
    assert set(f["study_id"]) == {"target"}


def test_wave1_zero_overlap_with_prior_scientific_bakeoff_ledger():
    g = _manifest("G")
    f = _manifest("F")
    target_ids = set(g["custom_id"]) | set(f["custom_id"])
    state_path = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "scientific_bakeoff_submission_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    prev_ids = set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", []))
    assert not (target_ids & prev_ids)


def test_wave1_partitions_reconstruct_exactly():
    summary = json.loads((WAVE1_ROOT / "summary.json").read_text(encoding="utf-8"))
    for role in ("G", "F"):
        info = summary[role]
        model_dir = info["model"].replace("/", "_")
        out_dir = WAVE1_ROOT / role / model_dir
        total_requests = sum(p["requests"] for p in info["partitions"])
        assert total_requests == info["requests"]
        reconstructed_ids = set()
        for p in info["partitions"]:
            part_manifest = pd.read_csv(out_dir / p["partition"] / "request_manifest.csv")
            reconstructed_ids |= set(part_manifest["custom_id"])
        canonical_ids = set(pd.read_csv(out_dir / "request_manifest.csv")["custom_id"])
        assert reconstructed_ids == canonical_ids


def test_wave1_partition_sizes_under_operational_ceiling():
    summary = json.loads((WAVE1_ROOT / "summary.json").read_text(encoding="utf-8"))
    for role in ("G", "F"):
        for p in summary[role]["partitions"]:
            assert p["jsonl_size_mb"] < 95
            assert p["requests"] < 50_000
