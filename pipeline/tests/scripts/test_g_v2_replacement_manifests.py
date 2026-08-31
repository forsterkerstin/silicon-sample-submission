"""Structural checks on the already-built (never submitted) G-v2 format-fix
replacement manifests: exact 17,000-request total, correct standard/
consensus_stage_a split, zero collision with the original v1 run and every
other manifest in outputs/, and the frozen amendment artifact's required
fields. Skipped if the manifests are not present in this environment."""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

V2_STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage"
AMENDMENT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_format_failure_amendment.json"

pytestmark = pytest.mark.skipif(not V2_STAGE_ROOT.exists(), reason="G-v2 replacement manifests not built in this environment")


def _load_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "custom_id" not in (r.fieldnames or []):
            return set()
        return {row["custom_id"] for row in r}


def test_standard_16000_consensus_a_1000():
    assert len(_load_ids(V2_STAGE_ROOT / "standard" / "request_manifest.csv")) == 16_000
    assert len(_load_ids(V2_STAGE_ROOT / "consensus_stage_a" / "request_manifest.csv")) == 1_000


def test_zero_collision_with_original_v1_run_and_every_other_manifest():
    v1_standard = _load_ids(PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "standard" / "G" / "request_manifest.csv")
    v1_consensus = _load_ids(PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "consensus_stage_a" / "G" / "request_manifest.csv")
    v2_ids = _load_ids(V2_STAGE_ROOT / "standard" / "request_manifest.csv") | _load_ids(V2_STAGE_ROOT / "consensus_stage_a" / "request_manifest.csv")
    assert v2_ids & v1_standard == set()
    assert v2_ids & v1_consensus == set()

    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "wave1_g_v2_replacement" in path or "g_v2_engineering_smoke" in path:
            continue  # both ARE the G-v2 replacement effort itself, not independent prior manifests
        prior |= _load_ids(Path(path))
    assert len(prior) > 250_000  # sanity: really scanned the full prior manifest history (threshold lowered after submission-cleanup removed several deprecated development manifests; the actual collision invariant below is unaffected)
    assert v2_ids & prior == set()


def test_amendment_artifact_records_required_fields():
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    assert amendment["root_cause_classification"] == "PROVIDER_SERVING_FORMAT_FAILURE"
    assert amendment["failed_backend_fingerprint"] == "vllm-0.21.0-8326ea74"
    assert amendment["original_standard_invalid"] == 14720
    assert amendment["original_consensus_a_invalid"] == 980
    assert amendment["scientific_target_outputs_inspected"] is False
    assert amendment["scientific_prompt_content_changed"] is False
    assert amendment["replacement_uses_fresh_draws"] is True
    assert amendment["original_failed_run_preserved"] is True
    assert amendment["full_replacement"]["total_requests"] == 17_000
    assert amendment["full_replacement"]["zero_collision_with_original_v1_run"] is True
    assert amendment["g_v2_format_only_change"]["scientific_prompt_equivalence_verified"] is True
    assert amendment["new_paid_inference_performed"] is False


def test_smoke_manifest_is_small_and_covers_both_stages():
    smoke_root = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
    summary = json.loads((smoke_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_requests"] == 10
    assert summary["per_stage"]["standard"]["requests"] == 5
    assert summary["per_stage"]["consensus_stage_a"]["requests"] == 5
