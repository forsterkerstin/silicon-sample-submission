"""Structural checks on the actual submission-ready G-v2 full-replacement
partitions: excludes exactly the already-submitted smoke ids, zero
collision with the original v1 run / smoke / every other manifest, and
correct total count. Skipped if not built in this environment."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "submission"
REPLACEMENT_STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage"
SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"

pytestmark = pytest.mark.skipif(not SUBMISSION_ROOT.exists(), reason="G-v2 full-replacement submission set not built in this environment")


def _load_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "custom_id" not in (r.fieldnames or []):
            return set()
        return {row["custom_id"] for row in r}


def test_excludes_exactly_the_ten_already_submitted_smoke_ids():
    full_std = _load_ids(REPLACEMENT_STAGE_ROOT / "standard" / "request_manifest.csv")
    full_cons = _load_ids(REPLACEMENT_STAGE_ROOT / "consensus_stage_a" / "request_manifest.csv")
    sub_std = _load_ids(SUBMISSION_ROOT / "standard" / "request_manifest.csv")
    sub_cons = _load_ids(SUBMISSION_ROOT / "consensus_stage_a" / "request_manifest.csv")

    smoke_std = _load_ids(SMOKE_ROOT / "standard" / "request_manifest.csv")
    smoke_cons = _load_ids(SMOKE_ROOT / "consensus_stage_a" / "request_manifest.csv")

    assert len(full_std) - len(sub_std) == 5
    assert len(full_cons) - len(sub_cons) == 5
    assert (full_std - sub_std) == smoke_std
    assert (full_cons - sub_cons) == smoke_cons
    assert sub_std & smoke_std == set()
    assert sub_cons & smoke_cons == set()


def test_total_submission_count_is_16990():
    sub_std = _load_ids(SUBMISSION_ROOT / "standard" / "request_manifest.csv")
    sub_cons = _load_ids(SUBMISSION_ROOT / "consensus_stage_a" / "request_manifest.csv")
    assert len(sub_std) == 15_995
    assert len(sub_cons) == 995
    assert len(sub_std) + len(sub_cons) == 16_990


def test_zero_collision_with_v1_run_smoke_and_every_other_manifest():
    sub_ids = _load_ids(SUBMISSION_ROOT / "standard" / "request_manifest.csv") | _load_ids(SUBMISSION_ROOT / "consensus_stage_a" / "request_manifest.csv")

    v1_std = _load_ids(PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "standard" / "G" / "request_manifest.csv")
    v1_cons = _load_ids(PIPELINE_ROOT / "outputs" / "target_production" / "wave1" / "by_stage" / "consensus_stage_a" / "G" / "request_manifest.csv")
    assert sub_ids & v1_std == set()
    assert sub_ids & v1_cons == set()

    smoke_std = _load_ids(SMOKE_ROOT / "standard" / "request_manifest.csv")
    smoke_cons = _load_ids(SMOKE_ROOT / "consensus_stage_a" / "request_manifest.csv")
    assert sub_ids & smoke_std == set()
    assert sub_ids & smoke_cons == set()

    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "wave1_g_v2_replacement" in path or "g_v2_engineering_smoke" in path:
            continue
        prior |= _load_ids(Path(path))
    # threshold reflects the current repo's known-manifest scan after
    # submission-cleanup removed several deprecated development manifests
    # (approach3 wave1, f_model_screen/f_reliability_* one-offs,
    # together_batch/together_smoke scratch, disabled Consensus-A completion) --
    # the actual collision-safety invariant below is unaffected.
    assert len(prior) > 250_000
    assert sub_ids & prior == set()


# NOTE: a test asserting the submission set has zero overlap with the
# ledger's live submitted_custom_ids was deliberately removed here. It was
# valid only up to the moment of real submission -- once the operator
# actually submitted this exact set (as happened), those ids correctly
# and permanently appear in submitted_custom_ids, so that assertion could
# never hold again. The timeless invariant (zero overlap with the smoke's
# specific 10 ids) is already covered by
# test_excludes_exactly_the_ten_already_submitted_smoke_ids above, which
# does not depend on the ledger's mutable current state.


def test_declared_phases_have_correct_request_stage_and_allowlist_size():
    ledger = json.loads((PIPELINE_ROOT / "outputs" / "target_production" / "target_production_submission_state.json").read_text(encoding="utf-8"))
    phases = ledger["phases"]
    assert phases["standard_g_v2"]["request_stage"] == "standard"
    assert len(phases["standard_g_v2"]["approved_custom_ids"]) == 15_995
    assert phases["consensus_stage_a_g_v2"]["request_stage"] == "consensus_stage_a"
    assert len(phases["consensus_stage_a_g_v2"]["approved_custom_ids"]) == 995
