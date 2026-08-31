"""Tests for the target-F Wave-1 reclassification provenance note. Confirms
the real submission ledger is never modified and the note correctly
identifies exactly the 9 real target-F Wave-1 batches (8 standard + 1
consensus_stage_a) already established elsewhere this session."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reclassify_target_f_wave1_provenance as reclass_mod  # noqa: E402


def test_ledger_untouched():
    before = reclass_mod.LEDGER_PATH.read_bytes()
    reclass_mod.main()
    assert reclass_mod.LEDGER_PATH.read_bytes() == before


def test_identifies_exactly_nine_f_wave1_batches():
    result = reclass_mod.main()
    assert result["target_f_wave1_batch_count"] == 9
    standard = [b for b in result["target_f_wave1_batches"] if b["phase"] == "standard"]
    consensus = [b for b in result["target_f_wave1_batches"] if b["phase"] == "consensus_stage_a"]
    assert len(standard) == 8
    assert all(b["request_count"] == 13000 for b in standard)
    assert len(consensus) == 1
    assert consensus[0]["request_count"] == 500


def test_status_and_usage_fields():
    result = reclass_mod.main()
    assert result["status_per_batch"] == "GENERATED_UNDER_SUPERSEDED_CANDIDATE_METHOD"
    assert result["usage_for_final_s2_predictions"] == "UNUSED_FOR_FINAL_S2_PREDICTIONS"
    assert result["target_f_dependence_of_final_method"] is False
    assert result["ledger_untouched_by_this_script"] is True
