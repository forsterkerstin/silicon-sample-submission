"""Tests confirming scripts/assemble_target_g_wave1_first_valid.py is
standard-only: no Consensus-A CLI argument or path is required or
referenced. The underlying round-building/ledger-merge logic is shared
with (and already tested via) scripts/score_target_g_wave1_completion.py's
round_from_report, exercised in test_target_g_completion_reconciliation.py
-- not duplicated here."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import assemble_target_g_wave1_first_valid as assembler  # noqa: E402

pytestmark = pytest.mark.skipif(not assembler.PROVENANCE_PATH.exists(), reason="target G standard completion provenance not built in this environment")


def test_assembler_never_requires_consensus_stage_a_as_input():
    source = inspect.getsource(assembler.main)
    assert "--consensus" not in source
    assert not hasattr(assembler, "STAGES")
    assert assembler.STAGE == "standard"


def test_assembler_reuses_shared_round_builder():
    import score_target_g_wave1_completion as reconciler

    assert assembler.reconciler is reconciler
    assert hasattr(reconciler, "round_from_report")
