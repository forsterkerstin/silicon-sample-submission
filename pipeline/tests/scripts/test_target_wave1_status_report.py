"""Structural tests for the offline, read-only target Wave-1 status report.
Confirms it reads only the local ledger/summary (no network calls, no
TOGETHER_API_KEY needed), reports exactly the 14 real submitted partitions,
and correctly identifies the two consensus_stage_a batches (G and F)."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from target_wave1_status_report import build_status_table  # noqa: E402


def test_reports_exactly_fourteen_partitions_totaling_121500():
    result = build_status_table()
    assert result["total_partitions"] == 14
    assert result["total_requests"] == 121_500


def test_identifies_both_consensus_stage_a_batches_uniquely():
    result = build_status_table()
    g = result["consensus_stage_a"]["G"]
    f = result["consensus_stage_a"]["F"]
    assert g["role"] == "G" and g["stage"] == "consensus_stage_a" and g["request_count"] == 1000
    assert f["role"] == "F" and f["stage"] == "consensus_stage_a" and f["request_count"] == 500
    assert g["batch_id"] != f["batch_id"]
    assert g["part_label"] == "part1"
    assert f["part_label"] == "part1"


def test_role_split_matches_authorized_wave1_shape():
    result = build_status_table()
    g_rows = [r for r in result["rows"] if r["role"] == "G"]
    f_rows = [r for r in result["rows"] if r["role"] == "F"]
    assert len(g_rows) == 5  # 4 standard + 1 consensus_stage_a
    assert len(f_rows) == 9  # 8 standard + 1 consensus_stage_a
    assert sum(r["request_count"] for r in g_rows) == 17_000
    assert sum(r["request_count"] for r in f_rows) == 104_500


def test_every_row_has_a_status_query_command():
    result = build_status_table()
    assert len(result["status_query_commands"]) == 14
    for cmd in result["status_query_commands"]:
        assert cmd.startswith("python scripts/together_batch.py status --batch-id ")
