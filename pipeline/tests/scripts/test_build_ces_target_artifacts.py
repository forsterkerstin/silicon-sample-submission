"""Tests for the CES profiles_core_1000.csv/simulation_roster_17000.csv
materialization step (scripts/build_ces_target_artifacts.py). No LLM calls;
structural verification of the already-materialized/live artifacts on disk,
skipped if not built in this environment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_ces_target_artifacts import PROFILE_PATH, ROSTER_PATH, SUPERSEDED_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not PROFILE_PATH.exists(), reason="CES target artifacts not materialized in this environment")


def test_profiles_core_1000_is_ces_sourced():
    core = pd.read_csv(PROFILE_PATH)
    assert len(core) == 1000
    assert core["latent_profile_id"].is_unique
    assert int((core["gender"] == "Other").sum()) == 8


def test_profiles_core_1000_has_pums_person_weight_alias():
    """build_g_master reads core.get('pums_person_weight') -- the CES
    roster's own ces_commonweight column must be renamed, not dropped."""
    core = pd.read_csv(PROFILE_PATH)
    assert "pums_person_weight" in core.columns
    assert core["pums_person_weight"].notna().all()


def test_roster_stub_matches_design_skeleton_profile_ids():
    roster = pd.read_csv(ROSTER_PATH)
    design = pd.read_csv(REPO_ROOT / "data" / "generated" / "tier1_design_skeleton.csv", usecols=["profile_id"])
    assert len(roster) == 17000
    assert roster["profile_id"].is_unique
    assert set(roster["profile_id"]) == set(design["profile_id"])


def test_old_pums_profiles_core_preserved():
    backup = SUPERSEDED_DIR / "profiles_core_1000.csv"
    assert backup.exists()
    old = pd.read_csv(backup)
    assert len(old) == 1000
    assert int((old["gender"] == "Other").sum()) == 0  # the historical PUMS build never had an Other level


def test_old_simulation_roster_preserved():
    backup = SUPERSEDED_DIR / "simulation_roster_17000.csv"
    assert backup.exists()
    assert len(pd.read_csv(backup)) == 17000
