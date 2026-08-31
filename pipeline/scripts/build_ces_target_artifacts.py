#!/usr/bin/env python3
"""scripts/build_ces_target_artifacts.py

Materializes data/processed/population/profiles_core_1000.csv and
simulation_roster_17000.csv from the frozen all-CES N=1000 roster
(config/population.yaml population_source_amendments,
data/derived/population/ces_production_roster_n1000.csv), preserving the
historical PUMS versions of both files, then runs the now CES/Other-aware
scripts/validate_personas.py -- the SAME script every other build/test path
already uses -- to build g_personas_master.csv, f_target_panel.csv, and
both 17-condition skeletons at their normal live paths.

This is deliberately a thin materialization step, not a second build
pipeline: profiles_core_1000.csv is a pure, deterministic RESHAPE of the
already-frozen CES roster (rename ces_commonweight -> pums_person_weight;
no new donor selection, no new randomness) -- validate_personas.py's own
build_g_master/build_f_panel_with_other/expand_skeleton (all reused
unmodified) do everything else, exactly as they do for the historical PUMS
core. Because those functions are now Other-aware and n_other=0 is an exact
no-op for the rescaled-quota logic, running validate_personas.py against a
PUMS-shaped profiles_core_1000.csv (if one is ever restored) reproduces the
historical PUMS build byte-for-byte; running it against this CES-shaped one
reproduces the CES build -- one script, one behavior, driven entirely by
what's on disk.

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (REPO_ROOT / "src", REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

CES_ROSTER_PATH = REPO_ROOT / "data" / "derived" / "population" / "ces_production_roster_n1000.csv"
POPULATION_CONFIG_PATH = REPO_ROOT / "config" / "population.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "population"
PROFILE_PATH = PROCESSED_DIR / "profiles_core_1000.csv"
ROSTER_PATH = PROCESSED_DIR / "simulation_roster_17000.csv"
SUPERSEDED_DIR = PROCESSED_DIR / "superseded_pums"


class CutoverBlocked(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_ces_source() -> pd.DataFrame:
    pop_cfg = yaml.safe_load(POPULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    amendments = [a for a in pop_cfg["population_source_amendments"] if a.get("amendment_type") == "TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN"]
    if not amendments:
        raise CutoverBlocked("no TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN amendment found in config/population.yaml")
    expected_sha = amendments[-1]["frozen_roster"]["sha256"]
    actual_sha = sha256_file(CES_ROSTER_PATH)
    if actual_sha != expected_sha:
        raise CutoverBlocked(f"CES roster on disk (sha256={actual_sha}) does not match the frozen amendment (sha256={expected_sha}) -- refusing to build from an unfrozen source")
    df = pd.read_csv(CES_ROSTER_PATH)
    if len(df) != 1000 or df["donor_id"].nunique() != 1000:
        raise CutoverBlocked(f"CES roster has {len(df)} rows / {df['donor_id'].nunique()} unique donors, expected 1000/1000")
    return df


def materialize_profiles_core(ces_roster: pd.DataFrame) -> None:
    """Deterministic reshape only -- no new donor selection."""
    core = ces_roster.rename(columns={"ces_commonweight": "pums_person_weight"})
    SUPERSEDED_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = SUPERSEDED_DIR / "profiles_core_1000.csv"
    if PROFILE_PATH.exists() and not backup_path.exists():
        backup_path.write_bytes(PROFILE_PATH.read_bytes())
    core.to_csv(PROFILE_PATH, index=False)


def materialize_roster_stub(design_profile_ids: pd.Series) -> None:
    """simulation_roster_17000.csv is consumed by validate_personas.py's
    duplication_audit ONLY for profile_id uniqueness (verified: no
    cross-comparison against other artifacts) -- the 17-condition design
    skeleton's own profile_id column already satisfies that, and IS what
    'simulation_roster_17000' conceptually represents (17,000 condition-level
    rows), so it is reused directly rather than re-deriving a second,
    parallel expansion."""
    SUPERSEDED_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = SUPERSEDED_DIR / "simulation_roster_17000.csv"
    if ROSTER_PATH.exists() and not backup_path.exists():
        backup_path.write_bytes(ROSTER_PATH.read_bytes())
    pd.DataFrame({"profile_id": design_profile_ids}).to_csv(ROSTER_PATH, index=False)


def main() -> int:
    ces_roster = verify_frozen_ces_source()
    materialize_profiles_core(ces_roster)

    # Run validate_personas.py once (real subprocess, identical to how the
    # test fixture invokes it) to get a real design skeleton for the roster
    # stub, then run it again for the actual live build -- both runs are
    # pure functions of profiles_core_1000.csv, so this is deterministic
    # and idempotent, not a hidden second build path.
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "validate_personas.py")], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise CutoverBlocked("validate_personas.py failed on first pass (before roster stub materialization)")

    design = pd.read_csv(REPO_ROOT / "data" / "generated" / "tier1_design_skeleton.csv", usecols=["profile_id"])
    materialize_roster_stub(design["profile_id"])

    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "validate_personas.py")], cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise CutoverBlocked("validate_personas.py failed on final pass (after roster stub materialization)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CutoverBlocked as exc:
        print(f"BLOCKED: {exc}")
        raise SystemExit(1)
