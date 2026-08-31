"""src/population/audit.py

Computational audit helpers: the raw-input source manifest, build metadata,
exclusion-flow / distribution / duplicate-household reports, and the
NC-EST-workbook / existing-census_cells.csv audit-only comparisons (§11-12,
§23). The narrative write-ups (pums_variable_audit.md, ces_variable_audit.md,
nc_est_audit.md, existing_census_cells_audit.md, population_report.md) are
authored directly as markdown -- this module produces the numbers and tables
they cite, and the two machine-readable manifests.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .io import file_metadata, get_logger, sha256_file

logger = get_logger("audit")

#: (relative path, role, operative) for every file directly under data/,
#: per §6/§23. "operative" distinguishes files the build actually depends on
#: from audit-only/reference files (NC-EST workbook, existing census_cells.csv).
RAW_INPUT_ROLES: list[tuple[str, str, bool]] = [
    ("data/csv_pus.zip", "2024 ACS 1-Year PUMS person-file archive (psam_pusa.csv, psam_pusb.csv)", True),
    ("data/csv_hus.zip", "2024 ACS 1-Year PUMS housing-file archive (HINCP source, joined via SERIALNO)", True),
    ("data/PUMS_Data_Dictionary_2024.pdf", "PUMS variable dictionary (verification source for recodes)", True),
    ("data/CCES24_Common_OUTPUT_vv_topost_final.csv", "2024 CES Common Content respondent-level data", True),
    ("data/CCES24_Common_pre.docx", "2024 CES Common Content pre-election questionnaire", True),
    ("data/CES_2024_GUIDE_vv.pdf", "2024 CES Guide (weight table, codebook appendix)", True),
    ("data/census_cells.csv", "pre-existing 40-cell census template in this repo", False),
    ("data/nc-est2024-asr6h.xlsx", "Census Bureau Vintage 2024 national population estimates workbook", False),
    ("data/ate_archive.csv", "treatment-effect archive for the elicitation pipeline (out of scope here)", False),
    ("data/README.md", "documentation for the elicitation pipeline's data/ directory", False),
]


def build_source_manifest(data_dir: Path | str = "data") -> list[dict[str, Any]]:
    """Describe every raw input file per §23: relative path, filename,
    SHA-256, size, modification timestamp, role, and operative-vs-audit-only
    status. Does not read/modify any of these files beyond hashing them.
    """
    data_dir = Path(data_dir)
    manifest = []
    for rel_path, role, operative in RAW_INPUT_ROLES:
        path = Path(rel_path)
        actual_path = data_dir / Path(*path.parts[1:]) if path.parts and path.parts[0] == "data" else data_dir / path
        manifest.append(file_metadata(actual_path, role=role, operative=operative))
    return manifest


def verify_raw_files_unchanged(baseline_manifest: list[dict[str, Any]]) -> list[str]:
    """Recompute SHA-256 for every file in `baseline_manifest` and return the
    list of paths whose hash changed (empty list == nothing changed). Used
    both by --validate-only and by the end-of-run self-check (§6).
    """
    changed = []
    for entry in baseline_manifest:
        current = sha256_file(entry["path"])
        if current != entry["sha256"]:
            changed.append(entry["path"])
    return changed


def build_metadata_dict(
    master_seed: int,
    rng_spawn_keys: dict[str, list[int]],
    row_counts: dict[str, Any],
    ipf_stats: dict[str, Any],
    milp_status: dict[str, Any],
    ces_selected_variables: list[str],
    ces_selected_weight: str,
    model_settings: dict[str, Any],
    output_hashes: dict[str, str],
    git_commit: str | None,
) -> dict[str, Any]:
    """Assemble build_metadata.json's full content (§23)."""
    import importlib.metadata as importlib_metadata

    packages = ["pandas", "numpy", "scipy", "scikit-learn", "openpyxl", "python-docx", "pypdf", "PyYAML", "joblib"]
    package_versions = {}
    for pkg in packages:
        try:
            package_versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            package_versions[pkg] = None

    from datetime import datetime, timezone

    return {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "package_versions": package_versions,
        "operating_system": platform.platform(),
        "master_seed": master_seed,
        "rng_spawn_keys": rng_spawn_keys,
        "row_counts": row_counts,
        "ipf_convergence": ipf_stats,
        "milp_status": milp_status,
        "ces_selected_variables": ces_selected_variables,
        "ces_selected_weight": ces_selected_weight,
        "model_settings": model_settings,
        "output_sha256": output_hashes,
        "git_commit": git_commit,
    }


def git_commit_hash(repo_dir: Path | str = ".") -> str | None:
    """Best-effort current commit hash if `repo_dir` is inside a Git
    repository; None otherwise (never raises)."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def state_distribution(core_profiles: pd.DataFrame) -> pd.DataFrame:
    return core_profiles.groupby("state_abbr").size().reset_index(name="n").sort_values("n", ascending=False)


def education_distribution(core_profiles: pd.DataFrame) -> pd.DataFrame:
    return core_profiles.groupby("education").size().reset_index(name="n").sort_values("n", ascending=False)


def income_distribution(core_profiles: pd.DataFrame) -> pd.DataFrame:
    return core_profiles.groupby("income").size().reset_index(name="n").sort_values("n", ascending=False)


def party_distribution(core_profiles: pd.DataFrame) -> pd.DataFrame:
    realized = core_profiles.groupby("party").size().reset_index(name="realized_n")
    expected = core_profiles[["party_prob_democrat", "party_prob_republican", "party_prob_independent", "party_prob_other"]].sum()
    expected = expected.rename(
        index={
            "party_prob_democrat": "Democrat",
            "party_prob_republican": "Republican",
            "party_prob_independent": "Independent",
            "party_prob_other": "Other",
        }
    ).reset_index()
    expected.columns = ["party", "expected_n"]
    return expected.merge(realized, on="party", how="left").fillna({"realized_n": 0})


def nonquota_margins(core_profiles: pd.DataFrame) -> pd.DataFrame:
    """Education/income/party/state distributions were not hard quota
    constraints (§10 only constrains gender x age_band x race); this reports
    what was achieved on those dimensions given the ACS-donor-preserved
    correlations, for transparency."""
    frames = []
    for col in ("education", "income", "party", "state_abbr"):
        d = core_profiles.groupby(col).size().reset_index(name="n")
        d.insert(0, "dimension", col)
        d = d.rename(columns={col: "level"})
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def duplicate_household_report(core_profiles: pd.DataFrame) -> pd.DataFrame:
    """Selected donors sharing a SERIALNO (i.e. two selected profiles came
    from the same PUMS household) -- informational, not an error: PUMS
    persons are sampled independently across cells, so this can legitimately
    happen. Reports any such duplicates for transparency."""
    dup_serial = core_profiles["SERIALNO"].value_counts()
    dup_serial = dup_serial[dup_serial > 1]
    return core_profiles.loc[core_profiles["SERIALNO"].isin(dup_serial.index)].sort_values("SERIALNO")


def audit_nc_est_workbook(xlsx_path: Path | str) -> dict[str, Any]:
    """Inspect the NC-EST workbook's structure only (sheet names, dimensions)
    for the audit-only report -- never used to alter operative quotas (§12).
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    return {"path": str(xlsx_path), "sheet_names": wb.sheetnames, "sha256": sha256_file(xlsx_path)}


def audit_existing_census_cells(csv_path: Path | str) -> dict[str, Any]:
    """Inspect the pre-existing data/census_cells.csv template's schema/row
    count only, for the audit-only report -- never used as an operative
    target (§11)."""
    df = pd.read_csv(csv_path)
    return {
        "path": str(csv_path),
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "sha256": sha256_file(csv_path),
    }
