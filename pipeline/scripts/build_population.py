#!/usr/bin/env python3
"""scripts/build_population.py

CLI entry point for the population/simulation-roster construction pipeline.
Orchestrates src/population/{pums,ces,raking,sampling,roster,audit}.py; all
substantive logic lives there, not here.

Usage:
    python scripts/build_population.py --config config/population.yaml
    python scripts/build_population.py --config config/population.yaml --audit-inputs-only
    python scripts/build_population.py --config config/population.yaml --validate-only

Scope: population and simulation-roster construction only. Never generates
LLM survey responses, never estimates treatment effects, never touches
inference/ / ate/, never writes into the benchmark's predictions output
directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from population import audit, ces, pums, raking, roster, sampling  # noqa: E402
from population.constants import load_benchmark_schema, spawn_rngs, validate_schema_against  # noqa: E402
from population.io import configure_logging, ensure_dir, get_logger, read_json, sha256_file, write_json  # noqa: E402

logger = get_logger("cli")


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_quota_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the gender x age_band and gender x race quota tables sized to
    `cfg["population"]["n_profiles"]` (config/quota_gender_{age,race}_<n>.csv).
    Validates both tables sum to n_profiles and agree with each other on the
    Male/Female split (rather than a hardcoded number), so this works for
    any n_profiles, not just the original 1,000-profile design.
    """
    n_profiles = cfg["population"]["n_profiles"]
    quota_age = pd.read_csv(REPO_ROOT / "config" / f"quota_gender_age_{n_profiles}.csv")
    quota_race = pd.read_csv(REPO_ROOT / "config" / f"quota_gender_race_{n_profiles}.csv")
    if quota_age["target_n"].sum() != n_profiles:
        raise ValueError(f"quota_gender_age_{n_profiles}.csv must total {n_profiles}, got {quota_age['target_n'].sum()}")
    if quota_race["target_n"].sum() != n_profiles:
        raise ValueError(f"quota_gender_race_{n_profiles}.csv must total {n_profiles}, got {quota_race['target_n'].sum()}")
    for gender in ("Male", "Female"):
        age_total = int(quota_age.loc[quota_age["gender"] == gender, "target_n"].sum())
        race_total = int(quota_race.loc[quota_race["gender"] == gender, "target_n"].sum())
        if age_total != race_total:
            raise ValueError(f"quota_gender_age_{n_profiles}.csv and quota_gender_race_{n_profiles}.csv disagree on {gender} total: {age_total} vs {race_total}")
    schema = load_benchmark_schema(REPO_ROOT / "config" / "benchmark_schema.yaml")
    for table, col in ((quota_age, "age_band"), (quota_race, "race")):
        bad = set(table[col]) - set(schema["moderators"][col])
        if bad:
            raise ValueError(f"quota table has label(s) not in benchmark_schema.yaml moderators.{col}: {bad}")
    return quota_age, quota_race


def run_audit_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    """§27 --audit-inputs-only: inspect files and documentation, generate
    variable audits, do not build profiles."""
    paths = cfg["paths"]
    reports_dir = ensure_dir(REPO_ROOT / paths["reports_dir"])
    derived_dir = ensure_dir(REPO_ROOT / paths["derived_dir"])

    manifest = audit.build_source_manifest(REPO_ROOT / paths["data_dir"])
    write_json(derived_dir / "source_manifest.json", manifest)
    logger.info("wrote source_manifest.json (%d raw input files hashed)", len(manifest))

    pums_header_report = _validate_headers_only(REPO_ROOT / paths["pums_zip"], pums.find_person_file_members, pums.CANONICAL_TO_ACTUAL, "person")
    housing_header_report = _validate_headers_only(REPO_ROOT / paths["pums_housing_zip"], pums.find_housing_file_members, pums.HOUSING_CANONICAL_TO_ACTUAL, "housing")

    nc_est_report = audit.audit_nc_est_workbook(REPO_ROOT / paths["nc_est_workbook"])
    census_cells_report = audit.audit_existing_census_cells(REPO_ROOT / paths["existing_census_cells"])

    ces_csv_path = _resolve_ces_csv(paths["ces_csv_glob"])
    ces_raw = ces.load_ces(ces_csv_path)
    ces_training = ces.build_ces_training_frame(ces_raw)

    audit_summary = {
        "pums_header_validation": pums_header_report,
        "housing_header_validation": housing_header_report,
        "nc_est_workbook": nc_est_report,
        "existing_census_cells": census_cells_report,
        "ces_csv_path": str(ces_csv_path),
        "ces_n_rows_total": len(ces_raw),
        "ces_n_rows_valid_for_training": len(ces_training),
    }
    write_json(derived_dir / "audit_inputs_summary.json", audit_summary)
    logger.info("audit-inputs-only complete; see %s", derived_dir / "audit_inputs_summary.json")
    return audit_summary


def _validate_headers_only(zip_path: Path, find_members_fn, column_map: dict[str, str], label: str) -> dict[str, Any]:
    """Header-only validation for one archive (person or housing), used by
    --audit-inputs-only so it can report the finding without reading any row
    data and without crashing the whole audit run on a real, expected gap.
    """
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = find_members_fn(zf)
            for member in members:
                pums.validate_header(zf, member, column_map)
        logger.info("%s-file header validation: all required columns present in %s", label, members)
        return {"status": "all_required_columns_present", "members": members}
    except pums.PumsColumnError as e:
        logger.warning("%s-file header validation: %s", label, e)
        return {"status": "missing_required_column", "member": e.member, "missing_canonical": e.missing_canonical, "missing_actual": e.missing_actual}
    except pums.PumsIngestionError as e:
        logger.warning("%s-file header validation: %s", label, e)
        return {"status": "ingestion_error", "detail": str(e)}


def _resolve_ces_csv(pattern: str) -> Path:
    import glob

    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    exact = REPO_ROOT / "data" / "CCES24_Common_OUTPUT_vv_topost_final.csv"
    if exact.exists():
        return exact
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one CES CSV matching {pattern!r}, found {len(matches)}: {matches}")
    return Path(matches[0])


def run_full_build(cfg: dict[str, Any]) -> int:
    """Normal mode: audit inputs, build the full population, train the party
    model, create the roster, run validations, write reports. Returns the
    process exit code (0 success, nonzero on any failure -- including a
    structural data gap that blocks construction, per §8's fail-fast rule).
    """
    paths = cfg["paths"]
    pop_cfg = cfg["population"]
    processed_dir = ensure_dir(REPO_ROOT / paths["processed_dir"])
    derived_dir = ensure_dir(REPO_ROOT / paths["derived_dir"])
    reports_dir = ensure_dir(REPO_ROOT / paths["reports_dir"])
    models_dir = ensure_dir(REPO_ROOT / paths["models_dir"])

    quota_age, quota_race = load_quota_tables(cfg)
    schema = load_benchmark_schema(REPO_ROOT / "config" / "benchmark_schema.yaml")
    generators, spawn_keys = spawn_rngs(pop_cfg["master_seed"])

    audit_summary = run_audit_inputs(cfg)

    row_counts: dict[str, Any] = {"ces_n_rows_total": audit_summary["ces_n_rows_total"], "ces_n_rows_valid_for_training": audit_summary["ces_n_rows_valid_for_training"]}

    # --- CES: train + diagnose + refit (this genuinely works end-to-end) ---
    ces_raw = ces.load_ces(_resolve_ces_csv(paths["ces_csv_glob"]))
    ces_training = ces.build_ces_training_frame(ces_raw)
    diagnostics = ces.run_diagnostics(ces_training, generators["ces_diagnostic_split"], test_fraction=cfg["ces_party_model"]["diagnostic_test_fraction"])
    write_json(reports_dir / "party_model_diagnostics.json", diagnostics)
    pd.DataFrame(diagnostics["weighted_confusion_matrix"], index=diagnostics["classes"], columns=diagnostics["classes"]).to_csv(reports_dir / "party_confusion_matrix.csv")
    party_model = ces.fit_final_model(ces_training, regularization_c=cfg["ces_party_model"]["regularization_c"], max_iterations=cfg["ces_party_model"]["max_iterations"])
    import joblib

    joblib.dump(party_model, models_dir / "ces_party_model.joblib")
    ces_training[["harmonized_education", "harmonized_income_ces", "party"]].to_csv(reports_dir / "party_mapping.csv", index=False)
    logger.info("CES party model fit and diagnosed; weighted log loss=%.4f, weighted accuracy=%.4f", diagnostics["weighted_log_loss"], diagnostics["weighted_accuracy"])

    # --- PUMS: person file + housing file (HINCP), joined via SERIALNO ---
    try:
        pums_raw, ingestion_report = pums.read_pums(REPO_ROOT / paths["pums_zip"], REPO_ROOT / paths["pums_housing_zip"])
    except (pums.PumsColumnError, pums.PumsIngestionError) as e:
        logger.error("FATAL: PUMS ingestion cannot proceed: %s", e)
        fatal_error: dict[str, Any] = {"stage": "pums_ingestion", "type": type(e).__name__, "message": str(e)}
        if isinstance(e, pums.PumsColumnError):
            fatal_error |= {"member": e.member, "missing_canonical": e.missing_canonical, "missing_actual": e.missing_actual}
        write_json(
            derived_dir / "build_metadata.json",
            audit.build_metadata_dict(
                master_seed=pop_cfg["master_seed"], rng_spawn_keys=spawn_keys, row_counts=row_counts,
                ipf_stats={"status": "not_reached"}, milp_status={"status": "not_reached"},
                ces_selected_variables=ces.REQUIRED_COLUMNS, ces_selected_weight="commonweight",
                model_settings=cfg["ces_party_model"], output_hashes={},
                git_commit=audit.git_commit_hash(REPO_ROOT),
            ) | {"fatal_error": fatal_error},
        )
        print("\n=== FATAL: population build cannot proceed ===")
        print(f"PUMS ingestion failed: {e}")
        print("See reports/population/pums_variable_audit.md.")
        return 1

    row_counts["pums_person_rows_ingested"] = ingestion_report["n_rows_combined"]
    row_counts["pums_housing_rows_ingested"] = ingestion_report["housing"]["n_rows_combined"]
    row_counts["pums_housing_join_match_rate"] = ingestion_report["join"]["match_rate"]

    filtered, exclusion_flow = pums.apply_inclusion_filters(pums_raw)
    pd.DataFrame(exclusion_flow).to_csv(reports_dir / "exclusion_flow.csv", index=False)
    row_counts["pums_rows_after_inclusion_filters"] = len(filtered)
    recoded = pums.recode_pums(filtered, reference_year=pop_cfg["reference_year"])

    joint_cells = raking.build_joint_cells_table(recoded, quota_age, quota_race, cfg["ipf"]["tolerance"], cfg["ipf"]["max_iterations"])
    joint_cells.to_csv(derived_dir / "joint_cells_40.csv", index=False)

    selected = sampling.sample_donors(recoded, joint_cells, generators["pums_selection"])
    profiles = sampling.assign_latent_profile_ids(selected)
    profiles = pd.concat([profiles, ces.predict_party_probabilities(party_model, _with_harmonized_predictors(profiles))], axis=1)
    profiles["party"] = sampling.assign_party(profiles, generators["party_sampling"])
    n_profiles = pop_cfg["n_profiles"]
    core_profiles = sampling.build_core_profiles(profiles, pop_cfg["master_seed"], spawn_keys["party_sampling"], n_profiles)
    profiles_filename = f"profiles_core_{n_profiles}.csv"
    core_profiles.to_csv(processed_dir / profiles_filename, index=False)

    quota_audit_result = sampling.quota_audit(core_profiles, quota_age, quota_race)
    quota_audit_result["gender_age"].to_csv(reports_dir / "quota_audit_gender_age.csv", index=False)
    quota_audit_result["gender_race"].to_csv(reports_dir / "quota_audit_gender_race.csv", index=False)

    roster_df = roster.build_simulation_roster(core_profiles, schema["conditions"], cfg["roster"]["intervention_replicates"], cfg["roster"]["control_replicates"])
    roster_filename = f"simulation_roster_{len(roster_df)}.csv"
    roster_df.to_csv(processed_dir / roster_filename, index=False)

    output_hashes = {
        profiles_filename: sha256_file(processed_dir / profiles_filename),
        roster_filename: sha256_file(processed_dir / roster_filename),
    }
    write_json(
        derived_dir / "build_metadata.json",
        audit.build_metadata_dict(
            master_seed=pop_cfg["master_seed"], rng_spawn_keys=spawn_keys, row_counts=row_counts,
            ipf_stats={"iterations": int(joint_cells["ipf_iterations"].max()), "max_error": float(joint_cells["ipf_max_error"].max())},
            milp_status={"status": "optimal"}, ces_selected_variables=ces.REQUIRED_COLUMNS, ces_selected_weight="commonweight",
            model_settings=cfg["ces_party_model"], output_hashes=output_hashes, git_commit=audit.git_commit_hash(REPO_ROOT),
        ),
    )

    max_quota_error = max(
        (quota_audit_result["gender_age"]["achieved_n"] - quota_audit_result["gender_age"]["target_n"]).abs().max(),
        (quota_audit_result["gender_race"]["achieved_n"] - quota_audit_result["gender_race"]["target_n"]).abs().max(),
    )
    print("\n=== population build complete ===")
    print(f"PUMS person rows selected: {len(core_profiles)} (from {row_counts['pums_person_rows_ingested']} ingested, {row_counts['pums_rows_after_inclusion_filters']} after inclusion filters)")
    print(f"CES rows selected: {row_counts['ces_n_rows_valid_for_training']} (of {row_counts['ces_n_rows_total']} total)")
    print("party model weight: commonweight")
    print(f"party model weighted log loss: {diagnostics['weighted_log_loss']:.4f}")
    print(f"IPF convergence: {int(joint_cells['ipf_iterations'].max())} max iterations, {float(joint_cells['ipf_max_error'].max()):.2e} max residual error")
    print(f"unique profiles: {core_profiles['latent_profile_id'].nunique()}")
    print(f"exact quota error (max |achieved - target|): {max_quota_error}")
    print(f"roster row count: {len(roster_df)}")
    print(f"output directory: {processed_dir}")
    return 0


def _with_harmonized_predictors(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add the two predictors the party model needs beyond what's already on
    a recoded profile (gender/age_band/race/state_abbr): harmonized_education
    (from the donor's raw SCHL code, preserved through sampling) and
    harmonized_income_ces (from the recoded income_adjusted_2024 amount).
    """
    out = profiles.copy()
    out["harmonized_education"] = out["SCHL"].apply(ces.harmonized_education_from_schl)
    out["harmonized_income_ces"] = out["income_adjusted_2024"].apply(ces.harmonized_income_bracket_from_amount)
    return out


def run_validate_only(cfg: dict[str, Any]) -> int:
    """§27 --validate-only: do not retrain or resample; validate existing
    outputs and hashes."""
    paths = cfg["paths"]
    processed_dir = REPO_ROOT / paths["processed_dir"]
    derived_dir = REPO_ROOT / paths["derived_dir"]
    ok = True

    manifest = audit.build_source_manifest(REPO_ROOT / paths["data_dir"])
    changed = audit.verify_raw_files_unchanged(manifest)
    if changed:
        logger.error("raw input file(s) changed since manifest was recorded: %s", changed)
        ok = False
    else:
        logger.info("all raw input files unchanged (%d checked)", len(manifest))

    metadata_path = derived_dir / "build_metadata.json"
    if not metadata_path.exists():
        logger.error("no build_metadata.json found -- the population has never been successfully built (see build_metadata.json's would-be 'fatal_error' key if a failed attempt was recorded)")
        return 1

    metadata = read_json(metadata_path)
    if "fatal_error" in metadata:
        logger.error("last build attempt recorded a fatal error: %s", metadata["fatal_error"])
        return 1

    for name, expected_hash in metadata.get("output_sha256", {}).items():
        path = processed_dir / name
        if not path.exists():
            logger.error("expected output missing: %s", path)
            ok = False
            continue
        actual = sha256_file(path)
        if actual != expected_hash:
            logger.error("hash mismatch for %s: expected %s, got %s", path, expected_hash, actual)
            ok = False
        else:
            logger.info("%s hash OK", path)

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-inputs-only", action="store_true")
    args = parser.parse_args()

    configure_logging()
    cfg = load_config(args.config)

    if args.validate_only and args.audit_inputs_only:
        print("--validate-only and --audit-inputs-only are mutually exclusive", file=sys.stderr)
        return 2

    if args.audit_inputs_only:
        summary = run_audit_inputs(cfg)
        print("\n=== audit-inputs-only summary ===")
        print(f"PUMS person-file header validation: {summary['pums_header_validation']['status']}")
        print(f"PUMS housing-file header validation: {summary['housing_header_validation']['status']}")
        print(f"CES rows: {summary['ces_n_rows_total']} total, {summary['ces_n_rows_valid_for_training']} valid for training")
        return 0

    if args.validate_only:
        return run_validate_only(cfg)

    return run_full_build(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
