"""Persona-panel validation and skeleton-output tests.

These tests intentionally exercise pre-inference artifacts only. They should
never require an LLM call or completed outcome predictions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import survey_content as sc  # noqa: E402


@pytest.fixture(scope="session")
def persona_artifacts(repo_root, tmp_path_factory):
    outputs = tmp_path_factory.mktemp("persona_validation")
    generated = tmp_path_factory.mktemp("generated")
    env = os.environ.copy()
    # validate_personas.py's own write_plots() sets MPLCONFIGDIR via
    # os.environ.setdefault(...) under --output-dir once that's redirected;
    # leaving it unset here (rather than pointing it at the real
    # outputs/persona_validation/.matplotlib) lets that redirection take
    # effect instead of forcing the real cache path.
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "validate_personas.py"),
            "--output-dir", str(outputs),
            "--generated-dir", str(generated),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "summary": json.loads((outputs / "summary.json").read_text(encoding="utf-8")),
        "g": pd.read_csv(generated / "g_personas_master.csv"),
        "f": pd.read_csv(generated / "f_target_panel.csv"),
        "design": pd.read_csv(generated / "tier1_design_skeleton.csv"),
        "submission": pd.read_csv(generated / "tier1_submission_skeleton.csv"),
        "mapping": pd.read_csv(outputs / "category_mapping_audit.csv"),
        "g_age": pd.read_csv(outputs / "quota_diagnostics_gender_age.csv"),
        "g_race": pd.read_csv(outputs / "quota_diagnostics_gender_race.csv"),
        "f_age": pd.read_csv(outputs / "f_quota_diagnostics_gender_age.csv"),
        "f_race": pd.read_csv(outputs / "f_quota_diagnostics_gender_race.csv"),
        "condition_balance": pd.read_csv(outputs / "condition_balance_audit.csv"),
        "states": pd.read_csv(outputs / "state_audit.csv"),
        "duplicates": pd.read_csv(outputs / "duplication_audit.csv"),
        "inventory": pd.read_csv(outputs / "existing_persona_files.csv"),
        "outputs_dir": outputs,
        "generated_dir": generated,
    }


def response_columns() -> list[str]:
    return list(dict.fromkeys([*[item["target_label"] for item in sc.load_items()], *sc.OUTCOME_COMPOSITES.keys()]))


def test_persona_validation_summary_passes(persona_artifacts):
    assert persona_artifacts["summary"]["status"] == "PASS"
    assert persona_artifacts["summary"]["failures"] == []


def test_g_master_has_1000_unique_donors(persona_artifacts):
    g = persona_artifacts["g"]
    assert len(g) == 1000
    assert g["donor_key"].nunique() == 1000


def test_g_master_required_fields_are_present_and_nonmissing(persona_artifacts):
    required = ["donor_key", "age", "age_band", "gender", "race", "education", "income", "party", "state"]
    g = persona_artifacts["g"]
    assert set(required) <= set(g.columns)
    assert not g[required].isna().any().any()


def test_g_master_uses_adult_age_bands(persona_artifacts, schema):
    g = persona_artifacts["g"]
    assert (g["age"] >= 18).all()
    assert set(g["age_band"]) <= set(schema["moderators"]["age_band"])


def test_g_master_uses_schema_moderator_levels(persona_artifacts, schema):
    g = persona_artifacts["g"]
    for variable, allowed in schema["moderators"].items():
        assert set(g[variable].dropna()) <= set(allowed), variable


def test_g_master_has_no_outcome_or_raw_item_columns(persona_artifacts):
    g = persona_artifacts["g"]
    assert not (set(response_columns()) & set(g.columns))


def test_category_mapping_audit_is_exact(persona_artifacts):
    mapping = persona_artifacts["mapping"]
    assert not mapping.empty
    assert set(mapping["mapping_status"]) == {"exact"}


def test_state_audit_has_valid_states_and_stimuli(persona_artifacts):
    states = persona_artifacts["states"]
    assert set(states["state_abbr"]) <= set(sc.STATE_NAME_TO_ABBR.values())
    assert states["extreme_weather_stimulus_available"].all()


def test_g_gender_age_quota_exact(persona_artifacts):
    assert persona_artifacts["g_age"]["exact_count_match"].all()
    assert persona_artifacts["g_age"]["abs_difference_pp"].max() == 0


def test_g_gender_race_quota_exact(persona_artifacts):
    assert persona_artifacts["g_race"]["exact_count_match"].all()
    assert persona_artifacts["g_race"]["abs_difference_pp"].max() == 0


def test_f_panel_has_500_unique_profiles(persona_artifacts):
    f = persona_artifacts["f"]
    assert len(f) == 500
    assert f["f_profile_id"].nunique() == 500
    assert f["donor_key"].nunique() == 500


def test_f_panel_is_subset_of_g_population(persona_artifacts):
    assert set(persona_artifacts["f"]["donor_key"]) <= set(persona_artifacts["g"]["donor_key"])
    assert persona_artifacts["f"]["target_population"].nunique() == 1


def test_f_gender_age_quota_exact(persona_artifacts):
    assert persona_artifacts["f_age"]["exact_count_match"].all()
    assert persona_artifacts["f_age"]["abs_difference_pp"].max() == 0


def test_f_gender_race_quota_exact(persona_artifacts):
    assert persona_artifacts["f_race"]["exact_count_match"].all()
    assert persona_artifacts["f_race"]["abs_difference_pp"].max() == 0


def test_internal_design_skeleton_has_17000_rows(persona_artifacts):
    assert len(persona_artifacts["design"]) == 17_000


def test_internal_design_reuses_each_donor_across_17_conditions(persona_artifacts):
    counts = persona_artifacts["design"].groupby("donor_key").size()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 17


def test_internal_design_has_1000_rows_per_condition(persona_artifacts, schema):
    counts = persona_artifacts["design"].groupby("condition").size()
    assert set(counts.index) == set(schema["conditions"])
    assert (counts == 1000).all()


def test_internal_design_demographics_are_invariant_across_conditions(persona_artifacts):
    cols = ["age", "age_band", "gender", "race", "education", "income", "party", "state", "state_abbr"]
    nunique = persona_artifacts["design"].groupby("donor_key")[cols].nunique()
    assert (nunique == 1).all().all()


def test_condition_balance_is_exact(persona_artifacts):
    balance = persona_artifacts["condition_balance"]
    assert (balance["count_difference_vs_control"] == 0).all()
    assert (balance["proportion_difference_vs_control"] == 0).all()


def test_internal_skeleton_response_columns_are_missing_not_zero(persona_artifacts):
    design = persona_artifacts["design"]
    cols = response_columns()
    assert design[cols].isna().all().all()
    assert not (design[cols] == 0).any().any()


def test_submission_skeleton_uses_official_columns_only(persona_artifacts, schema):
    submission = persona_artifacts["submission"]
    trust_items = sc.OUTCOME_COMPOSITES["trust_multidimensional"][1]
    expected = [
        "profile_id",
        "condition",
        *schema["moderators"].keys(),
        "trust_multidimensional",
        *trust_items,
        *[outcome for outcome in sc.OUTCOME_COMPOSITES if outcome != "trust_multidimensional"],
    ]
    assert list(submission.columns) == expected
    assert "donor_key" not in submission.columns
    assert "state" not in submission.columns
    assert "state_abbr" not in submission.columns


def test_submission_skeleton_outcomes_are_missing_not_zero(persona_artifacts, schema):
    submission = persona_artifacts["submission"]
    metadata_cols = {"profile_id", "condition", *schema["moderators"].keys()}
    response_cols = [col for col in submission.columns if col not in metadata_cols]
    assert submission[response_cols].isna().all().all()
    assert not (submission[response_cols] == 0).any().any()


def test_duplicate_audit_has_no_donor_or_profile_duplicates(persona_artifacts):
    duplicates = persona_artifacts["duplicates"]
    checked = duplicates[duplicates["key"].isin(["donor_key", "profile_id"])]
    assert (checked["duplicates"] == 0).all()


def test_active_processed_population_excludes_stale_g_design(repo_root):
    assert not (repo_root / "data" / "processed" / "population" / "profiles_core_500.csv").exists()
    assert not (repo_root / "data" / "processed" / "population" / "simulation_roster_18000.csv").exists()
    assert (repo_root / "data" / "processed" / "population" / "profiles_core_1000.csv").exists()
    assert (repo_root / "data" / "processed" / "population" / "simulation_roster_17000.csv").exists()


def test_inventory_records_active_sources(persona_artifacts):
    inventory = persona_artifacts["inventory"]
    statuses = set(inventory["status"])
    assert "active_g_source" in statuses
    assert "active_g_roster" in statuses
    assert "generated_by_persona_validator" in statuses


def test_report_and_plots_exist(persona_artifacts):
    outputs = persona_artifacts["outputs_dir"]
    assert (outputs / "persona_validation_report.md").exists()
    for name in [
        "target_vs_actual_gender_age.png",
        "error_gender_age.png",
        "target_vs_actual_gender_race.png",
        "error_gender_race.png",
        "moderator_marginals.png",
        "state_distribution.png",
        "source_vs_selected_nonquota_demographics.png",
        "condition_balance.png",
        "joint_distribution_checks.png",
    ]:
        assert (outputs / name).exists(), name


def test_population_construction_audit_reports_complete_donor_row_sampling(repo_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "audit_population_construction.py"), "--output-dir", str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    outputs = tmp_path
    report_path = outputs / "population_construction_audit.md"
    csv_path = outputs / "age_race_source_vs_selected.csv"
    assert report_path.exists()
    assert csv_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Binary scientific answer: A." in report
    assert "Donor rows stay intact: YES." in report
    assert "impose conditional independence" in report
    audit = pd.read_csv(csv_path)
    assert {"age_band", "race", "source_weighted_share", "selected_share"} <= set(audit.columns)
    assert len(audit) == 5 * 4
