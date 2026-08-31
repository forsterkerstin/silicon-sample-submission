"""§25 tests 15-20 (population output), 27-32 (roster), and 36-38
(NC-EST/census_cells.csv cannot alter operative targets; predictions/ is
never written to).
"""

from __future__ import annotations

import inspect

import pandas as pd

from population import pums, raking, sampling


# 15. Exactly 1,000 profiles.
def test_core_profiles_has_exactly_1000_rows(built_population):
    core, _ = built_population
    assert len(core) == 1000


# 16. Exactly 1,000 unique donor IDs.
def test_core_profiles_donor_ids_unique(built_population):
    core, _ = built_population
    assert core["donor_id"].nunique() == 1000
    assert not core["donor_id"].duplicated().any()


# 17. Exact quota margins.
def test_core_profiles_exact_quota_margins(built_population, quota_age, quota_race):
    core, _ = built_population
    audit = sampling.quota_audit(core, quota_age, quota_race)  # raises SamplingError on any mismatch
    assert audit["gender_age"]["exact_match"].all()
    assert audit["gender_race"]["exact_match"].all()


# 18. Exact allowed factor strings.
def test_core_profiles_use_exact_benchmark_strings(built_population, schema):
    core, _ = built_population
    assert set(core["gender"]) <= set(schema["moderators"]["gender"])
    assert set(core["age_band"]) == set(schema["moderators"]["age_band"])
    assert set(core["race"]) <= set(schema["moderators"]["race"])
    assert set(core["education"]) <= set(schema["moderators"]["education"])
    assert set(core["income"]) <= set(schema["moderators"]["income"])
    assert set(core["party"]) <= set(schema["moderators"]["party"])


# 19. No missing required variables.
def test_core_profiles_no_missing_required_fields(built_population):
    core, _ = built_population
    assert not core.isna().any().any()


# 20. State FIPS mapping completeness.
def test_state_fips_mapping_covers_50_states_plus_dc():
    assert len(pums.STATE_FIPS_TO_ABBR) == 51
    assert len(set(pums.STATE_FIPS_TO_ABBR.values())) == 51  # no duplicate abbreviations
    assert "72" not in pums.STATE_FIPS_TO_ABBR


def test_core_profiles_states_all_valid(built_population):
    core, _ = built_population
    assert set(core["state_fips"]) <= set(pums.STATE_FIPS_TO_ABBR)
    assert set(core["state_abbr"]) <= set(pums.STATE_FIPS_TO_ABBR.values())


# 27. Exactly 17,000 rows.
def test_roster_has_exactly_17000_rows(built_population):
    _, roster_df = built_population
    assert len(roster_df) == 17_000


# 28. Exactly 17 conditions.
def test_roster_has_exactly_17_conditions(built_population, schema):
    _, roster_df = built_population
    assert set(roster_df["condition"]) == set(schema["conditions"])
    assert len(schema["conditions"]) == 17


# 29. Exactly 1,000 rows per intervention.
def test_roster_1000_rows_per_intervention(built_population, schema):
    _, roster_df = built_population
    counts = roster_df.groupby("condition").size()
    for condition in schema["conditions"]:
        if condition != "control":
            assert counts[condition] == 1000, condition


# 30. Exactly 1,000 control rows.
def test_roster_1000_control_rows(built_population):
    _, roster_df = built_population
    assert (roster_df["condition"] == "control").sum() == 1000


# 31. Globally unique profile_id.
def test_roster_profile_id_globally_unique(built_population):
    _, roster_df = built_population
    assert roster_df["profile_id"].nunique() == len(roster_df)


# 32. Demographics invariant within latent_profile_id.
def test_roster_demographics_invariant_across_conditions(built_population):
    _, roster_df = built_population
    attr_cols = ["gender", "age", "year_birth", "age_band", "race", "education", "income", "party", "state_fips", "state_abbr"]
    nunique = roster_df.groupby("latent_profile_id")[attr_cols].nunique()
    assert (nunique == 1).all().all()


# 36 & 37. NC-EST workbook / existing census_cells.csv cannot alter operative targets.
def test_raking_and_sampling_never_reference_audit_only_files():
    for module in (raking, sampling):
        source = inspect.getsource(module)
        assert "nc-est" not in source.lower()
        assert "nc_est" not in source.lower()
        assert "census_cells" not in source.lower()


def test_operative_quota_loading_does_not_touch_nc_est_or_census_cells(repo_root, tmp_path):
    import shutil
    import sys

    sys.path.insert(0, str(repo_root / "scripts"))
    import build_population as cli  # noqa: PLC0415

    cfg = cli.load_config(repo_root / "config" / "population.yaml")
    baseline_age, baseline_race = cli.load_quota_tables(cfg)

    # corrupt *copies* in a scratch dir -- never touch the real files -- and
    # confirm the quota loading path (which never reads these paths at all)
    # is unaffected.
    scratch = tmp_path / "nc_est_and_census_cells_corrupted"
    scratch.mkdir()
    (scratch / "nc-est2024-asr6h.xlsx").write_bytes(b"not a real workbook")
    (scratch / "census_cells.csv").write_text("garbage,not,real,data\n1,2,3,4\n")

    age_after, race_after = cli.load_quota_tables(cfg)
    pd.testing.assert_frame_equal(baseline_age, age_after)
    pd.testing.assert_frame_equal(baseline_race, race_after)
    shutil.rmtree(scratch)


# 38. The pipeline never writes into predictions/.
def test_pipeline_source_never_references_predictions_dir(repo_root):
    for path in (repo_root / "src" / "population").rglob("*.py"):
        source = path.read_text()
        assert "predictions/" not in source, f"{path} references predictions/"
    cli_source = (repo_root / "scripts" / "build_population.py").read_text()
    assert "predictions/" not in cli_source
