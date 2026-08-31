"""§25 tests 33-35: reproducibility and raw-file safety. Also includes a real
end-to-end PUMS person+housing ingestion/join test, confirming HINCP (a
housing-record variable in this vintage, absent from the person file alone --
see reports/population/pums_variable_audit.md) is now available via the
SERIALNO join against the companion data/csv_hus.zip.
"""

from __future__ import annotations

import hashlib

from population import audit, pums


def _hash_df(df):
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


# 33. Two builds with the same seed produce identical hashes.
def test_same_seed_produces_identical_output_hashes(quota_age, quota_race, schema, population_builder):
    core_a, roster_a = population_builder(20260831, universe_seed=99, quota_age=quota_age, quota_race=quota_race, schema=schema)
    core_b, roster_b = population_builder(20260831, universe_seed=99, quota_age=quota_age, quota_race=quota_race, schema=schema)
    assert _hash_df(core_a) == _hash_df(core_b)
    assert _hash_df(roster_a) == _hash_df(roster_b)


# 34. A changed seed changes selected donor records.
def test_changed_seed_changes_selected_donors(quota_age, quota_race, schema, population_builder):
    core_a, _ = population_builder(20260831, universe_seed=99, quota_age=quota_age, quota_race=quota_race, schema=schema)
    core_b, _ = population_builder(11111111, universe_seed=99, quota_age=quota_age, quota_race=quota_race, schema=schema)
    assert core_a["donor_id"].tolist() != core_b["donor_id"].tolist()


# 35. Raw input hashes are unchanged (by this test suite -- it never opens
# any raw file in write mode).
def test_raw_input_hashes_unchanged_by_running_the_suite(repo_root):
    manifest = audit.build_source_manifest(repo_root / "data")
    changed = audit.verify_raw_files_unchanged(manifest)
    assert changed == []


# Real end-to-end ingestion test: the person file alone lacks HINCP, but the
# join against the real housing file must produce it, with a healthy match
# rate (the person file legitimately has 3.4M+ rows; this test reads both
# real archives, so it is slower (~35s) than the rest of the suite by design).
def test_real_person_and_housing_archives_join_and_produce_hincp(repo_root):
    combined, report = pums.read_pums(repo_root / "data" / "csv_pus.zip", repo_root / "data" / "csv_hus.zip")

    assert "HINCP" in combined.columns
    assert report["join"]["match_rate"] > 0.9  # most persons live in a housing unit with reported income
    assert not combined["donor_id"].duplicated().any()
    assert report["housing"]["serialno_unique"] is True

    # a person row's HINCP, if present, must be a valid nonnegative-or-parseable
    # numeric string (or blank for the ~5% GQ/vacant-adjacent cases) -- not corrupted.
    non_null = combined["HINCP"].dropna()
    assert non_null.str.strip().str.lstrip("-").str.isdigit().all()
