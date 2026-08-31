"""src/population/pums.py

2024 ACS 1-Year PUMS person-file and housing-file ingestion, join, and
recoding.

Person ingestion reads directly from the `csv_pus.zip` archive (no manual
extraction), validates every `psam_pus*.csv` part's header against
CANONICAL_TO_ACTUAL before reading any data, and concatenates them.
`HINCP` (household income) is a *housing*-record variable in this vintage --
absent from the person file -- so it is read separately from the companion
`csv_hus.zip` archive (every `psam_hus*.csv` part, however many components
the release ships), validated for SERIALNO uniqueness, and left-joined onto
the person records via SERIALNO. Recoding then maps
SEX/AGEP/HISP/RAC1P/SCHL/HINCP+ADJINC/STATE to the benchmark's
gender/age_band/race/education/income/state fields, verified against
data/PUMS_Data_Dictionary_2024.pdf (see reports/population/pums_variable_audit.md).

Every "canonical" name below is the variable name used in this project's
population-construction instructions; `ST` is documented there but the actual
2024-vintage column (both in the dictionary and in the real CSV header) is
named `STATE` -- CANONICAL_TO_ACTUAL records that one naming difference.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import get_logger

logger = get_logger("pums")

#: canonical (spec) name -> actual 2024-vintage PUMS *person*-file column
#: name. HINCP is deliberately not here -- it is a housing-record variable
#: in this vintage (see HOUSING_CANONICAL_TO_ACTUAL) -- read from csv_hus.zip
#: and joined onto the person records via SERIALNO by read_pums().
CANONICAL_TO_ACTUAL: dict[str, str] = {
    "SERIALNO": "SERIALNO",
    "SPORDER": "SPORDER",
    "AGEP": "AGEP",
    "SEX": "SEX",
    "HISP": "HISP",
    "RAC1P": "RAC1P",
    "SCHL": "SCHL",
    "ADJINC": "ADJINC",
    "PWGTP": "PWGTP",
    "ST": "STATE",
}

#: canonical -> actual column name for the required *housing*-file fields.
#: SERIALNO is the join key; HINCP is the field the person file lacks.
HOUSING_CANONICAL_TO_ACTUAL: dict[str, str] = {
    "SERIALNO": "SERIALNO",
    "HINCP": "HINCP",
}

PERSON_FILE_PATTERN = re.compile(r"psam_pus[a-z0-9]+\.csv", re.IGNORECASE)
HOUSING_FILE_PATTERN = re.compile(r"psam_hus[a-z0-9]+\.csv", re.IGNORECASE)

#: HISP == this (after zero-padded normalization) means "Not Spanish/Hispanic/Latino"
#: (PUMS_Data_Dictionary_2024.pdf, HISP code 01). Any other HISP value is Hispanic.
HISP_NOT_HISPANIC_CODE = "01"

#: RAC1P codes for the three race categories the benchmark names explicitly;
#: everything else (3 AIAN, 4 Alaska Native, 5 AIAN+other, 7 NHPI, 8 Some
#: Other Race, 9 Two or More Races) recodes to "Other" (verified against the
#: dictionary's RAC1P section).
RAC1P_WHITE = "1"
RAC1P_BLACK = "2"
RAC1P_ASIAN = "6"

#: State FIPS ("STATE" column, Character 2) -> USPS abbreviation, for the 50
#: states + DC, transcribed from PUMS_Data_Dictionary_2024.pdf's STATE code
#: list. 72 (Puerto Rico) is deliberately excluded (see EXCLUDE_STATE_FIPS)
#: and is not a key here.
STATE_FIPS_TO_ABBR: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
EXCLUDE_STATE_FIPS = {"72"}  # Puerto Rico


class PumsColumnError(Exception):
    """Raised when a required column is absent from a PUMS person- or
    housing-file part. Carries the exact missing canonical/actual column
    names so the caller can report them explicitly rather than failing on an
    obscure KeyError deeper in the pipeline.
    """

    def __init__(self, member: str, missing_canonical: list[str], column_map: dict[str, str] = CANONICAL_TO_ACTUAL):
        self.member = member
        self.missing_canonical = missing_canonical
        self.missing_actual = [column_map[c] for c in missing_canonical]
        super().__init__(
            f"{member}: required column(s) absent from header: "
            f"{[f'{c} (as {column_map[c]})' for c in missing_canonical]}"
        )


class PumsIngestionError(Exception):
    """Raised for structural PUMS ingestion problems other than a missing
    column (e.g. no person-file part found, or non-unique donor_id).
    """


def find_person_file_members(zf: zipfile.ZipFile) -> list[str]:
    """Return the archive's national person-level PUMS CSV part names
    (matching psam_pus*.csv), excluding housing files and non-CSV entries.
    Raises PumsIngestionError if none are found.
    """
    names = zf.namelist()
    person_members = sorted(n for n in names if PERSON_FILE_PATTERN.fullmatch(Path(n).name))
    housing_members = sorted(n for n in names if HOUSING_FILE_PATTERN.fullmatch(Path(n).name))
    if housing_members:
        logger.info("ignoring housing file(s) present in archive: %s", housing_members)
    if not person_members:
        raise PumsIngestionError(
            f"no national person-level PUMS CSV (psam_pus*.csv) found in archive; entries were: {names}"
        )
    return person_members


def find_housing_file_members(zf: zipfile.ZipFile) -> list[str]:
    """Return the archive's national housing-level PUMS CSV part names
    (matching psam_hus*.csv) -- however many components the release ships
    (A/B, or A/B/C/D, ...), not assumed to be exactly two. Raises
    PumsIngestionError if none are found.
    """
    names = zf.namelist()
    housing_members = sorted(n for n in names if HOUSING_FILE_PATTERN.fullmatch(Path(n).name))
    if not housing_members:
        raise PumsIngestionError(
            f"no national housing-level PUMS CSV (psam_hus*.csv) found in archive; entries were: {names}"
        )
    return housing_members


def validate_header(zf: zipfile.ZipFile, member: str, column_map: dict[str, str] = CANONICAL_TO_ACTUAL) -> list[str]:
    """Read just the header line of one archive member and check every
    required column in `column_map` is present. Returns the header (list of
    column names) on success; raises PumsColumnError listing every missing
    canonical column on failure. Reads only the first line -- never loads the
    member's data. `column_map` defaults to the person-file requirements;
    pass HOUSING_CANONICAL_TO_ACTUAL to validate a housing-file part instead.
    """
    with zf.open(member) as f:
        first_line = f.readline().decode("utf-8-sig").rstrip("\r\n")
    header = first_line.split(",")
    missing = [canon for canon, actual in column_map.items() if actual not in header]
    if missing:
        raise PumsColumnError(member, missing, column_map)
    return header


def _read_required_columns(zf: zipfile.ZipFile, members: list[str], actual_cols: list[str], chunksize: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shared read loop for both person and housing parts: read only
    `actual_cols` (as strings) from each member in chunks, tag each row with
    its source_component, and concatenate. Returns (combined_df, per_member_row_counts).
    """
    frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    for member in members:
        with zf.open(member) as f:
            part_chunks = [chunk for chunk in pd.read_csv(f, usecols=actual_cols, dtype=str, chunksize=chunksize)]
            part_df = pd.concat(part_chunks, ignore_index=True) if part_chunks else pd.DataFrame(columns=actual_cols)
        part_df["source_component"] = member
        row_counts[member] = len(part_df)
        logger.info("%s: read %d rows", member, len(part_df))
        frames.append(part_df)
    combined = pd.concat(frames, ignore_index=True)
    return combined, row_counts


def read_housing(zip_path: Path | str, chunksize: int = 250_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and concatenate every national housing-file part in `zip_path`
    (however many components: A/B, or A/B/C/D, ...), validating every part's
    header against HOUSING_CANONICAL_TO_ACTUAL first. Validates SERIALNO is
    unique across the *combined* housing file (one row per housing
    unit/group-quarters facility) -- raises PumsIngestionError if not.
    """
    zip_path = Path(zip_path)
    report: dict[str, Any] = {"zip_path": str(zip_path), "parts": {}}

    with zipfile.ZipFile(zip_path) as zf:
        members = find_housing_file_members(zf)
        logger.info("found %d housing-file part(s): %s", len(members), members)

        for member in members:
            header = validate_header(zf, member, HOUSING_CANONICAL_TO_ACTUAL)
            report["parts"][member] = {"n_columns": len(header), "header_valid": True}
        logger.info("all required housing columns present in all %d part(s)", len(members))

        actual_cols = list(HOUSING_CANONICAL_TO_ACTUAL.values())
        combined, row_counts = _read_required_columns(zf, members, actual_cols, chunksize)
        for member, n in row_counts.items():
            report["parts"][member]["n_rows"] = n

    report["n_rows_sum_of_parts"] = sum(row_counts.values())
    report["n_rows_combined"] = len(combined)
    if report["n_rows_combined"] != report["n_rows_sum_of_parts"]:
        raise PumsIngestionError(
            f"combined housing row count ({report['n_rows_combined']}) != sum of part row counts "
            f"({report['n_rows_sum_of_parts']})"
        )

    serialno = combined["SERIALNO"].str.strip()
    if serialno.duplicated().any():
        n_dupes = int(serialno.duplicated().sum())
        raise PumsIngestionError(f"housing SERIALNO is not unique across combined parts: {n_dupes} duplicate(s)")
    report["serialno_unique"] = True
    report["members_loaded_exactly_once"] = sorted(members) == sorted(set(members))

    return combined, report


def join_person_and_housing(person_df: pd.DataFrame, housing_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-join HINCP onto the person records via SERIALNO (whitespace-
    stripped on both sides). A person row with no matching housing SERIALNO
    ends up with a missing HINCP, which the "nonmissing HINCP" inclusion
    filter (apply_inclusion_filters) then correctly excludes -- this is not
    expected to happen for a genuine national release (every person record's
    SERIALNO should reference a housing/GQ-facility record that exists in
    the same release), so a low match rate is flagged as a likely data
    problem rather than silently accepted.
    """
    left = person_df.copy()
    left["_serialno_key"] = left["SERIALNO"].str.strip()
    right = housing_df[["SERIALNO", "HINCP"]].copy()
    right["_serialno_key"] = right["SERIALNO"].str.strip()
    right = right.drop(columns=["SERIALNO"])

    merged = left.merge(right, on="_serialno_key", how="left").drop(columns=["_serialno_key"])
    n_matched = int(merged["HINCP"].notna().sum())
    match_rate = n_matched / len(merged) if len(merged) else 0.0
    report = {"n_person_rows": len(merged), "n_matched_any_housing_row": n_matched, "match_rate": match_rate}
    logger.info("person-housing join: %d/%d person rows matched a housing SERIALNO (%.1f%%)", n_matched, len(merged), 100 * match_rate)
    if match_rate < 0.5:
        raise PumsIngestionError(
            f"only {match_rate:.1%} of person rows matched a housing SERIALNO -- "
            "this is far below what a genuine release should produce; check that "
            "csv_pus.zip and csv_hus.zip are the same release/vintage"
        )
    return merged, report


def read_pums(pums_zip_path: Path | str, housing_zip_path: Path | str, chunksize: int = 250_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and concatenate every national person-file part in
    `pums_zip_path`, read and concatenate every national housing-file part in
    `housing_zip_path`, and left-join HINCP onto the person records via
    SERIALNO.

    Validates every person part's header against CANONICAL_TO_ACTUAL and
    every housing part's header against HOUSING_CANONICAL_TO_ACTUAL *before*
    reading any row data, so a missing required column fails immediately and
    explicitly rather than partway through a multi-hundred-MB read. Reads
    only the required columns (usecols), in chunks, with every column loaded
    as a string so leading-zero categorical codes (SCHL, HISP, STATE,
    SPORDER) and the letter-containing SERIALNO are never corrupted by
    numeric type inference.

    Returns (joined_dataframe, ingestion_report) where ingestion_report
    records each part's row count, header validation, the person-side
    donor_id uniqueness check, the housing-side SERIALNO uniqueness check,
    and the join match rate -- everything
    reports/population/exclusion_flow.csv and build_metadata.json need.
    """
    pums_zip_path = Path(pums_zip_path)
    report: dict[str, Any] = {"pums_zip_path": str(pums_zip_path), "housing_zip_path": str(housing_zip_path), "parts": {}}

    with zipfile.ZipFile(pums_zip_path) as zf:
        members = find_person_file_members(zf)
        logger.info("found %d person-file part(s): %s", len(members), members)

        for member in members:
            header = validate_header(zf, member)
            report["parts"][member] = {"n_columns": len(header), "header_valid": True}
        logger.info("all required person columns present in all %d part(s)", len(members))

        actual_cols = list(CANONICAL_TO_ACTUAL.values())
        combined, row_counts = _read_required_columns(zf, members, actual_cols, chunksize)
        for member, n in row_counts.items():
            report["parts"][member]["n_rows"] = n

    report["n_rows_sum_of_parts"] = sum(row_counts.values())
    report["n_rows_combined"] = len(combined)
    if report["n_rows_combined"] != report["n_rows_sum_of_parts"]:
        raise PumsIngestionError(
            f"combined row count ({report['n_rows_combined']}) != sum of part row counts "
            f"({report['n_rows_sum_of_parts']})"
        )

    donor_id = combined["SERIALNO"].str.strip() + "-" + combined["SPORDER"].str.strip()
    if donor_id.duplicated().any():
        n_dupes = int(donor_id.duplicated().sum())
        raise PumsIngestionError(f"donor_id is not unique after concatenation: {n_dupes} duplicate(s)")
    combined["donor_id"] = donor_id
    report["donor_id_unique"] = True
    report["members_loaded_exactly_once"] = sorted(members) == sorted(set(members))

    housing_df, housing_report = read_housing(housing_zip_path, chunksize)
    report["housing"] = housing_report

    joined, join_report = join_person_and_housing(combined, housing_df)
    report["join"] = join_report

    return joined, report


# --- Scalar recode functions (each independently unit-testable) -------------


def normalize_hisp(value: Any) -> str:
    """Normalize a HISP value to its 2-character zero-padded string form so
    numeric 1, string "1", and string "01" are all treated identically.
    """
    return str(value).strip().zfill(2)


def recode_gender(sex_raw: Any) -> str:
    """SEX == "1" -> Male, "2" -> Female (PUMS_Data_Dictionary_2024.pdf,
    SEX). Any other value is invalid for primary construction.
    """
    s = str(sex_raw).strip()
    if s == "1":
        return "Male"
    if s == "2":
        return "Female"
    raise ValueError(f"invalid SEX code for primary construction: {sex_raw!r}")


def recode_age_band(age: int) -> str:
    """AGEP -> the benchmark's 4 age bands. Raises for age < 18 (out of scope
    -- the primary panel is adults only).
    """
    age = int(age)
    if age < 18:
        raise ValueError(f"age {age} is below the adult inclusion threshold (18)")
    if age <= 29:
        return "18-29"
    if age <= 44:
        return "30-44"
    if age <= 59:
        return "45-59"
    return "60+"


def recode_race(hisp_raw: Any, rac1p_raw: Any) -> str:
    """Hispanic-origin-first race/ethnicity recode: HISP != "01" (any
    Hispanic-origin code) takes priority over RAC1P and maps to
    "Hispanic / Latino"; otherwise RAC1P 1/2/6 map to White/Black/Asian and
    every other RAC1P value (3, 4, 5, 7, 8, 9) maps to "Other".
    """
    if normalize_hisp(hisp_raw) != HISP_NOT_HISPANIC_CODE:
        return "Hispanic / Latino"
    r = str(rac1p_raw).strip()
    if r == RAC1P_WHITE:
        return "White / Caucasian"
    if r == RAC1P_BLACK:
        return "Black / African American"
    if r == RAC1P_ASIAN:
        return "Asian / Asian American"
    if r in {"3", "4", "5", "7", "8", "9"}:
        return "Other"
    raise ValueError(f"invalid RAC1P code: {rac1p_raw!r}")


def recode_education(schl_raw: Any) -> str:
    """SCHL -> the benchmark's 6 education levels, per
    PUMS_Data_Dictionary_2024.pdf's SCHL code list (01 No schooling ..
    24 Doctorate degree). Raises for the "bb" (N/A, under 3 years old) code
    or any other non-numeric/out-of-range value -- neither should occur once
    the AGEP >= 18 filter has been applied.
    """
    s = str(schl_raw).strip()
    if not s.isdigit():
        raise ValueError(f"non-numeric/blank SCHL code (unexpected for an adult): {schl_raw!r}")
    code = int(s)
    if 1 <= code <= 15:
        return "Less than high school"
    if 16 <= code <= 17:
        return "High school diploma / GED"
    if 18 <= code <= 20:
        return "Some college or Associate's degree"
    if code == 21:
        return "Bachelor's degree"
    if 22 <= code <= 23:
        return "Master's degree / Professional degree"
    if code == 24:
        return "Doctorate degree / Ph.D."
    raise ValueError(f"SCHL code out of the documented 01-24 range: {code}")


def adjusted_household_income(hincp_raw: Any, adjinc_raw: Any) -> float:
    """income_adjusted_2024 = HINCP * ADJINC / 1_000_000 (ADJINC's 6 implied
    decimal places, per PUMS_Data_Dictionary_2024.pdf). No CPI adjustment
    from 2024 to 2026 is applied -- the benchmark's income categories are
    fixed thresholds, and adding an unregistered inflation assumption would
    add undisclosed researcher discretion.
    """
    return int(str(hincp_raw).strip()) * int(str(adjinc_raw).strip()) / 1_000_000


def recode_income(value: float) -> str:
    """Adjusted household income -> the benchmark's 5 income categories.
    Zero and negative household incomes fall into "Less than $30,000" (the
    lowest category is unbounded below, not floored at 0).
    """
    if value < 30_000:
        return "Less than $30,000"
    if value < 56_000:
        return "$30,000 to $55,999"
    if value < 100_000:
        return "$56,000 to $99,999"
    if value < 168_000:
        return "$100,000 to $167,999"
    return "$168,000 or more"


def recode_year_birth(age: int, reference_year: int = 2026) -> int:
    """year_birth = reference_year - age. Deliberately does NOT age a 2024
    PUMS donor forward by 2 years -- the donor's real AGEP is used directly as
    the profile's 2026-benchmark age, preserving the preregistered age-band
    targets (aging every donor forward would shift people across age-band
    boundaries and drift the achieved quotas away from the targets).
    """
    return reference_year - int(age)


def recode_state(state_fips_raw: Any) -> tuple[str, str]:
    """STATE (FIPS, "ST" in the population-construction instructions) -> the
    2-character FIPS code and its USPS abbreviation. Raises for Puerto Rico
    (72) or any code not in the 50-states-plus-DC list.
    """
    fips = str(state_fips_raw).strip().zfill(2)
    if fips in EXCLUDE_STATE_FIPS:
        raise ValueError(f"state FIPS {fips} is Puerto Rico, excluded from the primary construction")
    if fips not in STATE_FIPS_TO_ABBR:
        raise ValueError(f"state FIPS {fips} is not one of the 50 states + DC")
    return fips, STATE_FIPS_TO_ABBR[fips]


# --- Inclusion filters and full recode, applied to an ingested DataFrame ----


def apply_inclusion_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply every §8 inclusion criterion in sequence, recording the row
    count before/after each step for reports/population/exclusion_flow.csv.
    Assumes `df` already has all CANONICAL_TO_ACTUAL columns plus HINCP (i.e.
    has passed read_pums's header validation and the person-housing join) --
    this function is pure/synthetic-data testable independent of the real
    archive.
    """
    flow: list[dict[str, Any]] = []
    working = df

    def _step(name: str, mask: pd.Series) -> None:
        nonlocal working
        n_before = len(working)
        working = working[mask]
        flow.append({"step": name, "n_before": n_before, "n_after": len(working), "n_excluded": n_before - len(working)})

    _step("AGEP >= 18", pd.to_numeric(working["AGEP"], errors="coerce") >= 18)

    state_fips = working["STATE"].str.strip().str.zfill(2)
    _step(
        "ST in 50 states or DC (excludes Puerto Rico / ST==72)",
        state_fips.isin(STATE_FIPS_TO_ABBR) & ~state_fips.isin(EXCLUDE_STATE_FIPS),
    )

    pwgtp = pd.to_numeric(working["PWGTP"], errors="coerce")
    _step("PWGTP finite and > 0", pwgtp.notna() & np.isfinite(pwgtp) & (pwgtp > 0))

    _step("valid SEX", working["SEX"].str.strip().isin({"1", "2"}))
    _step("valid HISP", working["HISP"].apply(normalize_hisp).str.fullmatch(r"0[1-9]|1[0-9]|2[0-4]"))
    _step("valid RAC1P", working["RAC1P"].str.strip().isin({"1", "2", "3", "4", "5", "6", "7", "8", "9"}))
    _step("valid SCHL", working["SCHL"].str.strip().str.isdigit() & pd.to_numeric(working["SCHL"], errors="coerce").between(1, 24))
    _step("nonmissing HINCP", pd.to_numeric(working["HINCP"], errors="coerce").notna())
    _step("valid ADJINC", pd.to_numeric(working["ADJINC"], errors="coerce").notna() & (pd.to_numeric(working["ADJINC"], errors="coerce") > 0))

    return working, flow


def recode_pums(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Apply every scalar recode to a filtered PUMS DataFrame (post
    apply_inclusion_filters) and return a new DataFrame with the benchmark
    fields added: gender, age, year_birth, age_band, race, education,
    income_adjusted_2024, income, state_fips, state_abbr, pums_person_weight.
    """
    out = df.copy()
    out["gender"] = out["SEX"].apply(recode_gender)
    out["age"] = pd.to_numeric(out["AGEP"], errors="raise").astype(int)
    out["year_birth"] = out["age"].apply(lambda a: recode_year_birth(a, reference_year))
    out["age_band"] = out["age"].apply(recode_age_band)
    out["race"] = [recode_race(h, r) for h, r in zip(out["HISP"], out["RAC1P"])]
    out["education"] = out["SCHL"].apply(recode_education)
    out["income_adjusted_2024"] = [adjusted_household_income(h, a) for h, a in zip(out["HINCP"], out["ADJINC"])]
    out["income"] = out["income_adjusted_2024"].apply(recode_income)
    state_recoded = [recode_state(s) for s in out["STATE"]]
    out["state_fips"] = [s[0] for s in state_recoded]
    out["state_abbr"] = [s[1] for s in state_recoded]
    out["pums_person_weight"] = pd.to_numeric(out["PWGTP"], errors="raise").astype(int)
    return out
