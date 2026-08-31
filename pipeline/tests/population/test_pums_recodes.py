"""§25 tests 1-8: PUMS recode boundaries, verified against
data/PUMS_Data_Dictionary_2024.pdf (see reports/population/pums_variable_audit.md).
"""

from __future__ import annotations

import pytest

from population import pums


# 1. Every age-band boundary: 18, 29, 30, 44, 45, 59, 60.
@pytest.mark.parametrize(
    "age,expected",
    [(18, "18-29"), (29, "18-29"), (30, "30-44"), (44, "30-44"), (45, "45-59"), (59, "45-59"), (60, "60+")],
)
def test_age_band_boundaries(age, expected):
    assert pums.recode_age_band(age) == expected


def test_age_band_below_18_raises():
    with pytest.raises(ValueError):
        pums.recode_age_band(17)


# 2. Hispanic priority over race.
def test_hispanic_takes_priority_over_race():
    # HISP=02 (Mexican) with RAC1P=1 (White alone) -> Hispanic, not White.
    assert pums.recode_race("02", "1") == "Hispanic / Latino"
    # Even RAC1P=2 (Black alone) is overridden by Hispanic origin.
    assert pums.recode_race("24", "2") == "Hispanic / Latino"


# 3. HISP representations: 1, "1", "01".
@pytest.mark.parametrize("value", [1, "1", "01"])
def test_hisp_representation_equivalence(value):
    assert pums.normalize_hisp(value) == "01"
    assert pums.recode_race(value, "1") == "White / Caucasian"  # not-Hispanic in every representation


# 4. Race mappings: White, Black, Asian, Other.
@pytest.mark.parametrize(
    "rac1p,expected",
    [("1", "White / Caucasian"), ("2", "Black / African American"), ("6", "Asian / Asian American")]
    + [(code, "Other") for code in ("3", "4", "5", "7", "8", "9")],
)
def test_race_mappings_non_hispanic(rac1p, expected):
    assert pums.recode_race("01", rac1p) == expected


def test_race_invalid_rac1p_raises():
    with pytest.raises(ValueError):
        pums.recode_race("01", "0")


# 5. Education boundaries: 1, 15, 16, 17, 18, 20, 21, 22, 23, 24.
@pytest.mark.parametrize(
    "code,expected",
    [
        (1, "Less than high school"), (15, "Less than high school"),
        (16, "High school diploma / GED"), (17, "High school diploma / GED"),
        (18, "Some college or Associate's degree"), (20, "Some college or Associate's degree"),
        (21, "Bachelor's degree"),
        (22, "Master's degree / Professional degree"), (23, "Master's degree / Professional degree"),
        (24, "Doctorate degree / Ph.D."),
    ],
)
def test_education_boundaries(code, expected):
    assert pums.recode_education(code) == expected


def test_education_blank_code_raises():
    with pytest.raises(ValueError):
        pums.recode_education("bb")


# 6. Income boundaries.
@pytest.mark.parametrize(
    "value,expected",
    [
        (-1, "Less than $30,000"), (0, "Less than $30,000"), (29_999, "Less than $30,000"),
        (30_000, "$30,000 to $55,999"), (55_999, "$30,000 to $55,999"),
        (56_000, "$56,000 to $99,999"), (99_999, "$56,000 to $99,999"),
        (100_000, "$100,000 to $167,999"), (167_999, "$100,000 to $167,999"),
        (168_000, "$168,000 or more"),
    ],
)
def test_income_boundaries(value, expected):
    assert pums.recode_income(value) == expected


# 7. Exclusion of Puerto Rico.
def test_puerto_rico_excluded():
    with pytest.raises(ValueError):
        pums.recode_state("72")
    assert "72" in pums.EXCLUDE_STATE_FIPS
    assert "72" not in pums.STATE_FIPS_TO_ABBR


def test_valid_state_recodes():
    assert pums.recode_state("06") == ("06", "CA")
    assert pums.recode_state("11") == ("11", "DC")
    assert len(pums.STATE_FIPS_TO_ABBR) == 51  # 50 states + DC


# 8. Correct ADJINC calculation.
def test_adjinc_calculation():
    # 2024 factor from the dictionary: 1015250 (6 implied decimals) == 1.015250
    assert pums.adjusted_household_income(100_000, 1_015_250) == pytest.approx(101_525.0)
    assert pums.adjusted_household_income(0, 1_015_250) == 0.0
    assert pums.adjusted_household_income(-5_000, 1_015_250) == pytest.approx(-5_076.25)
