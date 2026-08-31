# PUMS variable audit

Source: `data/PUMS_Data_Dictionary_2024.pdf` (136 pages, plain-text-extracted
with `pypdf`, no OCR needed). Cross-checked against the real archive header
(`data/csv_pus.zip` -> `psam_pusa.csv`/`psam_pusb.csv`, both identical,
286 columns). Every mapping below was verified against the dictionary text,
not asserted from memory of prior ACS vintages -- one important difference
from earlier vintages was caught this way (see "ST" below).

| Variable | Role | Verified meaning | Valid coding (verified) | Source | Implementation recode |
|---|---|---|---|---|---|
| `SERIALNO` | ID | Housing unit/GQ person serial number | Character(13), e.g. `2024GQ0000001..2024GQ9999999`, `2024HU0000001..2024HU9999999` | dictionary line 11 | kept as string; `donor_id = SERIALNO + "-" + SPORDER` |
| `SPORDER` | ID | Person number within household | Numeric(2), `01..20` | dictionary line 1529 | kept as zero-padded string (not cast to int, to preserve donor_id's exact form) |
| `AGEP` | predictor | Age | Numeric(2), `0` = under 1 year, `1..99` (top-coded) | dictionary line 1608 | `pums.recode_age_band`; inclusion filter `AGEP >= 18` |
| `SEX` | predictor | Sex | Character(1): `1`=Male, `2`=Female | dictionary line 2172 | `pums.recode_gender` |
| `HISP` | predictor | Recoded detailed Hispanic origin | Character(2): `01`=Not Spanish/Hispanic/Latino, `02..24`=specific Hispanic-origin groups | dictionary line 3221 | `pums.normalize_hisp` / `pums.recode_race` (HISP != "01" takes priority over RAC1P) |
| `RAC1P` | predictor | Recoded detailed race code | Character(1): `1`=White alone, `2`=Black/African American alone, `3`=AIAN alone, `4`=Alaska Native alone, `5`=AIAN tribes specified/other, `6`=Asian alone, `7`=NHPI alone, `8`=Some Other Race alone, `9`=Two or More Races | dictionary line 5620 | `pums.recode_race`: 1/2/6 -> White/Black/Asian, {3,4,5,7,8,9} -> Other, only when HISP == "01" |
| `SCHL` | predictor | Educational attainment | Character(2): `bb`=N/A (<3 yrs old), `01..15`=below HS diploma, `16`=regular HS diploma, `17`=GED, `18`=some college <1yr, `19`=1+yr college credit no degree, `20`=Associate's, `21`=Bachelor's, `22`=Master's, `23`=Professional degree beyond bachelor's, `24`=Doctorate | dictionary line 2138 | `pums.recode_education`: 1-15/16-17/18-20/21/22-23/24 |
| `HINCP` | predictor | Household income (use ADJINC to adjust) | Numeric(7); documented in the dictionary's **housing-record** section (line 729), not the person-record section | dictionary line 729 | absent from the person file; read from `data/csv_hus.zip` (`psam_husa.csv`/`psam_husb.csv`, both 241 columns, `HINCP` present) and left-joined onto persons via `SERIALNO` in `pums.read_pums`/`pums.join_person_and_housing` -- see "Resolution" below |
| `ADJINC` | predictor | Income/earnings adjustment factor, 6 implied decimal places | Character(7), e.g. `1015250` = 2024 factor 1.015250 | dictionary line 97 / 1599 | `pums.adjusted_household_income`: `HINCP * ADJINC / 1_000_000` |
| `PWGTP` | weight | Person's weight | Numeric(5): `1..9999` | dictionary line 1603 | inclusion filter `PWGTP` finite and > 0; used as sampling weight throughout |
| `ST` (task spec name) | geography | State code | **Not a column in this vintage.** The dictionary and the real CSV header both name it `STATE`, Character(2), codes `01`=Alabama .. `56`=Wyoming (50 states + DC) plus `72`=Puerto Rico | dictionary line 1544 (`STATE Character 2`) | `pums.recode_state` / `pums.STATE_FIPS_TO_ABBR`; `72` explicitly excluded |

## Verification method

For every variable above except `HINCP`, I:
1. Located its definition in the extracted dictionary text (line numbers
   cited above).
2. Read its exact valid-code list from that definition (transcribed
   verbatim into `pums.py`'s constants, e.g. `STATE_FIPS_TO_ABBR`,
   `RAC1P_WHITE`/`_BLACK`/`_ASIAN`).
3. Cross-checked the actual CSV header (`data/csv_pus.zip`) contains a
   column of that exact name.
4. Round-tripped the boundary values through the implemented recode function
   (see the smoke tests run against real dictionary boundaries during
   development: `recode_age_band(18/29/30/44/45/59/60)`,
   `recode_education(1/15/16/17/18/20/21/22/23/24)`,
   `recode_income(-1/0/29999/30000/...)`, `recode_state("06")` -> `("06\", \"CA\")`, `recode_state("72")` -> raises).

## Resolution: HINCP via the housing file (data/csv_hus.zip)

**`HINCP` is documented in the dictionary but is not a column in the actual
person-file CSVs inside `data/csv_pus.zip`.** Both `psam_pusa.csv` and
`psam_pusb.csv` have 286 columns (verified identical), none named `HINCP`
(checked by exact match, case-insensitive match, and substring match). The
dictionary's own structure explains why: `HINCP`'s definition (line 729)
falls within the *housing-record* variable block (lines 1-1512), which
restarts with person-record variables at line 1513 (`SERIALNO` repeats
there because both record types share it) -- i.e. `HINCP` is a
housing-record variable, joined from the companion national housing file
via `SERIALNO`. This was independently confirmed by the Census README
extracted from inside `csv_pus.zip` itself (`ACS2024_PUMS_README.pdf`):
*"The Housing files contain data on housing units... The Person files
contain data on people."* Every row in both person-file parts has
`RT == "P"`, confirming there were no housing rows to fall back on within
that archive alone.

**With `data/csv_hus.zip` now added** (`psam_husa.csv` + `psam_husb.csv`,
241 columns each, `HINCP` and `SERIALNO` both present, every row `RT == "H"`),
`pums.read_housing()` validates both parts' headers, concatenates
1,631,969 rows, and confirms `SERIALNO` is unique across the combined
housing file. `pums.join_person_and_housing()` left-joins `HINCP` onto the
3,422,888 person rows via `SERIALNO`: **3,239,682 matched (94.6%)**. The
~5.4% that don't match are group-quarters residents and similar cases whose
housing-file record has a blank/N/A `HINCP` (the dictionary's own `HINCP`
code `bbbbbbb` = "N/A (GQ/vacant)") -- correctly excluded downstream by the
`nonmissing HINCP` inclusion filter, not evidence of a join bug.

Everything in this table now checks out against the real dictionary and the
real files (both archives) with no remaining ambiguity; the full real build
completed successfully -- see `population_report.md`.
