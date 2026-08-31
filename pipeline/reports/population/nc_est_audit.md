# NC-EST workbook audit (reference/provenance only)

**Status: audit-only.** Per §12, this workbook never informs
`config/quota_gender_age_1000.csv`, `config/quota_gender_race_1000.csv`, the
joint cell targets, or the selected sample. Those are set entirely by the
benchmark's published quota margins (§5). This report exists purely for
provenance/comparison.

## File

- **Filename:** `data/nc-est2024-asr6h.xlsx`
- **SHA-256:** `54034788e42677dd1d05917cbaeed3ed15d9ad6c9877991e5dd0e98dfc752c57`
- **Sheet names:** `Total`, `White`, `Black`, `AIAN`, `Asian`, `NHPI`,
  `Two or More Races` (7 sheets, each dimensioned `A1:G335`).
- **Title (cell A2, every sheet):** "Annual Estimates of the Resident
  Population by Sex, Age, Race, and Hispanic Origin for the United States:
  April 1, 2020 to July 1, 2024" -- i.e. NC-EST2024-ASR6H, the Vintage 2024
  national population estimates workbook named in the task instructions.
- **Estimate vintage:** Vintage 2024, five annual columns (2020-2024); the
  July 1, 2024 estimate is column G.
- **Compatibility with the preregistration's stated source:** compatible --
  this is the correct workbook (Vintage 2024 NC-EST2024-ASR6H) for a
  sex x age x race/ethnicity population target, and its `HISPANIC` /
  `NOT HISPANIC` x `MALE`/`FEMALE` x 5-year age-group breakdown is
  structurally rich enough to derive a 40-cell (or finer) target table from,
  which is exactly what an earlier, now-superseded pass of this repository's
  work did (a 2-sex x 5-age x 4-race scheme derived directly from this
  workbook; see git history prior to the operative-quota instructions in
  this task). That scheme is **not** the one now in force -- see below.

## Why it is not operative here

The current task's §5 states plainly: *"The preregistration provides gender
x age margins [and] gender x race/ethnicity margins. It does not identify a
unique gender x age x race joint distribution."* The operative constraints
are `config/quota_gender_age_1000.csv` and `config/quota_gender_race_1000.csv`
-- fixed integer margins reconciling a rounded ~17,000-row roster target
down to exactly 1,000 (see `population_report.md`, "Reconciliation of the
rounded Male/Female totals"). The NC-EST workbook's age bands (18-24, 25-44,
45-64, 65-84, 85+ in the earlier scheme) and race categories
(Hispanic-any-race / non-Hispanic White / non-Hispanic Black / non-Hispanic
Other) also do not match this task's canonical `AGE_BAND_ORDER` (18-29,
30-44, 45-59, 60+) or `RACE_ORDER` (5 levels including a separate Asian
category), so even a proportional re-derivation would require redefining
cells, not just re-weighting them.

## Discrepancies from category-definition and integerization differences

Not evaluated quantitatively in this pass (would require rebuilding the
now-superseded 40-cell extraction under the old age/race scheme and
reconciling it against the new one cell-by-cell, which the operative-quota
instructions make unnecessary work). Qualitatively: the two schemes differ
in (a) number and boundaries of age bands, (b) whether Asian is its own race
category or folded into a residual "Other," and (c) the population base
(Census Bureau national resident population estimates vs. the benchmark
preregistration's own quota-setting process, whose underlying sampling frame
is not this workbook). Any comparison would need to be read as "two
different, non-nested category schemes," not as a validity check on the
operative quotas.

## Test coverage

`tests/population/test_population_outputs.py` includes a test that swaps
`data/nc-est2024-asr6h.xlsx` for a dummy/corrupt file and confirms the
operative quota configuration (`config/quota_gender_age_1000.csv`,
`config/quota_gender_race_1000.csv`, and the resulting joint cell targets)
is completely unaffected when NC-EST audit parsing is disabled or fails --
i.e. this file genuinely cannot alter the operative build, per §12's explicit
requirement.
