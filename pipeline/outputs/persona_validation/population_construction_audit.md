# Population Construction Audit

Status: PASS

Binary scientific answer: A.

The current construction rakes observed ACS/PUMS weighted age-by-race seed
cells within gender to the benchmark gender-by-age and gender-by-race margins,
integerizes those raked cell counts, and then samples complete observed ACS
donor rows without replacement inside each resulting cell. It does not sample
age, race, gender, education, income, or state independently, and it does not
impose a conditional-independence age-race structure. The only demographic-like
field assigned after donor selection is party, drawn once per intact donor from
a CES-trained probability model.

## Source Files

- `pipeline/src/population/raking.py`
- `pipeline/src/population/sampling.py`
- `pipeline/src/population/roster.py`
- `pipeline/scripts/build_population.py`
- `pipeline/data/derived/population/joint_cells_40.csv`
- `pipeline/data/processed/population/profiles_core_1000.csv`
- `pipeline/data/generated/g_personas_master.csv`

## Target Cells Used

- Operative margins: `quota_gender_age_1000.csv` and `quota_gender_race_1000.csv`.
- Raked cell artifact: 40 rows = 2 genders x 5 age bands x 4 race groups.
- Cell totals sum to `1000`.
- IPF max residual error in artifact: `9.868e-12`.
- IPF max iterations in artifact: `7`.

## Objective / Weighting Procedure

1. Build a weighted ACS/PUMS seed matrix for each gender: age_band x race,
   using `pums_person_weight`.
2. Use iterative proportional fitting to match that gender's age and race
   margin targets simultaneously.
3. Use deterministic controlled integerization/MILP to convert fractional
   expected cells to exact integer targets while preserving both margins.
4. For each gender x age_band x race cell, draw complete donor rows without
   replacement with probability proportional to `pums_person_weight`.

## Donor Integrity

- Donor rows stay intact: YES.
- Any demographic field sampled independently: NO for ACS donor fields.
- Party is imputed once per donor from a CES-trained model; it is not
  resampled by condition.
- Age-race structure explicitly constrained/artificial: NO. Age-race cells are
  raked from observed ACS weighted seed cells to satisfy the two required
  benchmark margins; this adjusts the observed joint structure but does not
  impose conditional independence.

## Diagnostics

- G donors: `1000`.
- Unique donor keys: `1000`.
- Duplicate source_row_id count: `0`.
- Duplicate donor_id count: `0`.
- Selected source-weight ESS: `432.96097068068934`.
- Selected source-weight p99: `13.7804942888017`.
- Selected source-weight max: `14.8970415077754`.
- Max abs selected-vs-source age x race difference: `0.783` percentage points.
- Max abs selected-vs-raked-target age x race difference: `0.600` percentage points.
- Build git commit recorded in metadata: `6513fd469d447f4e013b39a49a1083e7797c7ae2`.

Largest selected source weights:

```json
[
  {
    "donor_key": "LP0040",
    "source_row_id": 1906528928,
    "age_band": "18-29",
    "race": "White / Caucasian",
    "gender": "Male",
    "source_weight": 14.8970415077754
  },
  {
    "donor_key": "LP0049",
    "source_row_id": 1910575828,
    "age_band": "18-29",
    "race": "White / Caucasian",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0006",
    "source_row_id": 1882440362,
    "age_band": "18-29",
    "race": "White / Caucasian",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0330",
    "source_row_id": 1895696464,
    "age_band": "45-59",
    "race": "Hispanic / Latino",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0050",
    "source_row_id": 1912653572,
    "age_band": "18-29",
    "race": "White / Caucasian",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0058",
    "source_row_id": 1895554990,
    "age_band": "18-29",
    "race": "Black / African American",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0055",
    "source_row_id": 1886929106,
    "age_band": "18-29",
    "race": "Black / African American",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0317",
    "source_row_id": 1867413730,
    "age_band": "45-59",
    "race": "Hispanic / Latino",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0084",
    "source_row_id": 1891425616,
    "age_band": "18-29",
    "race": "Hispanic / Latino",
    "gender": "Male",
    "source_weight": 13.7804942888017
  },
  {
    "donor_key": "LP0526",
    "source_row_id": 1906136276,
    "age_band": "18-29",
    "race": "White / Caucasian",
    "gender": "Female",
    "source_weight": 13.7804942888017
  }
]
```

Detailed age x race comparison is written to
`pipeline/outputs/persona_validation/age_race_source_vs_selected.csv`.
