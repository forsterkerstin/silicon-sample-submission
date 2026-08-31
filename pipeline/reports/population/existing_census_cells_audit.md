# Existing `data/census_cells.csv` audit (reference/provenance only)

**Status: audit-only.** Per §11, this file is never overwritten and never
used as an operative target for the population-construction pipeline in
this task.

## What this file actually is

- **Path:** `data/census_cells.csv`
- **SHA-256:** `2a3fe14b37d486747e25d81f306dd6620b1025f6c83f85d988a8aef266e20b49`
- **Schema (header row):** `sex,age_band,race,share`
- **Row count:** 0 data rows (header only).
- **Apparent source/purpose:** this is a *template* created for a different
  component of this repository -- `regularized_probability_elicitation.py`'s
  `load_census_cells()` (see `pipeline/data/README.md`), which builds a
  simple 40-cell (2 sex x 4 age_band x 5 race) profile panel for the LLM
  probability-elicitation pipeline, a system this task is explicitly
  scoped away from ("Do not modify the probability-elicitation
  implementation"). It was never populated with actual Census figures (a
  companion `data/nc-est2024-asr6h.xlsx` extraction for it was in progress
  in this same session but superseded by this task's operative-quota
  instructions before completion).
- **Does it represent a true gender x age x race table?** No -- it is
  presently empty (header only, 0 rows). Even fully populated per its own
  README, it would represent a 2 x 4 x 5 = 40-cell scheme using this task's
  exact `age_band`/`race` category labels (unlike the NC-EST workbook's
  differently-binned scheme) sourced from simple Census population shares,
  not from the benchmark preregistration's own quota margins.

## Comparison with the preregistered benchmark margins

Not meaningful in its current (empty) state -- there are no rows to
normalize to proportions or compare against `config/quota_gender_age_1000.csv`
/ `config/quota_gender_race_1000.csv`. Per §11's own framing, *"A
disagreement is not an implementation failure because the benchmark quotas,
not this file, are the operative constraints"* -- so even if it were
populated, a mismatch against the operative quotas would not indicate a bug
in either file.

## Why it is not suitable as an operative target

1. It is empty.
2. Even populated, it targets simple Census population shares, not the
   benchmark preregistration's own published quota margins, which are what
   §5 designates as operative.
3. Its `age_band`/`race` scheme (matching this task's canonical labels) is
   coincidentally compatible in *shape*, but that is not sufficient grounds
   to treat it as a substitute for the preregistration's quota targets --
   the values themselves would still be the wrong population reference.

## Test coverage

`tests/population/test_population_outputs.py` includes a test confirming
that the content of `data/census_cells.csv` (empty, or any other content)
cannot alter `config/quota_gender_age_1000.csv`,
`config/quota_gender_race_1000.csv`, or the resulting joint cell targets --
this file is read only for this audit report, never by
`src/population/raking.py` or `src/population/sampling.py`.
