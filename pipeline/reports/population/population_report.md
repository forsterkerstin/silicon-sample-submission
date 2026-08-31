# Population construction report

## 1. Executive summary

This pipeline builds a 1,000-profile, quota-aligned synthetic-respondent
panel and its 17,000-row simulation roster for a Tier-1 Silicon Sample
Benchmark submission, per the task's §1-30. **This is a real, complete
build**, run end-to-end against the actual data in `data/`: the 2024 ACS
1-Year PUMS person file (`csv_pus.zip`), the companion housing file
(`csv_hus.zip`, added after the initial pass -- see §6), and the 2024 CES
Common Content (`CCES24_Common_OUTPUT_vv_topost_final.csv`). All 117 tests
in `tests/population/` pass, and `scripts/build_population.py` completed
successfully in all three modes (`--audit-inputs-only`, normal, and
`--validate-only`).

**The one real gap found in an earlier pass of this build -- `HINCP`
(household income) missing from the ACS PUMS person file -- is now
resolved.** `HINCP` is a housing-record variable in this vintage; it is read
from `data/csv_hus.zip` (`psam_husa.csv`/`psam_husb.csv`) and left-joined
onto the person records via `SERIALNO`, matching 3,239,682/3,422,888
(94.6%) person rows -- the ~5.4% that don't match are group-quarters
residents and similar cases with no household income concept, which the
`nonmissing HINCP` inclusion filter correctly excludes.

## 2. Raw inputs

| File | Role | Operative? |
|---|---|---|
| `data/csv_pus.zip` | 2024 ACS 1-Year PUMS person file (`psam_pusa.csv`, `psam_pusb.csv`) | Yes -- used |
| `data/csv_hus.zip` | 2024 ACS 1-Year PUMS housing file (`psam_husa.csv`, `psam_husb.csv`) -- `HINCP` source | Yes -- used |
| `data/PUMS_Data_Dictionary_2024.pdf` | PUMS variable dictionary | Yes (verification) |
| `data/CCES24_Common_OUTPUT_vv_topost_final.csv` | 2024 CES Common Content, 60,000 respondents | Yes -- used |
| `data/CCES24_Common_pre.docx` | CES pre-election questionnaire | Yes (verification) |
| `data/CES_2024_GUIDE_vv.pdf` | CES Guide (weights, codebook) | Yes (verification) |
| `data/census_cells.csv` | pre-existing template for a different pipeline | No -- audit-only |
| `data/nc-est2024-asr6h.xlsx` | Census Vintage 2024 workbook | No -- audit-only |
| `data/ate_archive.csv`, `data/README.md` | belong to `regularized_probability_elicitation.py` | No -- out of scope, untouched |

Full detail with SHA-256/size/timestamp: `data/derived/population/source_manifest.json`.

## 3. Why the benchmark quota margins are operative

Per §5, the preregistration publishes gender x age and gender x race
margins but does not identify a unique gender x age x race joint
distribution -- so `config/quota_gender_age_1000.csv` and
`config/quota_gender_race_1000.csv` are the pipeline's operative targets,
consumed by `src/population/raking.py`. Neither the NC-EST workbook nor the
pre-existing `census_cells.csv` is read by `raking.py` or `sampling.py` at
all (verified by
`tests/population/test_population_outputs.py::test_raking_and_sampling_never_reference_audit_only_files`).

## 4. Why NC-EST and census_cells.csv are audit-only

See `reports/population/nc_est_audit.md` and
`reports/population/existing_census_cells_audit.md`. NC-EST's age/race
binning doesn't match this task's canonical categories and is a different
reference frame than the preregistration's own quota-setting process; the
existing `census_cells.csv` is an empty template belonging to a different
component of this repository, never populated, and never an operative
target even there.

## 5. Reconciliation of the rounded Male/Female totals

`config/quota_gender_age_1000.csv` and `config/quota_gender_race_1000.csv`
were supplied as the task's fixed, already-reconciled integer margins. Both
total exactly 1,000 and both agree on Male = 490 / Female = 510 (verified in
`load_quota_tables()`, which raises on any disagreement). The real build's
achieved margins match every one of the 18 target rows exactly (§10).

## 6. PUMS inclusion and exclusion flow

**Real, end to end.** Person file: 3,422,888 rows (1,743,751 from
`psam_pusa.csv` + 1,679,137 from `psam_pusb.csv`). Housing file:
1,631,969 rows (827,133 + 804,836), `SERIALNO` verified unique across the
combined housing file. Join: 94.6% of person rows matched a housing
`SERIALNO`. After all 8 inclusion criteria (`AGEP>=18`, state in
50-states-or-DC, `PWGTP` finite/>0, valid `SEX`/`HISP`/`RAC1P`/`SCHL`,
nonmissing `HINCP`, valid `ADJINC`): **2,612,172 rows** remain eligible
donors. Per-step counts: `reports/population/exclusion_flow.csv`.

## 7. Exact PUMS recodes

Verified against `data/PUMS_Data_Dictionary_2024.pdf` -- full table with
line-number citations in `reports/population/pums_variable_audit.md`
(updated for the housing join: `HINCP` verified present in
`psam_husa.csv`/`psam_husb.csv`'s header, 241 columns each, identical).
Summary: `SEX` 1/2->Male/Female; `AGEP` banded at 18/29/30/44/45/59/60;
`HISP != "01"` takes priority over `RAC1P` (1/2/6->White/Black/Asian, else
Other); `SCHL` banded at 1-15/16-17/18-20/21/22-23/24;
`income_adjusted_2024 = HINCP * ADJINC / 1e6` banded at 30k/56k/100k/168k;
`STATE` mapped via a 51-entry FIPS table, `72` (Puerto Rico) excluded. All
boundary values in `tests/population/test_pums_recodes.py` pass.

## 8. IPF method and convergence

`raking.ipf_2d()` rakes a 4 (age_band) x 5 (race) seed matrix -- summed real
`PWGTP` per cell, separately for Male and Female -- to the quota margins,
tolerance `1e-10`, max `10,000` iterations. **Real result:** Male converged
in **7 iterations** (max residual error **9.87e-12**); Female converged in
**7 iterations** (max residual error **8.34e-12**). Both well under
tolerance.

## 9. Controlled integerization method

`raking.controlled_integerize()`: floor every IPF cell, compute each row's
and column's remaining shortfall, solve a binary MILP (`scipy.optimize.milp`,
HiGHS) assigning exactly the shortfalls to specific cells, objective
preferring the largest fractional remainders with a canonical-order
tie-break. Status `0` (optimal) for both genders in the real run.

## 10. Exact achieved quota tables

**Every one of the 40 real cells matches its quota exactly** (full table:
`data/derived/population/joint_cells_40.csv`; margin audits:
`reports/population/quota_audit_gender_{age,race}.csv`, all 18 rows
`exact_match=True`).

Gender x age_band: Male 18-29 103/103, Female 18-29 99/99, Male 30-44
131/131, Female 30-44 129/129, Male 45-59 114/114, Female 45-59 115/115,
Male 60+ 142/142, Female 60+ 167/167.

Gender x race: Male Asian 32/32, Female Asian 35/35, Male Black 58/58,
Female Black 65/65, Male Hispanic 91/91, Female Hispanic 90/90, Male Other
13/13, Female Other 14/14, Male White 296/296, Female White 306/306.

## 11. Education, income, state, and age distributions

**Real, from `profiles_core_1000.csv`.**

Education: Some college or Associate's 277, HS diploma/GED 269, Bachelor's
193, Master's/Professional 132, Less than HS 108, Doctorate 21.

Income: $100k-$167,999 260, $56k-$99,999 254, $168k+ 221, $30k-$55,999 155,
Less than $30k 110.

State: 51 distinct states/DC represented; top 10 by count: CA 119, TX 94,
FL 68, IL 57, NY 53, PA 47, GA 39, NJ 29, OH 29, WA 26.

Age: mean 48.2, median 47, range 18-95 (not itself quota-constrained --
only the 4-band age_band margin is -- so within-band age varies naturally
with the real ACS donor pool).

## 12. CES variable and weight selection

**Real.** Weight: `commonweight` (verified nonmissing for all 60,000 rows).
Variables: `caseid`, `commonweight`, `inputstate`, `birthyr`, `gender4`,
`educ`, `race`, `hispanic`, `faminc_new`, `pid3`. Full evidence trail:
`reports/population/ces_variable_audit.md`,
`data/derived/population/ces_variable_mapping.yaml`.

## 13. CES party model diagnostics

**Real.** 54,857/60,000 CES rows retained after requiring complete
predictors/target/positive weight; stratified 80/20 split (43,886 train /
10,971 test) via the `ces_diagnostic_split` RNG stream. Test-set weighted
log loss **1.224**, weighted accuracy **0.432**, converged in 73 iterations
(`reports/population/party_model_diagnostics.json`,
`party_confusion_matrix.csv`). Refit on all 54,857 valid rows and saved to
`models/population/ces_party_model.joblib`.

## 14. CES, expected-profile, and realized-profile party shares

CES weighted (54,857 valid rows): Democrat 32.8%, Republican 29.9%,
Independent 27.2%, Other 10.1%. Mean predicted probability across the real
1,000 selected profiles: Democrat 34.1%, Republican 29.6%, Independent
26.9%, Other 9.4% -- close to the CES baseline, as expected since the panel
is now demographically representative (unlike the earlier synthetic
demonstration). Realized sampled draws (one categorical draw per latent
profile via the `party_sampling` RNG stream, not argmax): Democrat 33.2%,
Independent 29.2%, Republican 28.9%, Other 8.7%.

## 15. Other-gender limitation

`other_gender_mode: none`: the primary panel is Male/Female only (490/510),
because the preregistration publishes no target share for an "Other" gender
category. The `ces_sensitivity` mode (§20) is not implemented in this pass.

## 16. Reuse of latent profiles across conditions

Each of the 1,000 latent profiles appears exactly once in control and exactly
once in each of the 16 interventions -- verified both structurally
(`tests/population/test_population_outputs.py`) and in the real 17,000-row
roster (1,000 rows/condition x 17 = 17,000, all 17 exact
condition labels present).

## 17. Limitations

- The `ces_sensitivity` other-gender mode (§20) is not implemented.
- The human reference this benchmark scores against is an opt-in,
  quality-screened sample; ACS PUMS and CES cannot reproduce its
  attention-check, bot-screening, or differential-attrition selection --
  this panel matches published quota margins, not that selection process.
- Education, household income, and party were not hard quota constraints
  (only gender x age_band x race is raked); they are inherited from the
  real ACS donor (education/income, naturally correlated with the raked
  cells) or imputed probabilistically (party), not independently targeted.
- Party assignments are **model-based** (a weighted multinomial logistic
  regression fit on CES), not an observed ACS variable -- ACS collects no
  partisan-identity item.
- This benchmark's schema snapshot (`config/benchmark_schema.yaml`, dated
  2026-08-04) must be rechecked against the final preregistration materials
  before the prediction lock.

---

"This is a quota-aligned synthetic profile panel, not a probability sample
of the eventual analyzed human respondents. It exactly matches the
benchmark's published gender-by-age and gender-by-race quota margins. The
joint age-by-race distribution and the distributions of education,
household income, exact age, and state are inherited from weighted 2024 ACS
PUMS donor records. Partisan identity is imputed probabilistically from a
weighted model estimated using the 2024 CES Common Content. The primary
panel does not impose an externally assumed prevalence for the unidentified
Other-gender category."
