# CES 2024 Common Content variable audit

Machine-readable version: `data/derived/population/ces_variable_mapping.yaml`.
Sources: `data/CES_2024_GUIDE_vv.pdf` (the "Guide"), `data/CCES24_Common_pre.docx`
(the "Questionnaire"), `data/CCES24_Common_OUTPUT_vv_topost_final.csv` (the
"CSV", 60,000 rows). All frequencies below were computed directly from the CSV
and cross-checked against the Guide's own codebook appendix, which reports
identical counts.

## Respondent ID

- **Column:** `caseid`. Unique per-respondent identifier; used as-is.

## Adult-population analysis weight

- **Column:** `commonweight`.
- **Evidence (Guide, "Using Weights", p.17):** a table lists four weights --
  `commonweight` (All respondents / Adults), `commonpostweight` (Answered
  both waves / Adults), `vvweight` (Matched to validated registration record
  / Registered adults), `vvweight_post` (Answered both waves & matched to
  validated registration / Registered adults) -- followed by: *"We recommend
  the use of 'commonweight' any time researchers wish to characterize the
  opinions and behaviors of adult Americans... We recommend the use of
  'vvweight' or 'vvweight_post' any time researchers wish to characterize the
  opinions, behaviors, or traits of voters or registered voters."*
- **Rejected:** `vvweight`/`vvweight_post` (voter-only, explicitly excluded
  per task instructions); `commonpostweight` (restricted to post-wave
  completers; not needed since no post-election items are used here).
- **File-restriction check:** despite the CSV's `_topost` filename, it is
  **not** restricted to post-wave completers. `commonweight` is nonmissing
  for all 60,000 rows; `tookpost` splits 10,568 / 49,432 across two values,
  and `commonpostweight` is nonmissing for only the 49,432 subset. This
  matches instructions §14's stated exception condition failing to hold, so
  the general adult-population weight (`commonweight`) is used.

## State

- **Column:** `inputstate`.
- **Evidence (Questionnaire, item `inputstate`):** "What is your State of
  Residence?" -- coded 1=Alabama, 2=Alaska, 4=Arizona, 5=Arkansas, ...,
  56=Wyoming, using the same national FIPS-style numbering as ACS PUMS
  `STATE` (identical gaps at codes 3, 7, 14, 43, 52). Values in the CSV are
  numeric without zero-padding (e.g. `1`, not `01`); zero-padded before
  joining `pums.STATE_FIPS_TO_ABBR`.

## Age

- **Column:** `birthyr`.
- **Evidence (Questionnaire, item `birthyr`):** "In what year were you born?"
  (open integer). The module's screenout logic bounds it to 1924-2006 for the
  2024 fielding. Age is computed as 2024 (survey year) minus `birthyr` for
  CES-side model features; this is an approximation disclosed here, not the
  benchmark's own year_birth (which, for the submitted profiles, comes from
  the ACS donor's AGEP via `pums.recode_year_birth`, referenced to 2026).

## Gender

- **Column:** `gender4`.
- **Evidence (Questionnaire, item `gender4`):** "What is your gender?" --
  1=Man, 2=Woman, 3=Non-binary, 4=Other (open `gender4_t`), 8=Skipped,
  9=Not Asked.
- **Mapping:** 1->Male, 2->Female, {3,4}->Other, {8,9}->missing.
- **Observed frequencies:** Man 27,454; Woman 31,992; Non-binary 448; Other
  106 (Guide codebook appendix, p.28, and confirmed directly in the CSV).
- A separate binary `gender` item exists in the instrument (1=Male, 2=Female)
  but is gated `Show if 0` -- never administered in this wave -- and is not
  used.

## Race / Hispanic ethnicity

- **Columns:** `race`, `hispanic`.
- **Evidence (Questionnaire, item `race`):** "What racial or ethnic group
  best describes you?" -- 1=White, 2=Black or African-American,
  3=Hispanic or Latino, 4=Asian or Asian-American, 5=Native American,
  6=Two or more races, 7=Other (open `race_other`), 8=Middle Eastern.
- **Evidence (Questionnaire, item `hispanic`, shown only if `race != 3`):**
  "Are you of Spanish, Latino, or Hispanic origin or descent?" -- 1=Yes,
  2=No, 8=Skipped, 9=Not Asked.
- **Mapping rule (instructions §15):** Hispanic / Latino if `race == 3` OR
  `hispanic == 1`; otherwise non-Hispanic `race` 1->White / Caucasian,
  2->Black / African American, 4->Asian / Asian American,
  {5,6,7,8}->Other.
- **Observed frequencies (`race`):** White 41,443; Black 7,728; Hispanic
  5,150; Asian 1,949; Native American 582; Two or more races 1,947; Other
  1,035; Middle Eastern 166 (Guide codebook appendix, p.28-29; matches CSV
  exactly).
- **Observed frequencies (`hispanic`):** Yes 7,647; No 52,352; missing 1.

## Education

- **Column:** `educ`.
- **Evidence (Guide codebook appendix, p.28):** "What is the highest level
  of education you have completed?" -- No HS 2,133; High school graduate
  15,983; Some college 13,961; 2-year 6,666; 4-year 13,297; Post-grad 7,960
  (sums to 60,000; matches CSV codes 1-6 exactly). The Questionnaire docx
  does not carry this item's own text (it is a panel-profile item pulled from
  prior-wave data rather than re-administered), so the Guide's codebook
  appendix is the source of record here.
- **Mapping:** used only as a **harmonized_education** predictor inside the
  party model (instructions §17), not mapped 1:1 onto the benchmark's 6
  levels -- CES's single "Post-grad" bucket cannot be split into the
  benchmark's separate "Master's degree / Professional degree" vs. "Doctorate
  degree / Ph.D." categories, so forcing a 1:1 map would misclassify actual
  doctorate holders. Both sides are instead binned onto CES's native 6-level
  scheme (ACS side: SCHL 1-15->No HS, 16-17->HS grad, 18-19->Some college,
  20->2-year, 21->4-year, 22-24->Post-grad). The benchmark `education` field
  on the submitted profiles/roster comes only from the ACS SCHL recode
  (`pums.recode_education`), never from CES.

## Household income

- **Column:** `faminc_new`.
- **Evidence (Questionnaire, item `faminc_new`):** "Thinking back over the
  last year, what was your family's annual income?" -- 16 ordered brackets
  from "Less than $10,000" (1) to "$500,000 or more" (16), plus 97="Prefer
  not to say", 998=Skipped, 999=Not Asked.
- **Mapping:** per instructions §15, used only to build a
  **harmonized_income_ces** predictor: the ACS side's numeric
  `income_adjusted_2024` is binned onto these exact same 16 interval
  boundaries (top/bottom open-ended). 97/998/999 are treated as missing and
  excluded from party-model training. The benchmark `income` field on the
  submitted profiles/roster comes only from the ACS HINCP*ADJINC recode
  (`pums.recode_income`), never from CES.
- **Observed frequencies:** see `ces_variable_mapping.yaml`; 5,119 rows are
  code 97, 23 rows are genuinely missing.

## Party identification

- **Column:** `pid3`.
- **Evidence (Questionnaire, item `pid3`):** "Generally speaking, do you
  think of yourself as a...?" -- 1=Democrat, 2=Republican, 3=Independent,
  4=Other (open `pid3_t`), 5=Not sure, 8=Skipped, 9=Not Asked.
- **Mapping:** 1->Democrat, 2->Republican, 3->Independent, {4,5}->Other
  (both are substantive, documented response options per instructions §16's
  own example list, not inferred missing/refused).
- **Observed frequencies:** Democrat 22,982; Republican 15,913; Independent
  16,292; Other(open) 2,371; Not sure 2,442 -- sums to exactly 60,000; no
  Skipped/Not Asked rows are present in this file for `pid3`.
- **Explicitly not used:** `pid7` (7-point, derived from `pid3` plus a
  follow-up strength question), any `presvote*`/validated-vote/registration
  variable, and `ideo5`, per instructions §16.

## Unresolved documentation ambiguity

None for the variables audited above -- every value observed in the CSV for
`gender4`, `educ`, `race`, `hispanic`, and `pid3` was traceable to a
documented code in the Questionnaire or the Guide's codebook appendix, and
the frequencies matched exactly. The one open item is `HINCP`'s absence from
the ACS PUMS person file -- an ACS/PUMS issue, not a CES documentation issue;
see `reports/population/pums_variable_audit.md` and `population_report.md`.
