

# Silicon Sample Benchmark — team_10 submission

This repository **is** a submission to the [Silicon Sample Benchmark](https://janpfander.github.io/llm_predictions_megastudy/):
a multi-team benchmark of AI approaches for predicting the results of a behavioral megastudy on
trust in climate scientists, *before* the human data are revealed.

## This submission

**Approach.** Tier 1, respondent-level survey simulation with external calibration.
`google/gemma-4-31B-it`, served via Together AI, generates one complete native survey response
per respondent per condition — the same 1,000 respondent profiles under control and each of the
16 interventions, 17,000 submitted rows in total. Gemma was selected over
`deepseek-ai/DeepSeek-V4-Pro-0813` using a pre-specified and frozen American Trends Panel (ATP
W149) external screen on real respondent-level survey items (`pipeline/ate/g_atp_screen.py`;
results under `pipeline/outputs/scientific_bakeoff/g_model_screen/`).

**Respondent construction.** The 1,000 respondent profiles are drawn from the 2024 Cooperative
Election Study (CES) Common Content (`pipeline/data/generated/g_personas_master.csv`), built
and audited by `pipeline/src/population/` + `pipeline/scripts/build_ces_production_roster.py`
(construction/provenance reports under `pipeline/reports/population/` and
`pipeline/outputs/persona_validation/`). The same 1,000 respondents are reused across all 17
conditions to hold demographic composition fixed.

**Calibration.** The 208 range-normalized treatment effects (16 interventions × 13 outcomes) are
calibrated by an additive overall-mean calibration: their overall mean is shifted onto an
external target computed from 136 human treatment effects from 31 completed randomized
experiments (`pipeline/data/ate_archive.csv`), with each study given equal weight. The same
shift applies to every value, so it preserves their full relative pattern exactly before support
projection; the estimator does not use any LLM forecast. The calibrated values then pass through
a deterministic native-support projection (`pipeline/ate/target_projection.py`) to land back on
each outcome’s valid integer support and reconstruct composites. See `registration.md` §G–J for
the full estimator, equations, and selection procedure.

**Post-freeze external diagnostic.** An independent post-freeze external diagnostic was
conducted using Orchinik et al. (2024) after the final estimator and target predictions were
frozen. It did not influence model selection, calibration, or the submitted predictions.

**Production & special cases.** Control-condition respondents receive one of three neutral filler
texts (neckties / baseball / dances), assigned by a frozen, reproducible balanced randomization
matching the survey’s `EvenPresentation` design, realized as 334 / 333 / 333. The Consensus
condition is administered as four genuinely sequential, chained requests per respondent —
STEP_1, STEP_2, STEP_3, then OUTCOMES — each carrying forward the prior steps’ real answers and
feedback; only the OUTCOMES-stage response enters the submitted Consensus row. Any request that
fails delivery or schema validation may be retried up to a bounded maximum of three attempts,
with the earliest technically valid response retained; retry eligibility never depends on
response content.

**Reproducibility / offline checks.** No step above requires network access or paid inference
to audit: `python3 -m pytest -q` under `pipeline/` runs the full offline test suite (unit tests,
synthetic-fixture guard/reconciliation tests, and structural checks against the real frozen
manifests already in the repo); `make validate-personas`, `make validate-prompts`, and the
`pipeline/scripts/score_*`/`assemble_*` reconciliation scripts re-derive accounting from
already-retrieved provider output without recomputing anything scientific.

**Current final artifacts.** The submitted Tier-1 file is `predictions/team_10_T1_primary_v1.csv`
(SHA-256 `8c8ef8661565769f7183283b6d09343691432794d2de1bbf3f0ed3145584cfb3`), containing 17,000
rows for 1,000 respondent profiles across control and each of the 16 interventions. The
corresponding native respondent-level simulation output is deposited as
`raw_data_deposit/final_first_valid_native_g_responses.csv`. Detailed row-level provenance and
reproducibility artifacts are available under `pipeline/outputs/target_production/`.

**Disclosure / deposit.** [`registration.md`](registration.md) is the complete method
disclosure; `metadata.json` and `.zenodo.json` carry the deposit metadata.

## Survey instrument

The full instrument is provided as two files. **Both encode the same survey**; they differ only in
format and intended use:

|  | `survey/survey.qsf` | `survey/survey.json` |
|----|----|----|
| **What it is** | Qualtrics’ proprietary survey-export file | Qualtrics’ documented Survey-Definitions API output |
| **Format** | JSON, but an undocumented proprietary structure | JSON with a documented schema (`result.Questions`, `result.Blocks`, `result.SurveyFlow`, …) |
| **Best for** | re-importing into Qualtrics to **run** the survey yourself | **reading / parsing** the instrument programmatically — e.g. individual participant simulations that need the items, response scales, block/flow order, branching and randomization a respondent saw |
| **Qualtrics license** | required (to import and run) | not required (it is plain JSON anyone can read) |

In short: use `survey.qsf` if you want to *run* the survey in Qualtrics; use `survey.json`
if you want to *read* it without a Qualtrics account.

> **Scope note.** These files are the reduced *LLM-simulation* instrument: respondents are routed
> through the non-interactive conditions only (assigned by a block randomizer); the interactive chatbot
> arms have been removed. The condition labels you are scored on are defined in
> `survey/condition_codenames.csv`, the outcomes in `codebook.csv`, and both in
> `scripts/lib/submission_spec.R`; treat those as authoritative for scope, and the two survey files as
> the faithful record of the instrument.

A human-readable rendering is also provided as `survey/questionnaire.txt`, laid out in
chronological survey order (the order a respondent moves through the instrument). Every item is
annotated as `[qualtrics_label · answer values] question`, alongside the condition labels and the
intervention stimulus texts.

Tier-1 runs export raw Qualtrics column names; `make clean` maps them to the analysis schema
documented in `codebook.csv`.

## Licensing of the shipped survey materials

Your Zenodo license (default `CC-BY-4.0` in `metadata.json`) applies to **your** contribution —
your code, predictions, and documentation. The shipped `survey/` folder is different: several
intervention stimulus texts adapt previously published journalism and other copyrighted material,
included here for scholarly research use. Keep `survey/` in your deposit unchanged (it documents
what your respondents saw), but your license grant does not — and cannot — re-license those
underlying texts.

## More

Common questions — the multi-pair condition code names, attention checks,
what feedback you get and when — are answered in [`FAQ.md`](FAQ.md). Tiers, scoring, disclosure
classes, and the full timeline are described in the
[call for participation](https://janpfander.github.io/llm_predictions_megastudy/). Questions:
see the call’s Contact page.
