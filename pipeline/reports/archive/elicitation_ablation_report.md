# Elicitation ablation report

Per the approach's §9. Both the 70-study archive (`data/ate_archive.csv`, built by `scripts/build_ate_archive.py` from `data/capsule-9843791-data.zip`) and the external baseline-calibration source (`data/baseline_references.json`, built by `scripts/build_baseline_reference.py` from `data/wv7c3-osfstorage-archive.zip`) are now real, supplied by the user (see `data/README.md`). What's below is real throughout: an actual leave-one-study-out shrinkage fit on the real archive, real control-distribution baseline references, test suite results, and small real elicitation demos. Only the fine-tuning variant remains blocked -- by compute, not by missing data (see §3).

## 1. Test suite results (real)

- `tests/elicitation`: 39 passed in 3.31s
- `tests/population`: 117 passed in 50.94s

## 2. Real global-lambda shrinkage fit, on the real 70-study archive

- 156 (study, outcome, hypothesis) rows across 51 studies (of the archive's 70 -- the rest were dropped for single-sided hypotheses or an RCT/LLM/scale mismatch; see `scripts/build_ate_archive.py`'s printed `skipped` counts)
- lambda = **0.066** (unconstrained: 0.066)
- leave-one-study-out RMSE: shrunk = 11.93 pp vs. raw (lambda=1) = 17.72 pp -- shrinkage is retained (it improves held-out RMSE).
- Heavy shrinkage (lambda far below 1) on this real, diverse 70-study sample is consistent with the approach's own rationale [4,5]: GPT-4's predicted effects here are, if anything, weakly *negatively* correlated with the real human effects across this heterogeneous set of studies, so the fitted lambda pulls hard toward control rather than trusting the raw predicted magnitude.

## 3. Real external baseline references (climate-advocacy megastudy control group)

| outcome           |   reference_mean |   reference_variance |   reference_n | source                                        |
|:------------------|-----------------:|---------------------:|--------------:|:----------------------------------------------|
| belief_post       |            54.34 |              1301.11 |          1204 | climate-advocacy megastudy, control condition |
| policy_general    |            52.98 |              1309.37 |          1200 | climate-advocacy megastudy, control condition |
| donation_ams      |             4.77 |                14.38 |          1212 | climate-advocacy megastudy, control condition |
| newsletter_signup |             0.24 |                 0.18 |          1187 | climate-advocacy megastudy, control condition |

Covers 4 of this benchmark's 13 outcomes directly (same scale, real control-group respondents); the other 9 (trust subscales, institutional trust, concern, specific policy items, behavior_mean) have no corresponding item in this megastudy and are left uncalibrated, not guessed.

## 4. Small real elicitation demos across ablation arms (real)

One profile, one item (`trust_post`), one intervention, `n_permutations=2` for speed; `lambda=1` for this specific demo item (`trust_post` has no archive-fit counterpart -- the archive's own outcome vocabulary doesn't include it) -- this demonstrates each arm *executes correctly and produces distinct, real model output*, not comparative accuracy.

| meta_instruction   | shrinkage_rule   |   p0_mean |   pt_mean |   q_mean |   elicit_seconds |
|:-------------------|:-----------------|----------:|----------:|---------:|-----------------:|
| False              | mixture          |   66.4899 |   78.4067 |  78.4067 |             14.8 |
| False              | quantile         |   66.4899 |   78.4067 |  78.405  |             14.8 |
| False              | moment           |   66.4899 |   78.4067 |  78.4067 |             14.8 |
| True               | mixture          |   46.8206 |   61.1682 |  61.1682 |             16   |
| True               | quantile         |   46.8206 |   61.1682 |  61.175  |             16   |
| True               | moment           |   46.8206 |   61.1682 |  61.1682 |             16   |

## 5. Ablation arms: status

| arm                                                   | status                          | detail                                                                                                                                                                                                                                                                                                                                                                                        |
|:------------------------------------------------------|:--------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Uncalibrated (lambda=1, mixture rule)                 | REAL                            | demonstrated in §4; also the fallback for any item the real archive doesn't cover                                                                                                                                                                                                                                                                                                             |
| Global-lambda shrinkage                               | REAL                            | distributional_calibration.fit_global_shrinkage on the real 70-study archive: lambda=0.066 (§2); also unit-tested on synthetic data with a known ground truth (tests/elicitation/test_shrinkage.py)                                                                                                                                                                                           |
| Hierarchical (treatment/outcome-family) shrinkage     | TESTED (synthetic archive only) | distributional_calibration.fit_hierarchical_shrinkage/select_shrinkage_specification; recovers a known family difference on synthetic data; the real archive has no treatment_family (honestly left blank -- see data/README.md), so it falls back to the global fit in §2                                                                                                                    |
| Distributional (quantile) shrinkage                   | REAL + TESTED                   | distributional_calibration.quantile_shrink; demonstrated in §4 and unit-tested (boundary conditions, variance movement); used with the real lambda from §2 in scripts/generate_responses.py's default run                                                                                                                                                                                     |
| Moment (mean/dispersion) shrinkage fallback           | REAL + TESTED                   | distributional_calibration.moment_shrink; demonstrated in §4; precise on smooth shapes, direction-only on jagged ones (documented limitation)                                                                                                                                                                                                                                                 |
| Prompt meta-knowledge instruction                     | REAL, off by default            | hf_elicit.META_INSTRUCTION; demonstrated on/off in §4; kept off per §3 ('selected only using external validation data') -- neither archive resolves this, since it's about prompt content, not calibration                                                                                                                                                                                    |
| Label-permutation averaging (R>2)                     | REAL + TESTED                   | regularized_probability_elicitation.balanced_labels; full Latin-square balance verified in tests/elicitation/test_balanced_labels.py; default min(4, n_options)                                                                                                                                                                                                                               |
| External baseline calibration (control distributions) | REAL, for 4/13 outcomes         | baseline_calibration.py + data/baseline_references.json (§3); belief_post/policy_general/donation_ams/newsletter_signup only -- the other 9 outcomes have no corresponding item in the climate-advocacy megastudy                                                                                                                                                                             |
| Fine-tuning variant                                   | BLOCKED (compute, not data)     | a real, suitable training dataset exists (climate_advocacy_megastudy/data/advocacy_data.csv, 31,324 real respondents with demographics, on this repo's own scales) -- but the fine-tuning procedure itself has not been implemented or run (a separate, substantial compute undertaking not attempted in this pass, on top of the already-measured elicitation cost for the primary pipeline) |

