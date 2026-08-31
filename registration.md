# Silicon Sample Benchmark — method registration form

Fill in every item before the prediction lock; this file ships inside your repo's Zenodo release
(see the README's **Deposit** step). This form covers **one entry** (one repo / one Zenodo release,
`primary` or `secondary-k` — see the README's **What counts as a submission**); if you submit several
entries, fill one form per entry. Items marked **★**
must be disclosed **fully publicly** (never escrowed or withheld). Items marked **†** must be at
minimum escrowed — they may be sealed from the public, but never withheld from the core team. Items
not applicable to your approach: write `N/A`. When several models serve different pipeline stages, complete the model
sections (B) once per model. See the call's **Disclosure policy** for escrow rules.

---

## 0 · Approach identity and output

- **0.1 Team ★** — team ID: team_10; members: Kerstin Forster, Stefan Feuerriegel; affiliations: LMU Munich, Munich Center for Machine Learning; corresponding contact: Kerstin Forster, kerstin.forster@lmu.de.

- **0.2 Plain-language summary ★** — Our approach uses Gemma 4 31B (`google/gemma-4-31B-it`) to simulate 1,000 U.S. respondent profiles under the control condition and each of the 16 interventions, producing 17,000 individual survey responses and 208 treatment effects. We calibrate the overall treatment effect level using 136 human treatment effects from 31 completed randomized experiments, giving each study equal weight. The calibration shifts the overall mean of the range-normalized simulated effects to the external estimate while preserving their relative pattern. Final responses are then mapped to the survey response scales.

- **0.3 Submission tier & approach family ★** — Tier 1; population-grounded respondent-level survey simulation with external overall-mean calibration.

- **0.4 Pipeline diagram** — CES 2024 microdata → 1,000 respondent profiles → condition-specific Gemma survey simulation → 208 range-normalized treatment effects → external overall-mean calibration → native-scale response projection and composite reconstruction → 17,000-row Tier-1 file.

- **0.5 Coverage ★** — Full coverage: 1,000 submitted respondents in control and in each of 16 interventions, yielding 17,000 rows and all 16 × 13 = 208 intervention–outcome ATE values, with the required demographic moderators and respondent-level outcomes.

## A · Scope of LLM use

- **A.1 Purpose** — Gemma generates joint respondent-level survey responses under each experimental condition. The simulations provide baseline response distributions, cross-outcome dependence, subgroup heterogeneity, and intervention–outcome treatment effects. Completed human experiments provide an external reference for the overall treatment effect level; deterministic calibration and native-support projection produce the final benchmark responses.

- **A.2 Degree of automation ★** — Fully automated at prediction time. Post-processing is deterministic and reproducible from the frozen provider outputs. 

## B · Model / system details (once per model)

- **B.1 Model name(s)** — `google/gemma-4-31B-it` (Gemma 4 31B instruction-tuned; Google DeepMind), served by Together AI. Model card: https://ai.google.dev/gemma/docs/core/model_card_4

- **B.2 Access & context mode** — Together AI API v1 / Batch API using chat-completions requests. Each simulated respondent is generated in a fresh condition-specific interaction; the sequential Consensus intervention carries only the state required by its within-condition sequence.

- **B.3 Configuration** — temperature 1.0; top-p 0.95; `top_k` omitted from the provider request, with the effective provider-side default not independently established; max tokens 1,024; presence/frequency penalties 0; stop unset; one completion per production attempt; request-specific seeds recorded in manifests; `chat_template_kwargs={"enable_thinking": false}`; structured responses without chain-of-thought output.

- **B.4 Customization** — The released model is used without fine-tuning. Each request contains the respondent profile, relevant experimental material, questionnaire, and a structured response format. No retrieval, browsing, or external tools are used.

- **B.5 Persistent memory** — Interactions are condition-based. Respondents and experimental conditions do not share conversational state; the sequential Consensus condition retains only its own condition-specific state.

- **B.6 Inference stack** — N/A. Provider-hosted Together AI inference. Provider request metadata and serving fingerprints are retained with the raw logs.

- **B.7 Ensembles** — N/A. The final approach uses a single LLM model (`google/gemma-4-31B-it`).

## C · Prompts

- **C.1 Exact prompts** — Exact compiler code, rendered prompts, request manifests, and hashes are deposited. The final prompts reproduce the participant-visible benchmark materials and questionnaire. Consensus is implemented sequentially as described in E.2.

- **C.2 System-wide instructions** — Adopt the supplied demographic profile; answer the survey as that respondent; use the stated response scales exactly; do not explain or discuss the study design; return the requested structured responses.

- **C.3 Prompt-design rationale** — Prompts use observed pretreatment demographics, benchmark-native survey wording, and constrained response supports. This design follows evidence that demographic conditioning can recover meaningful subgroup structure while persona formulation and response-generation format can affect synthetic survey fidelity (Argyle et al., *Political Analysis*, 2023, doi:10.1017/pan.2023.2; Lutz et al., *Findings of EMNLP*, 2025, doi:10.18653/v1/2025.findings-emnlp.1261; Ahnert et al., ACL 2026, doi:10.18653/v1/2026.acl-long.1927).

## D · Persona / profile construction (Tiers 1–2)

- **D.1 Profile source** — 2024 Cooperative Election Study (CES) Common Content, Harvard Dataverse, doi:10.7910/DVN/X11EP6. The target population consists of 1,000 complete respondent rows selected to preserve observed joint demographic relationships while matching the benchmark population constraints.

- **D.2 Profile verbalization** — Deterministic template over observed pretreatment characteristics: age, gender, race/ethnicity, education, household income, party identification, state, and—when observed—political ideology and religion. Climate attitudes, trust outcomes, policy attitudes, and other target-adjacent variables are excluded.

- **D.3 Assignment & weighting** — The same 1,000 respondent profiles are used across all 17 conditions, eliminating arm-to-arm differences in respondent composition. Profiles are quota-aligned to the published gender × age and gender × race targets where defined. Submitted condition means are unweighted.

## E · Stimulus and survey administration

- **E.1 Stimulus presentation** — Participant-visible benchmark materials are reproduced verbatim. For the extreme-weather intervention, respondents receive the stimulus corresponding to their state. In the control condition, respondents are assigned one of the three neutral filler texts (neckties / baseball / dances) using a frozen reproducible balanced randomization matching the survey's `EvenPresentation` design. Condition names, study hypotheses, and intended treatment directions are excluded from participant-facing prompts.

- **E.2 Survey walk-through** — Control and the 15 non-Consensus interventions are administered in one fresh condition-specific interaction per respondent. Consensus is administered through four sequential requests: STEP_1, STEP_2, STEP_3, and OUTCOMES. Item 3 is always presented second, items 1 and 2 occupy the first and third positions according to the benchmark randomization, feedback follows each response, and the final Consensus summary is shown before the outcome questionnaire. Only the OUTCOMES response enters the submitted Consensus row.

- **E.3 Response elicitation** — Provider-supported structured generation on benchmark-native response supports. Raw sliders are generated as 0–100 integers, donation as a 0–10 integer, and newsletter signup as binary; multi-item outcomes are derived from their constituent responses.

## F · Stochasticity and aggregation

- **F.1 Runs & seeds** — One stochastic response is retained for each respondent profile and condition. We use a single draw rather than averaging multiple generations because the target is a respondent-level synthetic population; averaging repeated draws for the same profile would mechanically reduce stochastic response variation and compress the resulting response distributions. Request-specific seeds and inference parameters are logged. If a request fails delivery or schema validation, it may be retried up to the prespecified maximum of three attempts; the first technically valid response is retained.

- **F.2 Aggregation rule** — Population means and treatment effects are computed from the resulting 1,000 responses per condition.

## G · Validation & post-processing

- **G.1 Human validation** — N/A. Model responses are not manually reviewed, edited, or filtered for plausibility.

- **G.2 Post-processing** — Responses must satisfy their request-specific JSON schema and native response support. Technical delivery or schema failures may be retried as described in F.1; substantive response values never determine retry eligibility.

  After calibration, control responses remain unchanged, and treatment responses receive the outcome-specific correction defined in G.3. For ordinary items and mean composites, the correction is applied to the constituent raw items. For the reverse-scored outcome `funding_perceptions = 100 - funding_5`, the corresponding raw item is shifted in the opposite direction.

  Raw-item ideals are then projected to their valid integer supports while minimizing squared deviation and matching the nearest attainable target total. Equal-cost ties are resolved reproducibly using a frozen SHA-256 priority based on a fixed seed, intervention, raw item, and respondent identifier. Demographics and target human outcomes do not enter the tie-break. Multi-item outcomes are reconstructed from the projected constituent items using the benchmark definitions. Because response supports are discrete, final ATEs may differ slightly from their pre-projection targets.

- **G.3 Calibration corrections** — The final estimator calibrates the overall mean of the simulated range-normalized treatment effects using an external archive of completed human experiments while retaining the relative pattern of the simulated effects.

  Let $`N=1000`$, let $`a\in\{1,\ldots,16\}`$ index interventions, and let $`j\in\{1,\ldots,13\}`$ index outcomes. Let $`\widetilde{Y}^{\mathrm{sim}}_{iaj}`$ and $`\widetilde{Y}^{\mathrm{sim}}_{i0j}`$ denote the simulated scored outcome values for respondent $`i`$ under intervention $`a`$ and control, respectively, and let $`R_j>0`$ denote the attainable range width of outcome $`j`$.

  The simulated ATE and its range-normalized form are

$$\widetilde{\tau}^{\mathrm{sim}}_{aj} = \frac{1}{N} \sum_{i=1}^{N} \left( \widetilde{Y}^{\mathrm{sim}}_{iaj} - \widetilde{Y}^{\mathrm{sim}}_{i0j} \right), \qquad g_{aj} = 100 \frac{\widetilde{\tau}^{\mathrm{sim}}_{aj}}{R_j}.$$


  The overall mean across the $`16\times13=208`$ values is

$$\bar g = \frac{1}{208} \sum_{a=1}^{16} \sum_{j=1}^{13} g_{aj}.$$


  Target and external effects are computed on their scored outcome representations using the same frozen orientation convention. `funding_perceptions` is reverse-scored as `100 - funding_5` before effects are computed; no additional sign multiplier is applied during calibration.

  The external calibration archive contains 136 eligible treatment effects from 31 completed randomized experiments. Each external effect is normalized by its native outcome range, effects are averaged within study, and the 31 study means receive equal weight. The resulting external mean is

$$\widehat{\mu}_{\mathrm{ext}} = 1.9558595458395387 \approx 1.96$$


  percentage points of native outcome range.

  The calibrated pre-projection normalized ATE is

$$\widehat{\theta}_{aj} = \widehat{\mu}_{\mathrm{ext}} + \left( g_{aj}-\bar g \right).$$


  This is an additive overall-mean calibration: every normalized ATE receives the same shift $`\widehat{\mu}_{\mathrm{ext}}-\bar g`$. It therefore preserves all pairwise differences among the 208 normalized simulated effects before support projection.

  The corresponding native-scale target is

$$\widehat{\tau}_{aj} = \frac{R_j}{100} \widehat{\theta}_{aj},$$


  and the treatment-arm correction simplifies to

$$c_j = \widehat{\tau}_{aj} - \widetilde{\tau}^{\mathrm{sim}}_{aj} = \frac{R_j}{100} \left( \widehat{\mu}_{\mathrm{ext}}-\bar g \right).$$


  Thus the same native-scale correction $`c_j`$ is applied across interventions for a given outcome. Control responses are not shifted. Native-support projection may introduce small deviations between the calibrated pre-projection target and the final submitted ATE.

  The calibration is motivated by evidence that LLM simulations can recover experimental signal while misestimating treatment effect levels (Ashokkumar et al., *Nature*, 2026, doi:10.1038/s41586-026-10742-x) and by evidence of residual bias in unrectified synthetic survey responses (Krsteski et al., ACL 2026, doi:10.18653/v1/2026.acl-long.498). Target human outcomes do not enter calibration.

## H · Learning and conditioning components

- **H.1 Fine-tuning data** — N/A. Released Gemma weights are used directly.

- **H.2 Context & retrieval corpora** — N/A. Inference context is constructed directly from the respondent profile, participant-visible condition material, and benchmark questionnaire rather than an auxiliary retrieval corpus.

## I · Data inputs, blinding, and competing interests

- **I.1 Competing interests ★** — The authors declare no competing interests.

- **I.2 External human data †** — External human data comprise: (i) CES 2024 Common Content (doi:10.7910/DVN/X11EP6) for respondent-profile construction; (ii) two American Trends Panel W149 items distributed with Krsteski et al. (ACL 2026), `LIVSTAN_W149` and `SCOTUS_JOB_W149`, used for respondent-simulator model selection; (iii) Orchinik et al. (2024), *PNAS Nexus*, doi:10.1093/pnasnexus/pgae485, used only for a post-freeze diagnostic; and (iv) 136 eligible treatment effects from 31 randomized experiments in the Ashokkumar et al. (2026) 70-study archive, used to estimate the external calibration mean. The deposited calibration table records the included studies, treatment/control definitions, outcome ranges, and effect definitions.

- **I.3 Blinding attestation ★** — **mandatory.** We attest that no team member accessed, solicited, inferred, or was shown human outcome data from the target megastudy, including pilots, before the prediction lock. Target human outcomes do not enter model selection, external calibration, retry eligibility, post-processing, or method choice. Kerstin Forster, Stefan Feuerriegel, 31.08.2026.

- **I.4 Contamination note †** — The official Gemma 4 model card reports a January 2025 pretraining-data cutoff. This does not establish that individual pre-2025 external studies were absent from training, and we therefore do not characterize the external validation sets as contamination-free holdouts. The target human outcomes remained unreleased and inaccessible to the team throughout method development.

## J · Internal selection procedure

- **J.1 Design-space search †** — Gemma-4-31B-it and DeepSeek-V4-Pro-0813 were compared on two American Trends Panel W149 items using the equal-weight mean normalized Wasserstein-1 distance between simulated and human response distributions. Gemma scored 7.5048 percentage points versus 14.2626 for DeepSeek and was selected as the respondent simulator. Gemma is an open-weight model, which supports transparency and reproducibility.


  Treatment effect calibration was selected using whole-study validation on the frozen 31-study external archive. The final estimator uses the equal-weighted external study mean as an overall-mean calibration target while retaining the relative pattern of the 208 simulated treatment effects. After the estimator and target prediction values were frozen, the method was evaluated on Orchinik et al. (2024). This post-freeze diagnostic did not influence the submitted method.

## K · Reproducibility & frozen artifacts

- **K.1 Code & materials** — The complete code and frozen materials are deposited in the repository: https://github.com/forsterkerstin/silicon-sample-submission. The Zenodo DOI will be added after release archiving.

- **K.2 Raw output logs †** — Complete provider outputs, request manifests, serving metadata, and technical-failure logs are deposited under `pipeline/outputs/target_production/`. `pipeline/outputs/target_production/final_submission_row_provenance.csv` maps every submitted row to the provider response retained for that row.

- **K.3 Computational resources** — Final target generation comprises 20,000 logical requests: 16,000 standard requests for control and the 15 non-Consensus interventions, plus 4,000 sequential Consensus requests. Only the OUTCOMES response is submitted for each Consensus profile. Additional physical calls arose from bounded technical retries, serving-format remediation, and the final control-filler correction; exact request counts, submission timestamps, token counts, and provider provenance are retained in the deposited production ledgers.

## L · Disclosure class

Each item above is deposited as **public**, **escrowed** (sealed from the public but available to the
core team and auditors under confidentiality, with a public SHA-256 hash + timestamp so the lock is
still verifiable — an embargo with a sunset date is encouraged), or **withheld** (permitted only for
items marked neither ★ nor †). Your entry's class is set by its **most restricted item** and recorded
in `metadata.json` → `disclosure_class` (and `escrow_doi` if anything is escrowed):

- **A · Open** — all items public. Full results-table standing; all features enter the design-choice analysis.
- **B · Escrowed** — some items sealed but every item is available to the core team/auditors under confidentiality. Full standing with an *escrowed* badge; only publicly disclosed features enter the design-choice analysis.
- **C · Sealed** — one or more permitted items withheld even from escrow. Scored and reported with a *not independently verifiable* flag; excluded from the approach catalogue and design-choice analysis.

★ items must always be public (never escrowed or withheld). † items must be at minimum escrowed. Full
policy: <https://janpfander.github.io/llm_predictions_megastudy/#disclosure>

Entry class: **A · Open**. Every ★ and † item above is deposited fully publicly in this repository; nothing is escrowed or withheld, so no `escrow_doi` applies.
