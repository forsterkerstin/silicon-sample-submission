"""Freeze the domain-validation protocol (OFFLINE ONLY, before any model
prediction is observed for either study).

Writes outputs/domain_validation/frozen_domain_validation_protocol.json.
Records exactly what was verified against the real, downloaded source
materials for Howe (2019) and Orchinik (2024) -- including a genuine,
evidenced BLOCKER for Howe and the verified (not invented) condition-label
mapping for Orchinik -- rather than assuming everything specified in the
task is available. Does not invoke any model, does not touch target G/F
output or MU_EXTERNAL/S2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_DATA_DIR = PIPELINE_ROOT / "data" / "domain_validation"
OUT_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    howe_files = {
        "Analysis_Howe.R": _sha256_file(DOMAIN_DATA_DIR / "howe" / "Analysis_Howe.R"),
        "Data_Howe.csv": _sha256_file(DOMAIN_DATA_DIR / "howe" / "Data_Howe.csv"),
    }
    orchinik_files = {
        name: _sha256_file(DOMAIN_DATA_DIR / "orchinik" / name)
        for name in ("Bovitz_qualtrics.docx", "Bovitz_qualtrics.qsf", "final_bovitz_raw.csv", "bovitz_data_clean.R", "final_clean.csv", "analysis.Rmd")
    }

    protocol = {
        "status": "FROZEN_BEFORE_MODEL_PREDICTIONS_OBSERVED",
        "target_human_outcomes_used": False,
        "target_g_scientific_outputs_used": False,
        "mu_external_reestimated": False,
        "s2_gamma_changed": False,
        "s2_frozen_unchanged": {"mu_external": 1.9558595458395387, "gamma_g_shape": 1.0},
        "howe": {
            "citation": "Howe, MacInnis, Krosnick, Markowitz, and Socolow (2019), 'Acknowledging uncertainty impacts public acceptance of climate scientists' predictions', Nature Climate Change, DOI 10.1038/s41558-019-0587-5",
            "source_url": "https://osf.io/tgmyh/",
            "source_files": howe_files,
            "source_files_downloaded_via": "OSF API (api.osf.io/v2/nodes/tgmyh/files/osfstorage/), sha256 independently verified against the API-reported hash before use",
            "sample_n_total_rows": 1174,
            "randomized_condition_variable": "Condition (6 levels: Fully Bounded, Fully Bounded and Irreducible, Partially Bounded, Partially Bounded and Irreducible, No Uncertainty, Irreducible)",
            "status": "BLOCKED",
            "blocker": (
                "The publicly released Howe et al. (2019) data (Data_Howe et al_Acknowledging Uncertainty.csv, "
                "18 columns) contains ONLY the authors' post-treatment DICHOTOMIZED trust mediator (trustmed2, "
                "values {0,1}), which the task explicitly forbids using. The raw five-category native trust item "
                "('How much do you trust the things scientists say about the environment...') is NOT present under "
                "any column name in the released CSV, and Analysis_Howe.R never re-derives trustmed2 from a raw "
                "ordinal source column within this release -- confirmed by reading every line of the released "
                "analysis script. The OSF node (tgmyh) contains exactly these 2 files; no other file offers the "
                "raw item. This is a genuine data-availability gap, not an inference to guess around."
            ),
            "files_needed_to_unblock": [
                "A version of the Howe et al. (2019) dataset containing the raw ordinal ('completely'/'a lot'/'a "
                "moderate amount'/'a little'/'not at all') response to the environmental-scientist-trust item, "
                "keyed to the same CaseID -- obtainable, if at all, only by contacting the authors directly "
                "(the OSF release does not include it)."
            ],
            "not_proceeding_with_manifest_construction": True,
        },
        "orchinik": {
            "citation": "Orchinik et al. (2024), 'Learning from and about scientists: Consensus messaging shapes perceptions of climate change and climate scientists', PNAS Nexus, DOI 10.1093/pnasnexus/pgae485",
            "source_url": "https://osf.io/jynqh/",
            "source_files": orchinik_files,
            "source_files_downloaded_via": "OSF API (api.osf.io/v2/nodes/jynqh/files/osfstorage/), sha256 independently verified against the API-reported hash before use",
            "sample_used": "Bovitz (main nationally representative sample) -- final_bovitz_raw.csv / final_clean.csv, per the authors' own bovitz_data_clean.R",
            "eligible_n": 2545,
            "eligible_n_matches_paper_reported_n": True,
            "eligibility_rule": "drop == FALSE, where drop = (fails > 0 | flag > 0); fails = (nick != 5) OR (captcha != '15') -- taken verbatim from data/domain_validation/orchinik/bovitz_data_clean.R, lines defining `fails`/`flag`/`drop`",
            "arm_counts_eligible": {"control": 847, "skill": 837, "trust": 861},
            "condition_label_mapping": {
                "verification_method": (
                    "The task's stated condition labels ('control', 'History', 'Institutions') do NOT literally "
                    "match the raw `condition` column values in the Bovitz data ('control', 'skill', 'trust'). "
                    "This was resolved by extracting the actual Qualtrics instrument text (data/domain_validation/"
                    "orchinik/Bovitz_qualtrics.docx) rather than guessing: the 'Skill Intervention' block's passage "
                    "is entirely about the long HISTORY of climate science (Tyndall 1860, Fourier 1824, Mauna Loa "
                    "1958, Hansen 1988) -- i.e. condition=='skill' IS the paper's 'History' condition. The 'Trust "
                    "Intervention' block's passage is entirely about INSTITUTIONAL bias-safeguards (conflict-of-"
                    "interest disclosure, peer sanctioning, funding-source scrutiny) -- i.e. condition=='trust' IS "
                    "the paper's 'Institutions' condition. This mapping is read directly from the released "
                    "instrument text, not inferred from variable names alone."
                ),
                "control": "control",
                "History": "skill (Skill Intervention block)",
                "Institutions": "trust (Trust Intervention block)",
            },
            "focal_beliefs": {
                "human_caused_climate_change": "P_cc_given_cons{50,75,90,97,99}",
                "bias_of_pro_consensus_scientists": "P_pro_bias_given_cons{50,75,90,97,99}",
                "bias_of_anti_consensus_scientists": "P_anti_bias_given_cons{50,75,90,97,99}",
                "skill_of_pro_consensus_scientists": "P_pro_skill_given_cons{50,75,90,97,99}",
                "skill_of_anti_consensus_scientists": "P_anti_skill_given_cons{50,75,90,97,99}",
            },
            "consensus_levels": [50, 75, 90, 97, 99],
            "expected_effect_cells": 50,
            "human_ate_estimator": "unweighted randomized-arm mean difference among eligible respondents; no survey weighting used (none present in the authors' own Bovitz-sample analysis code -- verified by grep, not assumed)",
            "human_ate_surface_status": "COMPUTED (real data)",
            "human_ate_surface_script": "scripts/compute_orchinik_human_ate_surface.py",
            "missing_data_policy": "no imputation for the primary analysis; the real computation confirmed zero missing responses across all 50 cells for the 2,545 eligible respondents",
            "synthetic_donor_design_status": "NOT YET BUILT",
            "synthetic_donor_design_blocker": (
                "Persona-field mapping from Bovitz's observed pretreatment demographics (age, gender, race, edu, "
                "income, party, politics, politics_social, politics_econ, god) onto this project's frozen G persona "
                "philosophy (which additionally expects state of residence and a denominational religion field, "
                "neither present in Bovitz), plus embedding the exact condition-specific passage text and the "
                "5-consensus-level x 5-belief question battery into a from-scratch external-study G prompt "
                "compiler, is real remaining engineering work not completed in this pass -- deferred rather than "
                "rushed, per this project's standing rule against silently inventing scientific mappings."
            ),
            "manifest_construction_status": "NOT BUILT (blocked on synthetic_donor_design_status above)",
        },
        "governance": {
            "howe_included_in_mu_external": False,
            "orchinik_included_in_mu_external": False,
            "rationale": (
                "The 31-study calibration universe and its representability rules were frozen before the "
                "calibration result. Howe and Orchinik were identified later for domain-specific validation and "
                "are therefore assigned independent validation roles rather than retroactively appended to the "
                "level-calibration archive."
            ),
        },
        "howe_confirmation_consequence": (
            "If Gemma wins: DOMAIN_SPECIFIC_G_CONFIRMATION=PASS. If DeepSeek wins: "
            "DOMAIN_SPECIFIC_G_CONFIRMATION=FAIL_MIXED_EVIDENCE. If tied: TIE. Under all outcomes G* remains "
            "google/gemma-4-31B-it unless a later explicit human-approved new-method decision is made -- this "
            "study never automatically alters G* or S2."
        ),
        "orchinik_gamma_consequence": (
            "DELTA_RMSE<0 => EXTERNAL_G_SHAPE_SUPPORT=POSITIVE; >0 => NEGATIVE; ==0 => TIE. Regardless of outcome, "
            "gamma_G for S2 remains 1.0 and MU_EXTERNAL remains 1.9558595458395387 -- any later proposal to change "
            "gamma would be a new, separate, explicit methodological decision."
        ),
        "diagnostic_metrics_module": "ate/domain_validation_metrics.py",
        "diagnostic_metrics_module_sha256": _sha256_file(PIPELINE_ROOT / "ate" / "domain_validation_metrics.py"),
        "bootstrap_rule": "respondent/donor-cluster bootstrap, seed=20260826, 10000 replicates, resampling distinct cluster ids with replacement and carrying all repeated measures of a resampled cluster together -- never resampling the 50 cells as if independent studies",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "frozen_domain_validation_protocol.json"
    out_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(out_path)
    (OUT_DIR / "frozen_domain_validation_protocol.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    protocol["protocol_sha256"] = sha
    return protocol


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
