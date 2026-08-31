"""Compute the real Orchinik et al. (2024) Bovitz-sample human ATE surface
(50 cells: 2 intervention contrasts x 5 focal beliefs x 5 consensus levels)
from the authors' own released, already-cleaned data
(data/domain_validation/orchinik/final_clean.csv).

This uses ONLY external human validation data explicitly permitted by the
domain-validation protocol (outputs/domain_validation/
frozen_domain_validation_protocol.json) -- it is not target data, not used
to recompute MU_EXTERNAL, and does not touch the frozen 31-study
calibration archive.

Eligibility (drop == FALSE) and the condition/intervention-label mapping
(condition "skill" = "Skill Intervention" block = the paper's "History"
passage about the long history of climate science; condition "trust" =
"Trust Intervention" block = the paper's "Institutions" passage about
institutional bias-safeguards) are taken directly from the authors' own
data/scripts (data/domain_validation/orchinik/bovitz_data_clean.R and the
Qualtrics instrument text), not invented -- see the frozen protocol for the
verification trail. No survey weighting is applied: the authors' own
Bovitz-sample cleaning/analysis code applies none.

No target G/F output, no target human outcomes, no LLM inference of any
kind.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PIPELINE_ROOT / "data" / "domain_validation" / "orchinik" / "final_clean.csv"
OUT_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"

BELIEFS = ["P_cc_given_cons", "P_pro_bias_given_cons", "P_anti_bias_given_cons", "P_pro_skill_given_cons", "P_anti_skill_given_cons"]
BELIEF_LABELS = {
    "P_cc_given_cons": "human_caused_climate_change",
    "P_pro_bias_given_cons": "bias_of_pro_consensus_scientists",
    "P_anti_bias_given_cons": "bias_of_anti_consensus_scientists",
    "P_pro_skill_given_cons": "skill_of_pro_consensus_scientists",
    "P_anti_skill_given_cons": "skill_of_anti_consensus_scientists",
}
LEVELS = ["50", "75", "90", "97", "99"]
INTERVENTIONS = (("History", "skill"), ("Institutions", "trust"))
EXPECTED_ELIGIBLE_N = 2545  # the paper's own reported N; verified below, not assumed


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> dict:
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    eligible = [r for r in all_rows if r.get("drop") == "FALSE"]
    if len(eligible) != EXPECTED_ELIGIBLE_N:
        raise ValueError(f"eligible N mismatch: expected {EXPECTED_ELIGIBLE_N} (the paper's reported N), got {len(eligible)} -- STOP rather than silently using a different N")

    by_condition = {"control": [], "skill": [], "trust": []}
    for r in eligible:
        c = r.get("condition")
        if c not in by_condition:
            raise ValueError(f"unexpected condition value on an eligible row: {c!r}")
        by_condition[c].append(r)

    cells = []
    for interv_label, interv_code in INTERVENTIONS:
        for belief in BELIEFS:
            for lvl in LEVELS:
                col = f"{belief}{lvl}"
                treat_vals = [_to_float(r.get(col)) for r in by_condition[interv_code]]
                ctrl_vals = [_to_float(r.get(col)) for r in by_condition["control"]]
                treat_valid = [v for v in treat_vals if v is not None]
                ctrl_valid = [v for v in ctrl_vals if v is not None]
                if len(treat_valid) != len(treat_vals) or len(ctrl_valid) != len(ctrl_vals):
                    raise ValueError(f"missing response(s) in column {col!r} for condition {interv_code!r}/control -- primary analysis does not impute; investigate before proceeding")
                h_e = sum(treat_valid) / len(treat_valid) - sum(ctrl_valid) / len(ctrl_valid)
                cells.append(
                    {
                        "intervention": interv_label,
                        "belief": BELIEF_LABELS[belief],
                        "consensus_level": int(lvl),
                        "source_column": col,
                        "h_e": h_e,
                        "treat_n": len(treat_valid),
                        "control_n": len(ctrl_valid),
                    }
                )

    if len(cells) != 50:
        raise ValueError(f"expected exactly 50 cells, got {len(cells)}")

    arm_counts = {c: len(rows) for c, rows in by_condition.items()}

    result = {
        "note": "REAL human ATE surface from Orchinik et al. (2024) Bovitz sample -- external validation data only, never used to recompute MU_EXTERNAL or the 31-study archive",
        "source_file": str(DATA_PATH.relative_to(PIPELINE_ROOT)),
        "source_file_sha256": _sha256_file(DATA_PATH),
        "eligible_n": len(eligible),
        "eligible_n_matches_paper": True,
        "arm_counts_eligible": arm_counts,
        "human_ate_estimator": "unweighted randomized-arm mean difference: h_e = mean(Y_e | intervention, eligible) - mean(Y_e | control, eligible); no survey weights applied (none used in the authors' own Bovitz-sample analysis code)",
        "cells": cells,
        "n_cells": len(cells),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "orchinik_human_ate_surface.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output_sha256"] = _sha256_file(out_path)
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2))
