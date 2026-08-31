"""Prospectively freeze the shared common-shift + support-projection method
(OFFLINE ONLY). Writes
outputs/secondary_calibration_diagnostic/target_projection_method.json.

Does NOT invoke ate.target_projection on any real target G output -- only
records the frozen algorithm/provenance, before any target G model output
has been retrieved or inspected. Does not touch the primary calibration
artifact, frozen_method_manifest.json, the Secondary-1/Secondary-2
artifacts, or any target manifest/ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic"
MODULE_PATH = PIPELINE_ROOT / "ate" / "target_projection.py"
TEST_PATH = PIPELINE_ROOT / "tests" / "ate" / "test_target_projection.py"

PRIMARY_ARTIFACT_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
SECONDARY_1_ARTIFACT_PATH = OUT_DIR / "secondary_calibration_selected_model.json"
SECONDARY_2_ARTIFACT_PATH = OUT_DIR / "secondary_2_mconst_gshape_method.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    from ate.target_projection import REAL_BENCHMARK_SHARED_RAW_ITEMS

    for p in (PRIMARY_ARTIFACT_PATH, SECONDARY_1_ARTIFACT_PATH, SECONDARY_2_ARTIFACT_PATH):
        if not p.exists():
            raise FileNotFoundError(f"required upstream artifact missing: {p}")

    artifact = {
        "method_name": "SHARED_COMMON_SHIFT_SUPPORT_PROJECTION",
        "status": "PROSPECTIVELY_FROZEN_BEFORE_TARGET_G_OUTPUT_INSPECTION",
        "target_g_outputs_inspected_before_freeze": False,
        "target_human_outcomes_used": False,
        "shared_across_methods": ["PRIMARY_M2", "SECONDARY_1_MCONST", "SECONDARY_2_MCONST_GSHAPE"],
        "method_independence": "project_cell/project_composite_cell/project_target_ate_table take only tau_hat_aj and native G control/treatment responses -- no parameter or branch identifies which calibration method produced tau_hat_aj",
        "algorithm": {
            "common_shift": "tau_G_aj = mean_i(Y_treat_i - Y_control_i); c_aj = tau_hat_aj - tau_G_aj; v_i = Y_treat_i + c_aj",
            "bounded_integer_projection": (
                "base-round each v_i to the nearest support level (ties go to the lower level); "
                "adjust the total one +/-1 step at a time toward the nearest attainable integer total "
                "to sum(v_i) (clipped to [N*low, N*high]), always picking the respondent(s) with smallest "
                "marginal increase in squared distance (largest residual fractional part for +1, smallest "
                "for -1), ties broken by ascending profile_id -- proven optimal by convex separable "
                "resource-allocation argument, brute-force-verified on small-N synthetic fixtures"
            ),
            "binary_projection": "identical engine with low=0, high=1 -- equivalent to assigning K=achieved total ones to the K highest ideal values",
            "finite_discrete_non_consecutive_support": "NOT implemented (explicit NotImplementedError/STOP) -- no outcome in this benchmark has such a support; implementing a speculative general solver was judged itself an undisclosed scientific choice",
            "composite_constituent_adjustment": "equal-contribution rule: each constituent item's coefficient*(ideal item shift) = c_aj/K; for this benchmark's actual composite forms (uniform 1/K coefficients for means, or a single +1/-1 coefficient item) this simplifies to +c_aj per item for mean/item composites and -c_aj for the one reverse_100 composite (funding_perceptions); each item is then independently projected via the shared bounded-integer engine and the composite is mechanically recomputed from survey_content.OUTCOME_COMPOSITES",
            "shared_raw_item_joint_consistency": "audited via survey_content.OUTCOME_COMPOSITES; zero raw items are currently shared across outcomes (independently re-verified: 44 raw items, 44 distinct item->outcome mappings) so the joint-constraint check is implemented and tested against synthetic fixtures but not exercised on the real benchmark today",
            "controls": "never read into any projection function argument that could modify them; returned pass-through unchanged in every cell result",
        },
        "supported_outcome_types": {
            "bounded_integer": "sliders [0,100], donation_ams [0,10] -- 11 of 13 outcomes' constituent raw items",
            "binary": "newsletter_signup {0,1}",
            "finite_discrete_non_consecutive": "not implemented; not required by current benchmark metadata",
        },
        "tie_break": "ascending profile_id (lexical) for competitive adjustment steps; exact-half base-rounding ties go to the lower integer -- both deterministic, no randomness",
        "real_benchmark_shared_raw_items": REAL_BENCHMARK_SHARED_RAW_ITEMS,
        "implementation_module": "ate/target_projection.py",
        "implementation_module_sha256": _sha256_file(MODULE_PATH),
        "test_fixture_module": "tests/ate/test_target_projection.py",
        "test_fixture_module_sha256": _sha256_file(TEST_PATH),
        "upstream_provenance_pointers": {
            "primary_artifact_sha256": _sha256_file(PRIMARY_ARTIFACT_PATH),
            "secondary_1_artifact_sha256": _sha256_file(SECONDARY_1_ARTIFACT_PATH),
            "secondary_2_artifact_sha256": _sha256_file(SECONDARY_2_ARTIFACT_PATH),
        },
        "failure_policy": [
            "missing control/treatment pair (donor id set mismatch)",
            "duplicate donor id (build_donor_map on flat record input)",
            "non-finite target ATE",
            "unknown support kind",
            "finite_discrete non-consecutive support (always -- not implemented)",
            "missing/inconsistent constituent-item metadata for composites",
            "unresolved shared-raw-item joint constraint",
            "duplicate intervention/outcome cell in a batch",
        ],
        "not_executed_on_real_target_output_in_this_freeze": True,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = OUT_DIR / "target_projection_method.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(artifact_path)
    (OUT_DIR / "target_projection_method.sha256.txt").write_text(sha + "\n", encoding="utf-8")
    artifact["projection_artifact_sha256"] = sha
    return artifact


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
