"""Supplementary provenance layered on top of (never modifying)
validation.holdout.build_frozen_method_manifest's frozen schema --
binds the additional target/calibration provenance this task's freeze
requires (CES source, target correction equation, Wave-1 manifest hashes)
without touching the already-frozen holdout-lock manifest itself."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

FROZEN_METHOD_MANIFEST_PATH = PIPELINE_ROOT / "outputs" / "validation" / "frozen_method_manifest.json"
SUPPLEMENT_PATH = PIPELINE_ROOT / "outputs" / "validation" / "frozen_method_manifest_target_supplement.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    frozen = json.loads(FROZEN_METHOD_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "binds_frozen_method_manifest_sha256": sha256_file(FROZEN_METHOD_MANIFEST_PATH),
        "binds_method_hash": frozen["method_hash"],
        "ces_target_donor_source": {
            "source_sha256": "73cb40367efd0f92b8593cb9555660f68bdc95d8c1aad38f41aa13ab089ca43d",
            "freeze_commit": "bcc5717",
            "cutover_commit": "3a47541",
            "n_g": 1000,
            "n_f": 500,
        },
        "target_correction_equation": {
            "pp_scale_frozen_form": "calibrated_effect_pp = alpha_hat + lambda_hat * raw_f_ate_pp  (ate.target_effects.apply_calibration_to_target_ates, unmodified)",
            "native_scale_equivalent": "tau_hat_aj = lambda_hat * z_aj + (R_j / 100) * alpha_hat",
            "alpha_hat": frozen["final_alpha"],
            "lambda_hat": frozen["final_lambda"],
            "selected_model": frozen["selected_calibration_model"],
        },
        "support_projection_procedure": {
            "description": "minimum-distortion integer-response projection of the calibrated ideal onto native legal support -- unchanged, pre-existing",
            "modules": ["calibration.calibrate_arm.calibrate_arm_to_target_ate", "calibration.project_support.project_integer_to_total", "calibration.project_support.project_binary_to_count", "calibration.project_support.project_matrix_to_composite_total"],
        },
        "target_wave1_manifests": {
            "G": {"requests": 17000, "manifest_sha256": "167fa6eae7b9cd4e6328371b9b3f4e0745466279ffe3fda76b5a62fb23a77408", "jsonl_sha256": "3000df9fecd7f377c6e7a598718b16f19db53abadb3fcdc0f594b6f8597c397e"},
            "F": {"requests": 104500, "manifest_sha256": "0a45cccf91f3dcecf321317286f5266da0d39ca826c755c22848c56dc7968c87", "jsonl_sha256": "3ee0a8d8f1427e5338df0b7063758098db886b774e82cecd5777fd0955695e9f"},
            "freeze_commit": "a8ae43c",
        },
        "no_target_human_outcome_information_included": True,
    }
    SUPPLEMENT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"FINAL_METHOD_MANIFEST_SHA256 = {sha256_file(FROZEN_METHOD_MANIFEST_PATH)}")
    print(f"supplement sha256 = {sha256_file(SUPPLEMENT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
