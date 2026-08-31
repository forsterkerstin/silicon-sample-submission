"""S2 (MCONST_GSHAPE)-specific prerequisite guard for target G Wave 2
(Consensus Stage B).

Deliberately SEPARATE from inference.target_production_guard's
assert_target_production_prerequisites_frozen, which hard-requires F*/R_F/a
usable_for_production calibration artifact -- none of those are
prerequisites for S2, which has zero target-F dependence (see the
S2-promotion governance decision: outputs/validation/
frozen_final_submission_manifest_s2.json). This module checks only what S2
actually needs: G* frozen, the CES source frozen, the S2 final-submission
manifest itself frozen, and the shared projector prospectively frozen -- and
explicitly does NOT check F*, R_F, target-F completion, or F Consensus
Stage B.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from inference.model_config import selected_model

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
S2_FINAL_SUBMISSION_MANIFEST_PATH = PIPELINE_ROOT / "outputs" / "validation" / "frozen_final_submission_manifest_s2.json"
PROJECTOR_ARTIFACT_PATH = PIPELINE_ROOT / "outputs" / "secondary_calibration_diagnostic" / "target_projection_method.json"
POPULATION_CONFIG_PATH = PIPELINE_ROOT / "config" / "population.yaml"

CES_SOURCE_SHA256 = "73cb40367efd0f92b8593cb9555660f68bdc95d8c1aad38f41aa13ab089ca43d"


class S2GWave2NotAuthorized(RuntimeError):
    pass


def _frozen_ces_source_sha256() -> str | None:
    cfg = yaml.safe_load(POPULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    for amendment in cfg.get("population_source_amendments", []):
        if amendment.get("amendment_type") == "TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN":
            return amendment["frozen_roster"]["sha256"]
    return None


def assert_s2_g_wave2_prerequisites_frozen() -> dict:
    """Hard-stop unless G*, the frozen CES source, the S2 final-submission
    manifest, and the shared projector are ALL frozen. Never checks F*,
    R_F, or any F-related artifact -- by design, per the S2-promotion
    decision that target F is not a final-production dependency."""
    problems: list[str] = []

    g_star = None
    try:
        g_star = selected_model("g", require_frozen=True)
    except RuntimeError as exc:
        problems.append(f"G* is not frozen: {exc}")

    ces_sha = _frozen_ces_source_sha256()
    if ces_sha is None:
        problems.append("frozen CES source not found in config/population.yaml")
    elif ces_sha != CES_SOURCE_SHA256:
        problems.append(f"frozen CES source sha256 mismatch: expected {CES_SOURCE_SHA256}, got {ces_sha}")

    if not S2_FINAL_SUBMISSION_MANIFEST_PATH.exists():
        problems.append(f"S2 final-submission manifest is not frozen: missing {S2_FINAL_SUBMISSION_MANIFEST_PATH}")
    else:
        try:
            manifest = json.loads(S2_FINAL_SUBMISSION_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"S2 final-submission manifest could not be loaded: {exc}")
            manifest = {}
        if manifest.get("final_method") != "S2_MCONST_GSHAPE":
            problems.append(f"S2 final-submission manifest final_method is not S2_MCONST_GSHAPE (got {manifest.get('final_method')!r})")
        if manifest.get("target_f_dependence") is not False:
            problems.append("S2 final-submission manifest does not explicitly declare target_f_dependence=false")

    if not PROJECTOR_ARTIFACT_PATH.exists():
        problems.append(f"shared common-shift/support-projection method is not frozen: missing {PROJECTOR_ARTIFACT_PATH}")

    if problems:
        raise S2GWave2NotAuthorized("S2 target-G Wave-2 prerequisites are not met:\n- " + "\n- ".join(problems))

    return {"selected_g_model": g_star, "ces_source_sha256": ces_sha}
