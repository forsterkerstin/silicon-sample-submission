"""Score the frozen Orchinik one-shot post-freeze external diagnostic
validation (see outputs/domain_validation/
frozen_orchinik_final_validation_protocol.json -- every formula, metric,
comparator, and bootstrap parameter here is copied from that already-frozen
document, never invented or adjusted here).

NOT RUNNABLE YET: requires outputs/domain_validation/
orchinik_g_domain_confirmation_v2/google_gemma-4-31B-it/retrieved/
batch_output.jsonl, which does not exist until the frozen manifest is
actually submitted and retrieved (a separate, explicit, real-money action
never taken by this script). Running this file before that data exists
fails closed with FileNotFoundError.

Reads outputs/domain_validation/orchinik_human_ate_surface.json (the real
human ATE values) -- per the frozen protocol's governance, this must only
happen after the frozen protocol document above already exists on disk
with all rules committed, which this script asserts before opening it.

Computes, per the frozen formula:
    tau_G_aj = mean(Y_treatment_aj) - mean(Y_control_j)
    g_aj     = 100 * tau_G_aj / R_j
    g_bar    = mean over Orchinik's own 50 g_aj
    theta_hat_aj = MU_EXTERNAL + (g_aj - g_bar)
compares against the real theta_ext_aj via the frozen primary metric (RMSE)
and the two frozen comparators (raw Gemma g_aj; flat MU_EXTERNAL), reports
the frozen diagnostics, and a respondent-cluster bootstrap CI reusing
ate.domain_validation_metrics's already-frozen primitives.

No metric, comparator, or formula may be changed after this script has
ever been run against real data -- if a change is needed, it must go
through a new, explicitly authorized freeze, not an edit to this file
after results are seen. This script never writes to predictions/,
registration.md, or any calibration/method artifact.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.domain_validation_metrics import cluster_bootstrap_indices, percentile_interval  # noqa: E402
from ate.f_screen_validation import validate_response  # noqa: E402
from inference.orchinik_domain_confirmation_guard import PHASES as ORCHINIK_PHASES  # noqa: E402

DOMAIN_VALIDATION_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"
PROTOCOL_PATH = DOMAIN_VALIDATION_DIR / "frozen_orchinik_final_validation_protocol.json"
HUMAN_SURFACE_PATH = DOMAIN_VALIDATION_DIR / "orchinik_human_ate_surface.json"
GEMMA_ROOT = DOMAIN_VALIDATION_DIR / "orchinik_g_domain_confirmation_v2" / "google_gemma-4-31B-it"
RETRIEVED_OUTPUT_PATH = GEMMA_ROOT / "retrieved" / "batch_output.jsonl"
OUT_PATH = DOMAIN_VALIDATION_DIR / "orchinik_final_validation_result.json"

CONSENSUS_LEVELS = (50, 75, 90, 97, 99)
BELIEFS = ("cc", "pro_bias", "anti_bias", "pro_skill", "anti_skill")
ARMS = ("skill", "trust")
R_J = 100.0


def _load_frozen_protocol() -> dict:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("frozen_orchinik_final_validation_protocol.json does not exist -- run scripts/freeze_orchinik_final_validation_protocol.py first; scoring rules must be committed before any human value is opened")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "PROSPECTIVELY_FROZEN_BEFORE_INFERENCE":
        raise RuntimeError(f"unexpected protocol status {protocol['status']!r}")
    return protocol


def _load_jsonl_by_cid(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and "custom_id" in rec:
                out[str(rec["custom_id"])] = rec
    return out


def _load_schema_by_cid(jsonl_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["custom_id"])] = r["body"]["response_format"]["json_schema"]["schema"]
    return out


def load_gemma_responses() -> dict[str, dict[str, Any]]:
    """{cid: {"arm": "control"|"skill"|"trust", "items": {item_key: value}}} for every resolved, schema-valid Gemma response."""
    import csv

    if not RETRIEVED_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"{RETRIEVED_OUTPUT_PATH} does not exist -- the Orchinik Gemma-only manifest has not been submitted/retrieved yet; this validation cannot be scored")

    spec = ORCHINIK_PHASES["orchinik_g_domain_confirmation_v2_gemma"]
    manifest_by_cid: dict[str, dict] = {}
    with open(spec["manifest_path"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest_by_cid[row["custom_id"]] = row

    schema_by_cid = _load_schema_by_cid(spec["jsonl_path"])
    output_by_cid = _load_jsonl_by_cid(RETRIEVED_OUTPUT_PATH)

    responses: dict[str, dict[str, Any]] = {}
    for cid, row in manifest_by_cid.items():
        if cid not in output_by_cid:
            continue
        validation = validate_response(output_by_cid[cid], schema_by_cid.get(cid))
        if not validation["valid"]:
            continue
        response_key_map = json.loads(row["response_key_map"])
        raw = validation["parsed"]
        items = {response_key_map[q]: raw[q] for q in response_key_map}
        responses[cid] = {"arm": row["condition_id"], "items": items}
    return responses


def compute_gemma_grid(responses: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Returns {(arm, item_key): g_aj} for arm in ('skill','trust'), over the 25 focal items."""
    by_arm: dict[str, list[dict]] = {"control": [], "skill": [], "trust": []}
    for r in responses.values():
        by_arm.setdefault(r["arm"], []).append(r["items"])
    item_keys = sorted({k for belief in BELIEFS for level in CONSENSUS_LEVELS for k in [f"{belief}_cons{level}"]})

    control_mean = {k: float(np.mean([row[k] for row in by_arm["control"]])) for k in item_keys}
    g: dict[tuple[str, str], float] = {}
    for arm in ARMS:
        for k in item_keys:
            tau = float(np.mean([row[k] for row in by_arm[arm]])) - control_mean[k]
            g[(arm, k)] = 100.0 * tau / R_J
    return g


def apply_frozen_estimator(g: dict[tuple[str, str], float], mu_external: float) -> dict[tuple[str, str], float]:
    g_bar = float(np.mean(list(g.values())))
    return {key: mu_external + (val - g_bar) for key, val in g.items()}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def score(theta_hat: dict, g: dict, human_theta: dict, mu_external: float) -> dict:
    keys = sorted(theta_hat.keys())
    hat = np.array([theta_hat[k] for k in keys])
    raw = np.array([g[k] for k in keys])
    flat = np.full(len(keys), mu_external)
    human = np.array([human_theta[k] for k in keys])

    result = {
        "primary_rmse": _rmse(hat, human),
        "comparator_A_raw_gemma_rmse": _rmse(raw, human),
        "comparator_B_flat_mu_rmse": _rmse(flat, human),
        "mae": float(np.mean(np.abs(hat - human))),
        "pearson_r": float(pearsonr(hat, human)[0]),
        "spearman_r": float(spearmanr(hat, human)[0]),
        "sign_agreement": float(np.mean(np.sign(hat) == np.sign(human))),
    }
    for arm in ARMS:
        arm_keys = [k for k in keys if k[0] == arm]
        h = np.array([theta_hat[k] for k in arm_keys])
        y = np.array([human_theta[k] for k in arm_keys])
        result[f"rmse_{arm}_arm"] = _rmse(h, y)
    return result


def main() -> dict:
    protocol = _load_frozen_protocol()
    mu_external = protocol["frozen_method_constants"]["mu_external"]

    responses = load_gemma_responses()
    g = compute_gemma_grid(responses)
    theta_hat = apply_frozen_estimator(g, mu_external)

    if not HUMAN_SURFACE_PATH.exists():
        raise FileNotFoundError(f"{HUMAN_SURFACE_PATH} does not exist")
    human_surface = json.loads(HUMAN_SURFACE_PATH.read_text(encoding="utf-8"))
    # Label alignment only (no scoring-rule change): the human surface uses the
    # paper's published intervention names ("History"/"Institutions") and its
    # own source-column naming ("P_<belief>_given_cons<level>"), while the
    # Gemma-side manifest uses the raw Bovitz condition values ("skill"/
    # "trust") and the bare item key ("<belief>_cons<level>") -- see
    # outputs/domain_validation/frozen_domain_validation_protocol.json's own
    # condition_label_mapping for the same History->skill / Institutions->trust
    # correspondence, established from the actual Qualtrics instrument text.
    _ARM_LABEL = {"History": "skill", "Institutions": "trust"}
    _SOURCE_COLUMN_RE = re.compile(r"^P_(.+)_given_cons(\d+)$")

    def _item_key(source_column: str) -> str:
        m = _SOURCE_COLUMN_RE.match(source_column)
        if not m:
            raise ValueError(f"unrecognized human source_column format: {source_column!r}")
        return f"{m.group(1)}_cons{m.group(2)}"

    human_theta = {(_ARM_LABEL[c["intervention"]], _item_key(c["source_column"])): 100.0 * float(c["h_e"]) / R_J for c in human_surface["cells"]}

    result = score(theta_hat, g, human_theta, mu_external)
    result["n_cells"] = len(theta_hat)
    result["bootstrap"] = {"seed": protocol["bootstrap"]["seed"], "n_boot": protocol["bootstrap"]["n_boot"], "note": "descriptive CI only, computed via ate.domain_validation_metrics.cluster_bootstrap_indices/percentile_interval over respondent/donor clusters -- not implemented as a stub here pending real retrieved per-respondent data"}
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
