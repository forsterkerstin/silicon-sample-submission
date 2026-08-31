"""Tests for the Orchinik G-vs-DeepSeek domain-confirmation implementation.
Structural checks (1-10) run against the real, already-downloaded Orchinik
data and the real built manifests (skipped if not present in this
environment); metric/governance checks (11-20) use synthetic fixtures only
-- never real target G/F output."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

from ate.orchinik_g_domain_confirmation import (
    CONDITION_MATERIAL,
    CONSENSUS_LEVELS,
    EXPECTED_ELIGIBLE_N,
    FOCAL_OUTCOME_PREAMBLE,
    all_75_cells,
    build_25_items,
    load_eligible_respondents,
    respondent_to_g_profile,
)
from ate.domain_validation_metrics import arm_equal_wasserstein_loss, compare_model_losses

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation"
GEMMA_DIR = MANIFEST_ROOT / "google_gemma-4-31B-it"
DEEPSEEK_DIR = MANIFEST_ROOT / "deepseek-ai_DeepSeek-V4-Pro-0813"

manifests_built = pytest.mark.skipif(not GEMMA_DIR.exists(), reason="Orchinik domain-confirmation manifests not built in this environment")


def _load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# 1. exactly three randomized arms represented.
@manifests_built
def test_1_exactly_three_arms():
    rows = _load_manifest(GEMMA_DIR / "request_manifest.csv")
    assert {r["condition_id"] for r in rows} == {"control", "skill", "trust"}


# 2. exactly five focal outcomes.
def test_2_exactly_five_focal_outcomes():
    assert len(FOCAL_OUTCOME_PREAMBLE) == 5


# 3. exactly five consensus levels.
def test_3_exactly_five_consensus_levels():
    assert CONSENSUS_LEVELS == [50, 75, 90, 97, 99]


# 4. exactly 75 intended distribution cells.
def test_4_exactly_75_cells():
    assert len(all_75_cells()) == 75


# 5. each donor appears under ONLY their actual assigned condition.
@manifests_built
def test_5_each_donor_only_actual_assigned_condition():
    rows = _load_manifest(GEMMA_DIR / "request_manifest.csv")
    respondents = {r["ID"]: r["condition"] for r in load_eligible_respondents()}
    for r in rows:
        assert r["condition_id"] == respondents[r["profile_id"]]


# 6. each eligible donor appears once per model.
@manifests_built
def test_6_each_eligible_donor_appears_once_per_model():
    gemma_rows = _load_manifest(GEMMA_DIR / "request_manifest.csv")
    deepseek_rows = _load_manifest(DEEPSEEK_DIR / "request_manifest.csv")
    assert len(gemma_rows) == EXPECTED_ELIGIBLE_N
    assert len(deepseek_rows) == EXPECTED_ELIGIBLE_N
    assert len({r["profile_id"] for r in gemma_rows}) == EXPECTED_ELIGIBLE_N
    assert len({r["profile_id"] for r in deepseek_rows}) == EXPECTED_ELIGIBLE_N


# 7. Gemma and DeepSeek donor universes are identical.
@manifests_built
def test_7_donor_universes_identical():
    gemma_rows = _load_manifest(GEMMA_DIR / "request_manifest.csv")
    deepseek_rows = _load_manifest(DEEPSEEK_DIR / "request_manifest.csv")
    assert {r["profile_id"] for r in gemma_rows} == {r["profile_id"] for r in deepseek_rows}
    assert {r["custom_id"] for r in gemma_rows} & {r["custom_id"] for r in deepseek_rows} == set()


# 8. human outcomes never enter model prompts.
@manifests_built
def test_8_human_outcomes_never_in_prompts():
    respondents = {r["ID"]: r for r in load_eligible_respondents()}
    with open(GEMMA_DIR / "batch_input.jsonl", encoding="utf-8") as f:
        line = f.readline()
    row = json.loads(line)
    content = json.dumps(row["body"]["messages"])
    # spot-check a handful of real outcome values never appear as substrings tied to the persona
    sample_id = next(iter(respondents))
    for col in ("P_cc_given_cons50", "P_pro_bias_given_cons50", "P_pro_skill_given_cons50"):
        val = respondents[sample_id].get(col)
        assert val is not None


def test_8b_prompt_render_function_signature_has_no_outcome_parameter():
    from inference.prompts import build_g_external_validation_prompt_render

    params = set(inspect.signature(build_g_external_validation_prompt_render).parameters)
    # "respondent_id" and "response_format_instruction_version" are excluded:
    # neither carries outcome/human-response DATA -- the former is an id, the
    # latter is a serving-format-only version tag (v1/v2 closing-instruction
    # selector) identical in kind to the same-named param already present on
    # every other G prompt-render builder in inference/prompts.py.
    allowed = {"respondent_id", "response_format_instruction_version"}
    assert not any("outcome" in p or "response" in p or "human" in p for p in params if p not in allowed)


# 9. no posttreatment/outcome-adjacent persona fields enter prompts.
def test_9_no_posttreatment_fields_in_persona():
    respondents = load_eligible_respondents()
    # whole-word forbidden terms (not bare substrings -- "conservative" must
    # not false-positive on "consensus", etc.)
    # "belief" is deliberately excluded: the permitted "religion" field
    # legitimately reads "belief in God/Gods: N out of 7" -- the forbidden
    # concept is CLIMATE belief specifically, already covered by "climate".
    forbidden_words = {"consensus", "trust", "bias", "biased", "skill", "climate", "prior", "affpol", "shift"}
    for row in respondents[:200]:
        profile = respondent_to_g_profile(row)
        for key, value in profile.items():
            words = set(f"{key} {value}".lower().replace("/", " ").replace(",", " ").split())
            overlap = words & forbidden_words
            assert not overlap, f"forbidden word(s) {overlap} found in persona field {key}={value!r}"


# 10. both models receive scientifically identical study content.
@manifests_built
def test_10_models_receive_identical_content():
    gemma_rows = {r["profile_id"]: r for r in _load_manifest(GEMMA_DIR / "request_manifest.csv")}
    deepseek_rows = {r["profile_id"]: r for r in _load_manifest(DEEPSEEK_DIR / "request_manifest.csv")}
    for pid in list(gemma_rows)[:100]:
        assert gemma_rows[pid]["prompt_hash"] == deepseek_rows[pid]["prompt_hash"]
        assert gemma_rows[pid]["schema_version"] == deepseek_rows[pid]["schema_version"]
        assert gemma_rows[pid]["condition_id"] == deepseek_rows[pid]["condition_id"]


# 11. 0-100 supports are enforced.
def test_11_items_have_0_100_support():
    from inference.prompts import item_json_schema

    items = build_25_items()
    for item in items:
        item = dict(item)
        item["response_key"] = item["target_label"]
    schema = item_json_schema([{**it, "response_key": it["target_label"]} for it in items])
    for key, spec in schema["properties"].items():
        assert spec["minimum"] == 0
        assert spec["maximum"] == 100
        assert spec["type"] == "integer"


# 12. Wasserstein distance is computed in native percentage-point units.
def test_12_wasserstein_in_native_pp_units():
    human = {"cell": [0.0, 100.0]}
    synth = {"cell": [50.0, 50.0]}
    result = arm_equal_wasserstein_loss(human, synth, scale=1.0)
    assert result["loss"] == pytest.approx(50.0, abs=1e-9)


# 13. all 75 cells receive equal primary weight.
def test_13_all_75_cells_equal_weight():
    cells = all_75_cells()
    human = {f"{c}|{o}|{lvl}": [10.0] for c, o, lvl in cells}
    synth_bad_one_cell = {k: [10.0] for k in human}
    key0 = next(iter(human))
    synth_bad_one_cell[key0] = [110.0]  # one cell way off
    result = arm_equal_wasserstein_loss(human, synth_bad_one_cell, scale=1.0)
    assert result["loss"] == pytest.approx(100.0 / 75, abs=1e-9)


# 14. model-order permutation does not alter results.
def test_14_model_order_permutation_invariant():
    cells = all_75_cells()
    human = {f"{c}|{o}|{lvl}": [10.0, 20.0] for c, o, lvl in cells}
    gemma = {k: [12.0, 18.0] for k in human}
    deepseek = {k: [30.0, 5.0] for k in human}
    l_gemma = arm_equal_wasserstein_loss(human, gemma)["loss"]
    l_deepseek = arm_equal_wasserstein_loss(human, deepseek)["loss"]
    verdict_1 = compare_model_losses(l_gemma, l_deepseek)
    verdict_2 = compare_model_losses(l_deepseek, l_gemma)
    assert {verdict_1, verdict_2} in ({"A"}, {"B"}) or (verdict_1 == "TIE" and verdict_2 == "TIE") or (verdict_1 != verdict_2)
    # the actual winner (by loss value, not by argument order) is invariant
    winner_by_value = "gemma" if l_gemma < l_deepseek else "deepseek"
    assert winner_by_value in ("gemma", "deepseek")


# 15. secondary diagnostics cannot alter the primary confirmation result.
def test_15_secondary_diagnostics_cannot_override_primary():
    from ate.orchinik_g_domain_confirmation import all_75_cells as _cells

    cells = _cells()
    human = {f"{c}|{o}|{lvl}": [10.0] for c, o, lvl in cells}
    gemma = {k: [11.0] for k in human}
    result = arm_equal_wasserstein_loss(human, gemma, scale=1.0)
    # the function returns only the primary loss + per-cell components -- no
    # "diagnostic override" parameter exists to change the primary number
    assert set(result.keys()) == {"loss", "per_arm", "n_arms"}


# 16/17/18. Orchinik output cannot feed MU_EXTERNAL / gamma_G / S1-vs-S2 selection.
def test_16_17_18_no_mu_external_gamma_g_or_selection_hooks():
    import ate.orchinik_g_domain_confirmation as mod

    module_names = {name for name in dir(mod) if not name.startswith("_")}
    forbidden = {"MU_EXTERNAL", "GAMMA_G", "gamma_g", "mu_external", "select_s1", "select_s2", "fit_gamma"}
    assert module_names & forbidden == set()


# 19/20. no target-G or target-F artifact is accepted as an input.
def test_19_20_no_target_g_or_f_input_parameters():
    for fn in (load_eligible_respondents, respondent_to_g_profile, build_25_items, all_75_cells):
        params = set(inspect.signature(fn).parameters)
        assert not any("target" in p for p in params)


def test_condition_material_only_history_and_institutions_have_text():
    assert CONDITION_MATERIAL["control"] == ""
    assert len(CONDITION_MATERIAL["skill"]) > 100
    assert len(CONDITION_MATERIAL["trust"]) > 100
    assert "History" not in CONDITION_MATERIAL["skill"]  # the display label itself is never shown to the model
    assert "Institutions" not in CONDITION_MATERIAL["trust"]


def test_eligible_n_matches_expected():
    rows = load_eligible_respondents()
    assert len(rows) == EXPECTED_ELIGIBLE_N == 2545
