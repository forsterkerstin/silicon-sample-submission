from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ate.f_reliability import (
    DEFAULT_F_NUM_DRAWS,
    DEFAULT_N_F,
    convergence_by_effect,
    create_pilot_manifest,
    nested_profile_subsets,
    profile_level_summary,
    require_frozen_f_protocol,
    stochastic_reliability_by_effect,
    validate_external_target_protocol,
)
from ate.target_effects import estimate_target_ates_from_f


def _f_fixture(n: int = 500) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for condition in ["control", "treat"]:
            rows.append(
                {
                    "f_profile_id": f"F{i:04d}",
                    "condition": condition,
                    "gender": "Male",
                    "age_band": "18-29",
                    "race": "White / Caucasian",
                    "education": "Bachelor's degree",
                    "income": "$56,000 to $99,999",
                    "party": "Independent",
                    "trust_post": i % 100 + (2 if condition == "treat" else 0),
                }
            )
    return pd.DataFrame(rows)


def test_production_f_defaults_are_500_unique_profiles_and_one_draw():
    assert DEFAULT_N_F == 500
    assert DEFAULT_F_NUM_DRAWS == 1


def test_target_f_requires_500_same_profile_ids_under_control_and_treatment():
    raw, contrasts = estimate_target_ates_from_f(_f_fixture(), outcomes=["trust_post"], expected_n_f=500)

    assert raw["n_f"].item() == 500
    assert contrasts["f_profile_id"].nunique() == 500
    assert raw["raw_ate_native"].item() == pytest.approx(2.0)
    assert raw["profile_ate_se_native"].item() == pytest.approx(0.0)


def test_target_f_refuses_unfrozen_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="frozen F protocol"):
        require_frozen_f_protocol(tmp_path / "missing.json")


def test_protocol_consistency_requires_identical_frozen_values():
    external = {"selected_f_model": "m", "n_f": 500, "f_num_draws": 1, "temperature": 1.0, "top_p": 0.95, "reasoning_configuration": {"reasoning_effort": "low"}, "prompt_version": "ashokkumar_experiment_forecast_adapted_v1"}
    target = dict(external)
    validate_external_target_protocol(external, target)
    target["f_num_draws"] = 2
    with pytest.raises(RuntimeError, match="differ"):
        validate_external_target_protocol(external, target)


def test_nested_profile_subsets_are_deterministic_and_nested():
    ids = [f"F{i:04d}" for i in range(500)]
    subsets = nested_profile_subsets("study", ids)

    assert [len(subsets[n]) for n in [50, 100, 250, 500]] == [50, 100, 250, 500]
    assert set(subsets[50]) <= set(subsets[100]) <= set(subsets[250]) <= set(subsets[500])
    assert subsets == nested_profile_subsets("study", list(reversed(ids)))


def test_profile_level_summary_is_mean_delta_and_sd_over_sqrt_n():
    delta = np.arange(500, dtype=float)
    out = profile_level_summary(delta, outcome_range=100)

    assert out["raw_ate_native"] == pytest.approx(delta.mean())
    assert out["profile_ate_se_native"] == pytest.approx(delta.std(ddof=1) / np.sqrt(500))


def test_convergence_analysis_uses_nested_external_effects(tmp_path):
    rows = []
    for effect in ["e1", "e2"]:
        for i in range(500):
            rows.append({"study_id": "s", "effect_id": effect, "f_profile_id": f"F{i:04d}", "delta_native": i / 100 + (1 if effect == "e2" else 0), "outcome_range": 100})
    by_effect, summary = convergence_by_effect(pd.DataFrame(rows), outputs_dir=tmp_path)

    assert {"z_50_native", "z_100_native", "z_250_native", "z_500_native"} <= set(by_effect.columns)
    assert set(summary["n"]) == {50, 100, 250}
    assert (tmp_path / "convergence_by_effect.csv").exists()


def test_stochastic_reliability_second_draw_same_effects(tmp_path):
    df = pd.DataFrame(
        [
            {"study_id": "s1", "effect_id": "e1", "replicate": "replicate_1", "z_native": 1.0, "z_pp": 10.0},
            {"study_id": "s1", "effect_id": "e1", "replicate": "replicate_2", "z_native": 1.1, "z_pp": 11.0},
            {"study_id": "s2", "effect_id": "e2", "replicate": "replicate_1", "z_native": -1.0, "z_pp": -10.0},
            {"study_id": "s2", "effect_id": "e2", "replicate": "replicate_2", "z_native": -1.2, "z_pp": -12.0},
        ]
    )

    by_effect, summary = stochastic_reliability_by_effect(df, outputs_dir=tmp_path)

    assert len(by_effect) == 2
    assert summary["sign_agreement"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "stochastic_reliability_summary.json").read_text())["rmse"] == pytest.approx(summary["rmse"])


def test_pilot_manifest_selection_is_metadata_seeded_not_performance_based(tmp_path):
    effects = pd.DataFrame(
        [
            {"study_id": f"s{i}", "effect_id": f"e{i}", "outcome_type": "attitude", "population_type": "general_us_adult", "human_ate": 1000 - i}
            for i in range(20)
        ]
    )

    manifest = create_pilot_manifest(effects, n_effects=5, seed=1, outputs_dir=tmp_path)

    assert len(manifest) == 5
    assert "human_ate" not in manifest.columns
    assert (tmp_path / "pilot_manifest.csv").exists()
