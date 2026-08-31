"""Tests for scripts/score_orchinik_final_validation.py's scoring formulas,
using synthetic fixtures ONLY -- never real Orchinik human values or real
Gemma responses. Proves the frozen formula (g_aj, g_bar, theta_hat_aj) and
the frozen metrics (RMSE primary, two comparators, MAE/Pearson/Spearman/
sign-agreement/per-arm RMSE) are implemented exactly as specified in
outputs/domain_validation/frozen_orchinik_final_validation_protocol.json."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import score_orchinik_final_validation as sov  # noqa: E402

MU_EXTERNAL = 1.9558595458395387


def test_apply_frozen_estimator_recenters_to_mu_external():
    g = {("skill", "a"): 10.0, ("skill", "b"): 20.0, ("trust", "a"): -5.0, ("trust", "b"): 15.0}
    theta_hat = sov.apply_frozen_estimator(g, MU_EXTERNAL)
    assert abs(np.mean(list(theta_hat.values())) - MU_EXTERNAL) < 1e-9
    # relative structure (differences between cells) must be preserved exactly
    for k1 in g:
        for k2 in g:
            assert abs((theta_hat[k1] - theta_hat[k2]) - (g[k1] - g[k2])) < 1e-9


def test_apply_frozen_estimator_matches_closed_form():
    g = {("skill", f"j{i}"): float(i) for i in range(10)}
    g.update({("trust", f"j{i}"): float(-i) for i in range(10)})
    g_bar = float(np.mean(list(g.values())))
    theta_hat = sov.apply_frozen_estimator(g, MU_EXTERNAL)
    for k, v in g.items():
        assert abs(theta_hat[k] - (MU_EXTERNAL + (v - g_bar))) < 1e-12


def test_score_zero_error_when_theta_hat_equals_human():
    keys = [("skill", "a"), ("skill", "b"), ("trust", "a"), ("trust", "b")]
    theta_hat = {k: 2.0 * i for i, k in enumerate(keys)}
    g = {k: 1.0 * i for i, k in enumerate(keys)}
    human = dict(theta_hat)
    result = sov.score(theta_hat, g, human, MU_EXTERNAL)
    assert result["primary_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["sign_agreement"] == pytest.approx(1.0)


def test_score_comparators_are_independent_of_theta_hat():
    keys = [("skill", "a"), ("skill", "b"), ("trust", "a"), ("trust", "b")]
    g = {k: float(i + 1) for i, k in enumerate(keys)}
    human = {k: float(i + 1) + 0.5 for i, k in enumerate(keys)}
    theta_hat_1 = {k: 0.0 for k in keys}
    theta_hat_2 = {k: 100.0 for k in keys}
    r1 = sov.score(theta_hat_1, g, human, MU_EXTERNAL)
    r2 = sov.score(theta_hat_2, g, human, MU_EXTERNAL)
    assert r1["comparator_A_raw_gemma_rmse"] == r2["comparator_A_raw_gemma_rmse"]
    assert r1["comparator_B_flat_mu_rmse"] == r2["comparator_B_flat_mu_rmse"]
    assert r1["primary_rmse"] != r2["primary_rmse"]


def test_score_per_arm_rmse_isolates_its_own_arm():
    keys = [("skill", "a"), ("skill", "b"), ("trust", "a"), ("trust", "b")]
    theta_hat = {("skill", "a"): 0.0, ("skill", "b"): 0.0, ("trust", "a"): 10.0, ("trust", "b"): 10.0}
    human = {("skill", "a"): 0.0, ("skill", "b"): 0.0, ("trust", "a"): 0.0, ("trust", "b"): 0.0}
    g = {k: 0.0 for k in keys}
    result = sov.score(theta_hat, g, human, MU_EXTERNAL)
    assert result["rmse_skill_arm"] == pytest.approx(0.0, abs=1e-9)
    assert result["rmse_trust_arm"] == pytest.approx(10.0, abs=1e-9)


def test_compute_gemma_grid_normalization():
    responses = {
        "c1": {"arm": "control", "items": {"cc_cons50": 40.0}},
        "c2": {"arm": "control", "items": {"cc_cons50": 60.0}},
        "s1": {"arm": "skill", "items": {"cc_cons50": 80.0}},
        "s2": {"arm": "skill", "items": {"cc_cons50": 90.0}},
        "t1": {"arm": "trust", "items": {"cc_cons50": 20.0}},
        "t2": {"arm": "trust", "items": {"cc_cons50": 30.0}},
    }
    for belief in sov.BELIEFS:
        for level in sov.CONSENSUS_LEVELS:
            key = f"{belief}_cons{level}"
            if key == "cc_cons50":
                continue
            for r in responses.values():
                r["items"][key] = 50.0
    g = sov.compute_gemma_grid(responses)
    # control mean for cc_cons50 = 50; skill mean = 85 -> tau=35 -> g=35.0 (R_j=100)
    assert g[("skill", "cc_cons50")] == pytest.approx(35.0)
    # trust mean = 25 -> tau=-25 -> g=-25.0
    assert g[("trust", "cc_cons50")] == pytest.approx(-25.0)
    # every other item has zero simulated shift
    assert g[("skill", "pro_bias_cons50")] == pytest.approx(0.0)


def test_scoring_script_fails_closed_without_real_retrieved_data(monkeypatch, tmp_path):
    """Fail-closed contract: load_gemma_responses() must refuse to proceed
    when the retrieved batch output is absent -- regardless of whether real
    production retrieval happens to already exist elsewhere in this
    repository. Isolated via monkeypatch onto a guaranteed-nonexistent
    tmp_path location, never touching or depending on the real
    outputs/domain_validation/.../retrieved/batch_output.jsonl."""
    missing_path = tmp_path / "retrieved" / "batch_output.jsonl"
    assert not missing_path.exists()
    monkeypatch.setattr(sov, "RETRIEVED_OUTPUT_PATH", missing_path)
    with pytest.raises(FileNotFoundError):
        sov.load_gemma_responses()
