"""baseline_calibration.py: assess REALISM of a simulated arm's raw,
native-response control values against an external reference. Diagnostic
only -- no active rescaling exists in this module (distribution/"spread"
correction is explicitly deferred by the primary specification)."""

from __future__ import annotations

import numpy as np
import pytest

from baseline_calibration import BaselineReference, assess_baseline_realism


@pytest.fixture
def elicited_control():
    rng = np.random.default_rng(0)
    return rng.normal(loc=30, scale=15, size=500).round().clip(0, 100).astype(int).tolist()


def test_assess_with_no_reference_reports_that_clearly(elicited_control):
    report = assess_baseline_realism(elicited_control, reference=None)
    assert report["status"] == "no_reference_data"
    assert "elicited_mean" in report


def test_assess_with_reference_reports_gap(elicited_control):
    ref = BaselineReference(mean=50.0, variance=200.0, source="synthetic test reference")
    report = assess_baseline_realism(elicited_control, ref)
    assert report["status"] == "compared"
    assert report["mean_gap"] == pytest.approx(np.mean(elicited_control) - 50.0)


def test_variance_ratio_computed(elicited_control):
    ref = BaselineReference(mean=30.0, variance=225.0, source="synthetic test reference")
    report = assess_baseline_realism(elicited_control, ref)
    assert report["variance_ratio"] == pytest.approx(np.var(elicited_control) / 225.0, rel=0.05)


def test_ks_and_wasserstein_reported_as_normal_approximations(elicited_control):
    ref = BaselineReference(mean=30.0, variance=225.0, source="synthetic test reference")
    report = assess_baseline_realism(elicited_control, ref)
    assert 0.0 <= report["ks_vs_normal_approx"] <= 1.0
    assert report["wasserstein_vs_normal_approx"] >= 0.0


def test_ks_wasserstein_absent_without_variance(elicited_control):
    ref = BaselineReference(mean=30.0, source="synthetic test reference")  # no variance reported
    report = assess_baseline_realism(elicited_control, ref)
    assert "ks_vs_normal_approx" not in report
    assert "wasserstein_vs_normal_approx" not in report


def test_no_active_rescaling_function_exists():
    import baseline_calibration as bc

    assert not hasattr(bc, "calibrate_baseline")
