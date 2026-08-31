"""calibrate_arm_to_target_ate(): minimum-distortion integer adjustment of
one arm's raw native responses to hit a lambda_ate-calibrated target ATE."""

from __future__ import annotations

import pytest

from calibration.calibrate_arm import calibrate_arm_to_target_ate


def _mean(values):
    return sum(values) / len(values)


def test_zero_target_ate_matches_control_mean_exactly():
    # target_ate_pp=0.0 means "the calibrated arm's mean must equal control's",
    # not "leave the raw values untouched" -- those differ whenever the raw
    # values didn't already happen to average there.
    control = [40, 50, 60]  # control mean = 50
    raw = [42, 48, 55, 61]  # raw mean = 51.5, needs to come down to 50
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=0.0, low=0, high=100)
    assert _mean(out) == pytest.approx(50.0)


def test_already_at_target_makes_no_changes():
    control = [40, 50, 60]  # control mean = 50
    raw = [45, 50, 55, 50]  # raw mean already exactly 50
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=0.0, low=0, high=100)
    assert out == raw


def test_hits_exact_target_when_feasible():
    control = [50, 50, 50, 50]  # control mean = 50
    raw = [50, 50, 50, 50]
    # target_ate_pp=10 on a 0-100 scale -> target mean = 60 -> target sum = 240
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=10.0, low=0, high=100)
    assert sum(out) == 240
    assert _mean(out) == pytest.approx(60.0)


def test_minimum_total_distortion():
    control = [50] * 10
    raw = [50] * 10
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=5.0, low=0, high=100)
    # target sum = 550 (mean 55), so total absolute distortion must equal exactly the required delta.
    total_distortion = sum(abs(o - r) for o, r in zip(out, raw))
    assert total_distortion == abs(sum(out) - sum(raw))


def test_never_touches_control():
    control = [10, 20, 30]
    control_copy = list(control)
    calibrate_arm_to_target_ate(control, [40, 50], target_ate_pp=5.0, low=0, high=100)
    assert control == control_copy


def test_respects_bounds():
    control = [0]
    raw = [95, 98, 100]
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=50.0, low=0, high=100)
    assert all(0 <= v <= 100 for v in out)


def test_infeasible_target_saturates_gracefully_not_silently():
    control = [0]
    raw = [99, 100]  # already near the top
    # asking for an enormous positive effect that can't fit in [0,100]
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=1000.0, low=0, high=100)
    assert out == [100, 100]  # closest achievable, not a silent pretend-match


def test_binary_degenerates_to_flipping_minimum_count():
    control = [0, 0, 0, 0, 0, 0, 0, 0]  # control mean = 0
    raw = [0, 0, 0, 0, 0, 0, 0, 0]
    # target_ate_pp=25 on a 0/1 scale -> target mean=0.25 -> 2 of 8 flipped to 1
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=25.0, low=0, high=1)
    assert sum(out) == 2
    assert all(v in (0, 1) for v in out)


def test_donation_scale_0_to_10():
    control = [2, 2, 2]  # mean 2
    raw = [2, 2, 2, 2]
    out = calibrate_arm_to_target_ate(control, raw, target_ate_pp=10.0, low=0, high=10)  # target mean = 3
    assert _mean(out) == pytest.approx(3.0)
    assert all(0 <= v <= 10 for v in out)


def test_deterministic_across_repeated_calls():
    control = [50, 50]
    raw = [40, 45, 55, 60]
    out1 = calibrate_arm_to_target_ate(control, raw, target_ate_pp=8.0, low=0, high=100)
    out2 = calibrate_arm_to_target_ate(control, raw, target_ate_pp=8.0, low=0, high=100)
    assert out1 == out2


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        calibrate_arm_to_target_ate([1], [1], target_ate_pp=0.0, low=10, high=10)


def test_non_integer_raw_response_rejected_instead_of_rounded():
    with pytest.raises(ValueError):
        calibrate_arm_to_target_ate([50], [50.5], target_ate_pp=0.0, low=0, high=100)


def test_out_of_bounds_raw_response_rejected():
    with pytest.raises(ValueError):
        calibrate_arm_to_target_ate([50], [101], target_ate_pp=0.0, low=0, high=100)
