from __future__ import annotations

import numpy as np

from calibration.project_support import project_binary_to_count, project_integer_to_total, project_matrix_to_composite_total


def test_integer_projection_hits_nearest_attainable_total():
    ideal = np.array([10.2, 20.8, 30.1, 99.9])
    out = project_integer_to_total(ideal, low=0, high=100, target_total=175)

    assert out.dtype.kind in {"i", "u"}
    assert out.sum() == 175
    assert out.min() >= 0
    assert out.max() <= 100


def test_donation_projection_uses_0_to_10_support():
    ideal = np.array([9.8, 9.7, 2.1])
    out = project_integer_to_total(ideal, low=0, high=10, target_total=25)

    assert out.sum() == 25
    assert out.min() >= 0
    assert out.max() <= 10


def test_binary_projection_assigns_top_k_stably():
    ideal = np.array([0.2, 0.9, 0.9, -1.0])
    out = project_binary_to_count(ideal, target_count=2)

    assert out.tolist() == [0, 1, 1, 0]


def test_matrix_projection_hits_composite_aggregate_total():
    ideal = np.array([[40.2, 41.1, 42.8], [50.0, 51.0, 52.0]])
    out = project_matrix_to_composite_total(ideal, low=0, high=100, target_total=300)

    assert out.shape == ideal.shape
    assert out.sum() == 300
    assert out.min() >= 0
    assert out.max() <= 100
