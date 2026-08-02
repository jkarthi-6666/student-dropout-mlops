"""Tests for core.metrics.

precision@k is verified against hand-computed examples so we know the headline
number means what we think. Bootstrap functions are checked for structural
correctness (point inside CI, difference logic) rather than exact bounds.
"""

from __future__ import annotations

import numpy as np

from dropout_risk.core.metrics import (
    bootstrap_difference,
    bootstrap_metric,
    k_from_fraction,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)


def test_k_uses_ceil():
    assert k_from_fraction(885, 0.10) == 89  # 88.5 -> 89
    assert k_from_fraction(1000, 0.10) == 100
    assert k_from_fraction(5, 0.10) == 1  # floor would give 0; ceil+min gives 1


def test_precision_at_k_perfect_ranking():
    # 10 items, 3 positives, all ranked at the very top
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # k = ceil(0.30*10)=3 ; all top-3 are positive
    assert precision_at_k(y, scores, k_percent=0.30) == 1.0


def test_precision_at_k_worst_ranking():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # positives scored lowest -> top-3 are all negatives
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    assert precision_at_k(y, scores, k_percent=0.30) == 0.0


def test_precision_at_k_partial():
    # top-4 contains 2 positives -> 0.5
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.85, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1])
    # k = ceil(0.50*8) = 4 ; top-4 idx by score: 0,1,2,3 -> labels 1,0,1,0 -> 2/4
    assert precision_at_k(y, scores, k_percent=0.50) == 0.5


def test_lift_relative_to_base_rate():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # base rate 0.2
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # k=ceil(0.2*10)=2, both positive -> precision 1.0 ; lift = 1.0/0.2 = 5
    assert lift_at_k(y, scores, k_percent=0.20) == 5.0


def test_recall_capped_by_capacity():
    # 10 positives, k=2 -> max recall 0.2 even with perfect ranking
    y = np.array([1] * 10 + [0] * 90)
    scores = np.concatenate([np.ones(10), np.zeros(90)])
    r = recall_at_k(y, scores, k_percent=0.02)  # k = ceil(0.02*100)=2
    assert r == 0.2


def test_tie_breaking_is_stable():
    # all equal scores -> stable sort keeps input order; top-k are first k rows
    y = np.array([1, 0, 1, 0, 1, 0])
    scores = np.full(6, 0.5)
    # k=ceil(0.5*6)=3 -> first 3 rows: labels 1,0,1 -> 2/3
    assert abs(precision_at_k(y, scores, k_percent=0.5) - 2 / 3) < 1e-9


def test_bootstrap_point_inside_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = rng.random(500)
    res = bootstrap_metric(y, scores, n_iter=500)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]


def test_bootstrap_difference_detects_clear_winner():
    # A ranks perfectly, B ranks randomly -> A should clearly beat B
    rng = np.random.default_rng(1)
    y = np.array([1] * 100 + [0] * 400)
    scores_a = np.concatenate([np.ones(100), np.zeros(400)])  # perfect
    scores_b = rng.random(500)  # random
    res = bootstrap_difference(y, scores_a, scores_b, n_iter=500)
    assert res["point"] > 0
    assert res["excludes_zero"] is True


def test_bootstrap_difference_ties_do_not_exclude_zero():
    # identical scorers -> difference is exactly 0, CI cannot exclude zero
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    scores = rng.random(400)
    res = bootstrap_difference(y, scores, scores.copy(), n_iter=500)
    assert res["point"] == 0.0
    assert res["excludes_zero"] is False
