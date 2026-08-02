"""Metrics for the dropout ranking problem.

Pure functions, fully testable. These compute the numbers the project is judged
on, so correctness here matters more than anywhere else.

Design decisions (see problem_statement.md):
  - k uses ceil: capacity is "up to 10%", and ceil keeps k >= 1 on small
    bootstrap resamples, so precision is always defined.
  - k is computed in ONE place (k_from_fraction) so model and baseline are
    always measured on identically sized lists. A comparison across different
    k values is meaningless.
  - Ties in score are broken by a stable sort, so results are reproducible.
"""

from __future__ import annotations

import math

import numpy as np


def k_from_fraction(n: int, k_percent: float) -> int:
    """Number of students in the top k_percent of n, rounded up, at least 1."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < k_percent <= 1:
        raise ValueError("k_percent must be in (0, 1]")
    return max(1, math.ceil(k_percent * n))


def precision_at_k(
    y_true: np.ndarray, scores: np.ndarray, k_percent: float = 0.10
) -> float:
    """Fraction of true positives among the top-k highest-scored items.

    y_true: 1 = dropout (positive), 0 = not.
    scores: predicted risk, higher = more likely dropout.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must be the same length")

    n = y_true.shape[0]
    k = k_from_fraction(n, k_percent)

    # argsort descending; stable so equal scores keep input order (reproducible).
    order = np.argsort(-scores, kind="stable")
    top_k_idx = order[:k]
    return float(y_true[top_k_idx].sum() / k)


def recall_at_k(
    y_true: np.ndarray, scores: np.ndarray, k_percent: float = 0.10
) -> float:
    """Fraction of all true positives captured in the top-k.

    Capacity-capped: max achievable recall = k / (total positives), so on this
    dataset it is bounded well below 1. Report with that ceiling stated.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    n = y_true.shape[0]
    k = k_from_fraction(n, k_percent)
    order = np.argsort(-scores, kind="stable")
    return float(y_true[order[:k]].sum() / total_pos)


def lift_at_k(
    y_true: np.ndarray, scores: np.ndarray, k_percent: float = 0.10
) -> float:
    """precision@k divided by base rate. 1.0 = no better than random."""
    y_true = np.asarray(y_true)
    base_rate = float(y_true.mean())
    if base_rate == 0:
        return 0.0
    return precision_at_k(y_true, scores, k_percent) / base_rate


def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn=precision_at_k,
    n_iter: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> dict:
    """Bootstrap a metric's sampling distribution by resampling with replacement.

    Returns point estimate plus the CI bounds. Resampling rows (paired y_true,
    scores) preserves each item's label-score pairing.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)

    point = metric_fn(y_true, scores, **metric_kwargs)
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], scores[idx], **metric_kwargs)

    alpha = 1.0 - ci_level
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return {"point": float(point), "ci_low": lo, "ci_high": hi, "ci_level": ci_level}


def bootstrap_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn=precision_at_k,
    n_iter: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> dict:
    """Bootstrap the difference metric(A) - metric(B) on the same resampled rows.

    This is the gate the promotion decision uses: if the CI of the difference
    excludes zero, model A genuinely beats model B; otherwise the apparent gap
    is within noise. Both models are scored on the SAME resample each iteration,
    which correctly accounts for their correlation.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)

    point = metric_fn(y_true, scores_a, **metric_kwargs) - metric_fn(
        y_true, scores_b, **metric_kwargs
    )
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        diffs[i] = metric_fn(yt, scores_a[idx], **metric_kwargs) - metric_fn(
            yt, scores_b[idx], **metric_kwargs
        )

    alpha = 1.0 - ci_level
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci_level,
        "excludes_zero": lo > 0 or hi < 0,
    }
