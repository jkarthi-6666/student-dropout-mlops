"""Tests for core.drift."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from dropout_risk.core.drift import (
    check_prediction_volume,
    drift_verdict,
    run_drift_report,
)

warnings.filterwarnings("ignore")


def _ref(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "grade": rng.normal(120, 20, n),
        "age": rng.normal(25, 6, n),        # continuous, not low-card integer
        "score": rng.normal(50, 10, n),
    })


def test_no_drift_when_same_distribution():
    # large samples from the SAME generator: drift should not fire on the
    # majority of columns. (Low-cardinality integer columns can trip chi-square
    # from pure sampling noise, so the no-drift reference uses continuous ones.)
    ref = _ref(n=2000, seed=1)
    cur = _ref(n=2000, seed=2)
    summary = run_drift_report(ref, cur)
    verdict = drift_verdict(summary, share_threshold=0.5)
    assert not verdict["alert"]


def test_drift_detected_when_shifted():
    ref = _ref(seed=1)
    rng = np.random.default_rng(9)
    cur = pd.DataFrame({
        "grade": rng.normal(170, 20, 400),  # shifted up
        "age": rng.integers(35, 65, 400),   # shifted up
        "units": rng.integers(0, 7, 400),
    })
    summary = run_drift_report(ref, cur)
    verdict = drift_verdict(summary, share_threshold=0.5)
    assert verdict["alert"]
    assert "grade" in verdict["drifted_columns"]


def test_volume_normal():
    r = check_prediction_volume(89, 885, 0.10, 0.30)
    assert not r["alert"]


def test_volume_alert_when_extreme():
    r = check_prediction_volume(450, 885, 0.10, 0.30)
    assert r["alert"]
    assert r["actual_rate"] > 0.5


def test_drift_summary_structure():
    ref = _ref(n=800)
    cur = _ref(n=800, seed=5)
    summary = run_drift_report(ref, cur)
    assert "drifted_share" in summary
    assert "per_column" in summary
    assert summary["n_columns_compared"] == 3
