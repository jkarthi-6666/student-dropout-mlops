"""Tests for core.models, core.evaluation, core.gate.

These exercise everything the ZenML train step wraps, so a green run here means
only the orchestration layer is unverified. Uses a realistic UCI-shaped frame
with a planted signal so the models can actually learn something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_risk.core.evaluation import evaluate_model, evaluate_slices
from dropout_risk.core.gate import promotion_decision
from dropout_risk.core.models import build_model

CONFIG = {
    "features": {"use_engineered": True},
    "model": {
        "class_weight": None,
        "logistic_regression": {"C": 1.0, "max_iter": 2000},
        "hist_gradient_boosting": {
            "learning_rate": 0.1, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0,
            "max_iter": 100, "early_stopping": True,
        },
    },
}


def _signal_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    enrolled = rng.integers(1, 8, n)
    # dropouts pass fewer units: plant the signal in approved
    latent = rng.random(n)
    approved = np.where(latent < 0.32,
                        rng.integers(0, 2, n),      # dropouts: few approvals
                        enrolled - rng.integers(0, 2, n))  # others: most
    approved = np.clip(approved, 0, enrolled)
    df = pd.DataFrame({
        "Marital Status": rng.integers(1, 7, n),
        "Application mode": rng.integers(1, 18, n),
        "Application order": rng.integers(0, 10, n),
        "Course": rng.integers(1, 18, n),
        "Daytime/evening attendance": rng.integers(0, 2, n),
        "Previous qualification": rng.integers(1, 18, n),
        "Nacionality": rng.integers(1, 22, n),
        "Mother's qualification": rng.integers(1, 35, n),
        "Father's qualification": rng.integers(1, 35, n),
        "Mother's occupation": rng.integers(1, 47, n),
        "Father's occupation": rng.integers(1, 47, n),
        "Displaced": rng.integers(0, 2, n),
        "Educational special needs": rng.integers(0, 2, n),
        "Debtor": rng.integers(0, 2, n),
        "Tuition fees up to date": rng.integers(0, 2, n),
        "Gender": rng.integers(0, 2, n),
        "Scholarship holder": rng.integers(0, 2, n),
        "International": rng.integers(0, 2, n),
        "Previous qualification (grade)": rng.uniform(0, 200, n),
        "Admission grade": rng.uniform(0, 200, n),
        "Age at enrollment": rng.integers(17, 60, n),
        "Curricular units 1st sem (credited)": rng.integers(0, 7, n),
        "Curricular units 1st sem (enrolled)": enrolled,
        "Curricular units 1st sem (evaluations)": rng.integers(0, 10, n),
        "Curricular units 1st sem (approved)": approved,
        "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
        "Curricular units 1st sem (without evaluations)": rng.integers(0, 5, n),
        "Unemployment rate": rng.choice([7.6, 10.8], n),
        "Inflation rate": rng.choice([0.3, 1.4], n),
        "GDP": rng.choice([0.32, 1.79], n),
    })
    # dropout label correlates with low pass rate (the planted signal)
    pass_rate = approved / enrolled
    prob = 0.7 - 0.6 * pass_rate
    df["dropout"] = (rng.random(n) < prob).astype(int)
    return df


@pytest.mark.parametrize("name", ["majority_baseline", "passrate_baseline",
                                   "logistic", "histgb"])
def test_each_model_fits_and_scores(name):
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    model = build_model(name, CONFIG)
    model.fit(df, y)
    scores = model.predict_scores(df)
    assert len(scores) == len(df)
    assert np.isfinite(scores).all()


def test_majority_precision_equals_base_rate():
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    m = build_model("majority_baseline", CONFIG).fit(df, y)
    res = evaluate_model(y, m.predict_scores(df), n_bootstrap=200)
    assert abs(res["precision_at_k"] - res["base_rate"]) < 0.12


def test_passrate_beats_majority():
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    maj = build_model("majority_baseline", CONFIG).fit(df, y).predict_scores(df)
    pr = build_model("passrate_baseline", CONFIG).fit(df, y).predict_scores(df)
    p_maj = evaluate_model(y, maj, n_bootstrap=100)["precision_at_k"]
    p_pr = evaluate_model(y, pr, n_bootstrap=100)["precision_at_k"]
    assert p_pr > p_maj


def test_evaluate_model_has_all_keys():
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    m = build_model("histgb", CONFIG).fit(df, y)
    res = evaluate_model(y, m.predict_scores(df), n_bootstrap=100)
    for key in ["precision_at_k", "lift_at_k", "recall_at_k", "pr_auc",
                "brier", "precision_at_k_ci_low", "precision_at_k_ci_high"]:
        assert key in res


def test_slice_table_covers_all_attributes():
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    m = build_model("histgb", CONFIG).fit(df, y)
    slices = ["Gender", "Debtor", "Scholarship holder", "International",
              "Displaced", "Age at enrollment"]
    tbl = evaluate_slices(df, y, m.predict_scores(df), slices)
    assert set(tbl["slice"].unique()) == set(slices)


def test_gate_returns_decision():
    df = _signal_frame()
    y = df["dropout"].to_numpy()
    cand = build_model("histgb", CONFIG).fit(df, y).predict_scores(df)
    base = build_model("passrate_baseline", CONFIG).fit(df, y).predict_scores(df)
    decision = promotion_decision(y, cand, base, n_bootstrap=200)
    assert "promote" in decision
    assert isinstance(decision["promote"], bool)


def test_cross_val_returns_five_folds():
    from dropout_risk.core.evaluation import cross_val_precision_at_k
    df = _signal_frame(n=1000)
    cfg = dict(CONFIG)
    cfg["target"] = {"name": "dropout"}
    cfg["split"] = {"cv_folds": 5}
    r = cross_val_precision_at_k(df, "histgb", cfg, n_splits=5, k_percent=0.10)
    assert len(r["cv_folds"]) == 5
    assert 0.0 <= r["cv_precision_at_k_mean"] <= 1.0
    assert r["cv_precision_at_k_std"] >= 0.0
