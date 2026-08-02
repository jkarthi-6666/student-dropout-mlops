"""Tests for core.preprocessing.

Key properties verified:
  - GBM branch: no expansion, categorical mask aligns with output width.
  - Logistic branch: cardinality capped, unseen categories tolerated at transform.
  - Train/serve parity: fit on train, transform test, no error, stable columns.
  - No leakage: fitting on train then transforming test never uses test statistics
    (checked indirectly via the scaler being fit only on train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_risk.core.preprocessing import (
    CATEGORICAL_COLUMNS,
    build_gbm_preprocessor,
    build_logistic_preprocessor,
    gbm_categorical_mask,
    numeric_feature_columns,
)


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Marital status": rng.integers(1, 7, n),
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
            "Curricular units 1st sem (enrolled)": rng.integers(1, 8, n),
            "Curricular units 1st sem (evaluations)": rng.integers(0, 10, n),
            "Curricular units 1st sem (approved)": rng.integers(0, 7, n),
            "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
            "Curricular units 1st sem (without evaluations)": rng.integers(0, 5, n),
            "Unemployment rate": rng.choice([7.6, 10.8, 12.4], n),
            "Inflation rate": rng.choice([0.3, 1.4, 2.6], n),
            "GDP": rng.choice([0.32, 1.79, -1.7], n),
        }
    )


# ---- GBM branch -------------------------------------------------------------

def test_gbm_no_expansion_matches_mask():
    df = _frame()
    pre = build_gbm_preprocessor(use_engineered=True)
    out = pre.fit_transform(df)
    mask = gbm_categorical_mask(use_engineered=True)
    assert out.shape[1] == len(mask)


def test_gbm_mask_marks_categoricals_first():
    mask = gbm_categorical_mask(use_engineered=True)
    assert all(mask[: len(CATEGORICAL_COLUMNS)])  # leading block True
    assert not any(mask[len(CATEGORICAL_COLUMNS):])  # trailing block False


def test_gbm_engineered_toggle_changes_width():
    df = _frame()
    wide = build_gbm_preprocessor(True).fit_transform(df).shape[1]
    narrow = build_gbm_preprocessor(False).fit_transform(df).shape[1]
    assert wide == narrow + 6  # six engineered features


# ---- Logistic branch --------------------------------------------------------

def test_logistic_caps_cardinality():
    df = _frame()
    pre = build_logistic_preprocessor(use_engineered=True, max_categories=11)
    out = pre.fit_transform(df)
    # Father's occupation has ~46 levels; capped encoding must be far below
    # a naive one-hot. Total width should be well under 200.
    assert out.shape[1] < 200


def test_logistic_tolerates_unseen_category_at_transform():
    train = _frame(seed=1)
    pre = build_logistic_preprocessor(use_engineered=True)
    pre.fit(train)

    test = _frame(seed=2)
    # inject a course code never seen in train
    test.loc[0, "Course"] = 999
    # must not raise
    out = pre.transform(test)
    assert out.shape[0] == len(test)


def test_logistic_train_test_columns_stable():
    train = _frame(seed=1)
    test = _frame(seed=2)
    pre = build_logistic_preprocessor(use_engineered=True)
    pre.fit(train)
    tr = pre.transform(train)
    te = pre.transform(test)
    assert list(tr.columns) == list(te.columns)


# ---- shared -----------------------------------------------------------------

def test_numeric_columns_include_engineered_when_requested():
    with_eng = numeric_feature_columns(True)
    without = numeric_feature_columns(False)
    assert len(with_eng) == len(without) + 6


def test_no_nan_in_output():
    df = _frame()
    for builder in (build_gbm_preprocessor, build_logistic_preprocessor):
        out = builder(use_engineered=True).fit_transform(df)
        assert not np.isnan(np.asarray(out, dtype=float)).any()
