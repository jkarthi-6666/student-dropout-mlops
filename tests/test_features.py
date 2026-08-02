"""Tests for core.features.

Emphasis on the two footguns: divide-by-zero and the grade scale mismatch.
Values are chosen so the correct answers are computable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dropout_risk.core.features import (
    ENGINEERED_COLUMNS,
    add_engineered_features,
    grade_delta,
    parents_max_qualification,
    sem1_pass_rate,
    zero_enrolled_flag,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [6, 5, 0, 4],
            "Curricular units 1st sem (approved)": [3, 5, 0, 1],
            "Curricular units 1st sem (evaluations)": [6, 5, 0, 8],
            "Curricular units 1st sem (without evaluations)": [0, 0, 0, 2],
            "Curricular units 1st sem (grade)": [10.0, 20.0, 0.0, 5.0],
            "Admission grade": [100.0, 200.0, 120.0, 50.0],
            "Mother's qualification": [3, 19, 1, 34],
            "Father's qualification": [5, 12, 2, 30],
        }
    )


def test_pass_rate_basic():
    pr = sem1_pass_rate(_df())
    # 3/6, 5/5, 0-enrolled->0, 1/4
    assert np.allclose(pr.values, [0.5, 1.0, 0.0, 0.25])


def test_pass_rate_no_inf_or_nan_on_zero_enrolled():
    pr = sem1_pass_rate(_df())
    assert np.isfinite(pr.values).all()
    assert pr.iloc[2] == 0.0  # the zero-enrolled row


def test_zero_enrolled_flag():
    z = zero_enrolled_flag(_df())
    assert list(z.values) == [0, 0, 1, 0]


def test_grade_delta_uses_normalised_scales():
    gd = grade_delta(_df())
    # row0: 10/20 - 100/200 = 0.5 - 0.5 = 0.0
    # row1: 20/20 - 200/200 = 1.0 - 1.0 = 0.0
    # row2:  0/20 - 120/200 = 0.0 - 0.6 = -0.6
    # row3:  5/20 -  50/200 = 0.25 - 0.25 = 0.0
    assert np.allclose(gd.values, [0.0, 0.0, -0.6, 0.0])


def test_grade_delta_would_be_wrong_if_raw():
    # Guard against a regression to raw subtraction: raw row0 would be
    # 10 - 100 = -90, nothing like the correct 0.0.
    gd = grade_delta(_df())
    assert abs(gd.iloc[0]) < 1e-9


def test_parents_max():
    pm = parents_max_qualification(_df())
    assert list(pm.values) == [5, 19, 2, 34]


def test_add_engineered_appends_all_columns():
    out = add_engineered_features(_df())
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns


def test_add_engineered_is_idempotent():
    once = add_engineered_features(_df())
    twice = add_engineered_features(once)
    # no duplicate columns, same values
    assert list(once.columns) == list(twice.columns)
    assert np.allclose(
        once[ENGINEERED_COLUMNS].values, twice[ENGINEERED_COLUMNS].values
    )


def test_original_frame_not_mutated():
    df = _df()
    before = df.copy()
    _ = add_engineered_features(df)
    pd.testing.assert_frame_equal(df, before)
