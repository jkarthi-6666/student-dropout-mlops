"""Tests for core.schema.

Strict policy means every corruption type must raise. Tests build a minimal but
schema-valid frame, then mutate one thing at a time to confirm each gate fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dropout_risk.core.schema import (
    check_base_rate,
    check_no_nulls_anywhere,
    validate_dataframe,
)

EXPECTED_RATE = 0.321
TOL = 0.05


def _valid_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_drop = int(round(EXPECTED_RATE * n))
    target = ["Dropout"] * n_drop + ["Graduate"] * (n - n_drop)
    rng.shuffle(target)
    df = pd.DataFrame(
        {
            "Previous qualification (grade)": rng.uniform(0, 200, n),
            "Admission grade": rng.uniform(0, 200, n),
            "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
            "Age at enrollment": rng.integers(17, 60, n),
            "Curricular units 1st sem (credited)": rng.integers(0, 7, n),
            "Curricular units 1st sem (enrolled)": rng.integers(1, 8, n),
            "Curricular units 1st sem (evaluations)": rng.integers(0, 10, n),
            "Curricular units 1st sem (approved)": rng.integers(0, 7, n),
            "Curricular units 1st sem (without evaluations)": rng.integers(0, 5, n),
            "Displaced": rng.integers(0, 2, n),
            "Educational special needs": rng.integers(0, 2, n),
            "Debtor": rng.integers(0, 2, n),
            "Tuition fees up to date": rng.integers(0, 2, n),
            "Gender": rng.integers(0, 2, n),
            "Scholarship holder": rng.integers(0, 2, n),
            "International": rng.integers(0, 2, n),
            "Target": target,
        }
    )
    df["dropout"] = (df["Target"] == "Dropout").astype(int)
    return df


def test_valid_data_passes():
    df = _valid_df()
    out = validate_dataframe(df, EXPECTED_RATE, TOL)
    assert len(out) == len(df)


def test_null_anywhere_raises():
    df = _valid_df()
    df.loc[0, "Admission grade"] = np.nan
    with pytest.raises(ValueError, match="null"):
        check_no_nulls_anywhere(df)


def test_out_of_range_grade_raises():
    df = _valid_df()
    df.loc[0, "Admission grade"] = 250.0  # scale maxes at 200
    with pytest.raises(Exception):
        validate_dataframe(df, EXPECTED_RATE, TOL)


def test_negative_age_raises():
    df = _valid_df()
    df.loc[0, "Age at enrollment"] = -5
    with pytest.raises(Exception):
        validate_dataframe(df, EXPECTED_RATE, TOL)


def test_bad_binary_flag_raises():
    df = _valid_df()
    df.loc[0, "Debtor"] = 7  # must be 0/1
    with pytest.raises(Exception):
        validate_dataframe(df, EXPECTED_RATE, TOL)


def test_shifted_base_rate_raises():
    df = _valid_df()
    # flip most non-dropouts to dropout -> rate far above tolerance
    df["dropout"] = 1
    with pytest.raises(ValueError, match="base rate"):
        check_base_rate(df, EXPECTED_RATE, TOL)


def test_unknown_target_label_raises():
    df = _valid_df()
    df.loc[0, "Target"] = "Withdrawn"  # not a known class
    with pytest.raises(Exception):
        validate_dataframe(df, EXPECTED_RATE, TOL)


def test_missing_required_column_raises():
    df = _valid_df().drop(columns=["Admission grade"])
    with pytest.raises(Exception):
        validate_dataframe(df, EXPECTED_RATE, TOL)
