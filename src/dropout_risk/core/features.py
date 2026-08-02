"""Engineered features for the dropout model.

Pure functions over a DataFrame, no sklearn or ZenML. Each returns a new Series
so the originals are never mutated in place. The two known footguns are handled
explicitly and tested:

  1. Divide-by-zero when a student enrolled in 0 units.
  2. The scale mismatch in grade_delta: semester-1 grade is 0-20, admission
     grade is 0-200. Both are normalised to [0,1] before subtraction.

Column-name constants match the UCI 697 CSV exactly (verified in EDA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENROLLED = "Curricular units 1st sem (enrolled)"
APPROVED = "Curricular units 1st sem (approved)"
EVALUATIONS = "Curricular units 1st sem (evaluations)"
WITHOUT_EVAL = "Curricular units 1st sem (without evaluations)"
SEM1_GRADE = "Curricular units 1st sem (grade)"
ADMISSION_GRADE = "Admission grade"
MOTHER_QUAL = "Mother's qualification"
FATHER_QUAL = "Father's qualification"

SEM1_GRADE_MAX = 20.0
ADMISSION_GRADE_MAX = 200.0

# Names of the engineered columns, so downstream code has one source of truth.
ENGINEERED_COLUMNS = [
    "sem1_pass_rate",
    "sem1_eval_rate",
    "sem1_unevaluated_ratio",
    "zero_enrolled_flag",
    "grade_delta",
    "parents_max_qualification",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator/denominator, returning 0.0 wherever denominator is 0.

    A student enrolled in 0 units has no meaningful pass rate; 0.0 is the correct
    encoding (they approved nothing), and it also avoids inf/NaN propagating into
    the model.
    """
    num = numerator.astype(float)
    den = denominator.astype(float)
    result = np.where(den > 0, num / den.where(den > 0, np.nan), 0.0)
    return pd.Series(result, index=numerator.index).fillna(0.0)


def sem1_pass_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of enrolled units approved. The single strongest signal."""
    return _safe_ratio(df[APPROVED], df[ENROLLED])


def sem1_eval_rate(df: pd.DataFrame) -> pd.Series:
    """Evaluations taken per enrolled unit -- assessment participation."""
    return _safe_ratio(df[EVALUATIONS], df[ENROLLED])


def sem1_unevaluated_ratio(df: pd.DataFrame) -> pd.Series:
    """Units left without evaluation, per enrolled unit -- disengagement."""
    return _safe_ratio(df[WITHOUT_EVAL], df[ENROLLED])


def zero_enrolled_flag(df: pd.DataFrame) -> pd.Series:
    """1 if the student enrolled in no units at all. Near-certain dropout."""
    return (df[ENROLLED].astype(float) == 0).astype(int)


def grade_delta(df: pd.DataFrame) -> pd.Series:
    """Semester-1 grade minus admission grade, both normalised to [0,1].

    Positive => performing above entry expectation; negative => below. Raw
    subtraction would be a bug because the two grades live on different scales.
    """
    sem1_norm = df[SEM1_GRADE].astype(float) / SEM1_GRADE_MAX
    adm_norm = df[ADMISSION_GRADE].astype(float) / ADMISSION_GRADE_MAX
    return (sem1_norm - adm_norm).rename("grade_delta")


def parents_max_qualification(df: pd.DataFrame) -> pd.Series:
    """Higher of the two parental qualification codes.

    Collapses two high-cardinality columns into one, cutting one-hot width with
    minimal information loss. Treated as ordinal-ish here only for the max; the
    original columns remain available to the categorical branch.
    """
    return df[[MOTHER_QUAL, FATHER_QUAL]].max(axis=1).rename("parents_max_qualification")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all six engineered columns appended.

    Idempotent: safe to call more than once (overwrites rather than duplicates).
    """
    out = df.copy()
    out["sem1_pass_rate"] = sem1_pass_rate(df)
    out["sem1_eval_rate"] = sem1_eval_rate(df)
    out["sem1_unevaluated_ratio"] = sem1_unevaluated_ratio(df)
    out["zero_enrolled_flag"] = zero_enrolled_flag(df)
    out["grade_delta"] = grade_delta(df)
    out["parents_max_qualification"] = parents_max_qualification(df)
    return out
