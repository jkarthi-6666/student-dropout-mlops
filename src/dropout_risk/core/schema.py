"""Data validation schema for UCI 697.

Encodes the rules confirmed during EDA (Step 3):
  - zero nulls permitted anywhere (dataset is complete; any null = corruption)
  - grades on the Portuguese 0-200 scale
  - age at enrollment strictly positive
  - the binary target present and 0/1
  - base rate within 32.1% +/- 5pp (catches silent label-mapping corruption)

Strict policy: ANY violation raises. There is no warn-and-continue path. For a
fixed research dataset this is the safest choice and keeps behaviour simple to
test -- every bad input is expected to raise.

Import path note: Pandera 0.32 exposes the pandas API at `pandera.pandas`.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Grades (admission + previous qualification) use the Portuguese 0-200 scale.
GRADE_MIN, GRADE_MAX = 0.0, 200.0

# Semester-1 curricular grades use a 0-20 scale.
SEM_GRADE_MIN, SEM_GRADE_MAX = 0.0, 20.0


def build_schema() -> DataFrameSchema:
    """Return the strict validation schema for the raw dropout DataFrame.

    Only columns with well-defined public ranges are bounded. Integer-encoded
    categoricals are checked for type and non-nullness but not enumerated here,
    because their legal code sets are validated separately against values
    observed at fit time (Step 5), not hard-coded into the schema.
    """
    numeric_bounded = {
        "Previous qualification (grade)": (GRADE_MIN, GRADE_MAX),
        "Admission grade": (GRADE_MIN, GRADE_MAX),
        "Curricular units 1st sem (grade)": (SEM_GRADE_MIN, SEM_GRADE_MAX),
    }

    columns: dict[str, Column] = {}

    # Bounded numeric columns: not null, within documented range.
    for name, (lo, hi) in numeric_bounded.items():
        columns[name] = Column(
            float,
            checks=Check.in_range(lo, hi),
            nullable=False,
            required=True,
            coerce=True,
        )

    # Age must be strictly positive.
    columns["Age at enrollment"] = Column(
        int, checks=Check.greater_than(0), nullable=False, required=True, coerce=True
    )

    # Semester-1 unit counts are non-negative integers.
    for name in [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (without evaluations)",
    ]:
        columns[name] = Column(
            int, checks=Check.greater_than_or_equal_to(0),
            nullable=False, required=True, coerce=True,
        )

    # Binary flag columns: 0/1 only.
    for name in [
        "Displaced", "Educational special needs", "Debtor",
        "Tuition fees up to date", "Gender", "Scholarship holder", "International",
    ]:
        columns[name] = Column(
            int, checks=Check.isin([0, 1]),
            nullable=False, required=True, coerce=True,
        )

    # Binary target: present and 0/1.
    columns["dropout"] = Column(
        int, checks=Check.isin([0, 1]), nullable=False, required=True, coerce=True
    )

    # Original multiclass target: one of the three known labels.
    columns["Target"] = Column(
        str,
        checks=Check.isin(["Dropout", "Enrolled", "Graduate"]),
        nullable=False, required=True,
    )

    # strict=False: other documented integer-encoded categorical columns are
    # allowed through without per-column rules. They are still guarded globally
    # for nulls by the wide check in validate_dataframe().
    return DataFrameSchema(columns, strict=False, coerce=True)


def check_no_nulls_anywhere(df) -> None:
    """Fail if any null exists in any column. EDA confirmed zero nulls."""
    total = int(df.isnull().sum().sum())
    if total > 0:
        offending = df.isnull().sum()
        offending = offending[offending > 0].to_dict()
        raise ValueError(
            f"Data contains {total} null value(s); dataset must be complete. "
            f"Columns with nulls: {offending}"
        )


def check_base_rate(df, expected: float, tol: float) -> None:
    """Fail if the dropout base rate drifts outside expected +/- tol.

    This is the check that catches a silent target-mapping bug: if the positive
    class were mislabelled, the rate would jump and this would halt the run.
    """
    if "dropout" not in df.columns:
        raise KeyError("Column 'dropout' missing; cannot check base rate.")
    rate = float(df["dropout"].mean())
    if abs(rate - expected) > tol:
        raise ValueError(
            f"Dropout base rate {rate:.4f} outside expected "
            f"{expected} +/- {tol}. Possible label corruption."
        )


def validate_dataframe(df, expected_base_rate: float, base_rate_tol: float):
    """Run all validation gates. Raises on the first violation; returns df on success.

    Order: global null check first (cheapest, catches the most), then base rate,
    then the full column schema.
    """
    check_no_nulls_anywhere(df)
    check_base_rate(df, expected_base_rate, base_rate_tol)
    schema = build_schema()
    return schema.validate(df, lazy=False)
