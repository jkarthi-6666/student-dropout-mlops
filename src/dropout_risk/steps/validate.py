"""ZenML validation step.

Thin wrapper over core.schema.validate_dataframe. All logic lives in core/;
this file only adapts it to the pipeline. Keeping it thin is what lets the
validation logic be unit-tested without a ZenML stack.
"""

from __future__ import annotations

import pandas as pd
from zenml import step
from zenml.logger import get_logger

from dropout_risk.core.schema import validate_dataframe

logger = get_logger(__name__)


@step
def validate_data(
    df: pd.DataFrame,
    expected_base_rate: float,
    base_rate_tol: float,
) -> pd.DataFrame:
    """Validate the raw dataframe against the strict schema.

    Raises and halts the pipeline on any violation. Returns the validated
    dataframe (coerced dtypes applied) on success.
    """
    logger.info("Validating %d rows against strict schema", len(df))
    validated = validate_dataframe(df, expected_base_rate, base_rate_tol)
    logger.info(
        "Validation passed. base rate=%.4f", float(validated["dropout"].mean())
    )
    return validated
