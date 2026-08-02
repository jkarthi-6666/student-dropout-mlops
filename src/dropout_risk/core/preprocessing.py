"""Preprocessing: two transformer branches sharing one feature-engineering front end.

Both models see the same engineered features. They differ only in how categoricals
are handled:

  - GBM path: integer-encoded categoricals passed through; HistGradientBoosting
    splits on them natively via a categorical mask. No expansion -> ~36 columns.
  - Logistic path: categoricals one-hot encoded with max_categories capping
    (top-K + infrequent bucket) so cardinality can't explode. Numerics scaled.

Everything is wrapped so it fits on train only and serialises as one object,
which is what guarantees train/serve parity: inference loads the identical
fitted transformer.

sklearn 1.9: OneHotEncoder(max_categories=, handle_unknown="infrequent_if_exist")
does the top-K + OTHER bucketing internally and tolerates unseen categories at
inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dropout_risk.core.features import ENGINEERED_COLUMNS

# Integer-encoded categoricals (verified in EDA). Not ordinal; must not be scaled.
CATEGORICAL_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Macro indicators: kept, but EDA showed they are cohort-year proxies.
MACRO_COLUMNS = ["Unemployment rate", "Inflation rate", "GDP"]

# Raw numeric columns (excluding categoricals, target, and engineered).
RAW_NUMERIC_COLUMNS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
] + MACRO_COLUMNS


def _engineered_feature_names(transformer, input_features):
    """Feature names out = all input columns plus the six engineered ones.

    Providing this lets the FunctionTransformer preserve names under any global
    sklearn output config, which is what prevents columns from being silently
    dropped to a bare ndarray inside a composed Pipeline.
    """
    input_features = list(input_features)
    return input_features + [c for c in ENGINEERED_COLUMNS if c not in input_features]


def numeric_feature_columns(use_engineered: bool) -> list[str]:
    """Numeric columns fed to the model, optionally including engineered ones."""
    cols = list(RAW_NUMERIC_COLUMNS)
    if use_engineered:
        # parents_max_qualification is numeric-ish; the rest are ratios/flags.
        cols = cols + ENGINEERED_COLUMNS
    return cols


def build_gbm_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    """Preprocessor for the HistGradientBoosting path.

    Assumes engineered features are ALREADY present on the input frame (added
    upstream by add_engineered_features). This keeps the sklearn Pipeline free of
    a FunctionTransformer, which is what makes it robust to MLflow autolog's
    global sklearn config changes -- there is no stateless wrapper for autolog to
    corrupt, only a plain column-selecting ColumnTransformer.

    Categoricals pass through untouched (the model splits on them natively);
    numerics pass through unscaled (trees are scale-invariant).
    """
    numeric_cols = numeric_feature_columns(use_engineered)

    column_tf = ColumnTransformer(
        transformers=[
            ("categorical", "passthrough", CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_tf.set_output(transform="pandas")
    return column_tf


def gbm_categorical_mask(use_engineered: bool = True) -> list[bool]:
    """Boolean mask marking which output columns are categorical, for HistGB.

    Order matches build_gbm_preprocessor output: categoricals first, then numerics.
    """
    numeric_cols = numeric_feature_columns(use_engineered)
    return [True] * len(CATEGORICAL_COLUMNS) + [False] * len(numeric_cols)


def build_logistic_preprocessor(
    use_engineered: bool = True, max_categories: int = 11
) -> ColumnTransformer:
    """Preprocessor for the LogisticRegression path.

    Assumes engineered features are already present (added upstream). Categoricals
    one-hot encoded with capping (top-(max_categories-1) + an infrequent bucket);
    numerics standardised. handle_unknown="infrequent_if_exist" routes unseen
    categories at inference to the infrequent bucket rather than erroring.
    """
    numeric_cols = numeric_feature_columns(use_engineered)

    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        max_categories=max_categories,
        sparse_output=False,
        dtype=np.float64,
    )

    column_tf = ColumnTransformer(
        transformers=[
            ("categorical", ohe, CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_tf.set_output(transform="pandas")
    return column_tf
