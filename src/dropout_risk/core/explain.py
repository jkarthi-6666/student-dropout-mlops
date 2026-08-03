"""SHAP explanations for the HistGB candidate (Phase 2E interpretability).

Two outputs, both required by the problem statement:
  - global feature importance (mean |SHAP|) -> which features drive the model
  - per-student top-3 contributing features -> the "why flagged" reasons that go
    in the inference output contract

Design: SHAP must see the matrix the classifier actually consumes, i.e. AFTER
the ColumnTransformer. So we transform the engineered frame through the fitted
preprocessor, then run TreeExplainer on the final estimator. Feature names come
from the fitted ColumnTransformer's get_feature_names_out, so SHAP values map
back to human-readable columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dropout_risk.core.features import add_engineered_features


def _transform_for_shap(fitted_pipeline, X: pd.DataFrame):
    """Run the frame through engineering + the fitted preprocessor.

    Returns (transformed_matrix, feature_names, final_estimator). Mirrors exactly
    what the classifier sees at predict time, so SHAP attributions are faithful.
    """
    engineered = add_engineered_features(X)
    # pipeline steps: ("pre", ColumnTransformer), ("clf", HistGB)
    pre = fitted_pipeline.named_steps["pre"]
    clf = fitted_pipeline.named_steps["clf"]
    transformed = pre.transform(engineered)
    if hasattr(transformed, "columns"):
        feature_names = list(transformed.columns)
        transformed_values = transformed.to_numpy()
    else:
        feature_names = list(pre.get_feature_names_out())
        transformed_values = np.asarray(transformed)
    return transformed_values, feature_names, clf


def global_importance(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, descending. The global 'what matters'."""
    import shap

    values, names, clf = _transform_for_shap(fitted_pipeline, X)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(values)
    # binary classifier: shap_values may be a list [class0, class1] or a 2D array
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_reasons_per_student(
    fitted_pipeline, X: pd.DataFrame, top_n: int = 3
) -> pd.DataFrame:
    """For each row, the top_n features pushing risk UP, with their SHAP values.

    Returns a frame with columns reason_1..reason_n (feature names) and
    reason_1_value..reason_n_value. These are the "why flagged" explanations for
    the inference output contract.
    """
    import shap

    values, names, clf = _transform_for_shap(fitted_pipeline, X)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(values)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    names_arr = np.array(names)
    rows = []
    for i in range(sv.shape[0]):
        contribs = sv[i]
        # rank features by how much they PUSH RISK UP (most positive SHAP)
        order = np.argsort(-contribs)[:top_n]
        row = {}
        for rank, idx in enumerate(order, start=1):
            row[f"reason_{rank}"] = str(names_arr[idx])
            row[f"reason_{rank}_value"] = float(contribs[idx])
        rows.append(row)
    return pd.DataFrame(rows)
