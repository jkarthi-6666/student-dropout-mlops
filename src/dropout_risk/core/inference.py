"""Batch inference: produce the ranked intervention list (Phase 2E).

The deliverable. Given a fitted model and a cohort frame, score every student,
rank by risk, take the top k%, and attach each flagged student's top-3 SHAP
reasons. Output matches the problem statement's contract:

    student_id, risk_score, rank, reason_1..3, reason_1_value..3_value

Pure functions over a fitted pipeline; no ZenML. The step wrapper loads the
registered model and calls this.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dropout_risk.core.explain import top_reasons_per_student


def rank_cohort(
    fitted_pipeline,
    cohort: pd.DataFrame,
    k_percent: float = 0.10,
    id_column: str | None = None,
    add_reasons: bool = True,
) -> pd.DataFrame:
    """Score, rank, and return the top-k% highest-risk students with reasons.

    fitted_pipeline: the sklearn Pipeline from the trained HistGB model
        (i.e. model.pipeline_), which handles engineering + preprocessing + clf.
    cohort: raw student frame (same schema as training, minus the target).
    id_column: column to use as student identifier; if None, uses the row index.
    """
    cohort = cohort.reset_index(drop=True)
    n = len(cohort)
    k = max(1, math.ceil(k_percent * n))

    scores = fitted_pipeline.predict_proba(_engineer(cohort))[:, 1]

    # student id
    if id_column and id_column in cohort.columns:
        ids = cohort[id_column].to_numpy()
    else:
        ids = np.arange(n)

    ranked_idx = np.argsort(-scores, kind="stable")
    top_idx = ranked_idx[:k]

    out = pd.DataFrame({
        "student_id": ids[top_idx],
        "risk_score": scores[top_idx],
        "rank": np.arange(1, k + 1),
    })

    if add_reasons:
        reasons = top_reasons_per_student(
            fitted_pipeline, cohort.iloc[top_idx], top_n=3
        )
        reasons = reasons.reset_index(drop=True)
        out = pd.concat([out.reset_index(drop=True), reasons], axis=1)

    return out


def _engineer(cohort: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features before the pipeline (which expects them present)."""
    from dropout_risk.core.features import add_engineered_features
    return add_engineered_features(cohort)
