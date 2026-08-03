"""Tests for core.explain.

SHAP is stochastic-free for TreeExplainer (exact), so we can assert structure:
correct shape, all features covered, reasons ranked by descending contribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dropout_risk.core.explain import global_importance, top_reasons_per_student
from dropout_risk.core.models import build_model

warnings.filterwarnings("ignore")

CONFIG = {
    "features": {"use_engineered": True},
    "model": {"class_weight": None,
        "hist_gradient_boosting": {"learning_rate": 0.1, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0,
            "max_iter": 50, "early_stopping": True}},
}


@pytest.fixture(scope="module")
def fitted():
    # import here so the heavy frame builder is shared
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_models import _signal_frame
    df = _signal_frame(n=500, seed=3)
    y = df["dropout"].to_numpy()
    model = build_model("histgb", CONFIG).fit(df, y)
    return model, df


def test_global_importance_covers_all_features(fitted):
    model, df = fitted
    gi = global_importance(model.pipeline_, df)
    # every feature has a non-negative importance
    assert (gi["mean_abs_shap"] >= 0).all()
    # sorted descending
    assert gi["mean_abs_shap"].is_monotonic_decreasing
    # non-trivial number of features
    assert len(gi) > 20


def test_top_reasons_shape(fitted):
    model, df = fitted
    tr = top_reasons_per_student(model.pipeline_, df.head(10), top_n=3)
    assert len(tr) == 10
    for col in ["reason_1", "reason_2", "reason_3",
                "reason_1_value", "reason_2_value", "reason_3_value"]:
        assert col in tr.columns


def test_reasons_ranked_descending(fitted):
    model, df = fitted
    tr = top_reasons_per_student(model.pipeline_, df.head(5), top_n=3)
    # reason_1 value >= reason_2 value >= reason_3 value for every row
    for _, row in tr.iterrows():
        assert row["reason_1_value"] >= row["reason_2_value"] >= row["reason_3_value"]
