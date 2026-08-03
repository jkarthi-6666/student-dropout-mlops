"""Tests for core.inference — the ranked intervention list."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from dropout_risk.core.inference import rank_cohort
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
def fitted_and_cohort():
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_models import _signal_frame
    df = _signal_frame(n=800, seed=5)
    y = df["dropout"].to_numpy()
    model = build_model("histgb", CONFIG).fit(df, y)
    cohort = df.drop(columns=["dropout"])
    return model, cohort


def test_output_size_is_k_percent(fitted_and_cohort):
    model, cohort = fitted_and_cohort
    out = rank_cohort(model.pipeline_, cohort, k_percent=0.10)
    assert len(out) == math.ceil(0.10 * len(cohort))


def test_output_contract_columns(fitted_and_cohort):
    model, cohort = fitted_and_cohort
    out = rank_cohort(model.pipeline_, cohort, k_percent=0.10)
    for col in ["student_id", "risk_score", "rank",
                "reason_1", "reason_2", "reason_3"]:
        assert col in out.columns


def test_ranks_and_scores_ordered(fitted_and_cohort):
    model, cohort = fitted_and_cohort
    out = rank_cohort(model.pipeline_, cohort, k_percent=0.10)
    assert list(out["rank"]) == list(range(1, len(out) + 1))
    scores = out["risk_score"].to_numpy()
    assert (scores[:-1] >= scores[1:]).all()


def test_reasons_can_be_disabled(fitted_and_cohort):
    model, cohort = fitted_and_cohort
    out = rank_cohort(model.pipeline_, cohort, k_percent=0.10, add_reasons=False)
    assert "reason_1" not in out.columns
    assert "risk_score" in out.columns


def test_custom_id_column(fitted_and_cohort):
    model, cohort = fitted_and_cohort
    cohort = cohort.copy()
    cohort["student_ref"] = ["S%04d" % i for i in range(len(cohort))]
    out = rank_cohort(model.pipeline_, cohort, k_percent=0.10,
                      id_column="student_ref")
    assert out["student_id"].iloc[0].startswith("S")
