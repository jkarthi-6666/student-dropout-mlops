"""The model ladder (Phase 2D).

Four rankers, in increasing sophistication. All expose a uniform interface:
`.fit(df, y)` and `.predict_scores(df) -> np.ndarray` where higher = more likely
dropout. This uniformity lets the evaluation code treat every rung identically.

  0. MajorityBaseline   - constant score; precision@k = base rate. The floor.
  1. PassRateBaseline   - rank by (1 - sem1_pass_rate). No fitting. The real bar.
  2. Logistic           - one-hot + scaled, linear.
  3. HistGBModel        - native categorical gradient boosting. The candidate.

Rungs 2 and 3 wrap the preprocessors from core.preprocessing so the whole
transform+model chain is one fitted object (train/serve parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dropout_risk.core.features import add_engineered_features, sem1_pass_rate
from dropout_risk.core.preprocessing import (
    build_gbm_preprocessor,
    build_logistic_preprocessor,
    gbm_categorical_mask,
)


class MajorityBaseline:
    """Constant scorer. Every student gets the same score => precision@k == base rate."""

    name = "majority_baseline"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MajorityBaseline":
        self.base_rate_ = float(np.asarray(y).mean())
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        # constant; ranking is arbitrary but precision@k resolves to base rate
        return np.full(len(X), self.base_rate_)


class PassRateBaseline:
    """Rank by ascending semester-1 pass rate: worst performers scored highest.

    score = 1 - pass_rate, so a student who passed nothing scores 1.0. No fitting
    required, but .fit is provided for interface uniformity.
    """

    name = "passrate_baseline"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PassRateBaseline":
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return (1.0 - sem1_pass_rate(X)).to_numpy()


class Logistic:
    """Logistic regression over the one-hot + scaled preprocessing branch."""

    name = "logistic"

    def __init__(self, use_engineered: bool = True, C: float = 1.0,
                 max_iter: int = 2000, class_weight=None):
        self.use_engineered = use_engineered
        self.pipeline_ = Pipeline(
            steps=[
                ("pre", build_logistic_preprocessor(use_engineered=use_engineered)),
                ("clf", LogisticRegression(
                    C=C, max_iter=max_iter, class_weight=class_weight)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Logistic":
        self.pipeline_.fit(add_engineered_features(X), np.asarray(y))
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(add_engineered_features(X))[:, 1]


class HistGBModel:
    """HistGradientBoosting with native categorical handling.

    The preprocessor emits categoricals-first, then numerics; the categorical
    mask is passed to the classifier so it splits on them directly.
    """

    name = "histgb"

    def __init__(self, use_engineered: bool = True, class_weight=None, **hgb_params):
        self.use_engineered = use_engineered
        pre = build_gbm_preprocessor(use_engineered=use_engineered)
        mask = gbm_categorical_mask(use_engineered=use_engineered)
        clf = HistGradientBoostingClassifier(
            categorical_features=mask,
            class_weight=class_weight,
            random_state=hgb_params.pop("random_state", 42),
            **hgb_params,
        )
        self.pipeline_ = Pipeline(steps=[("pre", pre), ("clf", clf)])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HistGBModel":
        self.pipeline_.fit(add_engineered_features(X), np.asarray(y))
        return self

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(add_engineered_features(X))[:, 1]


def build_model(name: str, config: dict):
    """Factory: construct a model by name using the model section of config."""
    use_eng = config["features"]["use_engineered"]
    cw = config["model"].get("class_weight")

    if name == "majority_baseline":
        return MajorityBaseline()
    if name == "passrate_baseline":
        return PassRateBaseline()
    if name == "logistic":
        lr = config["model"]["logistic_regression"]
        return Logistic(use_engineered=use_eng, C=lr["C"],
                        max_iter=lr["max_iter"], class_weight=cw)
    if name == "histgb":
        hp = config["model"]["hist_gradient_boosting"]
        return HistGBModel(use_engineered=use_eng, class_weight=cw, **hp)
    raise ValueError(f"unknown model name: {name}")
