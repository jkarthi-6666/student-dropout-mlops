"""Evaluation logic (Phase 2E/2F).

Computes the full metric suite for a fitted model on a test set, plus the
slice breakdown across the six sensitive attributes. Pure functions over
arrays and frames; no ZenML, no MLflow. The step wrapper handles logging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from dropout_risk.core.metrics import (
    bootstrap_metric,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)


def evaluate_model(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_percent: float = 0.10,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Full metric suite for one model's scores on the test set."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    boot = bootstrap_metric(
        y_true, scores, metric_fn=precision_at_k,
        n_iter=n_bootstrap, ci_level=ci_level, seed=seed, k_percent=k_percent,
    )
    return {
        "precision_at_k": precision_at_k(y_true, scores, k_percent),
        "precision_at_k_ci_low": boot["ci_low"],
        "precision_at_k_ci_high": boot["ci_high"],
        "lift_at_k": lift_at_k(y_true, scores, k_percent),
        "recall_at_k": recall_at_k(y_true, scores, k_percent),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "base_rate": float(y_true.mean()),
    }


def evaluate_slices(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    slice_columns: list[str],
    k_percent: float = 0.10,
) -> pd.DataFrame:
    """Precision@k and selection rate within each level of each slice attribute.

    A model whose top decile concentrates on one demographic is a finding; this
    table is how it surfaces. Small slices (< k students) are reported but
    flagged, since precision@k is unstable there.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    k = max(1, int(np.ceil(k_percent * n)))
    top_k_idx = set(np.argsort(-scores, kind="stable")[:k].tolist())

    rows = []
    for col in slice_columns:
        if col not in df.columns:
            continue
        values = df[col].to_numpy()
        for level in np.unique(values):
            mask = values == level
            grp_n = int(mask.sum())
            # how many of this group's members are in the global top-k
            grp_idx = np.where(mask)[0]
            in_topk = [i for i in grp_idx if i in top_k_idx]
            n_flagged = len(in_topk)
            grp_dropouts_flagged = int(y_true[in_topk].sum()) if in_topk else 0
            precision = grp_dropouts_flagged / n_flagged if n_flagged else np.nan
            rows.append({
                "slice": col,
                "level": level,
                "group_n": grp_n,
                "group_base_rate": float(y_true[mask].mean()),
                "n_flagged": n_flagged,
                "selection_rate": n_flagged / grp_n if grp_n else np.nan,
                "precision_in_group": precision,
                "small_slice": grp_n < k,
            })
    return pd.DataFrame(rows)


def cross_val_precision_at_k(
    df,
    model_name: str,
    config: dict,
    n_splits: int = 5,
    k_percent: float = 0.10,
    seed: int = 42,
) -> dict:
    """5-fold stratified CV precision@k for one model on the full frame.

    This is the honest headline metric: a single train/test split can produce a
    misleadingly perfect score, but CV averaging across five folds reveals the
    true, reproducible performance. Returns mean, std, and per-fold scores.

    Imported lazily inside to avoid a circular import (models imports evaluation
    indirectly via nothing, but keeping build_model local is safest).
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    from dropout_risk.core.metrics import precision_at_k
    from dropout_risk.core.models import build_model

    target = config["target"]["name"]
    y = df[target].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_scores = []
    for train_idx, test_idx in skf.split(df, y):
        train_fold = df.iloc[train_idx]
        test_fold = df.iloc[test_idx]
        model = build_model(model_name, config)
        model.fit(train_fold, train_fold[target].to_numpy())
        scores = model.predict_scores(test_fold)
        fold_scores.append(
            precision_at_k(test_fold[target].to_numpy(), scores, k_percent)
        )

    fold_scores = np.array(fold_scores)
    return {
        "cv_precision_at_k_mean": float(fold_scores.mean()),
        "cv_precision_at_k_std": float(fold_scores.std()),
        "cv_folds": [float(s) for s in fold_scores],
    }
