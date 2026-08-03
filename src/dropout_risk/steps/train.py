"""ZenML train + evaluate step.

Fits the full model ladder on train, evaluates each on test, logs everything to
MLflow, runs the promotion gate on the HistGB candidate against the pass-rate
baseline, and returns the fitted candidate pipeline plus a results dict.

This is intentionally one step: the ladder members share a train/test split and
are only meaningful in comparison, so evaluating them together keeps the MLflow
run coherent (one run, all rungs as nested metrics).
"""

from __future__ import annotations

from typing import Annotated, Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator
from zenml import step
from zenml.logger import get_logger

from dropout_risk.core.evaluation import (
    cross_val_precision_at_k,
    evaluate_model,
    evaluate_slices,
)
from dropout_risk.core.gate import promotion_decision
from dropout_risk.core.models import build_model

logger = get_logger(__name__)

LADDER = ["majority_baseline", "passrate_baseline", "logistic", "histgb"]
CANDIDATE = "histgb"
BASELINE = "passrate_baseline"


@step(experiment_tracker="mlflow_tracker")
def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[
    Annotated[BaseEstimator, "candidate_pipeline"],
    Annotated[dict, "results"],
]:
    target = config["target"]["name"]
    k = config["metric"]["k_percent"]
    n_boot = config["metric"]["bootstrap_iterations"]
    ci = config["metric"]["ci_level"]
    seed = config["project"]["seed"]
    slice_cols = config["slices"]

    y_train = train_df[target].to_numpy()
    y_test = test_df[target].to_numpy()

    # log run-level params
    mlflow.log_param("use_engineered", config["features"]["use_engineered"])
    mlflow.log_param("class_weight", str(config["model"].get("class_weight")))
    mlflow.log_param("k_percent", k)
    mlflow.log_param("test_size", config["split"]["test_size"])
    mlflow.log_param("seed", seed)

    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    scores: dict[str, Any] = {}

    for name in LADDER:
        model = build_model(name, config)
        model.fit(train_df, y_train)
        s = model.predict_scores(test_df)
        metrics = evaluate_model(
            y_test, s, k_percent=k, n_bootstrap=n_boot, ci_level=ci, seed=seed
        )
        results[name] = metrics
        fitted[name] = model
        scores[name] = s

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{name}__{metric_name}", value)
        logger.info(
            "%s: precision@%.0f%%=%.3f [%.3f, %.3f]  lift=%.2f  pr_auc=%.3f",
            name, k * 100, metrics["precision_at_k"],
            metrics["precision_at_k_ci_low"], metrics["precision_at_k_ci_high"],
            metrics["lift_at_k"], metrics["pr_auc"],
        )

    # slice analysis on the candidate
    slice_table = evaluate_slices(
        test_df, y_test, scores[CANDIDATE], slice_cols, k_percent=k
    )

    # --- Cross-validated precision@k: the HONEST headline metric ---
    # A single split can produce a misleadingly perfect score; 5-fold CV on the
    # full frame reveals the true, reproducible performance. Reported for the
    # baseline and the candidate so they are compared on the robust number.
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    for name in (BASELINE, CANDIDATE):
        cv = cross_val_precision_at_k(
            full_df, name, config, n_splits=config["split"]["cv_folds"],
            k_percent=k, seed=seed,
        )
        results[name]["cv_precision_at_k_mean"] = cv["cv_precision_at_k_mean"]
        results[name]["cv_precision_at_k_std"] = cv["cv_precision_at_k_std"]
        mlflow.log_metric(f"{name}__cv_precision_at_k_mean", cv["cv_precision_at_k_mean"])
        mlflow.log_metric(f"{name}__cv_precision_at_k_std", cv["cv_precision_at_k_std"])
        logger.info(
            "%s: CV precision@%.0f%% = %.3f +/- %.3f  (single-split was %.3f)",
            name, k * 100, cv["cv_precision_at_k_mean"], cv["cv_precision_at_k_std"],
            results[name]["precision_at_k"],
        )

    # promotion gate: candidate vs baseline
    decision = promotion_decision(
        y_test,
        candidate_scores=scores[CANDIDATE],
        baseline_scores=scores[BASELINE],
        slice_table=slice_table,
        k_percent=k,
        require_ci_excludes_zero=config["gate"]["require_ci_excludes_zero"],
        slice_tolerance=config["gate"]["slice_tolerance"],
        n_bootstrap=n_boot,
        seed=seed,
    )
    results["gate"] = decision

    mlflow.log_metric("gate__difference", decision["difference"])
    mlflow.log_metric("gate__difference_ci_low", decision["difference_ci_low"])
    mlflow.log_metric("gate__difference_ci_high", decision["difference_ci_high"])
    mlflow.log_metric("gate__promote", int(decision["promote"]))

    # persist slice table + log the fitted candidate pipeline
    slice_csv = "slice_table.csv"
    slice_table.to_csv(slice_csv, index=False)
    mlflow.log_artifact(slice_csv)

    # --- SHAP interpretability (Phase 2E) ---
    # Global importance: which features drive the candidate. Logged as CSV so it
    # can be inspected and cited, and as a cross-check on the leakage
    # investigation -- if an administrative flag dominates, revisit it.
    try:
        from dropout_risk.core.explain import global_importance
        gi = global_importance(fitted[CANDIDATE].pipeline_, test_df)
        gi_csv = "shap_global_importance.csv"
        gi.to_csv(gi_csv, index=False)
        mlflow.log_artifact(gi_csv)
        logger.info("SHAP top-5 features: %s",
                    ", ".join(gi.head(5)["feature"].tolist()))
    except Exception as exc:  # SHAP is heavy; never let it break the run
        logger.warning("SHAP global importance skipped: %s", exc)

    # Force cloudpickle serialization: MLflow 3.x defaults toward skops, which
    # rejects functools.partial and some sklearn validation helpers as
    # "untrusted". cloudpickle handles them without a trusted-types allowlist.
    mlflow.sklearn.log_model(
        fitted[CANDIDATE].pipeline_,
        name="candidate_model",
        serialization_format="cloudpickle",
    )

    logger.info(
        "GATE: promote=%s  candidate=%.3f  baseline=%.3f  diff=%.3f [%.3f, %.3f]",
        decision["promote"], decision["candidate_precision_at_k"],
        decision["baseline_precision_at_k"], decision["difference"],
        decision["difference_ci_low"], decision["difference_ci_high"],
    )

    return fitted[CANDIDATE].pipeline_, results
