"""Batch inference pipeline: load the registered model, rank the cohort, write CSV.

Run:  uv run python -m dropout_risk.pipelines.inference

Loads the champion model from MLflow, scores the current cohort, and writes the
ranked top-k% intervention list to the outputs directory. This is the artifact
the student-support team consumes.

Model loading: reads the most recent version of the registered model by name.
The training pipeline logs the candidate; a one-time registration (see
register_champion below) promotes it to the named model with the champion alias.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import yaml
from zenml import pipeline, step
from zenml.logger import get_logger

from dropout_risk.core.checksum import assert_checksum
from dropout_risk.core.inference import rank_cohort

logger = get_logger(__name__)


def _load_champion(model_name: str, alias: str):
    """Load the aliased champion model from the MLflow registry.

    Falls back to the latest version if the alias is not set, so a first run
    after training still works. Raises a clear error if nothing is registered.

    The tracking URI is set explicitly from ZenML's active MLflow tracker so this
    step reads the same registry the training step wrote to (a plain mlflow client
    would otherwise default to a local ./mlruns with no registered models).
    """
    try:
        from zenml.client import Client
        tracker = Client().active_stack.experiment_tracker
        if tracker is not None and hasattr(tracker, "get_tracking_uri"):
            mlflow.set_tracking_uri(tracker.get_tracking_uri())
            logger.info("MLflow tracking URI set to %s", mlflow.get_tracking_uri())
    except Exception as exc:
        logger.warning("Could not resolve ZenML tracker URI: %s", exc)

    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        logger.info("Loaded %s@%s (version %s)", model_name, alias, mv.version)
        return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    except Exception:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(
                f"No registered model named {model_name!r}. Run register_champion "
                "after training to register the candidate."
            )
        latest = max(versions, key=lambda v: int(v.version))
        logger.info("Alias %s not set; using latest version %s", alias, latest.version)
        return mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")


@step
def load_cohort(csv_path: str, checksum_path: str, drop_prefix: str) -> pd.DataFrame:
    """Load and validate the scoring cohort, dropping leakage + target columns."""
    assert_checksum(csv_path, checksum_path)
    df = pd.read_csv(csv_path)
    if drop_prefix:
        df = df.drop(columns=[c for c in df.columns if c.startswith(drop_prefix)])
    # drop the outcome columns: at inference time we do not have them
    df = df.drop(columns=[c for c in ("Target", "dropout") if c in df.columns])
    logger.info("Cohort loaded: %d students, %d features", len(df), df.shape[1])
    return df


@step(experiment_tracker="mlflow_tracker")
def score_and_rank(cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Load the champion, rank the cohort, attach reasons."""
    model = _load_champion(
        config["registry"]["model_name"], config["registry"]["champion_alias"]
    )
    ranked = rank_cohort(model, cohort, k_percent=config["metric"]["k_percent"])
    logger.info("Ranked top %d students for intervention", len(ranked))
    return ranked


@step
def write_intervention_list(ranked: pd.DataFrame, out_path: str) -> str:
    """Write the ranked list to the outputs directory."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    logger.info("Intervention list written to %s", out_path)
    return out_path


@pipeline(enable_cache=False)
def inference_pipeline(config: dict):
    cohort = load_cohort(
        csv_path=config["data"]["raw_path"],
        checksum_path=config["data"]["checksum_path"],
        drop_prefix=config["data"]["drop_column_prefix"],
    )
    ranked = score_and_rank(cohort=cohort, config=config)
    write_intervention_list(
        ranked=ranked, out_path="outputs/intervention_list.csv"
    )


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    inference_pipeline(config=config)


if __name__ == "__main__":
    main()
