"""Drift check pipeline: compare a cohort against the training reference.

Run:  uv run python -m dropout_risk.pipelines.drift

Per-term monitoring. In production the "current" cohort would be the new term's
enrollees; here, lacking a second cohort, we demonstrate the check by splitting
the dataset and treating half as reference, half as current -- which correctly
reports NO drift, proving the mechanism runs. The value is the wiring: point it
at a genuinely new cohort file and it flags distribution shift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from zenml import pipeline, step
from zenml.logger import get_logger

from dropout_risk.core.checksum import assert_checksum
from dropout_risk.core.drift import drift_verdict, run_drift_report

logger = get_logger(__name__)


@step
def load_reference_and_current(
    csv_path: str, checksum_path: str, drop_prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data; split into a reference half and a current half for the demo.

    Replace the 'current' half with a new cohort file in production.
    """
    assert_checksum(csv_path, checksum_path)
    df = pd.read_csv(csv_path)
    if drop_prefix:
        df = df.drop(columns=[c for c in df.columns if c.startswith(drop_prefix)])
    df = df.drop(columns=[c for c in ("Target", "dropout") if c in df.columns])
    mid = len(df) // 2
    reference = df.iloc[:mid].reset_index(drop=True)
    current = df.iloc[mid:].reset_index(drop=True)
    logger.info("Reference: %d rows | Current: %d rows", len(reference), len(current))
    return reference, current


@step
def check_drift(
    reference: pd.DataFrame, current: pd.DataFrame, share_threshold: float
) -> dict:
    """Run the drift report and produce a verdict."""
    summary = run_drift_report(reference, current)
    verdict = drift_verdict(summary, share_threshold=share_threshold)
    logger.info(verdict["message"])
    if verdict["drifted_columns"]:
        logger.info("Drifted columns: %s", verdict["drifted_columns"])
    return verdict


@pipeline(enable_cache=False)
def drift_pipeline(config: dict):
    reference, current = load_reference_and_current(
        csv_path=config["data"]["raw_path"],
        checksum_path=config["data"]["checksum_path"],
        drop_prefix=config["data"]["drop_column_prefix"],
    )
    check_drift(
        reference=reference,
        current=current,
        share_threshold=0.5,
    )


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    drift_pipeline(config=config)


if __name__ == "__main__":
    main()
