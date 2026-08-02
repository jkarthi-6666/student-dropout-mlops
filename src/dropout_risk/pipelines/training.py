"""Training pipeline: chains load -> validate -> split -> train+evaluate.

Run:  uv run python -m dropout_risk.pipelines.training
"""

from __future__ import annotations

from pathlib import Path

import yaml
from zenml import pipeline

from dropout_risk.steps.ingest import load_data
from dropout_risk.steps.split import split_data
from dropout_risk.steps.train import train_and_evaluate
from dropout_risk.steps.validate import validate_data


@pipeline(enable_cache=False)
def training_pipeline(config: dict):
    df = load_data(
        csv_path=config["data"]["raw_path"],
        checksum_path=config["data"]["checksum_path"],
        drop_prefix=config["data"]["drop_column_prefix"],
    )
    validated = validate_data(
        df=df,
        expected_base_rate=config["target"]["expected_base_rate"],
        base_rate_tol=config["target"]["base_rate_tolerance"],
    )
    train_df, test_df = split_data(
        df=validated,
        test_size=config["split"]["test_size"],
        seed=config["project"]["seed"],
        target=config["target"]["name"],
    )
    train_and_evaluate(train_df=train_df, test_df=test_df, config=config)


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    training_pipeline(config=config)


if __name__ == "__main__":
    main()
