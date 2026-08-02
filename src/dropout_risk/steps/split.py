"""ZenML split step: stratified train/test split."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def split_data(
    df: pd.DataFrame, test_size: float, seed: int, target: str = "dropout"
) -> tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
]:
    """Stratified split on the binary target."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    logger.info("Split: %d train / %d test", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
