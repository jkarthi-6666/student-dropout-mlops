"""ZenML ingest step: load the checksummed raw snapshot from disk."""

from __future__ import annotations

import pandas as pd
from zenml import step
from zenml.logger import get_logger

from dropout_risk.core.checksum import assert_checksum

logger = get_logger(__name__)


@step
def load_data(csv_path: str, checksum_path: str, drop_prefix: str = "") -> pd.DataFrame:
    """Verify the raw file against its checksum, load it, drop leakage columns.

    Assumes run_ingest has already fetched the data once. In the pipeline this
    guarantees we train on exactly the committed, verified snapshot.

    drop_prefix removes columns that postdate the decision point (the 2nd-semester
    curricular columns). These are leakage: a mid-year dropout has near-zero
    2nd-sem activity, so keeping them lets the model "predict" an outcome that has
    already happened. Dropping them is what makes the task honest.
    """
    assert_checksum(csv_path, checksum_path)
    df = pd.read_csv(csv_path)
    if drop_prefix:
        leak_cols = [c for c in df.columns if c.startswith(drop_prefix)]
        df = df.drop(columns=leak_cols)
        logger.info("Dropped %d leakage columns matching %r: %s",
                    len(leak_cols), drop_prefix, leak_cols)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], csv_path)
    return df
