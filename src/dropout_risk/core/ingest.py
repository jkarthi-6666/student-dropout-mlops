"""Data ingestion for UCI dataset 697 (Predict Students' Dropout and Academic Success).

Pure functions, no ZenML imports, so this is unit-testable without a stack.
Fetches once, writes a raw CSV snapshot, and records a SHA-256 sidecar so every
later pipeline run can assert the data has not changed underneath it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def fetch_raw() -> pd.DataFrame:
    """Fetch UCI 697 and return features + target as one DataFrame.

    The ucimlrepo package splits features and target; we recombine them into a
    single frame with the target in its original 3-class text form. Binary
    mapping happens later, explicitly, so the raw snapshot stays faithful to source.
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=697)
    features = dataset.data.features
    target = dataset.data.targets

    # target is a single-column frame; its column is conventionally "Target"
    df = features.copy()
    target_col = target.columns[0]
    df[target_col] = target[target_col].values
    return df


def add_binary_target(
    df: pd.DataFrame,
    source_column: str = "Target",
    positive_class: str = "Dropout",
    new_column: str = "dropout",
) -> pd.DataFrame:
    """Add a binary `dropout` column: 1 for Dropout, 0 for Enrolled/Graduate.

    Leaves the original multiclass column in place so nothing is lost. The
    mapping is deliberately explicit rather than a label encoder, so the
    positive class is unambiguous and auditable.
    """
    if source_column not in df.columns:
        raise KeyError(
            f"Expected target column {source_column!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    classes = set(df[source_column].unique())
    if positive_class not in classes:
        raise ValueError(
            f"Positive class {positive_class!r} not present in target. "
            f"Found classes: {sorted(classes)}"
        )

    out = df.copy()
    out[new_column] = (out[source_column] == positive_class).astype(int)
    return out


def compute_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw_snapshot(
    df: pd.DataFrame,
    csv_path: str | Path,
    checksum_path: str | Path,
) -> str:
    """Write the DataFrame to CSV and its SHA-256 to a sidecar file.

    Returns the checksum. The CSV is written first, then hashed from disk, so
    the recorded hash matches exactly what a later run will read back.
    """
    csv_path = Path(csv_path)
    checksum_path = Path(checksum_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    checksum = compute_sha256(csv_path)
    checksum_path.write_text(checksum + "\n")
    return checksum


def ingest(
    csv_path: str | Path,
    checksum_path: str | Path,
    source_column: str = "Target",
    positive_class: str = "Dropout",
) -> tuple[pd.DataFrame, str]:
    """Full ingest: fetch, add binary target, snapshot to disk with checksum.

    Returns the DataFrame and its checksum. This is the function the ZenML
    ingest step wraps.
    """
    df = fetch_raw()
    df = add_binary_target(df, source_column=source_column, positive_class=positive_class)
    checksum = write_raw_snapshot(df, csv_path, checksum_path)
    return df, checksum
