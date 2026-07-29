"""Run ingestion once and print verification stats.

Usage:  uv run python -m dropout_risk.run_ingest

This is a plain script, not a ZenML pipeline. It exists so Step 2 can be run
and verified before any orchestration is wired up. The numbers it prints are
what we check against the dataset's documented shape.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dropout_risk.core.checksum import assert_checksum
from dropout_risk.core.ingest import ingest


def main() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    data_cfg = config["data"]
    target_cfg = config["target"]

    csv_path = data_cfg["raw_path"]
    checksum_path = data_cfg["checksum_path"]

    print("Fetching UCI dataset 697 ...")
    df, checksum = ingest(
        csv_path=csv_path,
        checksum_path=checksum_path,
        source_column=target_cfg["source_column"],
        positive_class=target_cfg["positive_class"],
    )

    # Immediately verify the snapshot reads back cleanly.
    assert_checksum(csv_path, checksum_path)

    n_rows = len(df)
    n_cols = df.shape[1]
    base_rate = df["dropout"].mean()
    class_counts = df[target_cfg["source_column"]].value_counts().to_dict()

    print("\n--- ingest verification ---")
    print(f"rows:           {n_rows}")
    print(f"columns:        {n_cols}  (includes original target + binary 'dropout')")
    print(f"checksum:       {checksum[:16]}...")
    print(f"original target: {class_counts}")
    print(f"binary base rate (dropout=1): {base_rate:.4f}")

    expected_rate = target_cfg["expected_base_rate"]
    tol = target_cfg["base_rate_tolerance"]
    status = "OK" if abs(base_rate - expected_rate) <= tol else "OUT OF RANGE"
    print(f"expected ~{expected_rate} +/- {tol}  ->  {status}")
    print(f"\nsaved: {csv_path}")
    print(f"saved: {checksum_path}")


if __name__ == "__main__":
    main()
