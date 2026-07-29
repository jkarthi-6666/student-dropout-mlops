"""Unit tests for core.ingest and core.checksum.

These do not hit the network. `fetch_raw` is the only networked function and is
not tested here; everything else is tested on synthetic frames so the suite
runs in milliseconds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dropout_risk.core.checksum import ChecksumMismatchError, assert_checksum
from dropout_risk.core.ingest import (
    add_binary_target,
    compute_sha256,
    write_raw_snapshot,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [1, 2, 3, 4],
            "Target": ["Dropout", "Graduate", "Enrolled", "Dropout"],
        }
    )


def test_add_binary_target_maps_dropout_to_one():
    out = add_binary_target(_toy_df())
    assert list(out["dropout"]) == [1, 0, 0, 1]


def test_add_binary_target_preserves_original_column():
    out = add_binary_target(_toy_df())
    assert "Target" in out.columns
    assert list(out["Target"]) == ["Dropout", "Graduate", "Enrolled", "Dropout"]


def test_add_binary_target_missing_column_raises():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        add_binary_target(df, source_column="Target")


def test_add_binary_target_missing_positive_class_raises():
    df = pd.DataFrame({"Target": ["Graduate", "Enrolled"]})
    with pytest.raises(ValueError):
        add_binary_target(df, positive_class="Dropout")


def test_write_snapshot_and_checksum_roundtrip(tmp_path):
    df = add_binary_target(_toy_df())
    csv = tmp_path / "raw.csv"
    chk = tmp_path / "raw.csv.sha256"

    checksum = write_raw_snapshot(df, csv, chk)

    assert csv.exists()
    assert chk.exists()
    assert chk.read_text().strip() == checksum
    # verifying the freshly written file must pass
    assert assert_checksum(csv, chk) == checksum


def test_assert_checksum_detects_tampering(tmp_path):
    df = add_binary_target(_toy_df())
    csv = tmp_path / "raw.csv"
    chk = tmp_path / "raw.csv.sha256"
    write_raw_snapshot(df, csv, chk)

    # mutate the data after the checksum was recorded
    csv.write_text(csv.read_text() + "5,Dropout,1\n")

    with pytest.raises(ChecksumMismatchError):
        assert_checksum(csv, chk)


def test_compute_sha256_is_deterministic(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert compute_sha256(p) == compute_sha256(p)
