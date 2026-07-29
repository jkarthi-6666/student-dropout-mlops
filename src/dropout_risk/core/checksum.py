"""Checksum guard.

Every pipeline calls `assert_checksum` at startup. If the raw data file has
changed since it was snapshotted, the run fails immediately with a clear
message, rather than silently producing different numbers weeks later.
"""

from __future__ import annotations

from pathlib import Path

from dropout_risk.core.ingest import compute_sha256


class ChecksumMismatchError(RuntimeError):
    """Raised when the data file's hash does not match the recorded sidecar."""


def assert_checksum(csv_path: str | Path, checksum_path: str | Path) -> str:
    """Verify csv_path matches the hash recorded in checksum_path.

    Returns the verified checksum on success. Raises with an actionable message
    on any mismatch or missing file.
    """
    csv_path = Path(csv_path)
    checksum_path = Path(checksum_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file missing: {csv_path}. Run the ingest step first."
        )
    if not checksum_path.exists():
        raise FileNotFoundError(
            f"Checksum sidecar missing: {checksum_path}. Run the ingest step first."
        )

    expected = checksum_path.read_text().strip()
    actual = compute_sha256(csv_path)

    if actual != expected:
        raise ChecksumMismatchError(
            f"Data file {csv_path} has changed.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "If this change is intended, re-run ingest to refresh the snapshot "
            "and commit the new checksum."
        )
    return actual
