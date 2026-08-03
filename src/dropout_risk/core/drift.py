"""Cohort drift detection (Phase 2F).

Per-term check: is the incoming cohort distributed differently enough from the
training reference that the model's assumptions may no longer hold? Since true
dropout labels arrive years late, this distribution check is the only early
warning available -- it carries more weight here than in fast-feedback domains.

Verified against Evidently 0.7.21:
  - build Datasets via Dataset.from_pandas(df, data_definition=DataDefinition())
  - Report([DataDriftPreset()]).run(current_data=, reference_data=)
  - result.dict()["metrics"] -> DriftedColumnsCount gives count + share
"""

from __future__ import annotations

import pandas as pd


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare current cohort against the training reference.

    Returns a summary dict: number and share of drifted columns, the per-column
    drift verdicts, and an overall boolean. Never raises on individual column
    issues -- a drift check should report, not crash the term's run.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # align columns: only compare features present in both
    common = [c for c in reference.columns if c in current.columns]
    ref = reference[common].copy()
    cur = current[common].copy()

    defn = DataDefinition()
    ref_ds = Dataset.from_pandas(ref, data_definition=defn)
    cur_ds = Dataset.from_pandas(cur, data_definition=defn)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)
    d = result.dict()

    drifted_count = 0.0
    drifted_share = 0.0
    per_column = {}
    for metric in d.get("metrics", []):
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            val = metric.get("value", {})
            drifted_count = float(val.get("count", 0.0))
            drifted_share = float(val.get("share", 0.0))
        elif name.startswith("ValueDrift"):
            col = metric.get("config", {}).get("column", "?")
            threshold = metric.get("config", {}).get("threshold", 0.05)
            p_value = metric.get("value", 1.0)
            per_column[col] = {
                "p_value": float(p_value),
                "drifted": bool(p_value < threshold),
            }

    return {
        "drifted_count": drifted_count,
        "drifted_share": drifted_share,
        "n_columns_compared": len(common),
        "per_column": per_column,
    }


def drift_verdict(summary: dict, share_threshold: float = 0.5) -> dict:
    """Decide whether cohort drift is severe enough to warrant investigation.

    share_threshold: if more than this fraction of columns drifted, flag it.
    Returns the verdict plus the list of drifted column names.
    """
    drifted_cols = [c for c, v in summary["per_column"].items() if v["drifted"]]
    alert = summary["drifted_share"] > share_threshold
    return {
        "alert": alert,
        "drifted_share": summary["drifted_share"],
        "drifted_columns": drifted_cols,
        "message": (
            f"DRIFT ALERT: {summary['drifted_count']:.0f}/"
            f"{summary['n_columns_compared']} columns drifted "
            f"({summary['drifted_share']:.0%}). Investigate before trusting output."
            if alert
            else f"No significant drift: "
            f"{summary['drifted_count']:.0f}/{summary['n_columns_compared']} "
            f"columns drifted, within tolerance."
        ),
    }


def check_prediction_volume(
    n_flagged: int, n_total: int, expected_k_percent: float, tolerance: float
) -> dict:
    """Guardrail: has the flagged-student count moved far from the expected rate?

    The highest-value alarm available: if the model suddenly flags a third of
    the cohort instead of ~10%, something upstream broke -- and we would know
    that years before any dropout label confirms it.
    """
    actual_rate = n_flagged / n_total if n_total else 0.0
    deviation = abs(actual_rate - expected_k_percent)
    alert = deviation > tolerance
    return {
        "alert": alert,
        "actual_rate": actual_rate,
        "expected_rate": expected_k_percent,
        "deviation": deviation,
        "message": (
            f"VOLUME ALERT: flagged {actual_rate:.1%} vs expected "
            f"{expected_k_percent:.0%} (deviation {deviation:.1%}). "
            "Likely upstream data issue."
            if alert
            else f"Flagged volume normal: {actual_rate:.1%}."
        ),
    }
