"""Promotion gate (Phase 2F).

A candidate model is promoted only if it beats the pass-rate baseline AND the
bootstrap CI of the difference excludes zero AND no slice falls too far below
the global precision. Implemented as a function returning a decision dict, so it
is testable and can block the project's own model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dropout_risk.core.metrics import bootstrap_difference, precision_at_k


def promotion_decision(
    y_true: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    slice_table: pd.DataFrame | None = None,
    k_percent: float = 0.10,
    require_ci_excludes_zero: bool = True,
    slice_tolerance: float = 0.20,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Return a decision dict with `promote: bool` and the reasons behind it."""
    y_true = np.asarray(y_true)

    cand_p = precision_at_k(y_true, candidate_scores, k_percent)
    base_p = precision_at_k(y_true, baseline_scores, k_percent)

    diff = bootstrap_difference(
        y_true, candidate_scores, baseline_scores,
        metric_fn=precision_at_k, n_iter=n_bootstrap, seed=seed,
        k_percent=k_percent,
    )

    beats_baseline = cand_p > base_p
    ci_ok = diff["excludes_zero"] if require_ci_excludes_zero else True

    # Slice fairness gate: no group's in-group precision may fall more than
    # slice_tolerance below the global candidate precision.
    slice_ok = True
    failing_slices = []
    if slice_table is not None and not slice_table.empty:
        floor = cand_p - slice_tolerance
        for _, r in slice_table.iterrows():
            p = r.get("precision_in_group")
            if pd.notna(p) and not r.get("small_slice", False) and p < floor:
                slice_ok = False
                failing_slices.append(
                    {"slice": r["slice"], "level": r["level"], "precision": float(p)}
                )

    promote = bool(beats_baseline and ci_ok and slice_ok)
    return {
        "promote": promote,
        "candidate_precision_at_k": float(cand_p),
        "baseline_precision_at_k": float(base_p),
        "difference": diff["point"],
        "difference_ci_low": diff["ci_low"],
        "difference_ci_high": diff["ci_high"],
        "difference_excludes_zero": diff["excludes_zero"],
        "beats_baseline": bool(beats_baseline),
        "ci_gate_passed": bool(ci_ok),
        "slice_gate_passed": bool(slice_ok),
        "failing_slices": failing_slices,
    }
