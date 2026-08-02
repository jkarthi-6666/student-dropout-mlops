"""Rank every feature by how strongly it alone predicts dropout.

A feature with near-perfect single-feature precision@10% is a leak: no honest
predictor should let one column perfectly identify the outcome.

Run:  uv run python diagnose_leakage.py
"""
import numpy as np, pandas as pd
from dropout_risk.core.metrics import precision_at_k

df = pd.read_csv("data/raw/dropout.csv")
# drop 2nd sem (already known leakage) + target cols
df = df[[c for c in df.columns if not c.startswith("Curricular units 2nd sem")]]
y = df["dropout"].to_numpy()
base = y.mean()

rows = []
for col in df.columns:
    if col in ("Target", "dropout"):
        continue
    s = df[col].astype(float).to_numpy()
    # try both directions (high=risk and low=risk), take the better
    p_hi = precision_at_k(y, s, 0.10)
    p_lo = precision_at_k(y, -s, 0.10)
    p = max(p_hi, p_lo)
    rows.append((col, p, p / base))

out = pd.DataFrame(rows, columns=["feature", "precision@10%", "lift"]).sort_values(
    "precision@10%", ascending=False)
pd.set_option("display.max_rows", 100)
print(f"base rate = {base:.3f}\n")
print(out.to_string(index=False))
print("\n>>> Any feature with precision@10% near 1.0 is a LEAK. "
      "0.80-0.90 is legitimate strong signal (like sem-1 performance).")
