"""Diagnose the perfect-score cause on the REAL data.

Three checks:
  1. Single-split vs 5-fold CV histgb precision@10%. If CV also ~1.0, it's real
     leakage, not overfitting. If CV drops a lot, it's split luck.
  2. Train vs test precision. If train=1.0 and test=1.0, suspicious; if train
     much higher than test, ordinary overfit.
  3. Perfect-separation check: how many test dropouts are perfectly ranked.

Run:  uv run python diagnose_perfect.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from dropout_risk.core.models import build_model
from dropout_risk.core.metrics import precision_at_k

CONFIG = {
    "features": {"use_engineered": True},
    "model": {"class_weight": None,
        "logistic_regression": {"C": 1.0, "max_iter": 2000},
        "hist_gradient_boosting": {"learning_rate": 0.1, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0,
            "max_iter": 300, "early_stopping": True}},
}

df = pd.read_csv("data/raw/dropout.csv")
df = df[[c for c in df.columns if not c.startswith("Curricular units 2nd sem")]]
y = df["dropout"].to_numpy()

# 1. single split
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
m = build_model("histgb", CONFIG).fit(tr, tr["dropout"].to_numpy())
p_train = precision_at_k(tr["dropout"].to_numpy(), m.predict_scores(tr), 0.10)
p_test = precision_at_k(te["dropout"].to_numpy(), m.predict_scores(te), 0.10)
print(f"single-split  train precision@10%: {p_train:.3f}")
print(f"single-split  test  precision@10%: {p_test:.3f}")

# 2. 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = []
for tri, tei in skf.split(df, y):
    trd, ted = df.iloc[tri], df.iloc[tei]
    mm = build_model("histgb", CONFIG).fit(trd, trd["dropout"].to_numpy())
    cv.append(precision_at_k(ted["dropout"].to_numpy(), mm.predict_scores(ted), 0.10))
print(f"\n5-fold CV     test  precision@10%: {np.mean(cv):.3f} +/- {np.std(cv):.3f}")
print(f"  per fold: {[round(s,3) for s in cv]}")

# 3. how separable is the top decile really
scores = m.predict_scores(te)
yte = te["dropout"].to_numpy()
k = int(np.ceil(0.10*len(te)))
order = np.argsort(-scores)
topk_labels = yte[order[:k]]
print(f"\ntop-{k} test students: {int(topk_labels.sum())}/{k} are real dropouts")
print(f"their scores range: {scores[order[k-1]]:.4f} to {scores[order[0]]:.4f}")
print(f"score at rank {k} vs rank {k+1}: {scores[order[k-1]]:.4f} vs {scores[order[k]]:.4f}")

print("\n>>> If CV ~1.0 too: real separability (maybe fees is that strong).")
print(">>> If CV much lower: single-split was lucky.")
