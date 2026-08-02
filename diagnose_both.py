"""Report precision@10% with and without 'Tuition fees up to date'.

Quantifies how much of the model's performance rests on the one temporally
ambiguous feature. Uses 5-fold CV (the honest metric), for the pass-rate
baseline and histgb, each with fees kept vs dropped.

Run:  uv run python diagnose_both.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
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
FEES = "Tuition fees up to date"


def cv_precision(df, model_name, seed=42):
    y = df["dropout"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tri, tei in skf.split(df, y):
        trd, ted = df.iloc[tri], df.iloc[tei]
        m = build_model(model_name, CONFIG).fit(trd, trd["dropout"].to_numpy())
        scores.append(precision_at_k(ted["dropout"].to_numpy(),
                                     m.predict_scores(ted), 0.10))
    return np.mean(scores), np.std(scores)


df = pd.read_csv("data/raw/dropout.csv")
df = df[[c for c in df.columns if not c.startswith("Curricular units 2nd sem")]]

# NOTE: dropping fees requires removing it from the model's categorical list too.
# We simulate its removal by zeroing it out (constant column -> no signal),
# which is equivalent to exclusion for a tree/linear model and needs no code change.
df_nofees = df.copy()
df_nofees[FEES] = 0

print("5-fold CV precision@10%  (base rate 0.321)\n")
print(f"{'model':20s} {'WITH fees':>18s} {'WITHOUT fees':>18s}")
for name in ["passrate_baseline", "logistic", "histgb"]:
    mw, sw = cv_precision(df, name)
    mn, sn = cv_precision(df_nofees, name)
    print(f"{name:20s} {mw:.3f} +/- {sw:.3f}   {mn:.3f} +/- {sn:.3f}")

print("\n>>> The WITH-vs-WITHOUT gap = how much rested on the fees feature.")
print(">>> passrate_baseline is unaffected by fees (uses only pass rate).")
