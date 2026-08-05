# Architecture: Student Dropout Risk Ranking System

**Status:** Phase 2 complete — build blueprint
**Companion document:** `problem_statement.md` (every decision below traces to it)
**Date:** 2026-07-29

---

## 1. System Overview

Three pipelines are built. One is deliberately cut.

| Pipeline | Status | Cadence | Purpose |
|---|---|---|---|
| Training | Build | On demand / annual | Validate → split → fit → evaluate → register |
| Batch inference | Build | Once per term | Score cohort, emit ranked top-10% list |
| Drift + data quality | Build | Once per term | Cohort vs training reference; input and prediction distributions |
| Monitoring (performance) | **Not possible** | — | Labels arrive ~3 years late |
| Automated retraining | **Cut** | — | One cohort per year; a human should approve support allocation |

### The constraint that shapes everything

Ground-truth dropout labels are unavailable for roughly three years after a prediction is
made. This is not a limitation to work around — it is a property of the domain, and it
determines three architectural facts:

1. Production performance cannot be measured. No accuracy dashboard is possible.
2. Retraining cannot be performance-triggered. Only schedule or drift can trigger it.
3. Drift detection is the only early-warning signal available. It carries more weight
   here than in domains with fast feedback.

**Maturity target: Level 1** (reproducible pipelines, versioned models, validated data),
plus two Level 2 practices that are cheap at this scale: unit tests on `core/`, and an
automated promotion gate. Level 3 is unreachable by construction — see above.

---

## 2. Repository Structure

```
dropout-risk/
├── pyproject.toml              # deps, pinned
├── uv.lock                     # reproducible environment
├── README.md
├── config/
│   └── config.yaml             # all hyperparameters, flags, paths, seeds
├── data/
│   ├── raw/dropout.csv         # committed; ~400 KB
│   └── raw/dropout.csv.sha256  # asserted at pipeline start
├── src/dropout_risk/
│   ├── core/                   # PURE PYTHON — no ZenML imports
│   │   ├── schema.py           # Pandera schema + validation
│   │   ├── features.py         # engineered feature functions
│   │   ├── preprocessing.py    # ColumnTransformer builders
│   │   ├── metrics.py          # precision_at_k, lift, bootstrap CI
│   │   ├── baselines.py        # majority-class, pass-rate rule
│   │   └── gate.py             # promotion decision function
│   ├── steps/                  # thin ZenML @step wrappers over core/
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   ├── split.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── register.py
│   │   ├── infer.py
│   │   └── drift.py
│   └── pipelines/
│       ├── training.py
│       ├── inference.py
│       └── drift.py
├── tests/                      # pytest over core/ only
└── notebooks/eda.ipynb
```

**The `core/` vs `steps/` split is the load-bearing decision.** Everything in `core/` is
plain functions with no orchestration imports, so it is unit-testable in milliseconds
without a ZenML stack. `steps/` contains thin `@step` wrappers that call into `core/`.
Business logic never lives in a step. This is what makes the tests meaningful and the
logic portable if ZenML is ever swapped out.

---

## 3. Data Plan

**Ingestion.** `ucimlrepo.fetch_ucirepo(id=697)` once, written to `data/raw/`. Every
pipeline run asserts the SHA-256 before proceeding. If the upstream file changes or
becomes unavailable, the run fails loudly instead of silently producing different numbers.

**Versioning: no DVC.** A single static ~400 KB CSV is versioned adequately by git. DVC
would add a remote, a cache, and a lock file to solve a problem that does not exist here.
Derived artifacts — which do change every run — are versioned by ZenML's artifact store.
Revisit if the data outgrows git or acquires an update cadence.

**Validation gates (Pandera).** Any failure halts the run.

| Check | Rule |
|---|---|
| Column presence | All 37 expected columns exist |
| Types | Match declared dtypes |
| Nulls | **Zero nulls permitted** — the source has none, so any null means upstream breakage |
| Duplicates | No duplicate rows |
| Admission grade | 0–200 (Portuguese scale) |
| Previous qualification grade | 0–200 |
| Age at enrollment | > 0 |
| Categorical codes | Within known category sets |
| Row count | Exact for the static file; ±30% for a new cohort |
| **Base rate** | dropout share 32.1% ± 5pp — catches label corruption |

The base-rate check is the one most projects omit and the one most likely to catch a
silent target-mapping bug.

**Splits.** No timestamp column exists, so a genuine temporal split is impossible. This
is a stated limitation, not something to fake using the macro indicators.

- **Test set:** 20% (~885 rows), stratified on target, held out and untouched until final evaluation
- **Model selection:** 5-fold stratified CV on the remaining 80% (~3,539 rows)
- **Reporting:** test-set metrics with bootstrap 95% CIs

**The small-k problem.** The test set's top decile is ~88 students. Precision@10% measured
on 88 observations carries a 95% CI of roughly ±8 percentage points. Single-number
reporting is therefore not evidence. Every headline metric is reported with its interval,
and every model-vs-baseline claim is made on the CI of the *difference*.

---

## 4. Feature Plan

**36 features into the model:** 30 retained raw + 6 engineered.

```
36 raw features
 −6  second-semester curricular columns (leakage: postdate the decision point)
────
 30  retained  (21 enrollment/demographic, 6 semester-1, 3 macro)
 +6  engineered
────
 36  model input
```

### Engineered features

| Feature | Definition | Rationale |
|---|---|---|
| `sem1_pass_rate` | approved ÷ enrolled | Strongest signal; also baseline 2 |
| `sem1_eval_rate` | evaluations ÷ enrolled | Assessment participation |
| `sem1_unevaluated_ratio` | without-evaluations ÷ enrolled | Disengagement, distinct from failure |
| `zero_enrolled_flag` | enrolled == 0 | Error handling *and* a strong signal |
| `grade_delta` | (sem1_grade ÷ 20) − (admission_grade ÷ 200) | Underperformance vs entry expectation |
| `parents_max_qualification` | max(mother, father) | Cardinality reduction |

**`grade_delta` scale warning:** semester-1 grade is on 0–20, admission grade on 0–200.
Both must be normalised to [0,1] before subtraction. Raw subtraction is a bug.

**Why engineer at all for a tree model?** Trees split on `feature ≤ threshold` and cannot
construct ratios. Approximating `approved ÷ enrolled` requires a separate branch per value
of `enrolled`, each needing sufficient samples. At 3,539 training rows that density is not
available. Engineered features matter here *because the dataset is small* — at 100k rows
they would add little. Additionally, the interpretability constraint requires them: SHAP
on `sem1_pass_rate` is actionable, SHAP on `approved` and `enrolled` separately is not.

**Controlled by a config flag.** `features.use_engineered: true|false`. Two runs, both
logged, so the contribution is measured rather than assumed.

### Categorical handling

High cardinality is the central risk: father's occupation ~46 levels, father's
qualification ~34, mother's ~29, nationality ~21. Naive one-hot expansion produces ~250
columns against 3,539 training rows.

- **Gradient boosting path:** `HistGradientBoostingClassifier` with `categorical_features`
  set. Native category splits, no expansion. **36 columns.**
- **Logistic regression path:** one-hot with **top-10 + `OTHER`** capping.
  **~125 columns.**

**No target encoding.** It requires out-of-fold computation to avoid leaking the label,
and getting that subtly wrong is a common silent failure. Top-K bucketing is safer at this
sample size.

**Explicitly excluded:** PCA (destroys interpretability), SMOTE (imbalance is mild at
32/68), automated feature selection before EDA.

**No feature store.** Feature stores solve cross-team reuse and online/offline consistency.
This project has one model, batch-only, one developer.

### Train/serve parity

All transformations — imputation, encoding, scaling, engineered ratios — live inside a
single `sklearn.Pipeline`, fit on train only, serialized as one artifact. The inference
pipeline loads that same fitted object. Training and serving cannot diverge because they
are the same code path. This is structural, not procedural.

---

## 5. Training Plan

### Model ladder

| # | Model | Role |
|---|---|---|
| 0 | `DummyClassifier(strategy="most_frequent")` | Floor. Precision@10% = 0.321 |
| 1 | **Pass-rate rule** — rank by `1 − sem1_pass_rate` | **The real bar.** No fitting |
| 2 | `LogisticRegression` (one-hot + scaled) | How much signal is linear? |
| 3 | `HistGradientBoostingClassifier` | Promotion candidate |

**Stopping at four is deliberate.** XGBoost and LightGBM are not meaningfully better than
HistGB at 3,539 rows and each adds a dependency and a separate categorical convention.
Add one only if HistGB visibly underperforms.

### Hyperparameter tuning

Randomized search, ~50 trials, over `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`,
`l2_regularization`, with early stopping on `max_iter`. Runs in under a minute.

**Reporting caveat:** the metric's CI is ~±8pp; tuning at this sample size typically moves
Precision@10% by 1–3pp. Tuning gains will sit inside the noise. Do not claim tuning
improved the model unless the bootstrap CI of the difference excludes zero.

### Class weights

Tested as a flag (`None` vs `balanced`), with a null result expected. The primary metric
is a *ranking* metric, and rankings are invariant to monotonic score rescaling. Class
weighting does alter the fitted model, so the effect is not exactly zero — but it is far
smaller than it would be for a threshold-based metric like F1. Report the null result.

### Reproducibility

Fixed `random_state` throughout. Every run logs: git SHA, data SHA-256, full config,
library versions, all metrics, the fitted pipeline, PR curve, SHAP summary, slice table.

---

## 6. Evaluation Plan

### Metric definitions

Let `n` = test rows, `k = ceil(0.10 × n)`, scores = predicted P(dropout).

```
Precision@10%  = (# true dropouts among top-k by score) / k
Lift@10%       = Precision@10% / 0.321
Recall@10%     = (# true dropouts among top-k) / (total dropouts)
                 ceiling ≈ 0.31 — REPORT THE CEILING ALONGSIDE
PR-AUC         = average precision, threshold-free
Brier          = mean((p − y)²), calibration
```

**Rejected:** accuracy (67.9% by predicting no-dropout), ROC-AUC (averages over a ranking
region never inspected), F2 (upweights budget-capped recall).

### Bootstrap procedure

1. Resample test predictions and labels with replacement, same size
2. Compute the metric
3. Repeat 2,000 times
4. Report 2.5th and 97.5th percentiles

For model-vs-baseline, bootstrap the **difference**. If the interval contains zero, no
superiority claim may be made.

### Slice evaluation (mandatory)

Precision@10% and selection rate broken out by: gender, age band, international status,
displaced status, scholarship holder, debtor.

The intervention is a scarce real resource. A model whose top decile concentrates heavily
on one demographic is a finding that must be reported regardless of aggregate performance.

### Promotion gate

```
promote  IFF
    test_precision_at_10 > baseline_passrate_precision_at_10
AND bootstrap_CI_95(difference).lower > 0
AND no slice has precision_at_10 below (global − 0.20)
```

Implemented in `core/gate.py` as a function returning a boolean. It is permitted to block
the project's own model. A gate that cannot fail is not a gate.

---

## 7. Deployment Plan

**Pattern:** batch. Once per term.

**Steps:** load `champion`-aliased model from MLflow → load cohort → validate against
schema → predict → rank → take top 10% → attach top-3 SHAP contributors per student →
write CSV.

**Output contract:** `student_id, risk_score, rank, reason_1, reason_2, reason_3`.
Reasons are required — a list of IDs without explanations is not actionable by an advisor
and not defensible if challenged.

**No API.** No real-time consumer exists. A FastAPI endpoint would exist to resemble
production rather than to serve one. If a demo surface is wanted, a Streamlit page reading
the CSV is more honest.

**Rollback:** reassign the `champion` alias to a prior model version.

> **MLflow 3 API note:** model *stages* (`Staging`, `Production`,
> `transition_model_version_stage`) were removed in favour of **aliases** and tags. Any
> tutorial predating 2025 uses the removed API. Verify current calls against the docs
> before implementing.

---

## 8. Monitoring and Drift Plan

**One Evidently run per term**, comparing the incoming cohort against the training
reference profile.

| Signal | Threshold | Action |
|---|---|---|
| Feature distribution (PSI) | > 0.25 on any feature | Investigate before trusting output |
| Prediction score distribution | Material shift vs reference | Investigate |
| Flagged-count deviation | > ±30% from expected 10% | **Halt** — upstream breakage likely |
| Null rate / schema | Any violation | Halt |

The flagged-count check is the highest-value alarm available: it catches real breakage
years before ground truth would.

**Performance monitoring is not implemented, by necessity.** Documented explicitly so its
absence reads as a justified decision rather than an oversight.

**Drift response is manual.** A drift alert opens a human decision about retraining. Drift
detection without a response plan is noise; automated response without labels is worse.

---

## 9. Versioning Plan

| Asset | Mechanism |
|---|---|
| Code | git |
| Raw data | git + SHA-256 assertion at run start |
| Derived artifacts | ZenML local artifact store |
| Models | MLflow registry, alias-based |
| Config | `config/config.yaml` in git, logged per run |
| Environment | `uv` + `uv.lock` |
| Experiments | MLflow runs (git SHA + data hash + config on each) |

---

## 10. Stack Specification

| Component | Choice | Version |
|---|---|---|
| Orchestrator | ZenML `local` | 0.96.2 |
| Artifact store | ZenML `local` | — |
| Experiment tracker | MLflow | 3.14.0 |
| Model registry | MLflow, aliases | 3.14.0 |
| Data validation | Pandera | latest |
| Drift | Evidently | 0.7.21 |
| Modelling | scikit-learn | 1.9.0 |
| Explainability | SHAP | latest |
| Env / deps | uv | latest |

**Version risk:** MLflow 3.x and Evidently 0.7.x are breaking rewrites of the APIs most
online material targets. Code for both is written against fetched current documentation,
not recalled patterns.

---

## 11. Implementation Sequence (Phase 3)

Each step ends with a working system. Nothing is written before the previous step runs.

| Step | Deliverable | Verified by |
|---|---|---|
| 1 | Repo, `uv` env, ZenML init, MLflow stack registered | `zenml stack describe` succeeds |
| 2 | Ingestion + checksum assertion | Raw CSV loads, hash matches |
| 3 | EDA notebook | Cardinalities, nulls, base rate, leakage checks confirmed |
| 4 | Pandera schema + validation step | Passes clean data, fails corrupted data |
| 5 | `core/features.py` + `core/preprocessing.py` | Unit tests pass |
| 6 | `core/metrics.py` — precision@k, lift, bootstrap | Unit tests on known inputs |
| 7 | Baselines + training pipeline + MLflow logging | Runs end to end, baseline numbers appear |
| 8 | Evaluation: CV, slices, SHAP, gate | Gate blocks a deliberately bad model |
| 9 | Registry + batch inference pipeline | Ranked CSV with reasons produced |
| 10 | Drift pipeline, tests, README | Full suite green |

**Step 3 has a hard checkpoint:** confirm exact categorical cardinalities and re-count the
one-hot width. The ~125-column figure in §4 is an estimate from documentation and may be
off by ten or more.

---

## 12. Explicit Non-Goals

Recorded so their absence reads as a decision:

- Admissions-time prediction and the enrollment-only ablation
- Three-class classification
- Real-time serving / REST API
- Feature store
- DVC
- SMOTE or aggressive resampling
- Automated retraining
- Production performance monitoring (impossible — label delay)
- Level 3 MLOps maturity
