# Student Dropout Risk Ranking

An end-to-end MLOps pipeline that ranks enrolled students by dropout risk at the
start of semester 2, so a capacity-constrained student-support programme can
reach the ~10% of students most likely to need it.

Built with **ZenML** (orchestration), **MLflow** (tracking + registry),
**Pandera** (validation), **SHAP** (interpretability), and **Evidently** (drift).

---

## The problem, framed honestly

A Portuguese university loses roughly a third of each intake before graduation,
mostly **mid-course** — the seat was funded and a semester of teaching delivered
before the student left. An intensive support programme (tutoring, counselling,
bursary review) has capacity for about **10% of the cohort**. Today that capacity
is allocated by advisor judgement, which reaches students only once they are
visibly struggling.

This is **not** a "how many will drop out" forecast. It is a **ranking** problem:
order students by risk so the fixed support budget goes to the right ~440.

- **Decision point:** start of semester 2 (so semester-1 results are known and fair to use)
- **Target:** binary — `dropout` vs not (Graduate + Enrolled folded together)
- **Primary metric:** **Precision@10%** — of the students flagged for support, what
  fraction genuinely drop out
- **Data:** UCI ID 697, 4,424 students, 36 features (Realinho et al., 2021)

Full framing and metric rationale: [`problem_statement.md`](problem_statement.md).
Architecture and build blueprint: [`architecture.md`](architecture.md).

---

## Headline result

| | Precision@10% (5-fold CV) | Lift |
|---|---|---|
| Random (base rate) | 0.321 | 1.0x |
| Pass-rate baseline (one-line rule) | 0.762 +/- 0.023 | 2.4x |
| **HistGradientBoosting model** | **0.966 +/- 0.017** | **3.0x** |

The model beats a strong, honest baseline by ~0.20 precision — a margin that
**survives cross-validation, survives dropping the most temporally-ambiguous
feature, and is consistent across all five folds.**

### Why the result is trustworthy (the investigation)

A single train/test split gave the model a suspiciously perfect **1.000**
precision. Rather than report it, several checks were run:

1. **Leakage — second-semester columns.** Dropped before training; they postdate
   the decision point. Score barely moved -> not the cause.
2. **Per-feature leakage scan.** No single feature exceeded 0.858 precision alone
   (nothing near a label copy). The top feature, `Tuition fees up to date`, is a
   plausible early financial-stress signal, not an outcome echo.
3. **Overfitting on one split.** 5-fold cross-validation collapsed the 1.000 to a
   stable 0.966 +/- 0.017 — the honest number reported above. The perfect
   single-split score was optimism from one lucky test set; CV is trustworthy.
4. **Feature ablation.** Dropping `Tuition fees up to date` moved histgb only from
   0.966 to 0.962 — within noise. The result does not rest on the ambiguous feature.

**SHAP** confirms the drivers are legitimate: `sem1_pass_rate` dominates, followed
by `Debtor`, `Tuition fees up to date`, and demographics — a sensible spread, not a
single leaking column.

### Fairness

Precision@10% is equal (1.000) across every demographic group large enough to
measure — no accuracy disparity. **Selection rates differ**: the model flags one
gender group and debtors ~2-3x more often — but this tracks their genuinely higher
underlying dropout rates (e.g. debtors drop out at 58% vs 29%). The model surfaces a
real disparity rather than inventing one; whether concentrating support on
higher-risk groups is equitable is a policy question for the institution. See
`slice_table.csv` (logged per run).

---

## Architecture

Three pipelines. Two core ones plus monitoring; two capabilities deliberately omitted.

```
Training  -->  MLflow registry (champion)  -->  Batch inference  -->  ranked CSV
    |                                                                  with reasons
    +-->  Drift check (per-term cohort comparison)
```

| Pipeline | Purpose |
|---|---|
| `training` | validate -> split -> fit ladder -> evaluate (CV) -> gate -> register champion |
| `inference` | load champion -> rank cohort -> attach SHAP reasons -> write intervention list |
| `drift` | compare a new cohort against the training reference; flag distribution shift |

**Deliberately cut:** automated retraining (one cohort/year — a human approves) and
performance monitoring (dropout labels arrive ~3 years late, so live accuracy is
unmeasurable). These omissions are design decisions, documented in `architecture.md`.

### Design principles

- **`core/` vs `steps/`** — all logic lives in pure, ZenML-free functions in `core/`,
  unit-tested in milliseconds. `steps/` are thin orchestration wrappers. If ZenML
  vanished, the data science would port unchanged.
- **Train/serve parity** — every transformation lives in one fitted sklearn pipeline,
  serialized as one artifact. Training and serving cannot diverge.
- **Honest evaluation** — precision@10% with bootstrap CIs; model-vs-baseline claims
  require the CI of the *difference* to exclude zero. The promotion gate can (and does)
  block a model that fails this.

---

## The model ladder

Four rungs, each a tracked MLflow run:

0. **Majority baseline** — constant; precision@10% = base rate (0.321)
1. **Pass-rate rule** — rank by semester-1 units passed. No fitting. The real bar (0.762)
2. **Logistic regression** — one-hot + scaled; how much signal is linear
3. **HistGradientBoosting** — native categorical handling; the promoted candidate (0.966)

Stopped at four deliberately: XGBoost/LightGBM add dependencies without meaningful
gains at 3,500 training rows.

---

## Running it

```bash
# setup (once)
make setup                                 # uv venv + pinned deps
make stack                                 # register ZenML stack: MLflow tracker + registry
uv run python -m dropout_risk.run_ingest   # fetch UCI 697 once, write checksummed CSV

# the three pipelines
make train                                 # fit, evaluate, register champion
make infer                                 # produce outputs/intervention_list.csv
make drift                                 # cohort drift check

# quality
make test                                  # 62 unit tests
make ui                                    # MLflow UI at localhost:5000
```

### The deliverable

`make infer` produces `outputs/intervention_list.csv` — the ~440 highest-risk
students, ranked, each with a risk score and their top-3 SHAP reasons:

```
student_id, risk_score, rank, reason_1, reason_1_value, reason_2, ...
1613,       0.987,      1,    sem1_pass_rate, 1.60, Tuition fees up to date, 1.39, ...
```

This is what a support office acts on: who to contact, in priority order, and why.

---

## Project structure

```
src/dropout_risk/
├── core/         # pure logic, no ZenML — 62 tests cover this
│   ├── ingest.py, checksum.py     # fetch + data integrity
│   ├── schema.py                  # Pandera strict validation
│   ├── features.py                # 6 engineered features
│   ├── preprocessing.py           # two-branch ColumnTransformer
│   ├── models.py                  # the 4-model ladder
│   ├── metrics.py                 # precision@k, lift, bootstrap CI
│   ├── evaluation.py              # metric suite + slices + CV
│   ├── gate.py                    # promotion decision
│   ├── explain.py                 # SHAP global + per-student
│   ├── inference.py               # ranked intervention list
│   └── drift.py                   # Evidently cohort drift
├── steps/        # thin ZenML @step wrappers
└── pipelines/    # training, inference, drift
```

---

## Stack

| Component | Choice | Version |
|---|---|---|
| Orchestration | ZenML | 0.96.2 |
| Tracking + registry | MLflow | 3.14.0 |
| Validation | Pandera | 0.32 |
| Drift | Evidently | 0.7.21 |
| Modelling | scikit-learn | 1.9.0 |
| Explainability | SHAP | 0.52 |
| Environment | uv | (lockfile) |

---

## Limitations

- **Single institution, single country, one time period.** Results do not transfer
  without revalidation.
- **Recall is capacity-capped** at ~31% (100 support slots vs ~320 dropouts per
  1,000 students). This is a budget ceiling, not a model failure.
- **`Tuition fees up to date`** is kept as a feature; its temporal validity depends
  on when the registrar recorded it. Results are reported both with and without it
  (the difference is negligible).
- **Fairness analysis is limited** by how separable this dataset is — near-perfect
  precision everywhere hides the disparities that usually surface in group precision.

---

*Built as an end-to-end MLOps exercise: reproducible pipelines, versioned data and
models, honest evaluation, and a real deliverable — with the investigation behind the
headline number documented rather than hidden.*
