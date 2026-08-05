# Problem Statement: Early Identification of Students at Risk of Dropout

**Status:** Phase 1 complete — revision 2 — awaiting approval before architecture design
**Date:** 2026-07-29

---

## Business Context

A higher-education institution loses roughly a third of each intake before graduation.
Most of that loss happens **mid-course, not at the gate**: the seat was funded, a
semester of teaching was delivered, and the student leaves anyway. The institution has
already spent the money by the time the outcome is visible.

The institution operates an **intensive student-support programme** (tutoring,
counselling, bursary review) with capacity for approximately **10% of the cohort per
term**. Today that capacity is allocated by advisor judgement and self-referral, which
reaches students who are already visibly struggling — often too late to change the
outcome.

The goal is not to predict how many students will drop out. It is to **rank students by
dropout risk at the point where intervention is still possible and teaching investment
is still recoverable**, so the fixed support budget reaches the 10% most likely to need
it.

## ML Formulation

> Given a student's enrollment record and first-semester results, predict whether they
> will drop out before completing their degree, for the student-support team, at the
> start of semester 2, in order to allocate a capacity-constrained intervention to the
> students most likely to benefit.

- **Problem type:** Binary classification, used as a **ranking** problem
- **Decision point:** Start of semester 2 — after semester-1 results are recorded
- **Target variable:** `dropout` = 1 if outcome is Dropout, 0 if Graduate or Enrolled
  - *Rationale:* "Enrolled" means still registered at the normal end date — behind
    schedule, not departed. Folding it into class 0 matches the actual decision (contact
    or don't). A 3-class model would spend capacity separating Enrolled from Graduate,
    which nobody acts on.
  - *Revisit if:* the intervention gains a third distinct response tier.

### Feature scope

**Included:** demographics, prior qualification, admission grade, application mode and
order, course, parental education and occupation, scholarship holder, debtor, tuition
fees up to date, macro indicators, and **all semester-1 curricular unit columns**
(credited, enrolled, evaluations, approved, grade, without evaluations).

**Excluded:** all semester-2 curricular unit columns. They postdate the decision point
and are pure leakage.

*Note:* choosing a post-semester-1 decision point removes the `Debtor` /
`Tuition fees up to date` leakage concern that would have applied to an
admissions-time model. At the start of semester 2 the institution legitimately observes
these fields.

### Metrics

**Primary: Precision@10%** — of the students in the top-ranked 10%, what fraction
actually drop out.

**Headline: Lift@10%** = Precision@10% ÷ 0.321 (base rate). Random selection yields
~0.321 precision. Lift answers "was the model worth building?" in a single number.

**Guardrails:**
- **PR-AUC** — threshold-free quality; keeps the operating point negotiable
- **Recall@10%** — *report with its ceiling stated:* capacity caps maximum achievable
  recall at ~31% (100 slots vs ~320 dropouts per 1,000 students). A reader who does not
  see the ceiling will read 31% as failure.
- **Brier score** — calibration; the ranking at the top of the list is what matters
- **Slice metrics** — Precision@10% broken out by gender, age band, international status,
  displaced status, scholarship holder, debtor

**Explicitly rejected:**
- *Accuracy* — 67.9% is achievable by predicting "no dropout" for everyone
- *ROC-AUC* — averages over the whole ranking including the bottom, which is never
  inspected; will read ~0.90 and say nothing about the only 100 rows that matter
- *F2* — upweights recall, which is budget-capped and cannot be bought

### Fairness constraint (hard)

The intervention is a scarce, real resource. If Precision@10% differs materially across
demographic slices, or if the selected decile is heavily skewed toward one group, this
must be reported as a finding — not buried. Allocation systems that concentrate support
on one demographic are a failure regardless of aggregate metrics.

### Baselines (must be beaten)

1. **Majority class** — predict no dropout for everyone. Precision@10% = 0.321 under
   random tie-breaking. Lift = 1.0.
2. **Semester-1 pass rate** — rank students by `approved units / enrolled units`,
   ascending. **This is the real bar.**

Baseline 2 is deliberately strong. Moving the decision point to after semester 1 makes
the prediction task easier, which means the naive heuristic gets better too — an advisor
can spot a student who passed one unit out of six without any model at all.

**Honesty clause:** if the trained model beats baseline 2 by only a small margin, that
result is reported as the finding, not buried. "A single ratio captures most of the
available signal; the model adds N points for the cost of a pipeline" is a legitimate
and useful conclusion.

## Data Summary

- **Source:** UCI ML Repository ID 697 — Realinho, Vieira Martins, Machado & Baptista
  (2021), *Predict Students' Dropout and Academic Success*. DOI 10.24432/C5MC89
- **Rows:** 4,424 students, single Portuguese higher-education institution
- **Features:** 36 raw — mixed numeric and encoded-categorical; semester-2 columns
  dropped, leaving ~30 usable
- **Original target:** Dropout 1,421 (32.1%) / Enrolled 794 (17.9%) / Graduate 2,209 (49.9%)
- **Binary target:** dropout 1,421 (32.1%) / not-dropout 3,003 (67.9%)
- **Labels:** complete, no missing target values

### Known issues to verify during EDA

- **Class imbalance is mild (32/68).** This is not a fraud-detection scenario. SMOTE and
  aggressive resampling are not warranted; class weights will be tested against no
  reweighting on evidence, not assumption.
- **Macro indicators** (unemployment rate, inflation, GDP) are cohort-level and nearly
  constant within an enrollment year. Expect near-zero contribution; check whether they
  act as a proxy for enrollment year.
- **Categoricals are pre-encoded as integers** (marital status, application mode, course,
  parental qualification and occupation). These must not be treated as ordinal numerics —
  course code 33 is not "greater than" course code 9.
- **Semester-1 features will dominate.** Expect them to carry most of the signal. This is
  expected, not a bug — but it is why baseline 2 exists.

## Constraints

- **Latency:** Batch. Predictions run once per term against the enrolled cohort. No
  real-time serving requirement.
- **Interpretability:** Required. Support staff must be able to see why a student was
  flagged, and the allocation must be defensible if challenged. This constrains model
  choice and makes SHAP attribution part of the deliverable, not a nice-to-have.
- **Regulatory / ethical:** Student records are personal data. The dataset is
  de-identified. Any real deployment would need a documented basis for automated
  risk-scoring of students.
- **Generalization:** Single institution, single country, fixed time period. Performance
  here does not transfer to another institution without revalidation. State this as a
  limitation rather than implying otherwise.

## Framework

- **Orchestration:** ZenML 0.96.2
- **Experiment tracking / registry:** MLflow 3.14.0
- **Drift detection:** Evidently 0.7.21 (pending architecture Phase 2F)
- **Modelling:** scikit-learn 1.9.0, gradient boosting to be selected in Phase 2D

*Note:* MLflow 3.x and Evidently 0.7.x carry breaking API changes from the versions most
online tutorials target. Code will be written against fetched current documentation.

## Success Criteria

This project is done when:

1. A ZenML training pipeline runs end to end and registers a versioned model in MLflow
2. The model beats both baselines on Precision@10% on a held-out test set — or the
   shortfall against baseline 2 is reported explicitly
3. Slice-level Precision@10% is reported across all six sensitive attributes
4. A batch inference pipeline loads the registered model and produces a ranked
   intervention list of the top 10% of the cohort
5. The README documents the problem, the metric rationale, and the known limitations —
   including the recall ceiling, the margin over baseline 2, and the single-institution
   caveat

## Out of Scope

- Admissions-time prediction and the enrollment-only ablation (considered, deliberately cut)
- Three-class classification (Dropout / Enrolled / Graduate)
- Real-time serving
- Semester-2 features
