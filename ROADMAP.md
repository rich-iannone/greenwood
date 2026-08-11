---
title: "Roadmap"
---

# Greenwood Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap.

## Planned — Medium Term

Regression model extensions and flexible semi-parametric approaches.

### Advanced Proportional-Hazards Tests

Extended testing of the Cox model assumptions.

- Smooth non-linear hazard ratio curves (`smoothHR`-style): spline-based visualization of covariate
effects on the log-hazard scale, complementing `cox_zph()` for diagnosing non-linearity in
continuous predictors

### Flexible Parametric Models

Semi-parametric and parametric spline-based hazard regression.

- Piecewise exponential models with optimal knot selection
- Generalized gamma regression (encompasses Weibull, log-normal, exponential)

---

## Planned — Long Term

Advanced estimators for complex survival problems and specialized applications.

### Additive Hazards & Cure Models

Alternative hazard structures and zero-inflated survival models.

- Aalen additive model for additive (vs. proportional) hazard regression with constrained
optimization to ensure non-negative hazards and proper survival functions
- Mixture cure models for populations with long-term survivors
- Non-parametric maximum likelihood estimation (NPMLE) for cure fractions
- Goodness-of-fit tests and model comparison for cure models

### Advanced Competing Risks & Multi-State

Extended methods for cause-specific and multi-state analyses.

- Gray's test for differences in cumulative incidence across groups
- Variance estimation for multi-state transition probabilities
- Pseudo-observation approach for CIF and multi-state occupancy regression
- Custom estimands via pseudo-observations framework

### Frailty Models

Random-effects Cox models for correlated survival data.

- Conditional and marginal predictions for known and new clusters

### Advanced Performance Metrics

Discrimination and calibration assessment beyond point-in-time.

- Integrated discrimination improvement (IDI) and net reclassification improvement (NRI)
- Time-dependent Brier score refinements and sensitivity analyses
- `concordance_index_ipcw`: inverse-probability-of-censoring-weighted concordance index for
time-dependent discrimination (more robust than Harrell's C under heavy censoring)
- `ipc_weights` / `CensoringDistributionEstimator`: IPC weights as first-class utilities,
enabling manual reweighting for metrics and models beyond concordance

### Machine Learning Survival Estimators

Tree-based and ensemble survival models that output individual survival functions rather than
scalar risk scores.

- `SurvivalTree`: single survival tree using the log-rank split criterion, which is the foundation
for ensemble methods and useful for interpretable non-parametric survival modeling
- `RandomSurvivalForest`: ensemble of survival trees with bootstrap aggregation (the most
widely used ML survival model, returns per-subject survival and cumulative hazard functions)
- `ExtraSurvivalTrees`: extremely randomized variant of RSF with additional split randomization
for faster training and lower variance on large datasets
- `GradientBoostingSurvivalAnalysis`: component-wise and tree-based boosting for survival (strong
predictive performance on structured tabular data)
- `IPCRidge`: IPC-weighted ridge regression (a linear survival model that corrects for censoring
bias via inverse-probability reweighting rather than the partial likelihood)

### Platform & Interop

Performance and ecosystem integration toward 1.0.

- Full backend matrix algebra with accelerated kernels (JAX/Numba) for ultra-large datasets
(100k+ rows)
- Finalized extension protocols and Narwhals dataframe backend completeness
- Full interoperability with Great Summaries (`tbl_survfit`, `tbl_regression`)
- Migration guides for users transitioning from R's **survival** package

---

## Feedback & Contributions

Have ideas for features not listed here? Open an issue with the `enhancement` label! Contributions
to any planned item are welcome so check existing issues first to avoid duplication.

_This roadmap is a living document. It is updated as features are added and new priorities emerge._
