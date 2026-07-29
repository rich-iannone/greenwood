---
title: "Roadmap"
---

# Greenwood Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap.

## Planned — Near Term

Descriptive and exploratory features building on the core estimators.

### Model Validation and Performance

Robust cross-validation and performance assessment for imbalanced survival data.

- Performance optimization for large datasets (memory efficiency, computation speed)
- Multi-metric `cross_validate`: accept `metrics` as a list so concordance, Brier score, and
time-dependent AUC can all be evaluated in a single CV run and returned as a keyed dict

### Survival Predictions

Point-in-time and distributional predictions from fitted models.

- `predict_median`: median survival time (time at which the survival curve crosses 0.5) for
regression models (CoxPH, AFT, RoystonParmar) with confidence intervals
- `predict_quantile`: generalized quantile survival time at any probability level
- `predict_expectation`: expected survival time using RMST as the tail assumption, with an explicit
`tau` (truncation time) parameter to make the tail assumption transparent

### Confidence Intervals & Inference

Systematic confidence interval and standard error support across all estimators.

- Bootstrap and analytical methods for uncertainty quantification
- Predictive intervals for time-varying Cox model forecasts
- Robust (sandwich) variance for Kaplan-Meier: correct CI coverage for inverse-probability-weighted
curves, replacing Greenwood's formula with a sandwich estimator when non-integer weights are present
(analogous to `robust=True` in R's **survfit**)
- Clustered robust variance for Kaplan-Meier: extend the sandwich estimator to handle correlated
observations (e.g., recurring events, patients nested in clusters), matching R's `cluster()` term in
**survfit** formulas

---

## Planned — Medium Term

Regression model extensions and flexible semi-parametric approaches.

### Cox Residual Diagnostics

Outlier detection and case-level assessment for Cox models.

- Leverage and hat-matrix diagnostics
- Visualizations for outlier and influential point detection

### Advanced Proportional-Hazards Tests

Extended testing of the Cox model assumptions.

- `cox_zph` with Kaplan-Meier and rank-based time transforms
- Time-stratified tests for non-proportional hazards
- Smooth non-linear hazard ratio curves (`smoothHR`-style): spline-based visualization of covariate
effects on the log-hazard scale, complementing `cox_zph()` for diagnosing non-linearity in
continuous predictors

### Flexible Parametric Models

Semi-parametric and parametric spline-based hazard regression.

- Piecewise exponential models with optimal knot selection
- Generalized gamma regression (encompasses Weibull, log-normal, exponential)
- AIC and BIC for penalized Cox models (CoxNet): useful for model selection on small datasets and
regulatory submissions where cross-validation is impractical

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
