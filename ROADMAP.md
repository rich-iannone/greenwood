---
title: "Roadmap"
---

# Greenwood Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap.

## Planned — Near Term

Descriptive and exploratory features building on the core estimators.

### Exploratory Plots

Visualization for grouped survival analyses.

- Cumulative incidence function curves (competing risks by group)
- Aligned at-risk tables for grouped curves

### Model Validation and Performance

Robust cross-validation and performance assessment for imbalanced survival data.

- Stratified k-fold cross-validation ensuring balanced event representation across folds
- Handling of highly imbalanced datasets (common in survival analysis) to prevent singular matrix errors
- Performance metrics (concordance, Brier score) computed reliably with stratified splits
- Documentation and examples for model selection with imbalanced event data
- Performance optimization for large datasets (memory efficiency, computation speed)
- Numerical stability benchmarks and recommendations for data scale

### Confidence Intervals & Inference

Systematic confidence interval and standard error support across all estimators.

- Confidence intervals for Cox model coefficients and hazard ratios (analytical)
- Kaplan-Meier survival function confidence intervals (Greenwood method and alternatives)
- Cumulative incidence function confidence intervals (competing risks)
- Standard errors and CIs for parametric model predictions
- Bootstrap and analytical methods for uncertainty quantification
- Predictive intervals for time-varying Cox model forecasts

### Univariate Parametric Models

Standalone parametric distributions for data exploration and model selection.

- Weibull, exponential, log-normal, and log-logistic distributional models
- Goodness-of-fit assessment and model comparison
- Maximum likelihood parameter estimation with standard errors
- Survival, hazard, and quantile predictions from fitted models

---

## Planned — Medium Term

Regression model extensions and flexible semi-parametric approaches.

### Time-Varying Covariates

Cox regression with covariates that evolve over follow-up time.

- Counting-process form integration for covariate changes
- Episode-splitting and data reshaping utilities
- Risk-set calculations with time-varying exposure
- Predictions at specified covariate trajectories

### Cox Residual Diagnostics

Outlier detection and case-level assessment for Cox models.

- Deviance and dfbeta residuals for influence assessment
- Scaled Schoenfeld residuals for proportional-hazards diagnosis
- Leverage and hat-matrix diagnostics
- Visualizations for outlier and influential point detection

### Advanced Proportional-Hazards Tests

Extended testing of the Cox model assumptions.

- `cox_zph` with Kaplan-Meier and rank-based time transforms
- Time-stratified tests for non-proportional hazards
- Robust sandwich variance for model misspecification

### Flexible Parametric Models

Semi-parametric and parametric spline-based hazard regression.

- Royston-Parmar restricted cubic spline models for hazard and survival
- Piecewise exponential models with optimal knot selection
- Generalized gamma regression (encompasses Weibull, log-normal, exponential)
- Parametric predictions: survival, hazard, and quantiles at arbitrary times

---

## Planned — Long Term

Advanced estimators for complex survival problems and specialized applications.

### Additive Hazards & Cure Models

Alternative hazard structures and zero-inflated survival models.

- Aalen additive model for additive (vs. proportional) hazard regression with constrained optimization to ensure non-negative hazards and proper survival functions
- Mixture cure models for populations with long-term survivors
- Non-parametric maximum likelihood estimation (NPMLE) for cure fractions
- Goodness-of-fit tests and model comparison for cure models

### Advanced Competing Risks & Multi-State

Extended methods for cause-specific and multi-state analyses.

- Gray's test for differences in cumulative incidence across groups
- Variance estimation for multi-state transition probabilities
- Pseudo-observation approach for CIF and multi-state occupancy regression
- Custom estimands via pseudo-observations framework

### Frailty and Penalized Regression

Random-effects and regularized Cox models.

- Shared frailty models (random intercept) for clustered or familial data
- Frailty variance estimation and inference
- Elastic-net Cox regression (ridge, lasso, elastic-net) for high-dimensional covariates
- Regularization parameter selection via cross-validation

### Advanced Performance Metrics

Discrimination and calibration assessment beyond point-in-time.

- Time-dependent AUC (area under cumulative/dynamic ROC curve)
- Integrated discrimination improvement (IDI) and net reclassification improvement (NRI)
- Calibration curves and calibration-in-the-large over follow-up time
- Time-dependent Brier score refinements and sensitivity analyses

### Platform & Interop

Performance and ecosystem integration toward 1.0.

- Full backend matrix algebra with accelerated kernels (JAX/Numba) for ultra-large datasets (100k+ rows)
- Finalized extension protocols and Narwhals dataframe backend completeness
- Full interoperability with Great Summaries (`tbl_survfit`, `tbl_regression`)
- Migration guides for users transitioning from R's `survival` package

---

## Feedback & Contributions

Have ideas for features not listed here? Open an issue with the `enhancement` label. Contributions to any planned item are welcome so check existing issues first to avoid duplication.

_This roadmap is a living document. It is updated as features ship and new priorities emerge._
