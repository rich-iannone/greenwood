# greenwood (development version)

* First development version: `Surv()` and `kaplan_meier()`.

* Every table from `kaplan_meier()` starts with a `strata` column (`"all"` for a single curve,
  `"sex=1"` and so on for groups), so the layout is the same for one curve or several.

* `kaplan_meier()` gains `robust` and `cluster` for robust (infinitesimal jackknife) standard
  errors.

* `kaplan_meier()` fits gain `predict()` at given times, `rmst()`, `rmrl()`, and `summary()`.

## Nonparametric estimators

* `nelson_aalen()` estimates the cumulative hazard, with `predict()` at given times.

* `event_table()` gives the risk sets and events at each observed time.

* `turnbull()` estimates survival from interval-censored data (any mix of exact, right-, left-,
  and interval-censored times), with `predict()`, `quantile()`, `median()`, `rmst()`, and
  `rmrl()`.

## Group comparisons

* `logrank_test()` (with Fleming-Harrington weights through `rho` and `gamma`),
  `pairwise_logrank_test()`, `trend_test()`, and `maxcombo_test()`.

* `rmst_test()` and `pairwise_rmst_test()` compare restricted mean survival time as a difference,
  ratio, or percentage difference. `rmst_diff()` gives the difference directly.

## Regression

* `cox_ph()` fits Cox models with Efron or Breslow ties, case weights, `strata()` terms, delayed
  entry, and robust or clustered variance. Fits work with `coef()`, `vcov()`, `logLik()`,
  `nobs()`, and `as.data.frame()`. `predict()` gives linear predictors, risks, survival, cumulative
  hazards, quantiles, and expected event counts. `residuals()` gives martingale, deviance, score,
  Schoenfeld, scaled Schoenfeld, dfbeta, dfbetas, and leverage residuals. `baseline_hazard()`
  gives the baseline cumulative hazard per stratum, and `cox_zph()` tests proportional hazards.

* `aft()` fits accelerated failure time models with Weibull, exponential, log-normal, and
  log-logistic errors, with predictions, residuals, and delta-method confidence limits.

* `parametric()` fits a parametric distribution to each group, and `compare_distributions()`
  compares the fits of several distributions.

## Competing risks and multi-state models

* `aalen_johansen()` estimates the cumulative incidence of each cause, and `grays_test()` compares
  it between groups.

* `fine_gray()` fits the Fine-Gray subdistribution hazard model, with `predict()` of cumulative
  incidence curves.

* `multi_state()` estimates state occupancy probabilities for multi-state data.

## Prediction metrics

* `concordance_index()` (Harrell) and `concordance_index_ipcw()` (Uno) measure how well risk
  scores rank survival times.

* `brier_score()`, `integrated_brier_score()`, `time_dependent_auc()`, `integrated_auc()`, and
  `calibration()` assess predicted survival probabilities.

* `brier_score_incidence()`, `integrated_brier_score_incidence()`,
  `concordance_index_incidence()`, `calibration_incidence()`, and `accuracy_in_time()` do the
  same for competing risks.
