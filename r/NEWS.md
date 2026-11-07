# greenwood (development version)

* First development version: `Surv()` and `kaplan_meier()`.

* `event_time()` builds an event-time response, with `new_event_time()`, `extract_time()`,
  `extract_status()`, and `as_surv()`. `duration()` gives the elapsed time between two dates, and
  `first_event()` builds a competing-risks response from several endpoints. Each works on the left
  of a formula.

* `split_episodes()` splits follow-up into intervals of constant time-varying covariates.

* The datasets used in the examples come with the package: `colon`, `e1684`, `lung`, `mgus2`,
  `ovarian`, `pbc`, `pbcseq`, and `veteran`.

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

* `cox_ph()` fits gamma and Gaussian frailty models, with `frailty_test()`, and `predict()` follows
  a subject along a covariate trajectory. `influence_diagnostics()` gathers the influence measures
  of a fit, and `smooth_hr()` estimates a spline curve for a continuous covariate's hazard ratio.

* `royston_parmar()` fits flexible parametric models on the proportional hazards or odds scale,
  `piecewise_exponential()` fits piecewise constant hazards, `aalen_additive()` fits Aalen's
  additive hazards model with `cumulative_coefficients()`, `buckley_james()` and `ipc_ridge()` fit
  censored linear regressions, and `mixture_cure()` fits a mixture cure model.

* `cox_net()` fits elastic-net Cox models, with `cv_coxnet()` to choose the penalizer.

## Trees, forests, and boosting

* `survival_tree()`, `random_survival_forest()`, and `extra_survival_trees()` grow log-rank
  survival trees and ensembles, and `competing_risk_forest()` grows them for competing risks.
  `variable_importance()` ranks the covariates.

* `gradient_boosting_survival_analysis()` boosts the Cox partial likelihood, and `survival_boost()`
  boosts competing-risks predictions.

## Competing risks and multi-state models

* `aalen_johansen()` estimates the cumulative incidence of each cause, and `grays_test()` compares
  it between groups.

* `fine_gray()` fits the Fine-Gray subdistribution hazard model, with `predict()` of cumulative
  incidence curves.

* `multi_state()` estimates state occupancy probabilities for multi-state data.

* `cause_specific_cox()` fits a Cox model for one cause, and `penalized_fine_gray()` fits
  elastic-net Fine-Gray models.

* `simulate_competing_risks()` simulates competing-risks data with known effects.

## Prediction metrics

* `concordance_index()` (Harrell) and `concordance_index_ipcw()` (Uno) measure how well risk
  scores rank survival times.

* `brier_score()`, `integrated_brier_score()`, `time_dependent_auc()`, `integrated_auc()`, and
  `calibration()` assess predicted survival probabilities.

* `brier_score_incidence()`, `integrated_brier_score_incidence()`,
  `concordance_index_incidence()`, `calibration_incidence()`, and `accuracy_in_time()` do the
  same for competing risks.

* `concordance_index_ci()` gives a confidence interval for the concordance, and
  `concordance_index_compare()` compares two risk scores. `score_cr()` scores predicted cumulative
  incidences.

* `censoring_distribution()` estimates the censoring distribution and its inverse probability
  weights.

* `cross_validate()` cross-validates a model, and `bootstrap()` gives bootstrap confidence
  intervals for Kaplan-Meier summaries.

* `logrank_n_events()`, `logrank_power()`, and `logrank_sample_size()` plan a log-rank test.

## Tidying and plotting

* `tidy()`, `glance()`, and `augment()` methods (on the generics package's generics) give the
  results of every model and test as data frames.

* ggplot2 plots: `plot_survival()` and `plot_cif()` for curves, with the numbers at risk stacked
  underneath through `risk_table = TRUE` (`risk_table()` and `get_risk_table_frame()` give them on
  their own), `plot_predicted_survival()`, `plot_forest()`, `plot_influence()`,
  `plot_schoenfeld()`, `plot_smooth_hr()`, and `plot_weibull()`. `theme_survival()` and
  `theme_forest()` style them.
