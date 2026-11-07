# R and Python APIs

Python estimators are classes with `fit()` and trailing-underscore attributes. R estimators are
functions that take a formula and return S3 objects, read with the usual generics (`coef()`,
`vcov()`, `predict()`, `residuals()`, `as.data.frame()`, `quantile()`, `median()`). Shared
functions (`Surv()`, the tests, and the metrics) keep their names. This file records the mapping
and the places where the R package differs from the Python one on purpose.

## Mapping

| Python | R | Contract or fixtures |
|---|---|---|
| `KaplanMeier` | `kaplan_meier()` | [`kaplan_meier.md`](kaplan_meier.md) |
| `NelsonAalen` | `nelson_aalen()` | the cumulative hazards in `km_*` |
| `Turnbull` | `turnbull()` | `turnbull_veteran_km_equivalence` |
| `event_table()` | `event_table()` | |
| `logrank_test()`, `pairwise_logrank_test()`, `trend_test()`, `maxcombo_test()` | same names | `logrank_*`, `grho_lung_sex_rho1`, `maxcombo_lung_sex`, `pairwise_logrank_veteran`, `trend_lung_ecog` |
| `rmst_test()`, `rmst_diff()`, `pairwise_rmst_test()` | same names | `rmst_twogroup_lung` |
| `km.rmst(tau)`, `km.rmrl(s, tau)` | `rmst(fit, tau)`, `rmrl(fit, s, tau)` | `rmst` |
| `CoxPH` | `cox_ph()` | `cox_*` (including `cox_weighted_residuals`), `predict_*` |
| `CoxPH.baseline_hazard()` | `baseline_hazard(fit)` | `cox_basehaz_ci`, `cox_diag_*` |
| `CoxPH.cox_zph()` | `cox_zph(fit)` | `cox_zph_windowed` |
| `CoxPH.fit(frailty=, frailty_cluster=)`, `CoxPH.frailty_test()` | `cox_ph()` with a `frailty()` term or the same arguments, `frailty_test(fit)` | checked against the Python values (lung by `inst`, and simulated clusters) |
| `RoystonParmar` | `royston_parmar()` | `rp_weibull_anchor`, `rp_loglogistic_anchor`, and the Python values for larger `df` |
| `AalenAdditive`, `AalenAdditive.cumulative_coefficients()` | `aalen_additive()`, `cumulative_coefficients(fit)` | `aalen_lung_age_sex`, `aalen_veteran`, and `aareg()` for weighted and counting-process fits |
| `CensoringDistribution` | `censoring_distribution()`, with `predict()`, `weights()`, and `as.data.frame()` | `ipcridge_lung`, and `survfit()` for weighted curves |
| `BuckleyJames` | `buckley_james()` | checked against the Python values, and against `lm()` without censoring |
| `PiecewiseExponential`, `PiecewiseExponential.baseline_hazard()` | `piecewise_exponential()`, `baseline_hazard(fit)` | `glm()` on the split data, the exponential `aft()`, and the Python values |
| `MixtureCure` | `mixture_cure()` | `cure_ph_e1684`, `glm()` and `coxph()` for the EM steps, and the Python values |
| `IPCRidge` | `ipc_ridge()` | `ipcridge_lung`, and `lm()` for the unpenalized and unstandardized fits |
| `AFT` | `aft()` | `aft_*`, `aft_residuals_*`, `predict_*` |
| `Parametric`, `compare_distributions()` | `parametric()`, `compare_distributions()` | `parametric_*` |
| `AalenJohansen`, `grays_test()` | `aalen_johansen()`, `grays_test()` | `cif_mgus2`, `grays_test_*` |
| `FineGray` | `fine_gray()` | `finegray_mgus2_pcm` |
| `CauseSpecificCox` | `cause_specific_cox()` | `cox_ph()` and `coxph()` on the recoded response, and the Python values |
| `MultiState` | `multi_state()` | `multistate_mgus2` |
| `CoxNet` | `cox_net()` | checked against the Python values, the optimality (KKT) conditions, and `cox_ph()` without a penalty |
| `cv_coxnet()`, `CoxNetCVResult` | `cv_coxnet()`, a `cv_coxnet` object | each fold's score rebuilt from the recorded folds, and the Python penalizer path |
| `PenalizedFineGray` | `penalized_fine_gray()` | checked against the Python values, the optimality (KKT) conditions, and `fine_gray()` without a penalty |
| `SurvivalTree` | `survival_tree()` | checked against the Python values, `survdiff()`'s log-rank statistic at every split, and `kaplan_meier()` and `nelson_aalen()` in every leaf |
| `RandomSurvivalForest`, `ExtraSurvivalTrees`, `variable_importance()` | `random_survival_forest()`, `extra_survival_trees()`, `variable_importance()` | the bagging, ensemble curves, out-of-bag concordance, and permutation importance rebuilt by hand |
| `GradientBoostingSurvivalAnalysis` | `gradient_boosting_survival_analysis()` | checked against the Python values, `coxph()`'s martingale residuals and Breslow baseline, and a brute-force search for each split |
| `SurvivalBoost` | `survival_boost()` | the sampled horizons, class targets, censoring weights, and trees of each round rebuilt by hand |
| `CompetingRiskForest` | `competing_risk_forest()` | checked against the Python values, the composite log-rank statistic computed from its definition for the best split, and `aalen_johansen()` in every leaf. The bagging, ensemble curves, out-of-bag concordance, and permutation importance rebuilt by hand |
| `concordance_index()`, `concordance_index_ipcw()` | same names | `cox_diag_efron`, `concordance_ipcw_lung` |
| `brier_score()`, `integrated_brier_score()` | same names | `brier_lung` |
| `time_dependent_auc()`, `integrated_auc()`, `calibration()` | same names | `td_auc_lung` |
| `*_incidence()` metrics, `accuracy_in_time()` | same names | `calibration_incidence_mgus2`, otherwise checked against the Python values |
| `concordance_index_ci()`, `concordance_index_compare()` | same names | `concordance_ci_ij` |
| `score_cr()` | same name | checked against the Python values |
| `logrank_n_events()`, `logrank_power()`, `logrank_sample_size()` | same names | checked against the Python values |
| `simulate_competing_risks()` | same name | the latent times checked against the data, and the parameters recovered by `cox_ph()` and `aft()` |
| `cross_validate()` | same name | each fold's score rebuilt from the recorded folds |
| `bootstrap()`, `BootstrapResult` | `bootstrap()`, a `greenwood_bootstrap` object | estimates checked against the Python values, standard errors to within the Monte Carlo error |

## Not yet in R

The tidy/glance/augment layer, plots, and the `EventTime` and `Outcome` layers.

## Adaptations

| Python | R | Why |
|---|---|---|
| `fit(surv, data=)` with a formula string or `Outcome` | `fn(formula, data, weights, subset, na.action)` | R's model-frame conventions. Rows dropped by `na.action` are counted in `fit$n_dropped`. |
| `se`, `lower_ci`, `upper_ci` in the RMST comparisons | `std_error`, `conf_low`, `conf_high` | One set of names across every R table. |
| `maxcombo_test(weights=[(rho, gamma), ...])` | `maxcombo_test(rho = c(...), gamma = c(...))` | `weights` means case weights in R. |
| MaxCombo p-value from Sobol points | from a deterministic lattice rule | Same value to about `1e-5`, reproducible without a seed or mvtnorm. |
| Cox and AFT predictions with one `subject_i` column per subject (and `subject_i_lower`, `subject_i_upper`) | long data frames: `subject`, `strata`, then `time` (or `p`, `tau`), `estimate`, and the limits. Expected event counts have no `std_error`, as in Python | Same layout for one subject or many, as in [`results.md`](results.md). |
| `CoxPH.cox_zph(transform="identity")` by default | `cox_zph(fit, transform = "km")` by default | The Kaplan-Meier transform is the usual default in R and is less sensitive to outlying times. |
| `frailty_effect_`, `frailty_levels_`, `frailty_theta_`, `frailty_lrt_stat_`, `frailty_lrt_p_value_` | `fit$frailty`: `distribution`, `cluster`, `theta`, `effects` (named by cluster), `test` (`statistic`, `df`, `p_value`), `iterations`, `converged` | One list for the frailty, with the test laid out as the other `cox_ph()` tests. |
| `frailty_test()` gives a dict (`theta`, `lr_statistic`, `df`, `p_value`) | a one-row data frame: `distribution`, `theta`, `statistic`, `df`, `p_value` | One set of names across every R table. |
| `strata=` argument when predicting from a stratified Cox model | strata read from the `strata()` variables in `newdata` | The formula already names them. |
| `baseline_hazard()` columns `strata`, `time`, `cumhaz`, `survival` | `strata`, `time`, `n_risk`, `n_event`, `estimate` (the cumulative hazard), `std_error`, `conf_low`, `conf_high` | The same columns as `nelson_aalen()`. |
| `predict(type="lp")` | a plain numeric vector for `cox_ph()`, `aft()`, and `fine_gray()`. Other types give data frames | As `predict.lm()` and the base generics do. |
| AFT `score`, `dfbeta`, and `dfbetas` residuals leave out the case weights | include them | Matches `cox_ph()`, where the weights enter the score. |
| `Parametric` limits on the natural scale | positive parameters (shape, scale, rate, sigma) get limits on the log scale | The limits stay positive. |
| `FineGray.predict_cumulative_incidence()` wide | `predict(fit, newdata, type = "cif", times)` long: `subject`, `time`, `estimate` | Same as the Cox predictions. |
| `FineGray.naive_vcov_` | `fit$naive_var`, with `vcov()` giving the robust variance | R names. |
| `MultiState.to_frame()` wide, one column per state | `as.data.frame()` long: `strata`, the grouping variables, `state`, `time`, `estimate` | Same layout for any number of states and groups. |
| `MultiState.fit(start, stop, state, event)` needs every row's starting state | `istate` is optional, and without it every row starts in the state `"(s0)"` | Competing-risks data then need no extra column. |
| metrics take a `Surv`, an `Outcome`, or a formula string with `data=` | metrics take a `Surv()` object, such as `Surv(lung$time, lung$status)` | The predictions are vectors or matrices, not columns of `data`, so a formula adds nothing. |
| `nan` for unidentified Turnbull values | `NA` | |
| `concordance_index_ci()` gives a `ConcordanceResult` (`estimate`, `se`, `ci_low`, `ci_high`, `conf_level`, `n`) | a one-row data frame: `estimate`, `std_error`, `conf_low`, `conf_high`, `n` | One set of names across every R table. |
| `concordance_index_compare()` gives a `ConcordanceCompareResult` (`delta`, `se`, `ci_low`, `ci_high`, `z`, `pvalue`) | a one-row data frame: `difference`, `std_error`, `conf_low`, `conf_high`, `statistic`, `p_value` | As `rmst_diff()`. |
| `score_cr()` labels `brier(t)` with `t` rounded to a whole number, and has an `se` column | `t` as `format()` gives it, and a `std_error` column | Fractional times keep distinct labels. |
| `bootstrap(seed=)` | no `seed` argument. Call `set.seed()` first | R's convention for random numbers. The resamples differ from Python's, so only the estimates agree exactly. |
| `bootstrap(by=)`, with the `"_diff"` statistics taking the first group minus the second sorted as text | groups from the formula's right-hand side, first minus second in the order of the `kaplan_meier()` curves | The same order as every other R table. The result records the two `groups`. |
| `BootstrapResult.se`, `to_frame()` | `std_error`, `as.data.frame()` | One set of names across every R table. |
| `cross_validate(CoxPH(), surv, covariates, seed=)` with an unfitted estimator | `cross_validate(cox_ph, formula, data, ...)` with the model function, its other arguments in `...`, and `set.seed()` for the folds | R passes functions, not estimator objects. `cox_ph()`, `cox_net()`, `aft()`, `royston_parmar()`, the survival trees and forests, and `gradient_boosting_survival_analysis()` are supported. |
| `cross_validate(metric=)` gives a flat dict and `metrics=` a nested one, with `std` | one `metrics` argument and one layout: `results` (`metric`, `mean`, `sd`) and long per-fold `scores` (`fold`, `metric`, `score`), plus the `folds` | One result layout whatever the number of metrics. |
| `simulate_competing_risks(seed=)` gives a `SimulatedCompetingRisks` (`surv`, `covariates`, `latent_times`, `censoring_time`) | a data frame (`time`, `event` as a factor with first level `"censor"`, and `x1`, `x2`, ...) with `latent_times`, `censoring_time`, and `coef` attributes. Call `set.seed()` first | The data frame goes straight into a formula, such as `fine_gray(Surv(time, event) ~ x1, data = sim, cause = "cause1")`. |
| `RoystonParmar.predict()`, `predict_quantile()`, `predict_median()`, `predict_expectation()`, with `newdata=None` for a subject whose covariates are all 0 | `predict(fit, newdata, type = "survival", "hazard", "cumhaz", "quantile", "expectation", or "lp")`, long, always with `std_error` and limits. Without `newdata`, the rows the model was fit to | The same layout and default as the other R models. |
| `RoystonParmar` fit by BFGS with a numerical Hessian | Newton-Raphson with the analytic Hessian | The likelihood is concave, so Newton's steps reach its maximum. Estimates agree to about `1e-5`. |
| `RoystonParmar` restricted-mean limits are the estimate plus or minus `z` standard errors | limits on the log scale | As `aft()`. The standard errors agree. |
| `RoystonParmar` drops rows with a time of 0, and takes no weights | an error for times of 0, and `weights` | As `aft()`. |
| `AalenAdditive(test=)` takes `"aalen"`, `"variance"`, or `"nrisk"`, with `"variance"` giving the same values as `"aalen"` | `test = "aalen"` or `"nrisk"` | An option that changes nothing is left out. |
| `AalenAdditive.to_frame()` columns `term`, `slope`, `coef`, `se`, `z`, `p` | `as.data.frame()`: `term`, `slope`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` | One set of names across every R table. |
| `AalenAdditive.cumulative_coefficients()` wide, one row per event (tied events have a row each) | long: `term`, `time`, `n_risk`, `n_event`, `estimate`, `std_error`, `conf_low`, `conf_high`, one row per distinct time | The same layout as the other curves. The standard errors are the root summed squared increments. |
| `AalenAdditive.predict()` wide, without standard errors | long: `subject`, `time`, `estimate`, `std_error`, `conf_low`, `conf_high` | As the other models. The variance of the cumulative hazard sums the outer products of the increments. |
| `AalenAdditive.n_event_times_used_` counts the events used | `fit$n_times` counts the distinct event times used | As `aareg()`. |
| `AalenAdditive` takes no weights | `weights`, with `nmin` compared with the weight at risk | As `aareg()`. |
| `CensoringDistribution.survival(times)`, `.survival_left(times)`, `.weights(tau=)` | `predict(object, times, left = FALSE)` and `weights(object, tau = Inf)` | R's generics. |
| `CensoringDistribution.to_frame()` lists only the censoring times | `as.data.frame()`, the same | |
| `IPCRidge` takes no weights | `weights` multiply the IPC weights and weight the censoring curve and the standard deviations, so whole-number weights act as copies | As the other R models. |
| `IPCRidge.predict()` | `predict(fit, newdata, type = c("lp", "response"))`, a numeric vector | As `predict.lm()`. |
| `BuckleyJames(n_boot=, seed=)` | `buckley_james(n_boot = )`, with `set.seed()` for the resamples | R's random number generator. Resamples with no events or collinear covariates are left out. |
| `BuckleyJames` drops rows with a time of 0, and takes no weights | an error for times of 0, and `weights` | As `aft()`. |
| `BuckleyJames.predict(type="survival")` wide, with `times` required | long: `subject`, `time`, `estimate`, with `times` required | As the other models. |
| `PiecewiseExponential` drops rows with a time of 0 | an error for times of 0 | As `aft()`. |
| `PiecewiseExponential.predict(type="survival" or "cumhaz")` wide, without standard errors | long: `subject`, `time`, `estimate`, `std_error`, `conf_low`, `conf_high`, with limits built on the log cumulative hazard | As the other models. |
| `PiecewiseExponential.baseline_hazard()` columns `start`, `stop`, `hazard`, `log_hazard` | the same, then `std_error` (of the log hazard), `conf_low`, `conf_high` | |
| `PiecewiseExponential.lr_stat_` | `fit$lr_test` (`statistic`, `df`, `p_value`) | Laid out as the `cox_ph()` tests. |
| `PiecewiseExponential` with breaks leaving an interval without follow-up fails in the linear algebra | an error naming the interval, and a warning for an interval with no events | |
| `MixtureCure().fit(surv, latency, cure=[...], nboot=100)`, with the resamples seeded by 42 | `mixture_cure(formula, cure = ~ ..., n_boot = 100)`, the latency covariates on the right of `formula`, and `set.seed()` for the resamples | R's formula and random number conventions. `n_boot = NULL` skips the bootstrap. |
| `MixtureCure(emmax=, eps=)` | `max_iter`, `tol` | The names of the other R models. A fit that runs out of iterations warns and records `converged = FALSE`. |
| `MixtureCure.predict_cure_prob()`, `predict_survival()` wide | `predict(fit, newdata, type = "uncured")` (a vector), `type = "survival"` (long: `subject`, `time`, `estimate`), and `type = "lp"` | As the other models. |
| `MixtureCure.to_frame()` columns `submodel`, `term`, `coef`, `se`, `z`, `p` | `as.data.frame()`: `submodel`, `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high`. `coef()` names them `cure.` and `latency.` | One set of names across every R table. |
| `MixtureCure` takes no weights | `weights`, in both models and the baseline survival, so whole-number weights act as copies | As the other R models. |
| `CauseSpecificCox(cause=...)`, with `cause` required | `cause_specific_cox(formula, data, cause = 1, ...)`, the arguments in `...` going to `cox_ph()` | The first cause is the usual one to model. |
| `CauseSpecificCox` wraps a `CoxPH` in `cox_`, and delegates attributes to it | the result is also a `cox_ph` object, so every `cox_ph()` method works on it, with `cause` and `states` added | S3 inheritance. `cross_validate()` rejects it, as it does the Python one. |
| `CauseSpecificCox.__repr__` header | `print()` starts `<cause_specific_cox>` and names the cause modeled | As the other R models. |
| `CoxNet` objective divides the log partial likelihood by `n`, and centers and scales the covariates without the weights | divides by the total weight, with weighted means and standard deviations, so whole-number weights act as copies | As the other R models. Without weights the fits are the same. |
| `CoxNet` stops at `max_iter` without a word | a warning, and `fit$converged` | As `mixture_cure()`. The stopping rule (a small relative change in the coefficients) is the Python one, so lasso fits agree to about `1e-9` and ridge and elastic-net fits to about `1e-5`. |
| `CoxNet.predict(type="survival")` wide, defaulting to the event times | long: `subject`, `time`, `estimate`, defaulting to every distinct time (`fit$times`) | As the other models. |
| `CoxNet.to_frame()`, `effective_df()`, `aic()`, `bic()` | `as.data.frame()` (`term`, `estimate`, `hazard_ratio`), `fit$effective_df`, and `logLik()`, `AIC()`, `BIC()` with the effective degrees of freedom and `nobs()` the number of events | R's generics. |
| `CoxNet` has no strata, clusters, or offsets | `strata()`, `cluster()`, and `offset()` terms are errors | They aren't taken as covariates by mistake. |
| `cv_coxnet(seed=)`, with the fold fits ignoring the weights | `set.seed()` for the folds, and the weights passed to every fold fit | R's convention for random numbers, and the weights mean the same in every fit. |
| `cv_coxnet()` takes counting-process data | right-censored data only | The concordance and the Brier score need right-censored data. |
| `CoxNetCVResult.penalizers_`, `mean_scores_`, `std_scores_`, `n_nonzero_`, `to_frame()` | `cv$path` (`penalizer`, `mean_score`, `std_score`, `n_nonzero`), `as.data.frame()`, and the per-fold `cv$scores` (`penalizer`, `fold`, `score`, `n_nonzero`) with the `folds` | As `cross_validate()`. A fit that doesn't converge is reported in one warning. |
| `PenalizedFineGray(cause=...)`, with `cause` required | `penalized_fine_gray(formula, data, cause = 1, ...)` | As `fine_gray()`. |
| `PenalizedFineGray.predict()` and `predict_cumulative_incidence()` (wide) | `predict(fit, newdata, type = c("lp", "risk", "cif"), times)`, the cumulative incidence long (`subject`, `time`, `estimate`) | As `fine_gray()`. |
| `PenalizedFineGray` stops at `max_iter` without a word, and keeps no likelihood | a warning, `fit$converged`, and `fit$loglik` (the log pseudo partial likelihood) | As `cox_net()`. The fits agree with Python's to about `1e-5`. |
| `PenalizedFineGray.to_frame()` | `as.data.frame()`: `term`, `estimate`, `hazard_ratio` | R's generics. |
| `random_state=` on the trees, forests, and boosting models, and `variable_importance(random_state=)` | no `random_state`. Call `set.seed()` first | R's convention for random numbers. The fits that draw none (a `"best"` tree trying every feature, and the Cox gradient boosting without `subsample` or `max_features`) agree with Python's. |
| `engine="numpy"` or `"numba"` for the split search | no `engine` argument | One implementation in base R. |
| A tree trying every feature still visits them in a random order | it visits them in column order and draws no random numbers | A tree is then the same whatever the state of the generator. Ties (equivalent columns, such as `ph.ecog` and `ph.karno` in lung) go to the first column, where Python's go to the first in its order, so their importance can be shared out differently. The predictions agree. |
| `SurvivalTree`, forest, and `GradientBoostingSurvivalAnalysis` `predict(type="survival")` and `"cumulative_hazard"`, wide | `predict(fit, newdata, type = c("risk", "survival", "cumhaz"), times)`, the curves long (`subject`, `time`, `estimate`), defaulting to the event times | As the other R models. The Cox gradient boosting also has `type = "lp"`. |
| Covariates that are NaN go right at every split | rows with a missing covariate get `NA` predictions | No prediction is made up for them. |
| `RandomSurvivalForest.oob_score_`, `variable_importance()` | `fit$oob_score` (`NA` unless `oob_score = TRUE`), and `variable_importance(fit, n_repeats = 5)`, a data frame (`term`, `importance`) sorted by importance | R's names. The Cox gradient boosting's `variable_importance()` is its normalized reduction in squared error, as in Python. |
| `survival_tree()` and the forests have no table of their splits | `as.data.frame()` of a tree: `node`, `depth`, `term`, `threshold`, `left`, `right`, `n`, `n_event` | So a tree can be read. |
| `SurvivalBoost` scores each subject's training rows with every round's trees over all of its horizons, not at the sampled horizons | each sampled (horizon, covariates) row is scored by the current ensemble | The gradient is then that of the model being fit. The fits differ from Python's. |
| `SurvivalBoost(hard_zero_fraction=0)` still sets one horizon to 0, and its time grid can repeat a quantile | `floor(hard_zero_fraction * n)` horizons set to 0, and the grid's repeated quantiles dropped | `0` means none, and every grid time is distinct. |
| `SurvivalBoost.predict_cumulative_incidence()` (a 3-D array or wide frame), `predict_survival_function()`, and `predict_proba(time_horizon=)`, with `newdata` required | `predict(fit, newdata, type = c("cif", "survival"), times)`, long (`subject`, `cause`, `time`, `estimate` for the causes, ordered by subject, cause, then time), defaulting to the rows the model was fit to and the time grid | As the other R competing-risks models. |
| `CompetingRiskForest.predict(cause=)` (wide, with `cause` required when there are several) and `predict_risk(cause=)`, with `newdata` required | `predict(fit, newdata, type = c("cif", "risk"), cause = 1, times)`, the curves long (`subject`, `time`, `estimate`), defaulting to the first cause, the rows the forest was grown on, and the event times | As `fine_gray()` and the other R forests. |
| `CompetingRiskForest.oob_score_` and `variable_importance()` for the first cause | `fit$oob_score` for the first cause (`NA` unless `oob_score = TRUE`), and `variable_importance(fit, cause = 1, n_repeats = 5)` | Any cause's importance can be had. |
