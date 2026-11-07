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
| `AFT` | `aft()` | `aft_*`, `aft_residuals_*`, `predict_*` |
| `Parametric`, `compare_distributions()` | `parametric()`, `compare_distributions()` | `parametric_*` |
| `AalenJohansen`, `grays_test()` | `aalen_johansen()`, `grays_test()` | `cif_mgus2`, `grays_test_*` |
| `FineGray` | `fine_gray()` | `finegray_mgus2_pcm` |
| `MultiState` | `multi_state()` | `multistate_mgus2` |
| `concordance_index()`, `concordance_index_ipcw()` | same names | `cox_diag_efron`, `concordance_ipcw_lung` |
| `brier_score()`, `integrated_brier_score()` | same names | `brier_lung` |
| `time_dependent_auc()`, `integrated_auc()`, `calibration()` | same names | `td_auc_lung` |
| `*_incidence()` metrics, `accuracy_in_time()` | same names | `calibration_incidence_mgus2`, otherwise checked against the Python values |

## Not yet in R

Penalized and machine-learning models (`CoxNet`, `cv_coxnet()`, `PenalizedFineGray`, the survival
trees, forests, and boosting models, `IPCRidge`, `SurvivalBoost`), other regression models
(`AalenAdditive`, `BuckleyJames`, `RoystonParmar`, `PiecewiseExponential`, `MixtureCure`,
`CauseSpecificCox`), frailty terms in `cox_ph()`, `concordance_index_ci()`,
`concordance_index_compare()`, `score_cr()`, `cross_validate()` and `bootstrap()`, sample size
and power (`logrank_power()` and friends), the tidy/glance/augment layer, plots, and the
`EventTime` and `Outcome` layers.

## Adaptations

| Python | R | Why |
|---|---|---|
| `fit(surv, data=)` with a formula string or `Outcome` | `fn(formula, data, weights, subset, na.action)` | R's model-frame conventions. Rows dropped by `na.action` are counted in `fit$n_dropped`. |
| `se`, `lower_ci`, `upper_ci` in the RMST comparisons | `std_error`, `conf_low`, `conf_high` | One set of names across every R table. |
| `maxcombo_test(weights=[(rho, gamma), ...])` | `maxcombo_test(rho = c(...), gamma = c(...))` | `weights` means case weights in R. |
| MaxCombo p-value from Sobol points | from a deterministic lattice rule | Same value to about `1e-5`, reproducible without a seed or mvtnorm. |
| Cox and AFT predictions with one `subject_i` column per subject (and `subject_i_lower`, `subject_i_upper`) | long data frames: `subject`, `strata`, then `time` (or `p`, `tau`), `estimate`, and the limits. Expected event counts have no `std_error`, as in Python | Same layout for one subject or many, as in [`results.md`](results.md). |
| `CoxPH.cox_zph(transform="identity")` by default | `cox_zph(fit, transform = "km")` by default | The Kaplan-Meier transform is the usual default in R and is less sensitive to outlying times. |
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
