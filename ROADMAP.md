# Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap.

## Now

- The `Surv` response object (right/left/interval/counting censoring, left truncation,
  weights, multi-state endpoints) and the narwhals risk-set/event-table kernel, validated
  against R's `survival`, plus the tidy protocol.
- **Kaplan-Meier & Nelson-Aalen**: survival and cumulative-hazard estimators with Greenwood
  variance, `plain`/`log`/`log-log` confidence intervals, median and quantiles, restricted
  mean survival time (RMST), step-function prediction, and stratification. R-validated.
- **Group comparisons**: the log-rank test and the G-rho (Fleming-Harrington) family for
  two or more groups. R-validated against `survdiff`.
- **Visualization (plotnine)**: `plot_survival` with confidence ribbons, censoring marks,
  and an aligned numbers-at-risk table.
- **Cox proportional hazards** (`CoxPH`): Efron/Breslow ties, hazard ratios, model-based
  standard errors, Wald z-tests, and the likelihood-ratio / Wald / score global tests;
  stratification and the robust (Lin-Wei) sandwich variance with clustering; baseline hazard
  and survival prediction, martingale and Schoenfeld residuals, the Grambsch-Therneau
  proportional-hazards test (`cox_zph`), and the concordance index. R-validated against
  `coxph`, `basehaz`, `residuals`, `cox.zph`, and `concordance`.
- **Parametric AFT models** (`AFT`): Weibull, exponential, log-normal, and log-logistic
  accelerated failure time models, R-validated against `survreg`.
- **Competing risks & multi-state** (`AalenJohansen`, `FineGray`, `MultiState`): per-cause
  cumulative incidence functions with delta-method standard errors, the Fine-Gray
  subdistribution hazard model with clustered robust standard errors, and Aalen-Johansen
  multi-state transition/occupancy probabilities. R-validated against `survfit` and
  `finegray`.
- **Prediction-performance metrics**: Harrell's concordance index and the IPCW (Graf)
  Brier score with its time integral. R-validated against `survival::concordance` and
  `survival::brier`.

## Descriptive

- **Stratified and trend tests**, pairwise comparisons with multiplicity control, and RMST
  differences.
- **More plots**: cumulative-incidence curves and forest plots as those estimators land.

## Regression

- **Cox extensions**: time-varying covariates, deviance/dfbeta residuals, Efron-consistent
  robust variance, and further `cox_zph` time transforms (Kaplan-Meier, rank).
- **Flexible-parametric**: Royston-Parmar spline models and piecewise exponential, plus
  parametric prediction (survival, hazard, quantiles) and generalized gamma.

## Advanced

- **Competing risks & multi-state**: Gray's test, multi-state standard errors, and
  pseudo-observations (building on the Aalen-Johansen CIF, Fine-Gray model, and multi-state
  occupancy probabilities already shipped).
- **Frailty, penalized, additive, cure, and further metrics**: shared frailty, elastic-net
  Cox, additive hazards, cure models, time-dependent AUC, and calibration.

## Toward 1.0

- Full backend matrix, accelerated kernels, finalized extension protocols, and migration
  guides R's `survival` package.

## Release train

| Release | Contents |
|---|---|
| `0.1` | Kaplan-Meier / Nelson-Aalen and core inference |
| `0.2` | Group comparisons |
| `0.3` | plotnine visualization |
| `0.4` | Cox proportional hazards |
| `0.6` | Parametric & flexible-parametric models |
| `0.8` | Competing risks & multi-state |
| `0.9` | Advanced estimators & prediction metrics |
| `1.0` | Interop, performance, docs, and API-stability guarantees |

## Ecosystem

Greenwood is the survival **engine**. [Great Summaries](https://github.com/rich-iannone/great-summaries)
is the table layer that calls it: greenwood ships broom-compatible `tidy`/`glance`/`augment`
so `gs.tbl_survfit(km)` and `gs.tbl_regression(cox)` work. Tables render through Great
Tables; figures are [plotnine](https://plotnine.org/) objects.
