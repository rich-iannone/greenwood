# Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap.

## Now

- The `Surv` response object (right/left/interval/counting censoring, left truncation,
  weights, multi-state endpoints) and the narwhals risk-set/event-table kernel, validated
  against R's `survival`, plus the tidy protocol.
- **Kaplan-Meier & Nelson-Aalen**: survival and cumulative-hazard estimators with Greenwood
  variance, `plain`/`log`/`log-log` confidence intervals, median and quantiles,
  step-function prediction, and stratification. R-validated.

## Descriptive

- **Restricted mean survival time (RMST)** with variance, and remaining CI transforms.
- **Group comparisons**: log-rank, the G-rho (Fleming-Harrington) family, stratified and
  trend tests, pairwise comparisons, RMST differences.
- **Visualization (plotnine)**: KM/CIF curves with CI ribbons, censoring marks, risk
  tables, and forest plots.

## Regression

- **Cox proportional hazards**: ties, strata, robust/cluster SE, time-varying covariates,
  proportional-hazards diagnostics, residuals, concordance.
- **Parametric & flexible-parametric**: AFT models, Royston-Parmar splines, piecewise
  exponential.

## Advanced

- **Competing risks & multi-state**: Aalen-Johansen, Fine-Gray, Gray's test, transition
  probabilities, pseudo-observations.
- **Frailty, penalized, additive, cure, and prediction metrics**: shared frailty,
  elastic-net Cox, additive hazards, cure models, time-dependent AUC, Brier/IPCW,
  calibration.

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
