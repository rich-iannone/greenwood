# Kaplan-Meier contract

The Kaplan-Meier estimate is validated against R's `survival::survfit()`. Both Greenwood packages
replay the same numeric fixtures.

| | |
|---|---|
| Reference | `survival::survfit(Surv(...) ~ ...)` with `robust = FALSE` |
| Fixtures | `km_lung_overall`, `km_lung_by_sex`, `km_veteran_overall`, `lung_km_overall`, `counting_truncation` in [`../fixtures/r/`](../fixtures/r/) |
| Generator | [`scripts/regenerate_r_fixtures.R`](../scripts/regenerate_r_fixtures.R) |
| Python | `KaplanMeier` in `python/greenwood/_nonparametric.py`, replayed in `python/tests/test_r_parity.py` |
| R | `kaplan_meier()` in `r/R/kaplan-meier.R`, replayed in `r/tests/testthat/test-kaplan-meier.R` |

## API

| Python | R |
|---|---|
| `gw.KaplanMeier(conf_type=, conf_level=).fit(surv, data=, by=, weights=)` | `kaplan_meier(formula, data, weights, subset, na.action, conf_type, conf_level)` |
| `by=` or the formula's right-hand side | the formula's right-hand side |
| `km.to_frame()` | `as.data.frame(fit)` |
| `km.median(ci=True)`, `km.quantile(p, ci=True)` | `median(fit)`, `quantile(fit, probs)` |

Python estimators are classes and R estimators are functions returning S3 objects. Argument names
match where both have them (`conf_type`, `conf_level`, `weights`).

## Estimate

- Times are the unique stop times (event and censoring times), sorted, per group.
- At risk at time \(t\): subjects with `start < t <= stop` (`stop >= t` for right-censored data),
  summed with case weights.
- \(S(t) = \prod_{t_i \le t} (1 - d_i / n_i)\).
- Greenwood: \(\mathrm{Var}(\log S) = \sum d_i / (n_i (n_i - d_i))\), and
  `std_error` \(= S \cdot \sqrt{\mathrm{Var}(\log S)}\). It is missing once \(S = 0\).
- Confidence limits on the `log` (default), `log-log`, or `plain` scale, clipped to \([0, 1]\).
  The interval is \([1, 1]\) while \(S = 1\) and missing once \(S = 0\).
- Groups are ordered by the grouping variables (factor levels, otherwise sorted values).

## Output columns

Per time: `time`, `n_risk`, `n_event`, `n_censor`, `estimate`, `std_error`, `conf_low`,
`conf_high`, preceded by the grouping information.

## Quantiles

The \(p\) quantile is the first time the estimate drops to \(1 - p\) or below, and its limits are the
first times `conf_low` and `conf_high` do. A quantile is missing when the curve never gets that low.

## Language adaptations

| Python | R | Why |
|---|---|---|
| a single `strata` column | one column per grouping variable | Keeps each variable's type and supports several variables cleanly. |
| `n_dropped_` | `fit$n_dropped`, from `na.action` | R's model-frame conventions. |

## Not yet in R

Robust (infinitesimal-jackknife) and cluster variance, `predict()` at arbitrary times, restricted
mean survival time, and tidy/glance methods. Python has all of these.
