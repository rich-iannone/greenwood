# Kaplan-Meier contract

The Kaplan-Meier estimate is validated against R's `survival::survfit()`. Both Greenwood packages
replay the same numeric fixtures.

| | |
|---|---|
| Reference | `survival::survfit(Surv(...) ~ ...)`, with `robust = FALSE` unless robust variance is requested |
| Fixtures | `km_lung_overall`, `km_lung_by_sex`, `km_veteran_overall`, `lung_km_overall`, `counting_truncation`, `km_robust_lung_overall`, `km_robust_lung_by_sex`, `km_robust_lung_weighted`, `km_cluster_lung_inst` in [`../fixtures/r/`](../fixtures/r/) |
| Generator | [`scripts/regenerate_r_fixtures.R`](../scripts/regenerate_r_fixtures.R) |
| Python | `KaplanMeier` in `python/greenwood/_nonparametric.py`, replayed in `python/tests/test_r_parity.py` |
| R | `kaplan_meier()` in `r/R/kaplan-meier.R`, replayed in `r/tests/testthat/test-kaplan-meier.R` |

## API

| Python | R |
|---|---|
| `gw.KaplanMeier(conf_type=, conf_level=, robust=).fit(surv, data=, by=, weights=, cluster=)` | `kaplan_meier(formula, data, weights, subset, na.action, conf_type, conf_level, robust, cluster)` |
| `by=` or the formula's right-hand side | the formula's right-hand side |
| `km.to_frame()` | `as.data.frame(fit)` |
| `km.median()`, `km.quantile(p)` | `median(fit)`, `quantile(fit, probs)` |
| `km.rmst(tau)`, `km.rmrl(s, tau)`, `km.predict(times)` | not yet |

Python estimators are classes and R estimators are functions returning S3 objects. Argument names
match where both have them (`conf_type`, `conf_level`, `weights`, `robust`, `cluster`).

## Estimate

- Times are the unique stop times (event and censoring times), sorted, per group.
- At risk at time \(t\): subjects with `start < t <= stop` (`stop >= t` for right-censored data),
  summed with case weights.
- \(S(t) = \prod_{t_i \le t} (1 - d_i / n_i)\).
- Greenwood: \(\mathrm{Var}(\log S) = \sum d_i / (n_i (n_i - d_i))\), and
  `std_error` \(= S \cdot \sqrt{\mathrm{Var}(\log S)}\). It is missing once \(S = 0\).
- Robust variance (`robust = TRUE`, implied by `cluster`): the infinitesimal jackknife. Subject
  \(i\)'s influence on \(\log S(t)\) is \(U_i(t) = -\sum_{t_j \le t} dM_i(t_j) / (n_j - d_j)\) with
  \(dM_i(t_j) = w_i (\mathbf{1}[i \text{ has the event at } t_j] - \mathbf{1}[i \text{ at risk at } t_j]\, d_j / n_j)\).
  Influences are summed within clusters, then \(\mathrm{Var}(\log S(t)) = \sum U(t)^2\). Rows with a
  missing cluster are dropped.
- A survivor weight that is zero up to rounding (\(|n - d| \le 10^{-10} n\), which non-integer
  weights can produce) is treated as exactly zero, so the estimate reaches exactly 0.
- Confidence limits on the `log` (default), `log-log`, or `plain` scale, clipped to \([0, 1]\).
  The interval is \([1, 1]\) while \(S = 1\) and missing once \(S = 0\).
- Groups are ordered by the grouping variables (factor levels, otherwise sorted values).

## Output columns

Per time: `strata`, then `time`, `n_risk`, `n_event`, `n_censor`, `estimate`, `std_error`,
`conf_low`, `conf_high`. The layout follows [`results.md`](results.md): the same for one curve
(`strata = "all"`) as for several (`"sex=1"`).

## Quantiles

The \(p\) quantile is the first time the estimate drops to \(1 - p\) or below, and its limits are the
first times `conf_low` and `conf_high` do. A quantile is missing when the curve never gets that low.

## Language adaptations

| Python | R | Why |
|---|---|---|
| `strata` column | `strata` column, then one column per grouping variable | Keeps each variable's type, so results can be filtered without parsing labels. |
| curves in order of first appearance | curves sorted by the grouping variables | Python's long-standing order. Labels and layout match. |
| `n_dropped_` | `fit$n_dropped`, from `na.action` | R's model-frame conventions. |

## Not yet in R

`predict()` at arbitrary times, restricted mean survival time, and tidy/glance methods. Python has
all of these.
