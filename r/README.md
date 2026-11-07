# greenwood

Greenwood is a toolkit for survival analysis, the study of how long it takes for an event to
happen. It's built for data where some outcomes are still unknown (censored), and it aims to cover
the whole workflow: describing survival, comparing groups, modeling, and checking how well models
predict.

Greenwood is in early development. It currently provides:

- `Surv()`, the survival response that everything else builds on
- Kaplan-Meier, Nelson-Aalen, and Turnbull (interval-censored) estimation, with predictions,
  quantiles, and restricted mean survival time
- group comparisons: log-rank, Fleming-Harrington, trend, MaxCombo, and restricted mean tests
- Cox, accelerated failure time, and parametric models
- competing risks and multi-state models: Aalen-Johansen, Gray's test, and Fine-Gray regression
- penalized, flexible parametric, additive hazards, cure, and Buckley-James models
- survival trees, random survival forests, and gradient boosting
- prediction performance: concordance, Brier scores, time-dependent AUC, and calibration
- `tidy()`, `glance()`, and `augment()` methods, and ggplot2 plots of curves, models, and
  diagnostics
- the datasets used in the examples, such as `lung`, `veteran`, and `mgus2`

Functions are idiomatic R: lower snake case names that return S3 objects, designed to work well in
formulas and pipelines. Every table starts with a `strata` column, so results have the same layout
whether a fit holds one curve or several.

## Example

```r
library(greenwood)

Surv(time = c(5, 6, 4, 9), event = c(1, 0, 1, 0))
#> <Surv[4]: right>
#> [1] 5  6+ 4  9+

fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
median(fit)
logrank_test(Surv(time, status) ~ sex, data = lung)
plot_survival(fit)

cox <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
tidy(cox, exponentiate = TRUE)
plot_forest(cox)
```

## Installation

Install the development version from GitHub:

```r
# install.packages("pak")
pak::pak("rich-iannone/greenwood/r")
```

## Development

Greenwood is developed alongside a Python package of the same name, in the same repository. See
[CONTRIBUTING](https://github.com/rich-iannone/greenwood/blob/main/.github/CONTRIBUTING.md) for
how to run the tests (`make r-test`) and `R CMD check` (`make r-check`).
