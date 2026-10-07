# greenwood

Greenwood is a toolkit for survival analysis, the study of how long it takes for an event to
happen. It's built for data where some outcomes are still unknown (censored), and it aims to cover
the whole workflow: describing survival, comparing groups, modeling, and checking how well models
predict.

Greenwood is in early development. It currently provides `Surv()`, the survival response that
everything else builds on. Planned areas include:

- Kaplan-Meier and Nelson-Aalen estimation
- group comparisons with log-rank and related tests
- Cox, parametric, and flexible regression models
- competing risks and multi-state models
- prediction performance: concordance, Brier scores, and calibration

Functions are idiomatic R: lower snake case names that return S3 objects, designed to work well in
formulas and pipelines.

## Example

```r
library(greenwood)

Surv(time = c(5, 6, 4, 9), event = c(1, 0, 1, 0))
#> <Surv[4]: right>
#> [1] 5  6+ 4  9+
```

## Installation

Install the development version from GitHub:

```r
# install.packages("pak")
pak::pak("rich-iannone/greenwood/r")
```

## Development

Greenwood is developed alongside a Python package of the same name, in the same repository. See
[CONTRIBUTING](../.github/CONTRIBUTING.md) for how to run the tests (`make r-test`) and
`R CMD check` (`make r-check`).
