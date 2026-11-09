<p align="center">
<a href="https://rich-iannone.github.io/greenwood/">
<img src="https://rich-iannone.github.io/greenwood/assets/logo.png" alt="Greenwood" width="350">
</a>
</p>
<p align="center">Survival analysis for Python and R, with the same API in both.</p>
<p align="center">
<a href="https://pypi.org/project/greenwood/"><img src="https://img.shields.io/pypi/v/greenwood?logo=python&logoColor=white&color=orange" alt="PyPI"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml"><img src="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
<a href="https://github.com/rich-iannone/greenwood/actions/workflows/r-check.yml"><img src="https://github.com/rich-iannone/greenwood/actions/workflows/r-check.yml/badge.svg?branch=main" alt="R-CMD-check"></a>
<a href="https://rich-iannone.github.io/greenwood/"><img src="https://img.shields.io/badge/docs-project_website-blue.svg" alt="Documentation"></a>
</p>

---

## What is Greenwood?

Greenwood is a toolkit for survival analysis: the study of how long it takes for something to
happen. That something might be a patient's relapse, a customer's cancellation, or a machine's
failure. The hard part of this kind of data is that the story is often unfinished. Some patients
are still well when the study ends, and some customers haven't left yet. Greenwood is built to make
the most of that incomplete information.

It covers the whole journey, from a first look at survival curves to regression models,
machine learning, and checking how well your predictions hold up.

## One API, two languages

Greenwood comes as a Python package and an R package that share one design. The same ideas have the
same names, the same options, and results laid out the same way, down to the column names of the
tables. Each one still feels at home in its language: Python uses classes and methods, and R uses
functions, formulas, and the generics you already know.

<table>
<tr>
<th>Python</th>
<th>R</th>
</tr>
<tr>
<td>

```python
import greenwood as gw

lung = gw.load_dataset("lung")

# Survival curves for each sex
km = gw.KaplanMeier().fit(
    "Surv(time, status) ~ sex", data=lung
)
km.median()
gw.plot_survival(km)

# Do the curves differ?
gw.logrank_test("Surv(time, status) ~ sex", data=lung)

# A Cox model, as hazard ratios
cox = gw.CoxPH().fit(
    "Surv(time, status) ~ age + sex", data=lung
)
gw.tidy(cox, exponentiate=True)
```

</td>
<td>

```r
library(greenwood)


# Survival curves for each sex
km <- kaplan_meier(
  Surv(time, status) ~ sex, data = lung
)
median(km)
plot_survival(km)

# Do the curves differ?
logrank_test(Surv(time, status) ~ sex, data = lung)

# A Cox model, as hazard ratios
cox <- cox_ph(
  Surv(time, status) ~ age + sex, data = lung
)
tidy(cox, exponentiate = TRUE)
```

</td>
</tr>
</table>

Both give the same answers. Median survival is 270 days for men and 426 days for women, the
log-rank test gives a chi-squared of 10.33 (p = 0.0013), and the hazard ratio for sex is 0.599. So
you can share an analysis with a colleague who works in the other language, move a project from one
to the other, or check one against the other, and the numbers will line up.

## What you can do with it

- **Describe survival.** Kaplan-Meier and Nelson-Aalen curves, median survival, and restricted mean
  survival time, with confidence intervals throughout.
- **Compare groups.** Log-rank tests and their weighted relatives, pairwise comparisons, trend
  tests, and comparisons of restricted mean survival.
- **Model survival.** Cox proportional hazards (with strata, frailty, and time-varying covariates),
  accelerated failure time and other parametric models, flexible spline models, penalized Cox, and
  cure models.
- **Handle competing risks.** Cumulative incidence curves, Fine-Gray regression, and multi-state
  models for when more than one kind of event can happen.
- **Use machine learning.** Survival trees, random survival forests, and gradient boosting.
- **Check predictions.** Concordance, Brier scores, time-dependent AUC, calibration, and
  cross-validation.
- **Report results.** Tidy summary tables and publication-ready plots of curves, models, and
  diagnostics.
- **Practice on real data.** Classic datasets such as `lung`, `veteran`, `pbc`, and `mgus2` come
  with the package.

In Python, Greenwood works with whichever data frame library you use, including pandas, Polars,
PyArrow, and DuckDB. In R, it works with ordinary data frames and the familiar `Surv()` formula.

## Installation

Python:

```bash
pip install "greenwood[all]"   # with formulas, plotting, pandas, Polars, and PyArrow
```

R (the development version, from GitHub):

```r
# install.packages("pak")
pak::pak("rich-iannone/greenwood/r")
```

## Learn more

- The [documentation site](https://rich-iannone.github.io/greenwood/) has a user guide that starts
  from the basics, and a full reference. The
  [R reference](https://rich-iannone.github.io/greenwood/r/) is part of the same site.
- The [Python README](python/README.md) gives a tour of the Python package.
- The [R README](r/README.md) does the same for R, and every R function has a help page (try
  `?kaplan_meier`).

## Built to be trusted

Both packages are tested against R's **survival** package, the long-standing reference for this
field, and they replay the same set of shared test cases. Estimates, standard errors, and test
statistics are checked to many decimal places on every change.

## Contributing

Contributions are welcome. The two packages live side by side in this repository, in
[`python/`](python/) and [`r/`](r/). From the repository root, `make install` sets up both and
`make test` runs both test suites. See [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for the rest.

## License

MIT (c) Richard Iannone. See the [`LICENSE`](LICENSE) file.
