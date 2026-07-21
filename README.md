<p align="center">
<a href="https://rich-iannone.github.io/greenwood/">
<img src="https://rich-iannone.github.io/greenwood/assets/logo.png" alt="Greenwood" width="350">
</a>
</p>
<p align="center">Modern survival analysis for Python: Narwhals-native, R-validated, beautifully visualized.</p>
<p align="center">
<a href="https://pypi.org/project/greenwood/"><img src="https://img.shields.io/pypi/v/greenwood?logo=python&logoColor=white&color=orange" alt="PyPI"></a>
<a href="https://pypi.org/project/greenwood/"><img src="https://img.shields.io/pypi/pyversions/greenwood.svg" alt="Python versions"></a>
<a href="https://pypistats.org/packages/greenwood"><img src="https://img.shields.io/pypi/dm/greenwood" alt="Downloads"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml"><img src="https://github.com/rich-iannone/greenwood/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
</p>
<p align="center">
<a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Repo Status"></a>
</p>
<p align="center">
<a href="https://rich-iannone.github.io/greenwood/"><img src="https://img.shields.io/badge/docs-project_website-blue.svg" alt="Documentation"></a>
<a href="https://github.com/rich-iannone/greenwood/graphs/contributors"><img src="https://img.shields.io/github/contributors/rich-iannone/greenwood" alt="Contributors"></a>
<a href="https://www.contributor-covenant.org/version/3/0/"><img src="https://img.shields.io/badge/Contributor%20Covenant-v3.0%20adopted-ff69b4.svg" alt="Contributor Covenant"></a>
</p>

---

## What is Greenwood?

Greenwood is a Python library for survival analysis, the statistical study of time-to-event outcomes. Greenwood gives you lots of powerful features for extracting insights from incomplete data where some observations are censored (i.e., we don't know if the event happened yet).

### Why Greenwood?

- it works with your dataframe library: Pandas, Polars, PyArrow, or anything supported by [Narwhals](https://narwhals-dev.github.io/narwhals/)
- it is rigorously validated, where every statistic is tested to tolerance against R's gold-standard `survival` package
- you get beautiful, interactive survival visualizations, with a choice of plotting backends so you can use whichever you prefer
- you also get publication-ready tables through integration with the [Great Tables](https://posit-co.github.io/great-tables/) library
- batteries are included: from simple Kaplan-Meier curves to Cox proportional hazards, competing risks, and beyond

## What's included

Descriptive statistics:

- **`Surv` response object**: Handle right-, left-, and interval-censored data; counting-process form; left truncation; weights; and multi-state endpoints with built-in validation.
- **Kaplan-Meier estimation** (`KaplanMeier`): Survival curves with Greenwood confidence intervals, median/quantile survival, restricted mean survival time, and step-function predictions.
- **Nelson-Aalen estimator** (`NelsonAalen`): Cumulative hazard curves.
- **Visualization** (`plot_survival()`, `forest_plot()`, `cif_plot()`): Interactive survival curves with confidence bands and censoring marks, publication-ready forest plots with aligned at-risk tables, and cumulative incidence plots for competing risks (all with a choice of plotting backends and Great Tables integration).

Hypothesis testing:

- **Log-rank tests** (`logrank_test()`, `pairwise_logrank_test()`): Standard log-rank test and the G-rho (Fleming-Harrington) family for 2+ groups with p-value adjustment for multiple comparisons.
- **Linear trend tests** (`trend_test()`): Test for linear trends across ordered groups with support for Fleming-Harrington weights and stratification.
- **RMST comparisons** (`rmst_test()`, `pairwise_rmst_test()`, `rmst_diff()`): Restricted mean survival time hypothesis tests, pairwise comparisons, and difference estimation.

Regression models:

- **Cox proportional hazards** (`CoxPH`): Model covariates as hazard ratios with stratification, robust sandwich variance, clustering, baseline hazard prediction with confidence intervals, shared gamma frailty by cluster (`frailty="gamma"`) with frailty-variance LR testing, and model diagnostics (residuals, proportional-hazards test, concordance).
- **Accelerated failure time** (`AFT`): Parametric models (Weibull, exponential, log-normal, log-logistic) with survival prediction confidence intervals, validated against R's `survreg`.
- **Competing risks**: Cumulative incidence functions (`AalenJohansen`), subdistribution hazards (`FineGray`), and multi-state transition probabilities.

Model performance:

- **Prediction metrics**: Concordance index (Harrell's C) and inverse-probability censoring weighted (IPCW) Brier score / integrated Brier score.
- **Cross-validation** (`cross_validate()`): K-fold cross-validation with stratification support for balanced outcome distributions.

Tidy & reproducible:

- **Tidy layer** (`greenwood.tidy`): Broom-compatible summaries aligned with Great Summaries for consistent reporting.
- **Built-in datasets** (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and **R-parity test harness**: Every statistic is validated to tolerance against R's `survival` package.

## Get started

Here's a simple example that loads survival data, estimates a survival curve, and visualizes it.

```python
import greenwood as gw

# Load the data and represent it as a survival object
lung = gw.load_dataset("lung", backend="polars")
y = gw.Surv.right(lung["time"], event=(lung["status"] == 2))

# Estimate the Kaplan-Meier survival curve
km = gw.KaplanMeier().fit(y)

# Visualize it
gw.plot_survival(km)

# Fit a Cox proportional hazards model
cox = gw.CoxPH().fit(y, lung[["age", "sex"]])
```

That's it! See the [user guide](user_guide/02-quick-start.qmd) for more details on each step, and scroll down for a comprehensive example covering more of Greenwood's capabilities.

## See more

Here's a comprehensive example showcasing more of Greenwood's capabilities:

```python
import greenwood as gw

df = gw.load_dataset("lung", backend="polars")
y = gw.Surv.right(df["time"], event=(df["status"] == 2))

# Kaplan-Meier with stratification and detailed summaries
km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
km.to_frame(format="polars")  # tidy: strata, time, n_risk, n_event, estimate, conf_low, conf_high
km.median(ci=True)         # median survival with confidence limits, per stratum
km.rmst(365, ci=True)      # restricted mean survival time up to 365 days
km.predict([180, 365])     # survival probability at specific times

# Statistical tests
gw.logrank_test(y, group=df["sex"])          # standard log-rank test
gw.logrank_test(y, group=df["sex"], rho=1)   # Peto-Peto (G-rho) test

# Visualization with risk tables
gw.plot_survival(km, risk_table=True)

# Cox proportional hazards regression
cox = gw.CoxPH().fit(y, df[["age", "sex"]])
gw.tidy(cox, exponentiate=True, format="polars")  # hazard ratios with confidence intervals
cox.cox_zph()                                # proportional-hazards test
cox.concordance()                            # C-statistic
cox.predict(df[["age", "sex"]].head(), type="survival", times=[180, 365], format="polars")

# Parametric accelerated failure time models
aft = gw.AFT("weibull").fit(y, df[["age", "sex"]])
gw.tidy(aft, format="polars")           # coefficients on the log-time scale

# Competing risks: cumulative incidence per cause
# (mgus2 loaded with Pandas here for the Series `.where` construction below)
mg = gw.load_dataset("mgus2", backend="pandas")
etime = mg["ptime"].where(mg["pstat"] == 1, mg["futime"])
cause = mg["pstat"].where(mg["pstat"] == 1, 2 * mg["death"])
cr = gw.Surv.multistate(etime, event=cause, states=("pcm", "death"))
gw.AalenJohansen().fit(cr).to_frame(format="polars")
gw.FineGray("pcm").fit(cr, mg[["age", "sex"]]).to_frame(format="polars")

# Model performance and prediction
gw.concordance_index(y, cox.predict(type="lp"))
S = cox.predict(df[["age", "sex"]], type="survival", times=[180, 365], format="pandas").iloc[:, 1:].to_numpy().T
gw.brier_score(y, S, times=[180, 365])
```

## License

MIT (c) Richard Iannone. See the [`LICENSE`](LICENSE) file.
