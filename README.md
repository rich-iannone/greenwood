# Greenwood

**Modern survival analysis for Python: Narwhals-native, R-validated, plotnine-visualized.**

## What is Greenwood?

Greenwood is a Python library for survival analysis, the statistical study of time-to-event outcomes. Greenwood gives you lots of powerful features for extracting insights from incomplete data where some observations are censored (i.e., we don't know if the event happened yet).

### Why Greenwood?

- it works with your dataframe library: Pandas, Polars, PyArrow, or anything supported by [Narwhals](https://narwhals-dev.github.io/narwhals/)
- it is rigorously validated, where every statistic is tested to tolerance against R's gold-standard `survival` package
- you get beautiful visualizations that are built on [plotnine](https://plotnine.org/) (for publication-quality survival curves)
- you also get publication-ready tables through integration with the [Great Tables](https://posit-co.github.io/great-tables/) library
- batteries are included: from simple Kaplan-Meier curves to Cox proportional hazards, competing risks, and beyond

## What's included

Descriptive statistics:

- **`Surv` response object**: Handle right-, left-, and interval-censored data; counting-process form; left truncation; weights; and multi-state endpoints with built-in validation.
- **Kaplan-Meier estimation** (`KaplanMeier`): Survival curves with Greenwood confidence intervals, median/quantile survival, restricted mean survival time, and step-function predictions.
- **Nelson-Aalen estimator** (`NelsonAalen`): Cumulative hazard curves.
- **Visualization** (`plot_survival`): Publication-ready curves with confidence ribbons, censoring marks, and aligned at-risk tables.

Hypothesis testing:

- **Log-rank tests** (`logrank_test`): Standard log-rank test and the G-rho (Fleming-Harrington) family for 2+ groups.

Regression models:

- **Cox proportional hazards** (`CoxPH`): Model covariates as hazard ratios with stratification, robust sandwich variance, clustering, baseline hazard prediction, and model diagnostics (residuals, proportional-hazards test, concordance).
- **Accelerated failure time** (`AFT`): Parametric models (Weibull, exponential, log-normal, log-logistic) validated against R's `survreg`.
- **Competing risks**: Cumulative incidence functions (`AalenJohansen`), subdistribution hazards (`FineGray`), and multi-state transition probabilities.

Model performance:

- **Prediction metrics**: Concordance index (Harrell's C) and inverse-probability censoring weighted (IPCW) Brier score / integrated Brier score.

Tidy & reproducible:

- **Tidy layer** (`greenwood.tidy`): Broom-compatible summaries aligned with Great Summaries for consistent reporting.
- **Built-in datasets** (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and **R-parity test harness**: Every statistic is validated to tolerance against R's `survival` package.

## Get started

Here's a simple example that loads survival data, estimates a survival curve, and visualizes it.

```python
import greenwood as gw

# Load the data and represent it as a survival object
lung = gw.load_dataset("lung")
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

df = gw.load_dataset("lung")
y = gw.Surv.right(df["time"], event=(df["status"] == 2))

# Kaplan-Meier with stratification and detailed summaries
km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
km.to_dataframe()          # tidy: strata, time, n_risk, n_event, estimate, conf_low, conf_high
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
gw.tidy.tidy(cox, exponentiate=True)         # hazard ratios with confidence intervals
cox.cox_zph()                                # proportional-hazards test
cox.concordance()                            # C-statistic
cox.predict(df[["age", "sex"]].head(), type="survival", times=[180, 365])

# Parametric accelerated failure time models
aft = gw.AFT("weibull").fit(y, df[["age", "sex"]])
gw.tidy.tidy(aft)                            # coefficients on the log-time scale

# Competing risks: cumulative incidence per cause
mg = gw.load_dataset("mgus2")
etime = mg["ptime"].where(mg["pstat"] == 1, mg["futime"])
cause = mg["pstat"].where(mg["pstat"] == 1, 2 * mg["death"])
cr = Surv.multistate(etime, event=cause, states=("pcm", "death"))
gw.AalenJohansen().fit(cr).to_dataframe()
gw.FineGray("pcm").fit(cr, mg[["age", "sex"]]).to_dataframe()

# Model performance and prediction
gw.concordance_index(y, cox.predict(type="lp"))
S = cox.predict(df[["age", "sex"]], type="survival", times=[180, 365]).iloc[:, 1:].to_numpy().T
gw.brier_score(y, S, times=[180, 365])
```

## License

MIT (c) Richard Iannone. See the [`LICENSE`](LICENSE) file.
