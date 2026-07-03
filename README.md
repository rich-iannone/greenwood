# Greenwood

**Modern survival analysis for Python: narwhals-native, R-validated, plotnine-visualized.**

Greenwood is a time-to-event modeling library built for 2026: it works on pandas, Polars,
PyArrow, and anything else [narwhals](https://narwhals-dev.github.io/narwhals/) supports;
it is validated to tolerance against R's `survival` package; it visualizes with
[plotnine](https://plotnine.org/); and it plugs into the Great Tables ecosystem for
publication-quality tables. Named after Greenwood's formula, the classic variance
estimator for the Kaplan-Meier curve.

## Status

This release ships:

- the **`Surv` response object** (right / left / interval censoring, counting-process form,
  left truncation, weights, multi-state endpoints) with eager validation;
- the narwhals-native **risk-set and event-table kernel** shared by Kaplan-Meier, the
  log-rank test, and Cox;
- the **`KaplanMeier`** estimator (survival function, Greenwood confidence intervals with
  `plain`/`log`/`log-log` transforms, median and quantiles, **restricted mean survival
  time**, step-function prediction) and the **`NelsonAalen`** cumulative-hazard estimator;
- the **`logrank_test`**, covering the standard log-rank test and the G-rho
  (Fleming-Harrington) family for two or more groups;
- **`CoxPH`** proportional hazards regression (Efron and Breslow ties, hazard ratios,
  model-based standard errors, Wald z-tests, and the likelihood-ratio / Wald / score global
  tests), with **stratification**, the **robust (Lin-Wei) sandwich variance** and clustering,
  **baseline hazard and survival prediction**, martingale and Schoenfeld residuals, the
  **Grambsch-Therneau proportional-hazards test** (`cox_zph`), and the concordance index;
- **`AFT`** parametric accelerated failure time models (Weibull, exponential, log-normal,
  log-logistic), validated against R's `survreg`;
- **`AalenJohansen`** cumulative incidence functions for competing risks (per-cause CIFs
  with delta-method standard errors), validated against R's `survfit`;
- **plotnine visualization** (`plot_survival`) with confidence ribbons, censoring marks, and
  an aligned numbers-at-risk table;
- bundled datasets (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and an **R-parity test
  harness**: every statistic above is validated to tolerance against R (`survfit`,
  `survdiff`, `coxph`, and the `survival` restricted mean);
- the **tidy layer** (`greenwood.tidy`), broom-compatible and aligned with Great Summaries.

The Fine-Gray subdistribution model and Gray's test, multi-state models,
flexible-parametric (spline) models, and further Cox extensions arrive in subsequent
releases. See [`ROADMAP.md`](ROADMAP.md).

## Quick look

```python
import greenwood as gw
from greenwood import Surv

df = gw.data.load_dataset("lung")                    # survival::lung (1 = censored, 2 = dead)
y = Surv.right(df["time"], event=(df["status"] == 2))

km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
km.to_dataframe()          # tidy: strata, time, n_risk, n_event, estimate, conf_low, conf_high
km.median(ci=True)         # median survival with confidence limits, per stratum
km.rmst(365, ci=True)      # restricted mean survival time up to 365 days
km.predict([180, 365])     # survival probability at specific times

gw.logrank_test(y, group=df["sex"])          # standard log-rank test
gw.logrank_test(y, group=df["sex"], rho=1)   # Peto-Peto (G-rho) test

gw.plot_survival(km, risk_table=True)        # plotnine curves + aligned risk table

cox = gw.CoxPH().fit(y, df[["age", "sex"]])  # Cox proportional hazards
gw.tidy.tidy(cox, exponentiate=True)         # hazard ratios with confidence intervals
cox.cox_zph()                                # proportional-hazards test
cox.concordance()                            # C-statistic
cox.predict(df[["age", "sex"]].head(), type="survival", times=[180, 365])

aft = gw.AFT("weibull").fit(y, df[["age", "sex"]])   # parametric AFT model
gw.tidy.tidy(aft)                                     # coefficients on the log-time scale

# Competing risks: cumulative incidence per cause.
mg = gw.data.load_dataset("mgus2")
etime = mg["ptime"].where(mg["pstat"] == 1, mg["futime"])
cause = mg["pstat"].where(mg["pstat"] == 1, 2 * mg["death"])   # 0 censor, 1 pcm, 2 death
cr = Surv.multistate(etime, event=cause, states=("pcm", "death"))
gw.AalenJohansen().fit(cr).to_dataframe()
```

## Development

```bash
make install   # pip install -e ".[dev]"
make check     # ruff + pyright + pytest
make docs      # build the Great Docs site locally
```

The version is derived from git tags by `setuptools_scm`; no version string is committed.

## License

MIT (c) Richard Iannone. See [`LICENSE`](LICENSE).
