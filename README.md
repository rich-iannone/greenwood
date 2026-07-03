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
- **plotnine visualization** (`plot_survival`) with confidence ribbons, censoring marks, and
  an aligned numbers-at-risk table;
- bundled datasets (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and an **R-parity test
  harness**: every statistic above is validated to tolerance against R (`survfit`,
  `survdiff`, and the `survival` restricted mean);
- the **tidy layer** (`greenwood.tidy`), broom-compatible and aligned with Great Summaries.

Cox regression, parametric models, and competing risks arrive in subsequent releases.
See [`ROADMAP.md`](ROADMAP.md).

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

# Coming in later releases:
# gw.CoxPH().fit("Surv(time, status) ~ age + sex", data=df)
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
