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
  `plain`/`log`/`log-log` transforms, median and quantiles, step-function prediction) and
  the **`NelsonAalen`** cumulative-hazard estimator, both validated to tolerance against R;
- bundled datasets (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and an **R-parity test
  harness** validating against R `survfit`;
- the **tidy layer** (`greenwood.tidy`), broom-compatible and aligned with Great Summaries.

Group tests, Cox regression, parametric models, competing risks, and visualization arrive
in subsequent releases. See [`ROADMAP.md`](ROADMAP.md).

## Quick look

```python
import greenwood as gw
from greenwood import Surv

df = gw.data.load_dataset("lung")                    # survival::lung (1 = censored, 2 = dead)
y = Surv.right(df["time"], event=(df["status"] == 2))

km = gw.KaplanMeier(conf_type="log-log").fit(y, by=df["sex"])
km.to_dataframe()          # tidy: strata, time, n_risk, n_event, estimate, conf_low, conf_high
km.median(ci=True)         # median survival with confidence limits, per stratum
km.predict([180, 365])     # survival probability at specific times
gw.tidy.glance(km)         # one-row-per-stratum summary

# Coming in later releases:
# gw.logrank_test(y, group=df["sex"])
# gw.CoxPH().fit("Surv(time, status) ~ age + sex", data=df)
# gw.plot_survival(km, risk_table=True)
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
