# Greenwood

**Modern survival analysis for Python: narwhals-native, R-validated, plotnine-visualized.**

Greenwood is a time-to-event modeling library built for 2026: it works on pandas, Polars,
PyArrow, and anything else [narwhals](https://narwhals-dev.github.io/narwhals/) supports;
it is validated to tolerance against R's `survival` package; it visualizes with
[plotnine](https://plotnine.org/); and it plugs into the Great Tables ecosystem for
publication-quality tables. Named after Greenwood's formula, the classic variance
estimator for the Kaplan-Meier curve.

## Status

This foundational release ships:

- the **`Surv` response object** (right / left / interval censoring, counting-process form,
  left truncation, weights, multi-state endpoints) with eager validation;
- the narwhals-native **risk-set and event-table kernel** shared by Kaplan-Meier, the
  log-rank test, and Cox;
- bundled datasets (`lung`, `veteran`, `ovarian`, `pbc`, `colon`) and an **R-parity test
  harness** validating the kernel against R `survfit`;
- the **tidy layer** skeleton (`greenwood.tidy`), broom-compatible and aligned with Great
  Summaries.

Estimators (`KaplanMeier`, `CoxPH`, parametric models, competing risks, ...) arrive in
subsequent phases. See [`ROADMAP.md`](ROADMAP.md).

## Quick look (target API)

```python
import greenwood as gw
from greenwood import Surv

y = Surv.right(time=df["time"], event=df["status"])

km = gw.KaplanMeier(ci="log-log").fit(y, by=df["trt"])            # (planned)
gw.logrank_test(y, group=df["trt"])                              # (planned)
cox = gw.CoxPH().fit("Surv(time, status) ~ age + trt", data=df)   # (planned)
gw.plot_survival(km, risk_table=True)                            # plotnine (planned)
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
