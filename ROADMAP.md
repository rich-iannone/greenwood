---
title: "Roadmap"
---

# Greenwood Roadmap

Greenwood is built in dependency-ordered, individually shippable steps. This is the
public capability roadmap. It lists only what hasn't shipped yet. See the API reference for
everything that has.

## Planned — Long Term

Advanced estimators for complex survival problems and specialized applications.

### Additive Hazards & Cure Models

Refinements to the existing `AalenAdditive` and `MixtureCure` models.

- Constrained estimation in `AalenAdditive` that guarantees non-negative hazards and proper
survival functions (predicted cumulative hazards are currently clipped at zero)
- Non-parametric maximum likelihood estimation (NPMLE) for cure fractions
- Goodness-of-fit tests and model comparison for cure models

### Advanced Competing Risks & Multi-State

Extended methods for cause-specific and multi-state analyses.

- Variance estimation for multi-state transition probabilities
- Pseudo-observation approach for CIF and multi-state occupancy regression
- Custom estimands via pseudo-observations framework

### Frailty Models

Random-effects Cox models for correlated survival data.

- Conditional and marginal predictions for known and new clusters

### Advanced Performance Metrics

Discrimination and calibration assessment beyond point-in-time.

- Integrated discrimination improvement (IDI) and net reclassification improvement (NRI)
- Time-dependent Brier score refinements and sensitivity analyses

### R survival Coverage

Features of R's **survival** package that Greenwood doesn't have yet.

- Expected and relative survival against population rate tables (`survexp()`)
- Person-years tables (`pyears()`)
- Building time-varying covariate data from event and covariate tables (`tmerge()`), beyond
what `split_episodes()` covers
- Time-transform terms in Cox models (`tt()`)
- Consistency checks for multi-state and counting-process data (`survcheck()`)

### Platform & Interop

Performance and ecosystem integration toward 1.0.

- Full backend matrix algebra with accelerated kernels (JAX/Numba) for ultra-large datasets
(100k+ rows), extending the Numba kernel the survival forests already use
- Finalized extension protocols and Narwhals dataframe backend completeness
- Full interoperability with Great Summaries (`tbl_survfit`, `tbl_regression`)
- Migration guides for users transitioning from R's **survival** package

---

## Feedback & Contributions

Have ideas for features not listed here? Open an issue with the `enhancement` label! Contributions
to any planned item are welcome so check existing issues first to avoid duplication.

_This roadmap is a living document. It is updated as features are added and new priorities emerge._
