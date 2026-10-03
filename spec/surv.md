# `Surv` contract

Greenwood's `Surv()` mirrors R's `survival::Surv()` exactly. R is the reference implementation:
the same arguments, the same type inference, the same status coding, the same columns, and the
same errors and warnings. A future R Greenwood uses `survival::Surv()` directly, so this is what
keeps the two languages in step. `as_surv()` (see [`event_time.md`](event_time.md)) builds its
result through `Surv()`.

| | |
|---|---|
| Reference | `survival::Surv()` (fixtures generated with the version in `conformance/surv/metadata.json`) |
| Fixtures | [`conformance/surv/`](conformance/surv/) |
| Generator | [`scripts/generate_surv_conformance.R`](../scripts/generate_surv_conformance.R) |
| Python harness | [`tests/test_surv_conformance.py`](../tests/test_surv_conformance.py) |

Regenerate from the repo root, then review the diff in `spec/conformance/surv/`:

```bash
Rscript scripts/generate_surv_conformance.R
```

## Signature

| R | Python |
|---|---|
| `Surv(time, time2, event, type, origin = 0)` | `gw.Surv(time, time2=None, event=None, *, type=None, origin=0.0)` |

A missing R argument is `None` in Python. As in R, two arguments without `type` mean
`(time, event)`, so `Surv(t, d)` works positionally.

## Type inference

| Arguments | `type` | Result type |
|---|---|---|
| `time` only | default | `right`, every row an event |
| `time`, `event` | default | `right` |
| `time`, `time2`, `event` | default | `counting` |
| a categorical `event` | default | `mright` / `mcounting` (first level is censoring) |
| `time`, `event` | `"left"` | `left` (status `1` exact, `0` left-censored) |
| `time`, `time2`, `event` | `"interval"` | `interval` (codes `0` right, `1` exact, `2` left, `3` interval) |
| `time`, `time2` | `"interval2"` | `interval` (missing or infinite bounds are open) |
| any | `"mstate"` | `mright` / `mcounting`, with levels sorted from the values |

## Status coding

For right, left, and counting data, status may be logical, `0`/`1`, or `1`/`2`. When the largest
non-missing value is `2`, R subtracts one, so `1`/`2` coding becomes `0`/`1`. Any other value
becomes missing with the warning "Invalid status value, converted to NA". This also means a
`0`/`1`/`2` column is not a valid status: write `status == 2` instead.

## Columns

`to_frame()` returns R's matrix columns: `time`, `status` (right, left, multi-state right),
`start`, `stop`, `status` (counting), and `time1`, `time2`, `status` (interval, where `time2` is
`1` for rows that aren't interval-censored, as in R). Status is a float with `nan` for missing
values, as R stores it as a double.

## Missing values

`Surv()` keeps missing values: missing inputs, invalid status codes, `start >= stop`, and
backwards intervals all give missing rows (the last three with R's warnings). Rows with any
missing value are dropped when a model is fitted, as R's default `na.action = na.omit` does,
together with the matching rows of the covariates, labels, and `data`. The count is reported in
`n_dropped_`.

## Not part of `Surv`

R's `Surv()` has no `data`, no event recoding, and no weights, so Greenwood's has none either.
Columns are named in a formula or an `Outcome` and resolved at `fit(..., data=)` (see
[`formula.md`](formula.md)). Recoding is a
comparison (`status == 2`). Case weights go to `fit(weights=)`.

## Python adaptations

| R | Python | Why |
|---|---|---|
| `NA` | `nan` in `Surv` columns, null in frames | |
| factor `event` | pandas `Categorical` or categorical `Series`, Polars `Categorical`/`Enum`, Arrow dictionary | Category order plays the role of factor levels. |
| `as.factor()` of characters under `type = "mstate"` | levels sorted with Python's `sorted()` | R sorts with the locale's collation, which can order case differently. |
| `difftime` times | not accepted | Python has no single equivalent. `duration()` in an `Outcome` covers date columns. |
| R's `max()` warning on an all-missing status | not raised | It comes from R itself, not from `Surv()`'s rules. |
| `Surv` matrix subsetting | `y[i]` with integers, slices, integer arrays, boolean masks | |

## Greenwood additions

`gw.first_event(endpoints, censor_at=None, start=None)` builds a multi-state response from one
`(time, event)` pair per endpoint, with the earliest observed event winning. R's `survival` has no
direct equivalent, so this helper is Greenwood's own and an R Greenwood would ship the same one.
