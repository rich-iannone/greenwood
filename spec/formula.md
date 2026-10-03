# Formula response contract

The left-hand side of a model formula describes the response by column names and expressions, and
is evaluated against the `data=` given to `fit()`. In R this is native: `coxph(Surv(time, status ==
2) ~ age, data = lung)`. Greenwood's Python side accepts the same left-hand sides as strings, and
`Outcome` is their structured form. Both are evaluated on the frame's own backend (pandas, Polars,
PyArrow, DuckDB, lazy frames) through Narwhals, so they are the portable way to start from a data
frame. `Surv()` and `event_time()` (see [`surv.md`](surv.md) and [`event_time.md`](event_time.md))
are the layer below, which takes values.

## Responses

| Formula | `Outcome` | Built with |
|---|---|---|
| `Surv(time, time2, event, type=, origin=)` | `Outcome.surv(time, time2=None, event=None, type=None, origin=0)` | `Surv()` |
| `event_time(time, status, time_max)` | `Outcome.event_time(time, status, time_max=None)` | `event_time()`, then `as_surv()` |
| (none) | `Outcome.first_event(endpoints, censor_at=None, start=None)` | `first_event()` (Greenwood's own) |

Arguments bind positionally or by name, as in R. `Surv(time, x)` without `event` reads `x` as the
status (R's rule), and an `Outcome` stores it that way, so `Outcome.from_formula("Surv(time, status
== 2)")` equals `Outcome.surv(time="time", event="status == 2")`.

## Expressions

Each argument is one of the following. They mean in Python what they mean in R.

| Expression | Example | Notes |
|---|---|---|
| column name | `time`, `ph.ecog`, `` `Survival time` `` | Backticks quote unusual names in a formula. As an `Outcome` argument, a string that isn't an expression is a column name. |
| comparison | `status == 2`, `status != 0` | Missing compared with anything is missing. |
| membership | `status %in% c(1, 2)`, `status in (1, 2)`, `status not in (0, 9)` | As in R, `NA %in% x` is `FALSE`. |
| `factor(x, levels, labels)` | `factor(cause, 0:2, c("censor", "pcm", "death"))` | Gives a multi-state status. The first label means censored. Repeated labels merge levels. Values not in `levels` are missing. |
| `duration(start, end, unit = "days")` | `duration(enroll, exit, unit = "years")` | Greenwood's own: elapsed time between two date columns. |

`c(...)` and integer ranges such as `0:2` are R's vector literals. Nothing else is evaluated: the
Python side parses with `ast` and rejects any other call or operator.

## Evaluation and missing rows

At `fit()`, every column the response and the covariates use is read from `data`, rows with a
missing value in any of them are dropped once (R's `na.omit`), the expressions are evaluated, and
the response is built. Rows the response itself makes missing (invalid status codes, backwards
intervals) are dropped next. The total is reported as `n_dropped_`. `Outcome.bind(data)` builds the
response without dropping anything, as `Surv()` does.

## Not part of the response

Case weights are an argument of `fit()` (`weights=`, a column name or values), not of the response.
