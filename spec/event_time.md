# `event_time` contract

Greenwood's event-time response follows Max Kuhn's [etd](https://github.com/topepo/etd) package
exactly. etd is the reference implementation. This document records which version we target, the
contract in a language-neutral form, and where Python must adapt. The companion contract for the
`Surv` response that `as_surv()` produces is [`surv.md`](surv.md).

## Reference implementation

| | |
|---|---|
| Package | etd `0.0.0.9000` (experimental) |
| Pinned commit | `3d5caff4f03d771a6b65b9605c9b0e48a912482f` |
| Fixtures | [`conformance/event_time/`](conformance/event_time/) |
| Generator | [`scripts/generate_event_time_conformance.R`](../scripts/generate_event_time_conformance.R) |
| Python harness | [`python/tests/test_event_time_conformance.py`](../python/tests/test_event_time_conformance.py) |
| R harness | [`r/tests/testthat/test-event-time-conformance.R`](../r/tests/testthat/test-event-time-conformance.R) |

To regenerate after etd moves, update the pinned commit in all three places (the generator, the
Python harness, and this table), then run from the repo root:

```bash
Rscript -e 'pak::pak("topepo/etd@3d5caff4f03d771a6b65b9605c9b0e48a912482f")'
Rscript scripts/generate_event_time_conformance.R
```

Review the diff in `spec/conformance/event_time/` before accepting it. Any change in a recorded
outcome is a change in the contract.

## API surface

| etd (R) | Greenwood (Python) | Greenwood (R) |
|---|---|---|
| `event_time(time, status, time_max = NULL)` | `gw.event_time(time, status, time_max=None)` | `event_time(time, status, time_max = NULL)` |
| `new_event_time(time = list(), status = character())` | `gw.new_event_time(time=(), status=())` | `new_event_time(time = list(), status = character())` |
| `extract_time(x)` | `gw.extract_time(x)` | `extract_time(x)` |
| `extract_status(x)` | `gw.extract_status(x)` | `extract_status(x)` |
| `as_surv(x)` | `gw.as_surv(x)` | `as_surv(x)` |
| `as_tibble(x)` | `x.to_frame(format=None)` | `tibble::as_tibble(x)` |
| class `event_time` | class `EventTime` | class `c("greenwood_event_time", "event_time")` |
| `length(x)`, `x[i]`, `print(x)`, `format(x)`, `is.na(x)` | `len(x)`, `x[i]`, `repr(x)`, `x.format()`, `x.is_na()` | the same base generics |

The rule: names from etd's own exports stay identical, and R base generics map to the Python
protocol or method that plays the same role.

## Status codes

| Code | Meaning | `time` | `time_max` | `format()` |
|---|---|---|---|---|
| `"e"` | exact event | event time | missing | `"7 "` (trailing space) |
| `"r"` | right-censored | censoring time | missing | `"5+"` |
| `"l"` | left-censored | censoring time | missing | `"3-"` |
| `"i"` | interval-censored | lower bound | upper bound, strictly greater | `"[2, 4]"` |

Codes are case-sensitive single letters. No long-form aliases are accepted. Missing rows format as
missing (`NA` in R, `None` in Python).

## Validation

`event_time()` runs these checks in this order. The first failing check is reported, with every
offending location where the check has locations.

| # | Check id | Fails when | Python exception | Locations |
|---|---|---|---|---|
| 1 | `time_not_numeric` | `time` can't be cast to double | `TypeError` | no |
| 1 | `time_max_not_numeric` | `time_max` can't be cast to double | `TypeError` | no |
| 2 | `status_not_character` | `status` is not character | `TypeError` | no |
| 3 | `status_length` | `status` length differs from `time` | `ValueError` | no |
| 4 | `time_max_length` | `time_max` length differs from `time` | `ValueError` | no |
| 5 | `time_negative` | a `time` is negative (including `-Inf`) | `ValueError` | yes |
| 6 | `status_missing` | `status` is missing where `time` is present | `ValueError` | yes |
| 7 | `status_invalid_value` | a `status` is not one of `e`, `r`, `l`, `i` | `ValueError` | yes |
| 8 | `time_max_not_allowed` | `time_max` is present on a non-interval row | `ValueError` | yes |
| 9 | `time_max_missing` | `time_max` is missing on an interval row | `ValueError` | yes |
| 10 | `interval_bounds_order` | `time >= time_max` on an interval row | `ValueError` | yes |

`time` and `time_max` are cast before `status` is checked, so check 1 covers both (`time` first).
Allowed casts follow vctrs: double, integer, and logical succeed, and anything else fails.

`new_event_time()` only checks types:

| Check id | Fails when | Python exception |
|---|---|---|
| `time_not_list` | `time` is not a list | `TypeError` |
| `status_not_character` | `status` is not character | `TypeError` |
| `fields_size` | `time` and `status` differ in length | `ValueError` |

`as_surv()` on anything other than an event-time vector fails with `as_surv_unsupported`
(`TypeError`).

In Python every error carries `.check` (the id above) and `.locations` (0-based positions, empty
when the check has none). Messages reuse etd's wording without cli markup.

## Conversion to `Surv`

`as_surv()` picks the `Surv` type from the codes present, ignoring missing rows:

- only `e` and `r`: `type = "right"`, with status `1` for `e` and `0` for `r`.
- only `e` and `l`: `type = "left"`, with status `1` for `e` and `0` for `l`.
- anything else: `type = "interval"`, with status `0` = `r`, `1` = `e`, `2` = `l`, `3` = `i`.
- zero-length input: an empty right-censored `Surv`.

Missing rows stay missing in the result. The resulting `Surv` columns use R's names: `time` and
`status` for right and left, `time1`, `time2`, and `status` for interval. Greenwood's `Surv` mirrors R's (see [`surv.md`](surv.md)), so the conversion matches column for
column.

## Python adaptations

Where Python cannot match R literally, it matches the meaning:

| R / etd | Python | Why |
|---|---|---|
| `NA_real_` in `time`, `time_max` | `nan`, and null in frames | Python has no typed NA for floats. |
| `NA_character_` in `status` | `None` | |
| 1-based locations | 0-based locations | Users index their data with the number. |
| logical NA vs character NA | both `None` | A status of `[None]` is a missing character status in Python. |
| atomic vector vs list (`new_event_time`) | a sequence of floats or 2-tuples | Python sequences don't make R's distinction, so `time_not_list` applies only to non-sequences. |
| factor `status` (rejected) | categorical `status` (pandas, Polars, Arrow dictionary) is rejected with `status_not_character` | Same meaning: categorical data isn't character data. |
| scalar inputs | a bare number or string is a length-1 vector | R has no scalars, so `event_time(3, "e")` is the usual R spelling. |
| `call =` argument | omitted | Python tracebacks identify the caller. |
| `print()` with `[1]` prefixes | `repr()` with the `<event_time[n]>` header and the same tokens | `format()` itself matches exactly. |
| pillar / tibble column display | not ported | |

The conformance harness skips the cases that exercise the R-only distinctions, naming the reason.

## R adaptations

The R package reimplements etd in base R, with no vctrs, rlang, or cli dependency. The vector is
a list of the two fields (`time`, `status`) with base S3 methods. Methods dispatch on the
`greenwood_event_time` subclass, so loading etd alongside Greenwood replaces neither package's
methods. Where it departs from etd:

| R / etd | Greenwood (R) | Why |
|---|---|---|
| cli messages with `x` and `i` bullets | the same sentences on one line, without bullets | No cli dependency. The harness strips the bullets before comparing. |
| unclassed errors | the class `greenwood_event_time_error`, with `check` (the id above) and `locations` (1-based) fields | Upstream issue 5. |
| `format()` of an empty vector is an error | `character(0)` | Upstream issue 1. |
| `x[NA_integer_]` gives a broken element | a missing element, as for any other vector | Upstream issue 2. |
| `sort()` and `order()` keep the input order | an error that points to `extract_time()` | Upstream issue 3. |
| `vctrs::vec_c(x, y)` | not available. `c(x, y)` works | The class is not a vctrs class. |
| `as_surv()` returns a survival `Surv` | Greenwood's own `Surv()`. `as_surv()` of a `Surv` returns it unchanged | No runtime dependency on survival. |
| `as_tibble()` | an S3 method for `tibble::as_tibble()`, registered when tibble is installed | tibble is suggested, not imported. |

An `event_time()` call on the left-hand side of a model formula is converted with `as_surv()`, so
every estimator takes it where it takes the matching `Surv()` response.

## Vector behaviour

`conformance/event_time/vector_ops.json` records what base R and vctrs do with an `event_time`
vector at the pinned commit. The R harness replays every operation except the four the R
adaptations above change (`vec_c`, `x[NA_integer_]`, `sort`, `order`). Python mirrors the operations that have a natural Python spelling and
leaves the rest out, rather than inventing APIs etd doesn't have:

| Operation | etd behaviour | Python |
|---|---|---|
| `length`, `is.na` | work. `is.na` is true when any time component is missing | `len(x)`, `x.is_na()` |
| `anyNA` | works | `x.is_na().any()` |
| `x[i]` with ranges, scalars, logical masks | work, returning `event_time` | `x[i]` with slices, integers, integer arrays, boolean masks, always returning `EventTime` |
| `x[-i]` (drop elements) | works | not mirrored. Negative integers follow Python and count from the end |
| `x[i]` past the end | error | `IndexError` |
| `x[NA]` | creates a broken element (upstream issue 2) | not applicable |
| `x[[i]]` | works, returning a length-1 `event_time` | `x[i]` with an integer |
| `c(x, y)`, `vec_c(x, y)` | work. Combining with a double is an error | `EventTime.concat([x, y])`. Combining with anything else is a `TypeError` |
| `==`, `!=` | elementwise, missing when either side is missing | whole-vector equality returning one `bool`, as for Python containers. `EventTime` is unhashable |
| `rev`, `head` | work | `x[::-1]`, `x[:n]` |
| `rep`, `unique`, `duplicated` | work | not mirrored |
| `sort`, `order` | run but don't order by time (upstream issue 3) | not mirrored |
| `as.character`, `as.double` | error | no conversions (`str(x)` is the `repr`) |
| `z[i] <- value` | works for `event_time` values | not mirrored. `EventTime` is immutable, like other Greenwood responses |
| `summary` | not implemented | not mirrored |
| `data.frame(times = x)` | stores the vector as one column | `x.to_frame()` gives the three component columns |

## Upstream issues

Found while generating the fixtures, to raise with etd rather than mirror. The harness tracks the
first one and fails when the fixtures stop showing it, as a reminder to update this list.

1. `format()` errors on a zero-length vector (`format(NULL)` returns `"NULL"`, which breaks the
   `split()` call). `print()` of an empty vector is unaffected.
2. Subsetting with a missing index (`x[NA_integer_]`) creates an element whose stored time is
   `NULL` rather than `NA`. `is.na()` then returns `FALSE`, and `format()`, `extract_time()`,
   `as_surv()`, and `as_tibble()` all fail on it.
3. `sort()` and `order()` don't order by time. The list-valued `time` field isn't ordered
   numerically, so `sort(x)` returns the input order. Either an explicit ordering or an error
   would be clearer.
4. `new_event_time()` with an unknown status formats with a literal `NA` suffix (`"1NA"`). This is
   expected for an unvalidated constructor, but worth confirming.
5. Errors have no condition classes, so tests in any language have to match message text.
   Classed conditions (for example `etd_error_status_invalid`) would make the check ids above
   official.
6. `extract_time()` returns a vector or a two-column matrix depending on whether any interval rows
   are present. Worth discussing a stable shape.
7. Weights: confirm they are out of scope by design. Greenwood plans to take them at `fit()`.
8. README typo: `pak::pak("topepo/etd)` is missing a closing quote.

## Fixture format

Each fixture file is a JSON array of cases. Inputs are recorded with their R type
(`{"r_type": "double", "values": [...]}`) so the replaying side can build the equivalent value.
Numeric missing values are the strings `"NA"` and `"NaN"`, infinities are `"Inf"` and `"-Inf"`,
and character or logical missing values are `null`. Locations are 1-based, as etd reports them.

| File | Contents |
|---|---|
| `metadata.json` | pinned commit and the R, etd, vctrs, and survival versions used |
| `constructor.json` | `event_time()` cases: every observation of valid input, or the failing check |
| `new_event_time.json` | `new_event_time()` cases |
| `conversion.json` | `as_surv()` on unsupported input |
| `vector_ops.json` | the vector behaviour inventory, with the R code for each operation |
| `format.json` | hand-picked and randomized `format()` cases for R's joint number formatting |
