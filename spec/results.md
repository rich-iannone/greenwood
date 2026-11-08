# Result structure

Every estimator in both packages follows these rules, so code that extracts, tabulates, or plots
results never has to branch on how many curves or groups a fit holds.

## Same layout for one curve or many

A fit's tables have the same columns, in the same order and with the same types, whether it holds
one curve or several. Summary methods (medians, quantiles, restricted means, predictions) return the
same type in both cases: a data frame in R, and a data frame (chosen with `format=`) in Python.

## The `strata` column

- Every curve-level and time-level table has a `strata` column, first, holding a string label.
- A single curve is labeled `"all"`.
- Grouped curves are labeled with `name=value` pairs joined by `", "`, such as `"sex=1"` or
  `"sex=1, ph.ecog=0"`.
- The same label appears in tables, printed output, and plot legends, with no prefixes added in one
  place and not another.

## Grouping variables

In R, the grouping variables follow `strata` as their own columns, with their original types, so
results can be filtered on them without parsing labels. Which columns appear depends only on the
formula, never on the number of curves.

## Label names

The name in `name=value` is the grouping column's name (from a formula or a column-name argument).
In Python, values passed directly use the Series name if they have one, and otherwise the argument's
name (`by=1`, `group=1`). Values are written as R prints them (`1`, not `1.0`, and `TRUE`/`FALSE`).

## Group order

Groups come in the order of their values, never the order of the rows, so shuffling the data
doesn't change a result.

- A categorical variable keeps its declared levels: an R factor, a pandas `Categorical` (bare or in
  a Series), or a Polars `Enum`. Levels that don't occur in the data are skipped.
- A Polars `Categorical` and an Arrow dictionary are sorted by value, the way their own libraries
  sort them.
- Other values are sorted. Numbers sort numerically (`2` before `10`), logicals put `FALSE` first,
  and text sorts by code point (`"B"` before `"a"`), the C locale's order, in every locale. Missing
  values come last.
- With several grouping variables, every combination that occurs is a group, and the first variable
  varies slowest (`sex=1, ph.ecog=0`, `sex=1, ph.ecog=1`, `sex=2, ph.ecog=0`, ...).

The same order applies everywhere: the curves of every estimator in the table below, the strata of
a stratified Cox model, the groups of the log-rank, trend, Gray, and RMST tests, and the default
trend scores (0, 1, 2, ... in group order). Two-group differences (`rmst_test()` and the bootstrap's
`"_diff"` statistics) take the first group minus the second.

## Status

| Estimator | R | Python |
|---|---|---|
| Kaplan-Meier | follows these rules | follows these rules |
| Nelson-Aalen | follows these rules | follows these rules |
| Turnbull | follows these rules | follows these rules |
| Aalen-Johansen | follows these rules | follows these rules |
| Event table | follows these rules (`event_table()`) | follows these rules (`event_table()`) |
| Cox baseline hazard | follows these rules (`baseline_hazard()`) | follows these rules (`baseline_hazard()`) |
| Multi-state occupancy | follows these rules, one row per state and time | one row per time and one column per state, without `strata` |

Group comparisons (log-rank tests, RMST comparisons) report the groups they compare rather than one
curve per group, so this layout doesn't apply to them. A stratified Cox model's coefficients are
shared across strata, but its baseline hazard is one curve per stratum and follows these rules.

## Known differences

- R keeps each grouping variable as its own column after `strata`. Python has only `strata`.
