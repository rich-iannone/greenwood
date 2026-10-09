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

## Status

| Estimator | R | Python |
|---|---|---|
| Kaplan-Meier | follows these rules | follows these rules |
| Nelson-Aalen | not implemented | follows these rules |
| Turnbull | not implemented | follows these rules |
| Aalen-Johansen | not implemented | follows these rules |
| Event table | not implemented | follows these rules (`event_table()`) |

Group comparisons (log-rank tests, RMST comparisons) and Cox stratification report the groups they
compare rather than one curve per group, so this layout doesn't apply to them.

## Known differences

- Group order: R sorts curves by the grouping variables (factor levels, otherwise sorted values).
  Python keeps the order of first appearance in the data. Layout and labels match either way.
- R keeps each grouping variable as its own column after `strata`. Python has only `strata`.
