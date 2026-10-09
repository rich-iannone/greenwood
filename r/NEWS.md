# greenwood (development version)

* First development version: `Surv()` and `kaplan_meier()`.

* Every table from `kaplan_meier()` starts with a `strata` column (`"all"` for a single curve,
  `"sex=1"` and so on for groups), so the layout is the same for one curve or several.

* `kaplan_meier()` gains `robust` and `cluster` for robust (infinitesimal jackknife) standard
  errors.
