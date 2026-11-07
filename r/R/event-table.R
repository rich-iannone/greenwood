# The event table: risk sets and events at each observed time, the counts that the Kaplan-Meier
# and Nelson-Aalen estimates and the log-rank tests are built from.

#' Risk sets and events at each observed time
#'
#' `event_table()` tabulates the event history: at each distinct observed time, the number at
#' risk, the number of events, and the number censored. It gives one block for the whole sample,
#' or one block per group when the right-hand side of the formula names grouping variables.
#'
#' A subject is at risk at time \eqn{t} when its start time is before \eqn{t} and its stop time is
#' at or after \eqn{t}. For right-censored data there is no start time, so the number at risk at
#' \eqn{t} is the number still under follow-up at \eqn{t}. A subject censored at \eqn{t} counts as
#' at risk for the events at \eqn{t}. With weights, every count is a sum of weights.
#'
#' @inheritParams kaplan_meier
#'
#' @return A data frame with one row per group and observed time: `strata`, the grouping
#'   variables, `time`, `n_risk`, `n_event`, and `n_censor`. The layout is the same for one block
#'   (`strata` is `"all"`) as for several.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
#' )
#' event_table(Surv(time, status) ~ 1, data = lung)
#' event_table(Surv(time, status) ~ sex, data = lung)
#' @export
event_table <- function(formula, data, weights, subset, na.action = stats::na.omit) {
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  parts <- surv_parts(frame$response, "event_table()")
  w <- frame$weights
  groups <- group_keys(frame$vars, nrow(frame$mf))
  tables <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    as.data.frame(count_events(parts$start[rows], parts$stop[rows], parts$event[rows], w[rows]))
  })
  bind_groups(groups$keys, tables)
}
