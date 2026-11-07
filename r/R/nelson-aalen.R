# Nelson-Aalen estimation of the cumulative hazard. The numeric fixtures are the cumhaz and
# cumhaz_var fields of fixtures/r/km_*, replayed by tests/testthat/test-nelson-aalen.R.

#' Nelson-Aalen estimate of the cumulative hazard
#'
#' `nelson_aalen()` estimates the cumulative hazard function from censored event times. It gives
#' one curve for the whole sample, or one curve per group when the right-hand side of the formula
#' names grouping variables.
#'
#' The estimate at time \eqn{t} is the sum of \eqn{d_i / n_i} over the event times
#' \eqn{t_i \le t}, where \eqn{n_i} is the number at risk and \eqn{d_i} the number of events. Its
#' variance is Aalen's estimator, the sum of \eqn{d_i / n_i^2}. Confidence intervals are built on
#' the log scale by default, or on the plain (identity) scale with the lower limit floored at 0.
#' The survival function can be recovered as \eqn{\exp(-H(t))}, though [kaplan_meier()] is the
#' usual estimate of survival itself.
#'
#' Right-censored data and counting-process data are supported. With counting-process data,
#' `Surv(start, stop, event)`, a subject is at risk only after its start time.
#'
#' @inheritParams kaplan_meier
#' @param conf_type The scale for confidence intervals: `"log"` (the default) or `"plain"`.
#'
#' @return A `nelson_aalen` object. Use [as.data.frame()] for the estimate at every time and
#'   [predict()] for the estimate at given times. Printing it shows the number of subjects, the
#'   number of events, and the largest cumulative hazard reached in each group.
#'
#'   The table from [as.data.frame()] has the columns `strata`, the grouping variables, `time`,
#'   `n_risk`, `n_event`, `estimate`, `std_error`, `conf_low`, and `conf_high`. The layout is the
#'   same for one curve (`strata` is `"all"`) as for several.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
#' )
#'
#' fit <- nelson_aalen(Surv(time, status) ~ 1, data = lung)
#' fit
#' as.data.frame(fit)
#'
#' # One curve per group, read at given times
#' by_sex <- nelson_aalen(Surv(time, status) ~ sex, data = lung)
#' predict(by_sex, times = c(200, 400))
#' @export
nelson_aalen <- function(formula,
                         data,
                         weights,
                         subset,
                         na.action = stats::na.omit,
                         conf_type = c("log", "plain"),
                         conf_level = 0.95) {
  conf_type <- match.arg(conf_type)
  check_conf_level(conf_level)

  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  parts <- surv_parts(frame$response, "nelson_aalen()")
  w <- frame$weights

  groups <- group_keys(frame$vars, nrow(frame$mf))
  z <- z_value(conf_level)
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    na_curve(parts$start[rows], parts$stop[rows], parts$event[rows], w[rows], conf_type, z)
  })

  structure(
    list(
      curves = curves,
      groups = groups$keys,
      conf_type = conf_type,
      conf_level = conf_level,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "nelson_aalen"
  )
}

na_curve <- function(start, stop, event, w, conf_type, z) {
  counts <- count_events(start, stop, event, w)
  n_risk <- counts$n_risk
  n_event <- counts$n_event
  estimate <- cumsum(ifelse(n_risk > 0, n_event / n_risk, 0))
  std_error <- sqrt(cumsum(ifelse(n_risk > 0, n_event / n_risk^2, 0)))

  if (conf_type == "plain") {
    low <- estimate - z * std_error
    high <- estimate + z * std_error
  } else {
    factor <- ifelse(estimate > 0, exp(z * std_error / estimate), 1)
    low <- estimate / factor
    high <- estimate * factor
  }
  data.frame(
    time = counts$time,
    n_risk = n_risk,
    n_event = n_event,
    estimate = estimate,
    std_error = std_error,
    conf_low = pmax(low, 0),
    conf_high = high
  )
}

#' @export
as.data.frame.nelson_aalen <- function(x, ...) {
  bind_groups(x$groups, x$curves)
}

#' Nelson-Aalen estimates at given times
#'
#' `predict()` reads the cumulative hazard of each group at the given times, with its standard
#' error and confidence limits. The curves are right-continuous step functions, so the value at
#' time \eqn{t} is the last step at a time \eqn{\le t}, and 0 before the first step.
#'
#' @param object A [nelson_aalen()] fit.
#' @param times One or more times.
#' @param ... Unused.
#'
#' @return A data frame with one row per group and time: `strata`, the grouping variables, `time`,
#'   `estimate`, `std_error`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2)
#' )
#' fit <- nelson_aalen(Surv(time, status) ~ 1, data = lung)
#' predict(fit, times = c(200, 400))
#' @name nelson_aalen_predict
#' @export
predict.nelson_aalen <- function(object, times, ...) {
  times <- check_times(times)
  tables <- lapply(object$curves, function(curve) {
    data.frame(
      time = times,
      estimate = step_at(curve$time, curve$estimate, times, 0),
      std_error = step_at(curve$time, curve$std_error, times, 0),
      conf_low = step_at(curve$time, curve$conf_low, times, 0),
      conf_high = step_at(curve$time, curve$conf_high, times, 0)
    )
  })
  bind_groups(object$groups, tables)
}

#' @export
print.nelson_aalen <- function(x, digits = getOption("digits"), ...) {
  cat("<nelson_aalen> ", deparse_formula(x$formula), "\n\n", sep = "")
  table <- data.frame(
    n = vapply(x$curves, function(curve) curve$n_risk[1L], numeric(1)),
    events = vapply(x$curves, function(curve) sum(curve$n_event), numeric(1)),
    max_cumhaz = vapply(x$curves, function(curve) curve$estimate[nrow(curve)], numeric(1))
  )
  rownames(table) <- x$groups$strata
  print(table, digits = digits)
  cat_dropped(x$n_dropped)
  invisible(x)
}
