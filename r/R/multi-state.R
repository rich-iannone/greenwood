# Multi-state models: multi_state() and its methods. The numeric fixture (multistate_mgus2) is
# replayed by tests/testthat/test-multi-state.R.

#' State occupancy in a multi-state model
#'
#' `multi_state()` estimates the probability of being in each state of a multi-state process over
#' time with the Aalen-Johansen estimator. At each distinct time \eqn{t}, the transition from state
#' \eqn{j} to state \eqn{k} has the estimated increment \eqn{d_{jk}(t) / Y_j(t)}, the number of
#' such transitions over the number in state \eqn{j} at risk. The transition probability matrix is
#' the product of the matrices \eqn{I + dA(t)} over the times up to \eqn{t}, and the occupancy
#' probabilities are the starting distribution times that product. They add up to 1 at every
#' time.
#'
#' The data are intervals `(start, stop]`, one row per stretch of time a subject spends in one
#' state. `istate` gives that state, and the event of the [Surv()] response the state entered at
#' `stop` (or censoring). A subject who moves through several states has one row per state. The
#' starting distribution is the proportion of the rows that begin at the earliest start time in
#' each state.
#'
#' Competing risks are the special case with one starting state and one absorbing state per
#' cause. Then `istate` can be left out: every subject starts in the state `"(s0)"`, its
#' occupancy is the Kaplan-Meier estimate of being free of any event, and the occupancy of each
#' other state is the cumulative incidence of [aalen_johansen()].
#'
#' @param formula A formula with a multi-state [Surv()] response on the left and either `1` or the
#'   grouping variables on the right. The response is `Surv(start, stop, event)`, or
#'   `Surv(time, event)` for competing risks, with `event` a factor whose first level means
#'   censored and whose other levels are the states entered.
#' @param data A data frame holding the variables in `formula` and `istate`.
#' @param istate The state each row's interval is spent in, a column of `data` or a vector. The
#'   default puts every row in the state `"(s0)"`.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#'
#' @return A `multi_state` object. [as.data.frame()] gives the occupancy probabilities with one row
#'   per group, state, and distinct stop time: `strata`, the grouping variables, `state`, `time`,
#'   and `estimate`. [predict()] reads them at given times.
#'
#' @examples
#' # An illness-death process: healthy -> ill -> dead, or healthy -> dead
#' intervals <- data.frame(
#'   id = c(1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8),
#'   start = c(0, 20, 0, 0, 35, 0, 0, 0, 50, 0, 0),
#'   stop = c(20, 60, 45, 35, 90, 70, 30, 50, 80, 100, 65),
#'   from = c("healthy", "ill", "healthy", "healthy", "ill", "healthy", "healthy", "healthy",
#'     "ill", "healthy", "healthy"),
#'   to = c("ill", "dead", "dead", "ill", "censor", "censor", "dead", "ill", "dead", "censor",
#'     "dead")
#' )
#' intervals$to <- factor(intervals$to, levels = c("censor", "ill", "dead"))
#'
#' fit <- multi_state(Surv(start, stop, to) ~ 1, data = intervals, istate = from)
#' fit
#' predict(fit, times = c(25, 50, 75))
#' @export
multi_state <- function(formula, data, istate, subset, na.action = stats::na.omit) {
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`multi_state()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  type <- attr(frame$response, "type")
  if (!type %in% c("mright", "mcounting")) {
    stop(
      paste0(
        "`multi_state()` needs a multi-state response, `Surv(start, stop, event)` or ",
        "`Surv(time, event)` with `event` a factor whose first level is censoring."
      ),
      call. = FALSE
    )
  }
  parts <- surv_parts(frame$response, "multi_state()", c("mright", "mcounting"))
  n <- length(parts$stop)
  start <- if (type == "mright") rep(0, n) else unname(parts$start)
  if (type == "mright" && any(parts$stop <= 0)) {
    stop("`multi_state()` needs positive times when there are no start times.", call. = FALSE)
  }
  entered <- attr(frame$response, "states")
  from <- frame$istate
  if (is.null(from)) {
    from_names <- "(s0)"
    from <- rep("(s0)", n)
  } else {
    from_names <- if (is.factor(from)) levels(droplevels(from)) else sort(unique(as.character(from)))
    from <- as.character(from)
  }
  states <- unique(c(from_names, entered))
  from <- match(from, states)
  to <- ifelse(parts$event > 0, match(entered[pmax(parts$event, 1)], states), NA_integer_)

  groups <- group_keys(frame$vars, n)
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    ms_occupancy(start[rows], unname(parts$stop[rows]), from[rows], to[rows], length(states))
  })
  structure(
    list(
      curves = curves,
      groups = groups$keys,
      states = states,
      n = n,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "multi_state"
  )
}

# The Aalen-Johansen occupancy probabilities of one group: the starting distribution, the
# distinct stop times, and a times-by-states matrix of occupancy.
ms_occupancy <- function(start, stop, from, to, n_states) {
  entry <- start == min(start)
  p0 <- tabulate(from[entry], n_states) / sum(entry)
  times <- sort(unique(stop))
  at_risk <- vapply(seq_len(n_states), function(j) {
    rows <- from == j
    weight_at_risk(start[rows], stop[rows], rep(1, sum(rows)), times)
  }, numeric(length(times)))
  at_risk <- matrix(at_risk, nrow = length(times))
  moved <- which(!is.na(to) & to != from)
  at <- match(stop[moved], times)

  prob <- diag(n_states)
  occupancy <- matrix(0, length(times), n_states)
  for (i in seq_along(times)) {
    increment <- matrix(0, n_states, n_states)
    here <- moved[at == i]
    if (length(here) > 0L) {
      counts <- table(factor(from[here], seq_len(n_states)), factor(to[here], seq_len(n_states)))
      y <- at_risk[i, ]
      increment <- matrix(as.numeric(counts), n_states) / y
      increment[y <= 0, ] <- 0
      diag(increment) <- -rowSums(increment)
      prob <- prob %*% (diag(n_states) + increment)
    }
    occupancy[i, ] <- drop(p0 %*% prob)
  }
  list(p0 = p0, time = times, occupancy = occupancy)
}

ms_table <- function(object, read) {
  tables <- lapply(object$curves, function(curve) {
    do.call(rbind, lapply(seq_along(object$states), function(j) {
      out <- read(curve, j)
      cbind(data.frame(state = object$states[j], stringsAsFactors = FALSE), out)
    }))
  })
  bind_groups(object$groups, tables)
}

#' @export
as.data.frame.multi_state <- function(x, row.names = NULL, optional = FALSE, ...) {
  ms_table(x, function(curve, j) {
    data.frame(time = curve$time, estimate = curve$occupancy[, j])
  })
}

#' State occupancy at given times
#'
#' `predict()` reads the state occupancy probabilities of a [multi_state()] fit at given times.
#' Before the first stop time they're the starting distribution.
#'
#' @param object A [multi_state()] fit.
#' @param times The times to read the probabilities at.
#' @param ... Unused.
#'
#' @return A data frame with one row per group, state, and time: `strata`, the grouping variables,
#'   `state`, `time`, and `estimate`.
#'
#' @examples
#' intervals <- data.frame(
#'   start = c(0, 20, 0, 0, 35, 0, 0),
#'   stop = c(20, 60, 45, 35, 90, 70, 30),
#'   from = c("healthy", "ill", "healthy", "healthy", "ill", "healthy", "healthy"),
#'   to = c("ill", "dead", "dead", "ill", "censor", "censor", "dead")
#' )
#' intervals$to <- factor(intervals$to, levels = c("censor", "ill", "dead"))
#' fit <- multi_state(Surv(start, stop, to) ~ 1, data = intervals, istate = from)
#' predict(fit, times = c(10, 40, 80))
#' @name multi_state_predict
#' @export
predict.multi_state <- function(object, times, ...) {
  times <- check_times(times)
  ms_table(object, function(curve, j) {
    data.frame(time = times, estimate = step_at(curve$time, curve$occupancy[, j], times,
      curve$p0[j]
    ))
  })
}

#' @export
print.multi_state <- function(x, digits = getOption("digits"), ...) {
  cat("<multi_state> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf("States: %s. n = %d\n\n", paste(x$states, collapse = ", "), x$n))
  table <- ms_table(x, function(curve, j) {
    last <- length(curve$time)
    data.frame(time = curve$time[last], estimate = curve$occupancy[last, j])
  })
  if (nrow(x$groups) == 1L) {
    table$strata <- NULL
  }
  cat("Occupancy at the last time:\n")
  print(table, digits = max(3L, digits - 3L), row.names = FALSE)
  cat_dropped(x$n_dropped)
  invisible(x)
}
