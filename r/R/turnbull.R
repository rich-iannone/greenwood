# Turnbull's nonparametric maximum likelihood estimate for interval-censored data. The numeric
# fixture (turnbull_veteran_km_equivalence) is replayed by tests/testthat/test-turnbull.R.

#' Turnbull estimate of survival for interval-censored data
#'
#' `turnbull()` estimates the survival function when event times are only known to lie in a window
#' `(lower, upper]`, as with periodic follow-up visits or inspections. Exact, right-censored,
#' left-censored, and interval-censored observations can be mixed in one fit. With only exact and
#' right-censored times, the estimate is the Kaplan-Meier curve.
#'
#' The nonparametric maximum likelihood estimate puts its probability mass on the maximal
#' intersection intervals of the data. These are found by splitting the time axis at every
#' observed bound and merging neighboring pieces that every observation either fully covers or
#' fully excludes, since only their combined mass is identified (Gentleman and Geyer, 1994). The
#' masses are then found with the self-consistency (EM) algorithm, which increases the likelihood
#' at every step.
#'
#' Only the total mass on each interval is identified, not where it sits inside the interval. So
#' the curve is known exactly outside the intervals and at those that are single points, and is
#' unknown strictly inside the others. [predict()] gives `NA` there, and [quantile()] gives the
#' bounds of the interval the curve crosses in rather than a single time. [rmst()] and [rmrl()]
#' need one curve, so they place each interval's mass at its upper end. That is the latest
#' placement the data allow, which makes them upper bounds rather than unbiased estimates.
#'
#' @param formula A formula with a [Surv()] response on the left and either `1` or the grouping
#'   variables on the right. The response is interval-censored, such as
#'   `Surv(lower, upper, type = "interval2")`, or right- or left-censored.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param tol The convergence tolerance: the EM stops when no interval's mass changes by more than
#'   this. The default is `1e-9`.
#' @param max_iter The largest number of EM iterations. The default is `10000`.
#'
#' @return A `turnbull` object. [as.data.frame()] gives one row per group and maximal intersection
#'   interval: `strata`, the grouping variables, `interval_low`, `interval_high`, `prob_mass`, and
#'   `estimate`, the survival just after the interval. [predict()], [quantile()], [median()],
#'   [rmst()], and [rmrl()] read the curve.
#'
#' @references Turnbull, B. W. (1976). The empirical distribution function with arbitrarily
#'   grouped, censored and truncated data. *Journal of the Royal Statistical Society, Series B*,
#'   38, 290-295.
#'
#'   Gentleman, R. and Geyer, C. J. (1994). Maximum likelihood for interval censored data:
#'   consistency and computation. *Biometrika*, 81, 618-623.
#'
#' @examples
#' # Six subjects seen at irregular visits. NA bounds are open, so (4, NA] is right-censored.
#' visits <- data.frame(
#'   lower = c(0, 4, 7, 0, 3, 5),
#'   upper = c(4, NA, 7, 2.5, 6, 5)
#' )
#'
#' fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
#' fit
#' as.data.frame(fit)
#' @export
turnbull <- function(formula,
                     data,
                     weights,
                     subset,
                     na.action = stats::na.omit,
                     tol = 1e-9,
                     max_iter = 10000) {
  check_number(tol, "tol")
  if (tol <= 0) {
    stop("`tol` must be positive.", call. = FALSE)
  }
  check_number(max_iter, "max_iter", min = 1)

  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`turnbull()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  bounds <- tb_bounds(frame$response)
  w <- frame$weights

  groups <- group_keys(frame$vars, length(bounds$lower))
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    tb_curve(bounds$lower[rows], bounds$upper[rows], w[rows], tol, max_iter)
  })

  structure(
    list(
      curves = curves,
      groups = groups$keys,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "turnbull"
  )
}

# Each observation's window (lower, upper]. An exact time t is (t, t], a right-censored one has
# upper = Inf, and a left-censored one has lower = 0.
tb_bounds <- function(response) {
  type <- attr(response, "type")
  if (type %in% c("mright", "mcounting")) {
    stop("`turnbull()` doesn't support multi-state or competing-risks data.", call. = FALSE)
  }
  if (type == "counting") {
    stop(
      "`turnbull()` doesn't support delayed entry (counting-process data).",
      call. = FALSE
    )
  }
  y <- unclass(response)
  if (type == "interval") {
    time1 <- unname(y[, "time1"])
    code <- unname(y[, "status"])
    lower <- ifelse(code == 2, 0, time1)
    upper <- ifelse(code == 0, Inf, ifelse(code == 3, unname(y[, "time2"]), time1))
  } else if (type == "right") {
    time <- unname(y[, "time"])
    lower <- time
    upper <- ifelse(y[, "status"] == 1, time, Inf)
  } else {
    time <- unname(y[, "time"])
    lower <- ifelse(y[, "status"] == 1, time, 0)
    upper <- time
  }
  list(lower = lower, upper = upper)
}

# The maximal intersection intervals and the observations-by-intervals matrix of which interval
# lies inside which window. Right-censored observations also cover an implicit interval beyond
# every finite bound, added as the last column, so mass after the last finite bound stays
# unresolved (as in Kaplan-Meier) instead of being forced onto the last interval.
tb_support <- function(lower, upper) {
  n <- length(lower)
  bp <- sort(unique(c(lower, upper[is.finite(upper)])))
  m <- length(bp)
  if (m == 0L) {
    low <- high <- numeric(0)
    alpha <- matrix(FALSE, n, 0L)
  } else {
    # Alternating candidates: each bound as a point, and the open stretch to the next bound.
    atom_low <- c(bp[1L], as.vector(rbind(bp[-m], bp[-1L])))
    atom_high <- c(bp[1L], as.vector(rbind(bp[-1L], bp[-1L])))
    is_point <- atom_low == atom_high
    point_cover <- outer(lower, atom_low, "<") & outer(upper, atom_low, ">=")
    open_cover <- outer(lower, atom_low, "<=") & outer(upper, atom_high, ">=")
    signature <- point_cover
    signature[, !is_point] <- open_cover[, !is_point]
    # An exact time is the degenerate window (t, t], matched to its own point.
    exact <- lower == upper
    signature <- signature | (outer(lower, atom_low, "==") & exact & rep(is_point, each = n))

    covered <- colSums(signature) > 0
    atom_low <- atom_low[covered]
    atom_high <- atom_high[covered]
    signature <- signature[, covered, drop = FALSE]
    k <- ncol(signature)
    if (k == 0L) {
      low <- high <- numeric(0)
      alpha <- matrix(FALSE, n, 0L)
    } else {
      changed <- if (k > 1L) {
        colSums(signature[, -1L, drop = FALSE] != signature[, -k, drop = FALSE]) > 0
      } else {
        logical(0)
      }
      first <- which(c(TRUE, changed))
      last <- c(first[-1L] - 1L, k)
      low <- atom_low[first]
      high <- atom_high[last]
      alpha <- signature[, first, drop = FALSE]
    }
  }
  if (any(is.infinite(upper))) {
    alpha <- cbind(alpha, is.infinite(upper))
  }
  list(low = low, high = high, alpha = alpha)
}

# The self-consistency iteration for the interval masses s_j: each observation's weight is shared
# among the intervals inside its window in proportion to their current mass.
tb_em <- function(alpha, w, tol, max_iter) {
  m <- ncol(alpha)
  if (m == 0L) {
    return(list(s = numeric(0), n_iter = 0L, converged = TRUE))
  }
  a <- alpha * 1
  total <- sum(w)
  s <- rep(1 / m, m)
  n_iter <- 0L
  converged <- FALSE
  for (it in seq_len(max_iter)) {
    n_iter <- it
    denom <- drop(a %*% s)
    share <- ifelse(denom > 0, w / denom, 0)
    s_new <- s * drop(crossprod(a, share)) / total
    done <- max(abs(s_new - s)) < tol
    s <- s_new
    if (done) {
      converged <- TRUE
      break
    }
  }
  list(s = s, n_iter = n_iter, converged = converged)
}

tb_curve <- function(lower, upper, w, tol, max_iter) {
  support <- tb_support(lower, upper)
  em <- tb_em(support$alpha, w, tol, max_iter)
  # The mass beyond every finite bound (if any) isn't reported.
  mass <- em$s[seq_along(support$low)]
  list(
    table = data.frame(
      interval_low = support$low,
      interval_high = support$high,
      prob_mass = mass,
      estimate = 1 - cumsum(mass)
    ),
    n = length(lower),
    n_iter = em$n_iter,
    converged = em$converged
  )
}

#' @export
as.data.frame.turnbull <- function(x, row.names = NULL, optional = FALSE, ...) {
  bind_groups(x$groups, lapply(x$curves, function(curve) curve$table))
}

# The interval in which the curve first drops to `level` or below: its time when the interval is a
# point, and its bounds either way.
tb_crossing <- function(table, level) {
  hit <- which(table$estimate <= level)
  if (length(hit) == 0L) {
    return(c(time = NA_real_, time_low = NA_real_, time_high = NA_real_))
  }
  low <- table$interval_low[hit[1L]]
  high <- table$interval_high[hit[1L]]
  c(time = if (low == high) low else NA_real_, time_low = low, time_high = high)
}

#' Survival time quantiles from a Turnbull fit
#'
#' `quantile()` gives the time by which a given proportion of subjects have had the event.
#' `median()` is the 0.5 quantile. When the curve drops past the level inside an interval that
#' isn't a single point, the data don't say where in the interval, so `time` is `NA` and
#' `time_low` and `time_high` are the interval's bounds. These bound what the data identify and
#' aren't confidence limits. Otherwise all three are the same. All three are `NA` when the curve
#' never drops far enough.
#'
#' @param x A [turnbull()] fit.
#' @param probs Proportions of subjects who have had the event, between 0 and 1.
#' @param na.rm Unused. Present for compatibility with the generic.
#' @param ... Unused.
#'
#' @return A data frame with one row per group and probability: `strata`, the grouping variables,
#'   `prob`, `time`, `time_low`, and `time_high`. `median()` omits `prob`.
#'
#' @examples
#' visits <- data.frame(
#'   lower = c(0, 4, 7, 0, 3, 5),
#'   upper = c(4, NA, 7, 2.5, 6, 5)
#' )
#' fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
#' median(fit)
#' quantile(fit, probs = c(0.25, 0.5))
#' @name turnbull_quantile
#' @export
quantile.turnbull <- function(x, probs = 0.5, ...) {
  if (!is.numeric(probs) || length(probs) == 0L || anyNA(probs) || any(probs < 0 | probs > 1)) {
    stop("`probs` must be numbers between 0 and 1.", call. = FALSE)
  }
  tables <- lapply(x$curves, function(curve) {
    cuts <- vapply(1 - probs, function(level) tb_crossing(curve$table, level), numeric(3))
    data.frame(
      prob = probs, time = cuts["time", ], time_low = cuts["time_low", ],
      time_high = cuts["time_high", ]
    )
  })
  bind_groups(x$groups, tables)
}

#' @rdname turnbull_quantile
#' @export
median.turnbull <- function(x, na.rm = FALSE, ...) {
  out <- stats::quantile(x, probs = 0.5)
  out$prob <- NULL
  out
}

#' Turnbull estimates at given times
#'
#' `predict()` reads the survival curve of each group at the given times. Outside the maximal
#' intersection intervals that aren't single points, the curve is a right-continuous step
#' function, 1 before the first interval. Strictly inside such an interval it isn't identified by
#' the data, and the estimate is `NA`.
#'
#' @param object A [turnbull()] fit.
#' @param times One or more times.
#' @param ... Unused.
#'
#' @return A data frame with one row per group and time: `strata`, the grouping variables, `time`,
#'   and `estimate`.
#'
#' @examples
#' visits <- data.frame(
#'   lower = c(0, 4, 7, 0, 3, 5),
#'   upper = c(4, NA, 7, 2.5, 6, 5)
#' )
#' fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
#'
#' # NA at time 1, inside the interval (0, 2.5]
#' predict(fit, times = c(1, 2.5, 5, 7))
#' @name turnbull_predict
#' @export
predict.turnbull <- function(object, times, ...) {
  times <- check_times(times)
  tables <- lapply(object$curves, function(curve) {
    data.frame(time = times, estimate = tb_predict(curve$table, times))
  })
  bind_groups(object$groups, tables)
}

tb_predict <- function(table, times) {
  m <- nrow(table)
  if (m == 0L) {
    return(rep(1, length(times)))
  }
  value <- step_at(table$interval_high, table$estimate, times, 1)
  # The first interval ending at or after each time, and whether the time is strictly inside it.
  nxt <- findInterval(times, table$interval_high, left.open = TRUE) + 1L
  inside <- nxt <= m
  j <- pmin(nxt, m)
  low <- table$interval_low[j]
  high <- table$interval_high[j]
  value[inside & times > low & times < high & low != high] <- NA_real_
  value
}

#' @rdname rmst
#' @export
rmst.turnbull <- function(object, tau, ...) {
  check_number(tau, "tau")
  tables <- lapply(object$curves, function(curve) {
    data.frame(tau = tau, estimate = tb_rmst(curve$table, tau))
  })
  bind_groups(object$groups, tables)
}

#' @rdname rmst
#' @export
rmrl.turnbull <- function(object, s, tau, ...) {
  check_number(s, "s", min = 0)
  check_number(tau, "tau")
  if (tau <= s) {
    stop("`tau` must be greater than `s`.", call. = FALSE)
  }
  tables <- lapply(object$curves, function(curve) {
    data.frame(s = s, tau = tau, estimate = tb_rmst(curve$table, tau, s))
  })
  bind_groups(object$groups, tables)
}

# The area under the curve over [s, tau] divided by S(s), with every interval's mass placed at its
# upper end.
tb_rmst <- function(table, tau, s = 0) {
  t <- table$interval_high
  surv <- table$estimate
  s_at <- step_at(t, surv, s, 1)
  if (s_at <= 0) {
    return(NA_real_)
  }
  starts <- c(0, t)
  ends <- c(t, tau)
  heights <- c(1, surv)
  width <- pmax(pmin(pmax(ends, s), tau) - pmin(pmax(starts, s), tau), 0)
  sum(heights * width) / s_at
}

#' @export
print.turnbull <- function(x, digits = getOption("digits"), ...) {
  cat("<turnbull> ", deparse_formula(x$formula), "\n\n", sep = "")
  table <- data.frame(
    n = vapply(x$curves, function(curve) curve$n, integer(1)),
    intervals = vapply(x$curves, function(curve) nrow(curve$table), integer(1)),
    ambiguous = vapply(x$curves, function(curve) {
      sum(curve$table$interval_low != curve$table$interval_high)
    }, integer(1)),
    iterations = vapply(x$curves, function(curve) curve$n_iter, integer(1)),
    converged = vapply(x$curves, function(curve) curve$converged, logical(1))
  )
  rownames(table) <- x$groups$strata
  print(table, digits = digits)
  cat_dropped(x$n_dropped)
  invisible(x)
}
