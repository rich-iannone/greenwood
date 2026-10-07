# Kaplan-Meier estimation. The contract is spec/kaplan_meier.md, and the numeric fixtures in
# fixtures/r/ (km_*, lung_km_*, counting_truncation) are replayed by
# tests/testthat/test-kaplan-meier.R.

#' Kaplan-Meier estimate of survival
#'
#' `kaplan_meier()` estimates the survival function from censored event times with the
#' product-limit method. It gives one curve for the whole sample, or one curve per group when the
#' right-hand side of the formula names grouping variables.
#'
#' The estimate at time \eqn{t} is the product of \eqn{(1 - d_i / n_i)} over the event times
#' \eqn{t_i \le t}, where \eqn{n_i} is the number at risk and \eqn{d_i} the number of events.
#' Standard errors use Greenwood's formula. Confidence intervals are built on the log scale by
#' default, or on the log-log or plain (identity) scale.
#'
#' Right-censored data and counting-process data are supported. With counting-process data,
#' `Surv(start, stop, event)`, a subject is at risk only after its start time, which handles
#' delayed entry (left truncation).
#'
#' @param formula A formula with a [Surv()] response on the left and either `1` (one curve) or
#'   one or more grouping variables on the right, such as `Surv(time, status) ~ sex`.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param conf_type The scale for confidence intervals: `"log"` (the default), `"log-log"`, or
#'   `"plain"`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return A `kaplan_meier` object. Use [as.data.frame()] for the estimate at every time, and
#'   [median()] or [quantile()] for survival times. Printing it shows the number of subjects,
#'   the number of events, and the median survival time with its confidence interval for each
#'   group.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
#' )
#'
#' fit <- kaplan_meier(Surv(time, status) ~ 1, data = lung)
#' fit
#'
#' # The estimate at every time
#' as.data.frame(fit)
#'
#' # One curve per group
#' kaplan_meier(Surv(time, status) ~ sex, data = lung)
#' @export
kaplan_meier <- function(formula,
                         data,
                         weights,
                         subset,
                         na.action = stats::na.omit,
                         conf_type = c("log", "log-log", "plain"),
                         conf_level = 0.95) {
  conf_type <- match.arg(conf_type)
  if (!is.numeric(conf_level) || length(conf_level) != 1 || !(conf_level > 0 && conf_level < 1)) {
    stop("`conf_level` must be a single number between 0 and 1.", call. = FALSE)
  }

  # Build the model frame the way lm() does, so `weights` and `subset` are evaluated in `data`.
  call <- match.call()
  mf <- match.call(expand.dots = FALSE)
  keep <- match(c("formula", "data", "weights", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, keep)]
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  response <- stats::model.response(mf)
  if (!inherits(response, "Surv")) {
    stop("The left-hand side of `formula` must be a `Surv()` response.", call. = FALSE)
  }
  type <- attr(response, "type")
  if (!type %in% c("right", "counting")) {
    stop(
      sprintf(
        "`kaplan_meier()` needs right-censored or counting-process data, not \"%s\".",
        type
      ),
      call. = FALSE
    )
  }

  n <- nrow(mf)
  w <- stats::model.weights(mf)
  if (is.null(w)) {
    w <- rep(1, n)
  } else if (!is.numeric(w) || any(w < 0)) {
    stop("`weights` must be non-negative numbers.", call. = FALSE)
  }

  y <- unclass(response)
  if (type == "right") {
    start <- rep(-Inf, n)
    stop <- y[, "time"]
  } else {
    start <- y[, "start"]
    stop <- y[, "stop"]
  }
  event <- y[, "status"]

  # Grouping variables are every model-frame column except the response and the weights.
  group_vars <- setdiff(names(mf), c(names(mf)[1L], "(weights)"))
  groups <- km_groups(mf[group_vars])

  z <- stats::qnorm(1 - (1 - conf_level) / 2)
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    km_curve(start[rows], stop[rows], event[rows], w[rows], conf_type, z)
  })

  structure(
    list(
      curves = curves,
      groups = groups$keys,
      conf_type = conf_type,
      conf_level = conf_level,
      n_dropped = length(attr(mf, "na.action")),
      formula = stats::formula(stats::terms(mf)),
      call = call
    ),
    class = "kaplan_meier"
  )
}

# One row per group (sorted by the grouping variables), and each subject's group number.
km_groups <- function(vars) {
  if (length(vars) == 0) {
    return(list(keys = data.frame(row.names = 1L), index = rep(1L, nrow(vars))))
  }
  key <- interaction(lapply(vars, factor), drop = TRUE, lex.order = TRUE)
  first <- match(levels(key), key)
  keys <- vars[first, , drop = FALSE]
  rownames(keys) <- NULL
  list(keys = keys, index = as.integer(key))
}

# The estimate for one group. A subject is at risk at time t when start < t <= stop, which for
# right-censored data (start = -Inf) is stop >= t.
km_curve <- function(start, stop, event, w, conf_type, z) {
  time <- sort(unique(stop))
  n_risk <- weight_at_or_above(stop, w, time) - weight_at_or_above(start, w, time)
  at <- factor(stop, levels = time)
  n_event <- as.vector(tapply(w * (event == 1), at, sum, default = 0))
  n_censor <- as.vector(tapply(w * (event == 0), at, sum, default = 0))

  estimate <- cumprod(ifelse(n_risk > 0, 1 - n_event / n_risk, 1))

  # Greenwood's variance of log S. A time where everyone at risk has the event drives S to zero,
  # and its variance is undefined.
  denom <- n_risk * (n_risk - n_event)
  sigma <- sqrt(cumsum(ifelse(denom > 0, n_event / denom, Inf)))
  std_error <- estimate * sigma
  std_error[is.nan(std_error)] <- NA_real_

  limits <- km_limits(estimate, sigma, conf_type, z)
  data.frame(
    time = time,
    n_risk = n_risk,
    n_event = n_event,
    n_censor = n_censor,
    estimate = estimate,
    std_error = std_error,
    conf_low = limits$low,
    conf_high = limits$high
  )
}

# Total weight of the subjects with x >= t, for each t.
weight_at_or_above <- function(x, w, t) {
  o <- order(x)
  cumulative <- c(0, cumsum(w[o]))
  below <- findInterval(t, x[o], left.open = TRUE)
  sum(w) - cumulative[below + 1L]
}

km_limits <- function(estimate, sigma, conf_type, z) {
  suppressWarnings({
    if (conf_type == "plain") {
      se <- estimate * sigma
      low <- estimate - z * se
      high <- estimate + z * se
    } else if (conf_type == "log") {
      low <- estimate * exp(-z * sigma)
      high <- estimate * exp(z * sigma)
    } else {
      log_s <- log(estimate)
      se_log_log <- sigma / abs(log_s)
      low <- exp(-exp(log(-log_s) + z * se_log_log))
      high <- exp(-exp(log(-log_s) - z * se_log_log))
    }
  })
  # Before the first event the interval is [1, 1]. Once the estimate reaches zero the interval
  # is undefined.
  at_one <- estimate >= 1
  at_zero <- estimate <= 0
  low[at_one] <- 1
  high[at_one] <- 1
  low <- pmin(pmax(low, 0), 1)
  high <- pmin(pmax(high, 0), 1)
  low[at_zero] <- NA_real_
  high[at_zero] <- NA_real_
  list(low = low, high = high)
}

# The first time a decreasing curve drops to `level` or below, or NA if it never does.
km_crossing <- function(time, curve, level) {
  hit <- which(curve <= level)
  if (length(hit) == 0) NA_real_ else time[hit[1L]]
}

km_with_groups <- function(x, rows) {
  if (ncol(x$groups) == 0) {
    return(rows)
  }
  cbind(x$groups, rows)
}

#' @export
as.data.frame.kaplan_meier <- function(x, ...) {
  tables <- lapply(seq_along(x$curves), function(g) {
    curve <- x$curves[[g]]
    if (ncol(x$groups) == 0) {
      return(curve)
    }
    keys <- x$groups[rep(g, nrow(curve)), , drop = FALSE]
    rownames(keys) <- NULL
    cbind(keys, curve)
  })
  out <- do.call(rbind, tables)
  rownames(out) <- NULL
  out
}

#' Survival time quantiles from a Kaplan-Meier fit
#'
#' `quantile()` gives the time by which a given proportion of subjects have had the event, with
#' confidence limits read from the confidence band. `median()` is the 0.5 quantile. A quantile is
#' `NA` when the curve (or the band) never drops far enough.
#'
#' @param x A [kaplan_meier()] fit.
#' @param probs Proportions of subjects who have had the event, between 0 and 1.
#' @param na.rm Unused. Present for compatibility with the generic.
#' @param ... Unused.
#'
#' @return A data frame with one row per group and probability: the grouping variables, `prob`,
#'   `time`, `conf_low`, and `conf_high`. `median()` omits `prob`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2)
#' )
#' fit <- kaplan_meier(Surv(time, status) ~ 1, data = lung)
#' median(fit)
#' quantile(fit, probs = c(0.25, 0.5, 0.75))
#' @name kaplan_meier_quantile
#' @export
quantile.kaplan_meier <- function(x, probs = 0.5, ...) {
  if (!is.numeric(probs) || any(is.na(probs)) || any(probs < 0 | probs > 1)) {
    stop("`probs` must be numbers between 0 and 1.", call. = FALSE)
  }
  rows <- lapply(seq_along(x$curves), function(g) {
    curve <- x$curves[[g]]
    level <- 1 - probs
    out <- data.frame(
      prob = probs,
      time = vapply(level, function(l) km_crossing(curve$time, curve$estimate, l), numeric(1)),
      conf_low = vapply(level, function(l) km_crossing(curve$time, curve$conf_low, l), numeric(1)),
      conf_high = vapply(level, function(l) km_crossing(curve$time, curve$conf_high, l), numeric(1))
    )
    if (ncol(x$groups) == 0) {
      return(out)
    }
    keys <- x$groups[rep(g, length(probs)), , drop = FALSE]
    rownames(keys) <- NULL
    cbind(keys, out)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname kaplan_meier_quantile
#' @export
median.kaplan_meier <- function(x, na.rm = FALSE, ...) {
  out <- stats::quantile(x, probs = 0.5)
  out$prob <- NULL
  out
}

#' @export
print.kaplan_meier <- function(x, digits = getOption("digits"), ...) {
  cat("<kaplan_meier> ", paste(deparse(x$formula), collapse = " "), "\n\n", sep = "")
  med <- stats::median(x)
  level <- format(x$conf_level)
  table <- data.frame(
    n = vapply(x$curves, function(curve) curve$n_risk[1L], numeric(1)),
    events = vapply(x$curves, function(curve) sum(curve$n_event), numeric(1)),
    median = med$time,
    conf_low = med$conf_low,
    conf_high = med$conf_high
  )
  names(table)[4:5] <- paste0(level, c(" LCL", " UCL"))
  if (ncol(x$groups) > 0) {
    rownames(table) <- do.call(
      paste,
      c(Map(function(name, value) paste0(name, "=", value), names(x$groups), x$groups), sep = ", ")
    )
  } else {
    rownames(table) <- ""
  }
  print(table, digits = digits)
  if (x$n_dropped > 0) {
    cat(sprintf(
      "\n%d %s dropped for missing values.\n",
      x$n_dropped, if (x$n_dropped == 1) "row was" else "rows were"
    ))
  }
  invisible(x)
}
