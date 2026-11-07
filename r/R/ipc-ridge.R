# Inverse-probability-of-censoring weights and IPC-weighted ridge regression:
# censoring_distribution(), ipc_ridge(), and their methods. The ipcridge_lung fixture in
# fixtures/r/ is replayed by tests/testthat/test-ipc-ridge.R.

#' The censoring distribution and its weights
#'
#' `censoring_distribution()` gives the Kaplan-Meier estimate of the censoring distribution,
#' \eqn{G(t)}, the probability of not being censored by time \eqn{t}. The censorings are the
#' events of this curve, and the events count as censorings. At a time with both, everyone with
#' that time is at risk.
#'
#' Its main use is inverse-probability-of-censoring (IPC) weighting. Weighting each subject with
#' an event by \eqn{1 / G(T_i-)}, the curve just before their time, makes up for the subjects lost
#' to censoring before then. Use [weights()] for these weights, [predict()] to read the curve at
#' any times, and [as.data.frame()] for the curve itself.
#'
#' @param response A right-censored [Surv()] object.
#' @param weights Optional case weights, one non-negative number per subject.
#'
#' @return A `censoring_distribution` object holding `time` (the distinct censoring times),
#'   `survival` (\eqn{G} just after each), `n`, and `n_censored`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2)
#' )
#'
#' censoring <- censoring_distribution(Surv(lung$time, lung$status))
#' censoring
#'
#' # The probability of staying uncensored through one and two years
#' predict(censoring, times = c(365, 730))
#'
#' # The IPC weights, 0 for the censored subjects
#' weights(censoring)
#' @export
censoring_distribution <- function(response, weights = NULL) {
  y <- metric_response(response, "censoring_distribution()")
  n <- length(y$time)
  if (is.null(weights)) {
    weights <- rep(1, n)
  } else if (!is.numeric(weights) || length(weights) != n || any(!is.finite(weights)) ||
    any(weights < 0)) {
    stop(sprintf("`weights` must be %d non-negative numbers, one per subject.", n), call. = FALSE)
  }
  censoring_km(y$time, y$status, as.numeric(weights))
}

# The weighted Kaplan-Meier curve of the censorings (status 0), with everyone whose time is at or
# after t at risk at t.
censoring_km <- function(time, status, w) {
  censored <- status == 0
  times <- sort(unique(time[censored]))
  censored_at <- tabulate_sum(match(time[censored], times), w[censored], length(times))
  at_risk <- weight_at_or_above(time, w, times)
  survival <- cumprod(ifelse(at_risk > 0, 1 - censored_at / at_risk, 1))
  structure(
    list(
      time = times,
      survival = survival,
      n = length(time),
      n_censored = sum(censored),
      stop = time,
      status = status
    ),
    class = "censoring_distribution"
  )
}

# G(t-), the curve just before each t.
censoring_left <- function(object, t) {
  i <- findInterval(t, object$time, left.open = TRUE)
  out <- rep(1, length(t))
  out[i > 0] <- object$survival[i[i > 0]]
  out
}

#' @describeIn censoring_distribution \eqn{G} at `times`, or just before them with `left = TRUE`.
#'
#' @param object A `censoring_distribution` object.
#' @param times The times to read the curve at.
#' @param left Whether to read the curve just before each time, \eqn{G(t-)}, rather than at it.
#'   The default is `FALSE`.
#' @param ... Unused.
#'
#' @export
predict.censoring_distribution <- function(object, times, left = FALSE, ...) {
  times <- check_times(times)
  check_flag(left, "left")
  if (left) {
    return(censoring_left(object, times))
  }
  step_at(object$time, object$survival, times, 1)
}

#' @describeIn censoring_distribution The IPC weights, one per subject: \eqn{1 / G(T_i-)} for a
#'   subject with an event, and 0 for a censored subject, a subject whose time is after `tau`, or
#'   one with \eqn{G(T_i-) = 0}. Case weights given to `censoring_distribution()` shape \eqn{G}
#'   but aren't multiplied in.
#'
#' @param tau A time after which subjects get a weight of 0. The default, `Inf`, keeps them all.
#'
#' @export
weights.censoring_distribution <- function(object, tau = Inf, ...) {
  check_number(tau, "tau", finite = FALSE)
  g <- censoring_left(object, object$stop)
  unname(ifelse(object$status == 1 & g > 0 & object$stop <= tau, 1 / g, 0))
}

#' @describeIn censoring_distribution The curve as a data frame with the columns `time` and
#'   `survival`, one row per distinct censoring time.
#'
#' @param x A `censoring_distribution` object.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#'
#' @export
as.data.frame.censoring_distribution <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(time = x$time, survival = x$survival)
}

#' @export
print.censoring_distribution <- function(x, digits = getOption("digits"), ...) {
  cat("<censoring_distribution>\n")
  last <- if (length(x$survival) > 0L) x$survival[length(x$survival)] else 1
  cat(sprintf(
    "n = %d, censored = %s, censoring times = %d, G after the last = %s\n",
    x$n, format(x$n_censored), length(x$time), format(last, digits = max(3L, digits - 3L))
  ))
  invisible(x)
}

#' IPC-weighted ridge regression
#'
#' `ipc_ridge()` fits a linear model for the log survival time, correcting for censoring by
#' inverse-probability-of-censoring (IPC) weighting rather than by a likelihood. Only the subjects
#' with an event enter the fit, each weighted by \eqn{w_i = 1 / G(T_i-)}, where \eqn{G} is the
#' Kaplan-Meier estimate of the censoring distribution (see [censoring_distribution()]). The
#' coefficients minimize
#' \deqn{\sum_{i: \delta_i = 1} w_i (\log T_i - \beta_0 - x_i^\top \beta)^2 +
#'   \alpha \sum_j (s_j \beta_j)^2,}
#' so no proportional hazards or error distribution is assumed.
#'
#' The intercept isn't penalized. With `standardize = TRUE` (the default), \eqn{s_j} is the
#' standard deviation of covariate \eqn{j} among the subjects with an event, so each covariate is
#' penalized on its own scale. With `standardize = FALSE`, \eqn{s_j = 1}. Either way, the
#' coefficients are on the scale of the covariates. Case weights multiply the IPC weights and
#' also weight the censoring distribution, so whole-number weights act as copies of rows.
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the
#'   covariates on the right, such as `Surv(time, status) ~ age + sex`. Factors are coded with
#'   treatment contrasts.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param alpha The ridge penalty, a non-negative number. `0` gives the unpenalized IPC-weighted
#'   least squares fit. The default is `1`.
#' @param standardize Whether to penalize each covariate on the scale of its standard deviation.
#'   The default is `TRUE`.
#'
#' @return An `ipc_ridge` object. Use [coef()] or [as.data.frame()] for the coefficients and
#'   [predict()] for the predicted log times or times. The object also holds `ipc_weights` (one per
#'   row, 0 for the censored), `censoring` (the [censoring_distribution()]), `scale` (the
#'   \eqn{s_j}), `n`, and `n_event` (the number of events in the fit).
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#'
#' fit <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = 1)
#' fit
#'
#' # Predicted survival times for two new subjects
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "response")
#' @export
ipc_ridge <- function(formula,
                      data,
                      weights,
                      subset,
                      na.action = stats::na.omit,
                      alpha = 1,
                      standardize = TRUE) {
  check_number(alpha, "alpha", min = 0)
  check_flag(standardize, "standardize")
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`ipc_ridge()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`ipc_ridge()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "ipc_ridge()", types = "right")
  design <- covariate_design(frame)
  x <- design$x
  p <- ncol(x)
  if (p == 0L) {
    stop("`ipc_ridge()` needs at least one covariate.", call. = FALSE)
  }
  time <- parts$stop
  event <- parts$event
  w <- frame$weights
  censoring <- censoring_km(time, event, w)
  ipc <- weights.censoring_distribution(censoring)
  use <- ipc > 0 & w > 0
  if (!any(use)) {
    stop("`ipc_ridge()` needs at least one event.", call. = FALSE)
  }
  if (any(time[use] <= 0)) {
    stop("`ipc_ridge()` needs positive event times.", call. = FALSE)
  }

  xf <- x[use, , drop = FALSE]
  yf <- log(time[use])
  wf <- w[use] * ipc[use]
  # The intercept is unpenalized, so the covariates and the log times are centred at their
  # weighted means. The scale is the case-weighted standard deviation among the events.
  centre <- colSums(wf * xf) / sum(wf)
  y_centre <- sum(wf * yf) / sum(wf)
  scale <- rep(1, p)
  if (standardize) {
    wc <- w[use]
    m <- colSums(wc * xf) / sum(wc)
    scale <- sqrt(colSums(wc * (xf - rep(m, each = nrow(xf)))^2) / (sum(wc) - 1))
    scale[!is.finite(scale) | scale <= 0] <- 1
  }
  xs <- sweep(sweep(xf, 2L, centre), 2L, scale, "/")
  gram <- crossprod(xs * sqrt(wf)) + diag(alpha, p)
  beta <- solve(gram, crossprod(xs, wf * (yf - y_centre))) / scale
  beta <- stats::setNames(drop(beta), colnames(x))
  intercept <- y_centre - sum(centre * beta)

  structure(
    list(
      coefficients = c("(Intercept)" = intercept, beta),
      alpha = alpha,
      standardize = standardize,
      scale = stats::setNames(scale, colnames(x)),
      ipc_weights = ipc,
      censoring = censoring,
      n = length(time),
      n_event = sum(use),
      x = x,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "ipc_ridge"
  )
}

#' @export
coef.ipc_ridge <- function(object, ...) {
  object$coefficients
}

#' @export
nobs.ipc_ridge <- function(object, ...) {
  object$n
}

#' Predictions and coefficients of an IPC-weighted ridge regression
#'
#' `predict()` gives the predicted log time (`type = "lp"`, the default), or the predicted time
#' (`type = "response"`), for each row of `newdata`. Higher values mean longer survival.
#' `as.data.frame()` gives the coefficients.
#'
#' @param object,x An [ipc_ridge()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"lp"` (the default) or `"response"`.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return `predict()` gives a numeric vector with one value per row. `as.data.frame()` gives a
#'   data frame with the columns `term` and `estimate`, starting with the intercept.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = 1)
#' as.data.frame(fit)
#' predict(fit, newdata = data.frame(age = 60, sex = 2), type = "response")
#' @name ipc_ridge_predict
#' @export
predict.ipc_ridge <- function(object, newdata, type = c("lp", "response"), ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    x <- object$x
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    x <- cox_new_design(
      list(
        terms = object$terms, xlevels = object$xlevels, contrasts = object$contrasts,
        coefficients = stats::setNames(numeric(ncol(object$x)), colnames(object$x))
      ),
      newdata
    )
  }
  lp <- unname(object$coefficients[[1L]] + linear_predictor(x, object$coefficients[-1L]))
  if (type == "lp") lp else exp(lp)
}

#' @rdname ipc_ridge_predict
#' @export
as.data.frame.ipc_ridge <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(
    term = names(x$coefficients),
    estimate = unname(x$coefficients),
    stringsAsFactors = FALSE
  )
}

#' @export
print.ipc_ridge <- function(x, digits = getOption("digits"), ...) {
  cat("<ipc_ridge> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "IPC-weighted ridge, alpha = %s. n = %d, events = %d (used for fitting)\n\n",
    format(x$alpha), x$n, x$n_event
  ))
  coefs <- data.frame(estimate = unname(x$coefficients), row.names = names(x$coefficients))
  print(coefs, digits = max(3L, digits - 3L))
  cat_dropped(x$n_dropped)
  invisible(x)
}
