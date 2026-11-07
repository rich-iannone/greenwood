# Royston-Parmar flexible parametric models: royston_parmar() and its methods. The Weibull and
# log-logistic anchors in fixtures/r/ (rp_*) are replayed by tests/testthat/test-royston-parmar.R,
# which also checks the fits against Greenwood's Python values.

#' Royston-Parmar flexible parametric survival models
#'
#' `royston_parmar()` fits a flexible parametric model: a restricted cubic spline in log time on a
#' transformed survival scale, with the covariate effects added to it. With
#' \eqn{u = \log t}, the model is
#' \deqn{\eta(t \mid x) = s(u; \gamma) + x^\top \beta,}
#' and the survival function is \eqn{S(t \mid x) = g(\eta)} for the link \eqn{g} of the scale.
#'
#' - `scale = "hazard"` (the default): \eqn{S = \exp(-\exp(\eta))}, so \eqn{\eta} is the log
#'   cumulative hazard and \eqn{\exp(\beta_j)} is a hazard ratio, constant over time.
#' - `scale = "odds"`: \eqn{S = 1 / (1 + \exp(\eta))}, so \eqn{\eta} is the log odds of having had
#'   the event and \eqn{\exp(\beta_j)} is an odds ratio, constant over time. This suits groups whose
#'   survival curves converge rather than stay apart.
#'
#' The spline has boundary knots at the smallest and largest log event times, and `df - 1` internal
#' knots at evenly spaced quantiles of the log event times between them. It has `df + 1` terms:
#' `gamma0` (the intercept), `gamma1` (the coefficient of \eqn{u}), and one term per internal knot.
#' With `df = 1` the spline is linear in \eqn{u}, so the model is a Weibull model on the hazard
#' scale and a log-logistic model on the odds scale. A larger `df` lets the baseline take more
#' shapes while the curves stay smooth.
#'
#' The parameters are estimated by maximum likelihood with Newton-Raphson iterations, halving the
#' step whenever it lowers the likelihood or makes the spline decrease in time at an observed time.
#' The iterations start from a least squares fit of the spline to the Nelson-Aalen log cumulative
#' hazard (or the Kaplan-Meier log odds). The variance is the inverse of the observed information.
#' The log-likelihood is on the time scale, so it's comparable with [aft()] and [parametric()].
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the covariates
#'   on the right, such as `Surv(time, status) ~ age + sex`, or `~ 1` for no covariates. Factors
#'   are coded with treatment contrasts. The spline's intercept takes the place of the formula's.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param df The spline's degrees of freedom, a whole number of at least 1: one more than the number
#'   of internal knots. The default is `3`.
#' @param scale `"hazard"` (the default, proportional hazards) or `"odds"` (proportional odds).
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param max_iter The largest number of Newton-Raphson iterations. The default is `100`.
#' @param tol The convergence tolerance: iterations stop once the log-likelihood changes by less
#'   than `tol` times its size. The default is `1e-10`.
#'
#' @return A `royston_parmar` object. Use [as.data.frame()] for the coefficient table, [coef()] and
#'   [vcov()] for the estimates and their variance, and [predict()] for survival, hazard, and
#'   cumulative hazard curves, quantiles, and restricted means. [logLik()], [AIC()], [BIC()], and
#'   [nobs()] work as usual, with the number of events as the number of observations, as for
#'   [cox_ph()].
#'
#'   The object also holds `knots` (on the log-time scale), `scale`, `df`, `loglik`, `n`,
#'   `n_event`, `iterations`, and `converged`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#'
#' fit <- royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2)
#' fit
#'
#' # Survival curves for two new subjects
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(200, 400))
#'
#' # A proportional odds model
#' royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2, scale = "odds")
#' @export
royston_parmar <- function(formula,
                           data,
                           weights,
                           subset,
                           na.action = stats::na.omit,
                           df = 3,
                           scale = c("hazard", "odds"),
                           conf_level = 0.95,
                           max_iter = 100,
                           tol = 1e-10) {
  scale <- match.arg(scale)
  check_number(df, "df", min = 1)
  if (df != round(df)) {
    stop("`df` must be a whole number.", call. = FALSE)
  }
  df <- as.integer(df)
  check_conf_level(conf_level)
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`royston_parmar()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`royston_parmar()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "royston_parmar()", types = "right")
  time <- parts$stop
  event <- parts$event
  w <- frame$weights
  if (any(time <= 0)) {
    stop("`royston_parmar()` needs positive times.", call. = FALSE)
  }
  if (sum(event * w) == 0) {
    stop("`royston_parmar()` needs at least one event.", call. = FALSE)
  }
  design <- covariate_design(frame)
  u <- log(time)
  knots <- stats::quantile(u[event == 1], probs = seq(0, 1, length.out = df + 1L), names = FALSE)
  if (knots[1L] == knots[length(knots)]) {
    stop("`royston_parmar()` needs events at two or more distinct times.", call. = FALSE)
  }
  if (anyDuplicated(knots)) {
    stop(
      "Some knots coincide, because many events share a time. Use a smaller `df`.",
      call. = FALSE
    )
  }
  basis <- rp_basis(u, knots)
  fit <- rp_newton(
    rp_start(time, event, w, knots, scale, ncol(design$x)),
    basis, u, design$x, event, w, scale, max_iter, tol
  )
  if (!fit$converged) {
    warning(
      sprintf("`royston_parmar()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }
  var <- tryCatch(solve(fit$info), error = function(e) NULL)
  if (is.null(var)) {
    stop("The information matrix is singular, so the model can't be fit.", call. = FALSE)
  }
  names_theta <- c(paste0("gamma", seq_len(df + 1L) - 1L), colnames(design$x))
  coefficients <- fit$theta
  names(coefficients) <- names_theta
  dimnames(var) <- list(names_theta, names_theta)
  structure(
    list(
      coefficients = coefficients,
      var = var,
      knots = knots,
      scale = scale,
      df = df,
      loglik = fit$loglik,
      n = length(time),
      n_event = sum(event),
      iterations = fit$iterations,
      converged = fit$converged,
      conf_level = conf_level,
      x = design$x,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "royston_parmar"
  )
}

# The restricted cubic spline basis at u, with columns 1, u, and one per internal knot k_j:
# (u - k_j)^3_+ - lambda_j (u - k_min)^3_+ - (1 - lambda_j) (u - k_max)^3_+, where
# lambda_j = (k_max - k_j) / (k_max - k_min). `d` holds the derivatives in u.
rp_basis <- function(u, knots) {
  k_min <- knots[1L]
  k_max <- knots[length(knots)]
  internal <- knots[-c(1L, length(knots))]
  n <- length(u)
  b <- cbind(1, u)
  d <- cbind(rep(0, n), rep(1, n))
  cube <- function(a) pmax(a, 0)^3
  square <- function(a) pmax(a, 0)^2
  for (k in internal) {
    lambda <- (k_max - k) / (k_max - k_min)
    b <- cbind(b, cube(u - k) - lambda * cube(u - k_min) - (1 - lambda) * cube(u - k_max))
    d <- cbind(d, 3 * (square(u - k) - lambda * square(u - k_min) - (1 - lambda) * square(u - k_max)))
  }
  dimnames(b) <- NULL
  dimnames(d) <- NULL
  list(b = b, d = d)
}

# The link of each scale, as functions of eta: `log_s` is log S, `log_f` is log(-d log S / d eta),
# so that the hazard is exp(log_f) s'(u) / t, and `d1`, `d2` are the derivatives of log_f.
rp_link <- function(scale) {
  if (scale == "hazard") {
    list(
      log_s = function(eta) -exp(eta),
      log_f = function(eta) eta,
      ds = function(eta) -exp(eta),
      d2s = function(eta) -exp(eta),
      df = function(eta) rep(1, length(eta)),
      d2f = function(eta) rep(0, length(eta)),
      target = function(p) log(-log1p(-p))
    )
  } else {
    list(
      log_s = function(eta) stats::plogis(eta, lower.tail = FALSE, log.p = TRUE),
      log_f = function(eta) stats::plogis(eta, log.p = TRUE),
      ds = function(eta) -stats::plogis(eta),
      d2s = function(eta) -stats::dlogis(eta),
      df = function(eta) stats::plogis(eta, lower.tail = FALSE),
      d2f = function(eta) -stats::dlogis(eta),
      target = function(p) stats::qlogis(p)
    )
  }
}

# The log-likelihood with its gradient and Hessian in theta = (gamma, beta). Each row contributes
# log S(t) and, for an event, the log density's other terms log(-d log S / d eta) + log s'(u) - u.
# The spline must increase at every observed time, or the likelihood is -Inf.
rp_terms <- function(theta, basis, u, x, event, w, link, derivatives = TRUE) {
  k <- ncol(basis$b)
  gamma <- theta[seq_len(k)]
  beta <- theta[-seq_len(k)]
  slope <- drop(basis$d %*% gamma)
  if (any(slope <= 0)) {
    return(list(loglik = -Inf))
  }
  eta <- drop(basis$b %*% gamma)
  if (length(beta) > 0L) {
    eta <- eta + drop(x %*% beta)
  }
  loglik <- sum(w * (link$log_s(eta) + event * (link$log_f(eta) + log(slope) - u)))
  if (!derivatives) {
    return(list(loglik = loglik))
  }
  z <- cbind(basis$b, x)
  a1 <- link$ds(eta) + event * link$df(eta)
  a2 <- link$d2s(eta) + event * link$d2f(eta)
  d <- cbind(basis$d, matrix(0, nrow(x), ncol(x)))
  gradient <- drop(crossprod(z, w * a1) + crossprod(d, w * event / slope))
  hessian <- crossprod(z, z * (w * a2)) - crossprod(d, d * (w * event / slope^2))
  list(loglik = loglik, gradient = gradient, hessian = hessian)
}

# Starting values: the spline fit by least squares to the Nelson-Aalen log cumulative hazard (on
# the hazard scale) or to the log odds of the Kaplan-Meier estimate (on the odds scale) at the
# event times, with the coefficients at 0. When that spline decreases somewhere, a straight line
# in log time is used instead.
rp_start <- function(time, event, w, knots, scale, p) {
  counts <- count_events(rep(-Inf, length(time)), time, event, w)
  hazard <- ifelse(counts$n_risk > 0, counts$n_event / counts$n_risk, 0)
  if (scale == "hazard") {
    cumhaz <- cumsum(hazard)
    keep <- cumhaz > 0
    target <- log(cumhaz[keep])
  } else {
    survival <- cumprod(1 - hazard)
    keep <- survival > 0 & survival < 1
    target <- stats::qlogis(1 - survival[keep])
  }
  u <- log(counts$time[keep])
  basis <- rp_basis(u, knots)
  gamma <- stats::lm.fit(basis$b, target)$coefficients
  gamma[is.na(gamma)] <- 0
  if (any(drop(rp_basis(log(time), knots)$d %*% gamma) <= 0)) {
    line <- stats::lm.fit(basis$b[, 1:2, drop = FALSE], target)$coefficients
    gamma <- c(line[1L], max(line[2L], 0.1), numeric(length(knots) - 2L))
  }
  c(unname(gamma), numeric(p))
}

# Newton-Raphson from `theta`, halving the step whenever it lowers the log-likelihood. The
# log-likelihood is concave in theta, so the steps head for its one maximum.
rp_newton <- function(theta, basis, u, x, event, w, scale, max_iter, tol) {
  link <- rp_link(scale)
  terms_at <- function(theta, derivatives = TRUE) {
    rp_terms(theta, basis, u, x, event, w, link, derivatives)
  }
  current <- terms_at(theta)
  converged <- FALSE
  iterations <- 0L
  for (iter in seq_len(max_iter)) {
    iterations <- iter
    step <- aft_step(-current$hessian, current$gradient)
    candidate <- terms_at(theta + step, derivatives = FALSE)
    halvings <- 0L
    while (!(is.finite(candidate$loglik) && candidate$loglik >= current$loglik - 1e-12) &&
      halvings < 30L) {
      step <- step / 2
      halvings <- halvings + 1L
      candidate <- terms_at(theta + step, derivatives = FALSE)
    }
    change <- candidate$loglik - current$loglik
    theta <- theta + step
    current <- terms_at(theta)
    if (abs(change) <= tol * (abs(current$loglik) + tol)) {
      converged <- TRUE
      break
    }
  }
  list(
    theta = theta,
    loglik = current$loglik,
    info = -current$hessian,
    iterations = iterations,
    converged = converged
  )
}

# -- methods ---------------------------------------------------------------------------------------

#' @export
coef.royston_parmar <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.royston_parmar <- function(object, ...) {
  object$var
}

#' @export
nobs.royston_parmar <- function(object, ...) {
  object$n_event
}

#' @export
logLik.royston_parmar <- function(object, ...) {
  structure(
    object$loglik,
    df = length(object$coefficients),
    nobs = object$n_event,
    class = "logLik"
  )
}

#' Coefficient table of a Royston-Parmar model
#'
#' `as.data.frame()` gives one row per parameter, the spline terms (`gamma0`, `gamma1`, ...) first
#' and then the covariates: the estimate, its standard error, the Wald statistic, the two-sided
#' p-value, and the confidence limits. A covariate's estimate is a log hazard ratio on the hazard
#' scale and a log odds ratio on the odds scale.
#'
#' @param x A [royston_parmar()] fit.
#' @param row.names,optional Unused, for compatibility with the generic.
#' @param exponentiate Give `exp()` of the estimates and limits (hazard ratios or odds ratios for
#'   the covariates). The standard error, the statistic, and the p-value are those of the
#'   coefficient either way. The default is `FALSE`.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return A data frame with the columns `term`, `estimate`, `std_error`, `statistic`, `p_value`,
#'   `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2)
#' as.data.frame(fit)
#' @name royston_parmar_table
#' @export
as.data.frame.royston_parmar <- function(x,
                                         row.names = NULL,
                                         optional = FALSE,
                                         ...,
                                         exponentiate = FALSE,
                                         conf_level = x$conf_level) {
  check_flag(exponentiate, "exponentiate")
  check_conf_level(conf_level)
  estimate <- x$coefficients
  std_error <- sqrt(diag(x$var))
  statistic <- estimate / std_error
  z <- z_value(conf_level)
  low <- estimate - z * std_error
  high <- estimate + z * std_error
  if (exponentiate) {
    estimate <- exp(estimate)
    low <- exp(low)
    high <- exp(high)
  }
  data.frame(
    term = names(x$coefficients),
    estimate = unname(estimate),
    std_error = unname(std_error),
    statistic = unname(statistic),
    p_value = unname(2 * stats::pnorm(-abs(statistic))),
    conf_low = unname(low),
    conf_high = unname(high),
    stringsAsFactors = FALSE
  )
}

#' @export
print.royston_parmar <- function(x, digits = getOption("digits"), ...) {
  cat("<royston_parmar> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%s scale (exp(coef) is %s ratio), df = %d. n = %d, events = %s\n\n",
    if (x$scale == "hazard") "Hazard" else "Odds",
    if (x$scale == "hazard") "a hazard" else "an odds",
    x$df, x$n, format(x$n_event)
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(
    estimate = table$estimate,
    std_error = table$std_error,
    statistic = table$statistic,
    p_value = table$p_value
  )
  rownames(coefs) <- table$term
  print(coefs, digits = max(3L, digits - 3L))
  short <- max(3L, digits - 4L)
  cat(sprintf("\nKnots (log time): %s\n", paste(format(x$knots, digits = short), collapse = ", ")))
  cat(sprintf("Log-likelihood: %s\n", format(x$loglik, digits = max(short, 6L))))
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- predictions -----------------------------------------------------------------------------------

#' Predictions from a Royston-Parmar model
#'
#' `predict()` gives, for each row of `newdata` (or of the data the model was fit to), the survival,
#' hazard, or cumulative hazard curve, quantiles of the survival time, the restricted mean survival
#' time, or the linear predictor \eqn{x^\top \beta}.
#'
#' Standard errors use the delta method with the variance of all the parameters. The limits for
#' survival and the cumulative hazard are built on the scale of \eqn{\eta}, so survival stays
#' within 0 and 1. The limits for the hazard, the quantiles, and the restricted mean are built on
#' the log scale. A quantile is the time at which the curve reaches \eqn{1 - p}, found by root
#' finding on the spline within 5 units of log time beyond the boundary knots, and `NA` when it's
#' not reached there. The restricted mean is \eqn{\int_0^\tau S(t) dt}, by numerical integration.
#'
#' @param object A [royston_parmar()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"survival"` (the default), `"hazard"`, `"cumhaz"`, `"quantile"`,
#'   `"expectation"` (the restricted mean), or `"lp"`.
#' @param times For `"survival"`, `"hazard"`, and `"cumhaz"`, the times to read the curves at.
#' @param p For `"quantile"`, the proportions of subjects who have had the event, between 0 and 1.
#'   The default, `0.5`, gives the median.
#' @param tau For `"expectation"`, the end of the window for the restricted mean.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return For `"lp"`, a numeric vector with one value per row of `newdata`. Otherwise a data
#'   frame with one row per subject (and per time or proportion): `subject` (the row of
#'   `newdata`), then `time`, `p`, or `tau`, then `estimate`, `std_error`, `conf_low`, and
#'   `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2)
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#'
#' predict(fit, newdata = newdata, type = "hazard", times = c(200, 400))
#' predict(fit, newdata = newdata, type = "quantile", p = c(0.25, 0.5))
#' predict(fit, newdata = newdata, type = "expectation", tau = 365)
#' @name royston_parmar_predict
#' @export
predict.royston_parmar <- function(object,
                                   newdata,
                                   type = c(
                                     "survival", "hazard", "cumhaz", "quantile", "expectation", "lp"
                                   ),
                                   times,
                                   p = 0.5,
                                   tau,
                                   conf_level = object$conf_level,
                                   ...) {
  type <- match.arg(type)
  check_conf_level(conf_level)
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
  k <- object$df + 1L
  beta <- object$coefficients[-seq_len(k)]
  lp <- linear_predictor(x, beta)
  if (type == "lp") {
    return(lp)
  }
  if (type == "quantile") {
    if (!is.numeric(p) || length(p) == 0L || anyNA(p) || any(p <= 0 | p >= 1)) {
      stop("`p` must be numbers between 0 and 1 (exclusive).", call. = FALSE)
    }
  } else if (type == "expectation") {
    if (missing(tau)) {
      stop("`type = \"expectation\"` needs `tau`.", call. = FALSE)
    }
    check_number(tau, "tau", min = 0)
  } else {
    if (missing(times)) {
      stop(sprintf("`type = \"%s\"` needs `times`.", type), call. = FALSE)
    }
    times <- check_times(times)
    if (any(times < 0)) {
      stop("`times` can't be negative.", call. = FALSE)
    }
  }
  z <- z_value(conf_level)
  rows <- lapply(seq_len(nrow(x)), function(i) {
    out <- switch(type,
      quantile = rp_quantiles(object, x[i, ], lp[[i]], p, z),
      expectation = rp_rmst(object, x[i, ], lp[[i]], tau, z),
      rp_curves(object, x[i, ], lp[[i]], times, type, z)
    )
    cbind(data.frame(subject = i), out)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# eta at log times u for a subject with linear predictor lp, with its gradient in the parameters
# (one row per u) and the spline's derivative in u.
rp_eta <- function(object, xi, lp, u) {
  k <- object$df + 1L
  gamma <- object$coefficients[seq_len(k)]
  basis <- rp_basis(u, object$knots)
  gradient <- cbind(basis$b, matrix(xi, length(u), length(xi), byrow = TRUE))
  list(
    eta = drop(basis$b %*% gamma) + lp,
    gradient = gradient,
    d = basis$d,
    slope = drop(basis$d %*% gamma)
  )
}

# Survival, cumulative hazard, or hazard curves. At time 0, survival is 1 and the cumulative
# hazard is 0, and the hazard is left undefined.
rp_curves <- function(object, xi, lp, times, type, z) {
  link <- rp_link(object$scale)
  positive <- times > 0
  u <- log(ifelse(positive, times, 1))
  at <- rp_eta(object, xi, lp, u)
  se_eta <- sqrt(pmax(rowSums((at$gradient %*% object$var) * at$gradient), 0))
  if (type == "hazard") {
    log_h <- link$log_f(at$eta) + log(at$slope) - u
    d <- cbind(at$d, matrix(0, length(u), length(xi)))
    gradient <- at$gradient * link$df(at$eta) + d / at$slope
    se_log <- sqrt(pmax(rowSums((gradient %*% object$var) * gradient), 0))
    estimate <- exp(log_h)
    out <- data.frame(
      time = times,
      estimate = estimate,
      std_error = estimate * se_log,
      conf_low = estimate * exp(-z * se_log),
      conf_high = estimate * exp(z * se_log)
    )
    out[!positive, -1L] <- NA_real_
    return(out)
  }
  log_s <- link$log_s(at$eta)
  # -d log S / d eta, the change in the cumulative hazard per unit of eta.
  rate <- -link$ds(at$eta)
  low_s <- link$log_s(at$eta + z * se_eta)
  high_s <- link$log_s(at$eta - z * se_eta)
  if (type == "survival") {
    out <- data.frame(
      time = times,
      estimate = exp(log_s),
      std_error = exp(log_s) * rate * se_eta,
      conf_low = exp(low_s),
      conf_high = exp(high_s)
    )
    out[!positive, -1L] <- data.frame(estimate = 1, std_error = 0, conf_low = 1, conf_high = 1)
  } else {
    out <- data.frame(
      time = times,
      estimate = -log_s,
      std_error = rate * se_eta,
      conf_low = -high_s,
      conf_high = -low_s
    )
    out[!positive, -1L] <- data.frame(estimate = 0, std_error = 0, conf_low = 0, conf_high = 0)
  }
  out
}

# Quantiles: the log time where eta reaches the link's target, with the delta method giving the
# variance of the log quantile as Var(eta) / s'(u)^2.
rp_quantiles <- function(object, xi, lp, p, z) {
  link <- rp_link(object$scale)
  k <- object$df + 1L
  gamma <- object$coefficients[seq_len(k)]
  lower <- object$knots[1L] - 5
  upper <- object$knots[length(object$knots)] + 5
  u_q <- vapply(p, function(pk) {
    gap <- function(u) drop(rp_basis(u, object$knots)$b %*% gamma) + lp - link$target(pk)
    ends <- gap(c(lower, upper))
    if (anyNA(ends) || sign(ends[1L]) == sign(ends[2L])) {
      return(NA_real_)
    }
    stats::uniroot(gap, c(lower, upper), f.lower = ends[1L], f.upper = ends[2L], tol = 1e-12)$root
  }, numeric(1))
  estimate <- exp(u_q)
  std_error <- conf_low <- conf_high <- rep(NA_real_, length(p))
  found <- !is.na(u_q)
  if (any(found)) {
    at <- rp_eta(object, xi, lp, u_q[found])
    se_eta <- sqrt(pmax(rowSums((at$gradient %*% object$var) * at$gradient), 0))
    se_log <- se_eta / abs(at$slope)
    std_error[found] <- estimate[found] * se_log
    conf_low[found] <- estimate[found] * exp(-z * se_log)
    conf_high[found] <- estimate[found] * exp(z * se_log)
  }
  data.frame(p = p, estimate = estimate, std_error = std_error, conf_low = conf_low,
    conf_high = conf_high
  )
}

# The restricted mean, the integral of S(t) from 0 to tau, with the delta method through the
# integral of dS/dtheta = (d S / d eta) times the gradient of eta.
rp_rmst <- function(object, xi, lp, tau, z) {
  link <- rp_link(object$scale)
  n_par <- length(object$coefficients)
  survival_at <- function(t) {
    out <- rep(1, length(t))
    live <- t > 0
    out[live] <- exp(link$log_s(rp_eta(object, xi, lp, log(t[live]))$eta))
    out
  }
  slope_at <- function(t, j) {
    out <- numeric(length(t))
    live <- t > 0
    at <- rp_eta(object, xi, lp, log(t[live]))
    out[live] <- exp(link$log_s(at$eta)) * link$ds(at$eta) * at$gradient[, j]
    out
  }
  integral <- function(f) {
    if (tau == 0) {
      return(0)
    }
    stats::integrate(f, 0, tau, rel.tol = 1e-10, abs.tol = 0, subdivisions = 1000L)$value
  }
  estimate <- integral(survival_at)
  gradient <- vapply(seq_len(n_par), function(j) integral(function(t) slope_at(t, j)), numeric(1))
  std_error <- sqrt(max(drop(gradient %*% object$var %*% gradient), 0))
  se_log <- if (estimate > 0) std_error / estimate else 0
  data.frame(
    tau = tau,
    estimate = estimate,
    std_error = std_error,
    conf_low = estimate * exp(-z * se_log),
    conf_high = estimate * exp(z * se_log)
  )
}
