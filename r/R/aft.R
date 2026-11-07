# Accelerated failure time models: aft() and its methods, and the location-scale engine that
# parametric() shares. The numeric fixtures in fixtures/r/ (aft_*, predict_*) are replayed by
# tests/testthat/test-aft.R.

#' Accelerated failure time regression
#'
#' `aft()` fits a parametric accelerated failure time (AFT) model by maximum likelihood. The log of
#' the event time is modeled as \eqn{\log T = x^\top \beta + \sigma \varepsilon}, where the error
#' \eqn{\varepsilon} has a fixed distribution and \eqn{\sigma} is the scale. A coefficient
#' \eqn{\beta_j} multiplies every quantile of the survival time by \eqn{\exp(\beta_j)} for a
#' one-unit increase in its covariate, so \eqn{\exp(\beta_j)} is a time ratio.
#'
#' The distributions are the Weibull (an extreme-value error), the exponential (the Weibull with
#' \eqn{\sigma = 1}), the log-normal (a normal error), and the log-logistic (a logistic error). The
#' estimates come from Newton-Raphson iterations on \eqn{(\beta, \log \sigma)}, halving the step
#' whenever it lowers the likelihood, and the variance is the inverse of the observed information
#' on that scale. The log-likelihood is on the time scale, so it's comparable across
#' distributions and with other parametric models.
#'
#' With `threshold = TRUE` the model has a third parameter, a threshold \eqn{\gamma \ge 0} below
#' which no events happen: \eqn{T = \gamma + T'}, where \eqn{T'} follows the AFT model. The
#' threshold is estimated by maximizing the profile likelihood over \eqn{0 \le \gamma < t_{(1)}},
#' the smallest observed time, and its variance comes from the observed information of all the
#' parameters. For some distributions the likelihood grows without bound as \eqn{\gamma} nears
#' \eqn{t_{(1)}}, and a warning says so when the estimate lands there.
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the covariates
#'   on the right, such as `Surv(time, status) ~ age + sex`. Factors are coded with treatment
#'   contrasts, and an intercept is always included.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param dist The error distribution: `"weibull"` (the default), `"exponential"`, `"lognormal"`,
#'   or `"loglogistic"`.
#' @param threshold Estimate a threshold (a third, location parameter). The default is `FALSE`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param max_iter The largest number of Newton-Raphson iterations. The default is `50`.
#' @param tol The convergence tolerance: iterations stop once the log-likelihood changes by less
#'   than `tol` times its size. The default is `1e-10`.
#'
#' @return An `aft` object. Use [as.data.frame()] for the coefficient table, [coef()] and [vcov()]
#'   for the estimates and their variance, [predict()] for linear predictors, quantiles, survival
#'   curves, and means, and [residuals()] for diagnostics. [logLik()], [AIC()], [BIC()], and
#'   [nobs()] work as usual.
#'
#'   The object also holds `scale` (\eqn{\sigma}), `threshold` (\eqn{\gamma}, 0 without
#'   `threshold = TRUE`), `loglik` (the log-likelihood of the intercept-only model and of the fit),
#'   the likelihood ratio test `lr_test` (with `statistic`, `df`, and `p_value`), `n`, `n_event`,
#'   `iterations`, and `converged`. `vcov()` covers \eqn{\beta}, \eqn{\log \sigma} (except for the
#'   exponential), and \eqn{\gamma} (with `threshold = TRUE`).
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#'
#' fit <- aft(Surv(time, status) ~ age + sex, data = lung)
#' fit
#'
#' # Time ratios with confidence limits
#' as.data.frame(fit, exponentiate = TRUE)
#'
#' # A log-normal model, and the median survival time of two new subjects
#' fit_lognormal <- aft(Surv(time, status) ~ age + sex, data = lung, dist = "lognormal")
#' predict(fit_lognormal, newdata = data.frame(age = c(55, 70), sex = c(1, 2)), type = "quantile")
#' @export
aft <- function(formula,
                data,
                weights,
                subset,
                na.action = stats::na.omit,
                dist = c("weibull", "exponential", "lognormal", "loglogistic"),
                threshold = FALSE,
                conf_level = 0.95,
                max_iter = 50,
                tol = 1e-10) {
  dist <- match.arg(dist)
  check_flag(threshold, "threshold")
  check_conf_level(conf_level)
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`aft()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`aft()` doesn't support `offset()` terms.", call. = FALSE)
  }
  design <- aft_design_matrix(frame)
  object <- aft_engine(
    design$x, frame$response, frame$weights, dist, threshold, max_iter, tol, "aft()"
  )
  object <- c(
    object,
    list(
      conf_level = conf_level,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    )
  )
  class(object) <- "aft"
  object
}

# The design matrix (with the intercept column) and what's needed to rebuild it for new data.
aft_design_matrix <- function(frame) {
  tt <- attr(frame$mf, "terms")
  mm <- stats::model.matrix(tt, frame$mf)
  if (ncol(mm) > 1L && qr(mm)$rank < ncol(mm)) {
    stop(
      "The covariates are collinear (or one is constant), so the model can't be fit.",
      call. = FALSE
    )
  }
  contrasts <- attr(mm, "contrasts")
  x <- unclass(mm)
  attr(x, "assign") <- NULL
  attr(x, "contrasts") <- NULL
  storage.mode(x) <- "double"
  rownames(x) <- NULL
  list(
    x = x,
    terms = stats::delete.response(tt),
    xlevels = stats::.getXlevels(tt, frame$mf),
    contrasts = contrasts
  )
}

# The design matrix of `newdata`, with the columns and coding of the fit.
aft_new_design <- function(object, newdata) {
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame.", call. = FALSE)
  }
  mf <- stats::model.frame(
    object$terms, newdata, na.action = stats::na.pass, xlev = object$xlevels
  )
  x <- stats::model.matrix(object$terms, mf, contrasts.arg = object$contrasts)
  x <- x[, names(object$coefficients), drop = FALSE]
  rownames(x) <- NULL
  x
}

# -- the error distributions -----------------------------------------------------------------------

# The standard error distribution of each family on the log-time scale. `event` and `censor` give
# the log density and the log survival at z with their first two derivatives in z.
aft_family <- function(dist) {
  switch(dist,
    weibull = ,
    exponential = list(
      event = function(z) {
        e <- exp(z)
        list(g = z - e, d1 = 1 - e, d2 = -e)
      },
      censor = function(z) {
        e <- exp(z)
        list(g = -e, d1 = -e, d2 = -e)
      },
      survival = function(z) exp(-exp(z)),
      density = function(z) exp(z - exp(z)),
      quantile = function(p) log(-log1p(-p)),
      mean = function(sigma) gamma(1 + sigma)
    ),
    lognormal = list(
      event = function(z) list(g = stats::dnorm(z, log = TRUE), d1 = -z, d2 = rep(-1, length(z))),
      censor = function(z) {
        log_s <- stats::pnorm(z, lower.tail = FALSE, log.p = TRUE)
        h <- exp(stats::dnorm(z, log = TRUE) - log_s)
        list(g = log_s, d1 = -h, d2 = -h * (h - z))
      },
      survival = function(z) stats::pnorm(z, lower.tail = FALSE),
      density = stats::dnorm,
      quantile = stats::qnorm,
      mean = function(sigma) exp(sigma^2 / 2)
    ),
    loglogistic = list(
      event = function(z) {
        f <- stats::plogis(z)
        list(g = stats::dlogis(z, log = TRUE), d1 = 1 - 2 * f, d2 = -2 * f * (1 - f))
      },
      censor = function(z) {
        f <- stats::plogis(z)
        list(g = stats::plogis(z, lower.tail = FALSE, log.p = TRUE), d1 = -f, d2 = -f * (1 - f))
      },
      survival = function(z) stats::plogis(z, lower.tail = FALSE),
      density = stats::dlogis,
      quantile = stats::qlogis,
      mean = function(sigma) if (sigma < 1) pi * sigma / sin(pi * sigma) else Inf
    )
  )
}

dist_label <- function(dist) {
  switch(dist,
    weibull = "Weibull",
    exponential = "Exponential",
    lognormal = "Log-normal",
    loglogistic = "Log-logistic"
  )
}

# -- fitting ---------------------------------------------------------------------------------------

# Fit a location-scale model for log(time) to the design `x`, and package what aft() and
# parametric() share. `response` is a right-censored Surv() response.
aft_engine <- function(x, response, w, dist, threshold, max_iter, tol, fn) {
  parts <- surv_parts(response, fn, types = "right")
  time <- parts$stop
  event <- parts$event
  if (any(time <= 0)) {
    stop(sprintf("`%s` needs positive times.", fn), call. = FALSE)
  }
  if (sum(event * w) == 0) {
    stop(sprintf("`%s` needs at least one event.", fn), call. = FALSE)
  }
  family <- aft_family(dist)
  fixed_scale <- dist == "exponential"

  fit_at <- function(gamma, x) {
    aft_newton(x, log(time - gamma), event, w, family, fixed_scale, max_iter, tol)
  }
  gamma <- 0
  if (threshold) {
    t_min <- min(time)
    bound <- t_min - 1e-6 * max(t_min, 1)
    profile <- function(g) fit_at(g, x)$loglik
    # A coarse grid first, so the search starts in the right basin.
    grid <- seq(0, bound, length.out = 11L)
    values <- vapply(grid, function(g) tryCatch(profile(g), error = function(e) -Inf), numeric(1))
    best <- which.max(values)
    lower <- grid[max(best - 1L, 1L)]
    upper <- grid[min(best + 1L, length(grid))]
    gamma <- stats::optimize(profile, c(lower, upper), maximum = TRUE, tol = 1e-10 * t_min)$maximum
    if (bound - gamma < 1e-4 * t_min) {
      warning(
        paste(
          "The threshold estimate is at the smallest observed time, where the likelihood can grow",
          "without bound. The estimate may not be meaningful."
        ),
        call. = FALSE
      )
    }
  }
  fit <- fit_at(gamma, x)
  if (!fit$converged) {
    warning(sprintf("`%s` didn't converge in %d iterations.", fn, as.integer(max_iter)),
      call. = FALSE
    )
  }
  null_fit <- if (ncol(x) > 1L) fit_at(gamma, x[, 1L, drop = FALSE]) else fit

  p <- ncol(x)
  names_theta <- c(colnames(x), if (!fixed_scale) "log(scale)")
  info <- fit$info
  if (threshold) {
    theta <- c(fit$theta, gamma)
    names_theta <- c(names_theta, "threshold")
    gradient <- function(par) {
      k <- length(par)
      -aft_terms(par[-k], x, time, event, w, family, fixed_scale, par[k], TRUE)$gradient
    }
    info <- stats::optimHess(theta, function(par) {
      k <- length(par)
      -aft_terms(par[-k], x, time, event, w, family, fixed_scale, par[k])$loglik
    }, gradient)
    info <- (info + t(info)) / 2
  }
  var <- tryCatch(solve(info), error = function(e) NULL)
  if (is.null(var)) {
    stop("The information matrix is singular, so the model can't be fit.", call. = FALSE)
  }
  dimnames(var) <- list(names_theta, names_theta)
  beta <- fit$theta[seq_len(p)]
  names(beta) <- colnames(x)
  df <- p - 1L
  list(
    coefficients = beta,
    scale = if (fixed_scale) 1 else unname(exp(fit$theta[p + 1L])),
    threshold = gamma,
    var = var,
    loglik = c(null_fit$loglik, fit$loglik),
    lr_test = if (df > 0L) cox_chisq(2 * (fit$loglik - null_fit$loglik), df) else NULL,
    linear_predictors = drop(x %*% beta),
    n = length(time),
    n_event = sum(event),
    iterations = fit$iterations,
    converged = fit$converged,
    dist = dist,
    has_threshold = threshold,
    x = x,
    time = time,
    event = event,
    weights = w
  )
}

# The log-likelihood of a location-scale model for log(time - gamma), with its gradient and
# Hessian in theta = (beta, log sigma) (beta only when the scale is fixed at 1). The density is
# on the time scale, so events carry the Jacobian -log(sigma) - log(time - gamma). With
# `gamma_gradient`, the gradient also has the derivative in gamma as its last element.
aft_terms <- function(theta, x, time, event, w, family, fixed_scale, gamma = 0,
                      gamma_gradient = FALSE) {
  p <- ncol(x)
  s <- if (fixed_scale) 0 else theta[p + 1L]
  sigma <- exp(s)
  shifted <- time - gamma
  if (any(shifted <= 0)) {
    return(list(loglik = -Inf))
  }
  y <- log(shifted)
  z <- (y - drop(x %*% theta[seq_len(p)])) / sigma
  is_event <- event == 1
  pieces <- aft_pieces(family, z, is_event)
  loglik <- sum(w * (pieces$g - is_event * (s + y)))
  d_eta <- -pieces$d1 / sigma
  gradient <- drop(crossprod(x, w * d_eta))
  hessian <- crossprod(x, x * (w * pieces$d2 / sigma^2))
  if (!fixed_scale) {
    d_s <- -pieces$d1 * z - is_event
    h_bs <- drop(crossprod(x, w * (pieces$d2 * z + pieces$d1) / sigma))
    h_ss <- sum(w * (pieces$d2 * z^2 + pieces$d1 * z))
    gradient <- c(gradient, sum(w * d_s))
    hessian <- rbind(cbind(hessian, h_bs), c(h_bs, h_ss))
  }
  if (gamma_gradient) {
    d_y <- pieces$d1 / sigma - is_event
    gradient <- c(gradient, sum(w * d_y * (-1 / shifted)))
  }
  list(loglik = loglik, gradient = gradient, hessian = hessian, z = z, pieces = pieces)
}

aft_pieces <- function(family, z, is_event) {
  g <- d1 <- d2 <- numeric(length(z))
  if (any(is_event)) {
    e <- family$event(z[is_event])
    g[is_event] <- e$g
    d1[is_event] <- e$d1
    d2[is_event] <- e$d2
  }
  if (any(!is_event)) {
    c <- family$censor(z[!is_event])
    g[!is_event] <- c$g
    d1[!is_event] <- c$d1
    d2[!is_event] <- c$d2
  }
  list(g = g, d1 = d1, d2 = d2)
}

# Newton-Raphson for the location-scale model of y = log(time - gamma). Starts from least squares
# on all the rows, and halves the step whenever it lowers the log-likelihood. When the observed
# information isn't positive definite, a ridge is added until it is.
aft_newton <- function(x, y, event, w, family, fixed_scale, max_iter, tol) {
  time <- exp(y)
  start <- stats::lm.wfit(x, y, w)
  theta <- start$coefficients
  theta[is.na(theta)] <- 0
  if (!fixed_scale) {
    spread <- sqrt(sum(w * start$residuals^2) / sum(w))
    theta <- c(theta, log(max(spread, 1e-3)))
  }
  current <- aft_terms(theta, x, time, event, w, family, fixed_scale)
  converged <- FALSE
  iterations <- 0L
  for (iter in seq_len(max_iter)) {
    iterations <- iter
    info <- -current$hessian
    step <- aft_step(info, current$gradient)
    candidate <- aft_terms(theta + step, x, time, event, w, family, fixed_scale)
    halvings <- 0L
    while (!(is.finite(candidate$loglik) && candidate$loglik >= current$loglik - 1e-12) &&
      halvings < 30L) {
      step <- step / 2
      halvings <- halvings + 1L
      candidate <- aft_terms(theta + step, x, time, event, w, family, fixed_scale)
    }
    change <- candidate$loglik - current$loglik
    theta <- theta + step
    current <- candidate
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

aft_step <- function(info, gradient) {
  ridge <- 0
  scale <- max(abs(diag(info)), 1)
  for (attempt in 1:30) {
    factor <- tryCatch(chol(info + diag(ridge, nrow(info))), error = function(e) NULL)
    if (!is.null(factor)) {
      return(backsolve(factor, forwardsolve(t(factor), gradient)))
    }
    ridge <- if (ridge == 0) 1e-8 * scale else ridge * 10
  }
  gradient / scale
}

# -- methods ---------------------------------------------------------------------------------------

#' @export
coef.aft <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.aft <- function(object, ...) {
  object$var
}

#' @export
nobs.aft <- function(object, ...) {
  object$n
}

#' @export
logLik.aft <- function(object, ...) {
  structure(
    object$loglik[2L],
    df = nrow(object$var),
    nobs = object$n,
    class = "logLik"
  )
}

#' Coefficient table of an accelerated failure time model
#'
#' `as.data.frame()` gives one row per coefficient, including the intercept: the estimate (on the
#' log-time scale, or a time ratio with `exponentiate = TRUE`), its standard error, the Wald
#' statistic, the two-sided p-value, and the confidence limits.
#'
#' @param x An [aft()] fit.
#' @param row.names,optional Unused, for compatibility with the generic.
#' @param exponentiate Give time ratios instead of coefficients on the log-time scale. The standard
#'   error, the statistic, and the p-value are those of the coefficient either way. The default is
#'   `FALSE`.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return A data frame with the columns `term`, `estimate`, `std_error`, `statistic`, `p_value`,
#'   `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- aft(Surv(time, status) ~ age + sex, data = lung)
#' as.data.frame(fit)
#' as.data.frame(fit, exponentiate = TRUE)
#' @name aft_table
#' @export
as.data.frame.aft <- function(x,
                              row.names = NULL,
                              optional = FALSE,
                              ...,
                              exponentiate = FALSE,
                              conf_level = x$conf_level) {
  check_flag(exponentiate, "exponentiate")
  check_conf_level(conf_level)
  p <- length(x$coefficients)
  estimate <- x$coefficients
  std_error <- sqrt(diag(x$var))[seq_len(p)]
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
print.aft <- function(x, digits = getOption("digits"), ...) {
  cat("<aft> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%s distribution. n = %d, events = %s\n\n",
    dist_label(x$dist), x$n, format(sum(x$event))
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(
    estimate = table$estimate,
    time_ratio = exp(table$estimate),
    std_error = table$std_error,
    statistic = table$statistic,
    p_value = table$p_value
  )
  rownames(coefs) <- table$term
  print(coefs, digits = max(3L, digits - 3L))
  short <- max(3L, digits - 4L)
  cat("\n")
  if (x$dist != "exponential") {
    cat(sprintf(
      "Scale: %s (std. error of log(scale) %s)\n",
      format(x$scale, digits = short),
      format(sqrt(x$var["log(scale)", "log(scale)"]), digits = short)
    ))
  }
  if (x$has_threshold) {
    cat(sprintf(
      "Threshold: %s (std. error %s)\n",
      format(x$threshold, digits = short),
      format(sqrt(x$var["threshold", "threshold"]), digits = short)
    ))
  }
  cat(sprintf("Log-likelihood: %s\n", format(x$loglik[2L], digits = max(short, 6L))))
  if (!is.null(x$lr_test)) {
    cat(sprintf(
      "Likelihood ratio test: %s on %d df, p = %s\n",
      format(x$lr_test$statistic, digits = short),
      as.integer(x$lr_test$df),
      format_p(x$lr_test$p_value)
    ))
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- predictions -----------------------------------------------------------------------------------

#' Predictions from an accelerated failure time model
#'
#' `predict()` gives, for each row of `newdata` (or of the data the model was fit to), the linear
#' predictor, quantiles of the survival time, the survival or cumulative hazard curve, the mean
#' survival time, or the restricted mean survival time.
#'
#' The linear predictor is \eqn{\eta = x^\top \beta}, the location of \eqn{\log T}. With the
#' standard error distribution's survival function \eqn{S_0} and quantile function \eqn{w_p},
#' survival is \eqn{S(t) = S_0((\log(t - \gamma) - \eta) / \sigma)} and the \eqn{p} quantile is
#' \eqn{\gamma + \exp(\eta + \sigma w_p)}, with \eqn{\gamma = 0} unless the model has a threshold.
#' The mean is \eqn{\gamma + \int_0^\infty S(t) dt}, and the restricted mean is the same integral
#' up to `tau`.
#'
#' Standard errors use the delta method with the full variance of the parameters (the
#' coefficients, the log scale, and the threshold). Confidence limits for quantiles, means, and
#' restricted means are built on the log scale. Limits for survival and the cumulative hazard are
#' built on the scale of \eqn{z = (\log(t - \gamma) - \eta) / \sigma}, which for the Weibull is the
#' log of the cumulative hazard, so they stay within 0 and 1.
#'
#' @param object An [aft()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"lp"` (the default), `"quantile"`, `"survival"`, `"cumhaz"`,
#'   `"mean"`, or `"expectation"` (the restricted mean).
#' @param p For `"quantile"`, the proportions of subjects who have had the event, between 0 and 1.
#'   The default, `0.5`, gives the median.
#' @param times For `"survival"` and `"cumhaz"`, the times to read the curves at.
#' @param tau For `"expectation"`, the end of the window for the restricted mean.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return For `"lp"`, a numeric vector with one value per row of `newdata`. Otherwise a data
#'   frame with one row per subject (and per proportion or time): `subject` (the row of
#'   `newdata`), then `p` for `"quantile"`, `time` for `"survival"` and `"cumhaz"`, or `tau` for
#'   `"expectation"`, then `estimate`, `std_error`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- aft(Surv(time, status) ~ age + sex, data = lung)
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#'
#' predict(fit, newdata = newdata, type = "quantile", p = c(0.25, 0.5, 0.75))
#' predict(fit, newdata = newdata, type = "survival", times = c(200, 400))
#' predict(fit, newdata = newdata, type = "expectation", tau = 730)
#' @name aft_predict
#' @export
predict.aft <- function(object,
                        newdata,
                        type = c("lp", "quantile", "survival", "cumhaz", "mean", "expectation"),
                        p = 0.5,
                        times,
                        tau,
                        conf_level = object$conf_level,
                        ...) {
  type <- match.arg(type)
  check_conf_level(conf_level)
  x <- if (missing(newdata)) object$x else aft_new_design(object, newdata)
  if (type == "lp") {
    return(drop(x %*% object$coefficients))
  }
  z <- z_value(conf_level)
  if (type == "quantile") {
    if (!is.numeric(p) || anyNA(p) || any(p <= 0 | p >= 1)) {
      stop("`p` must be numbers between 0 and 1 (exclusive).", call. = FALSE)
    }
    out <- aft_quantiles(object, x, p, z)
  } else if (type %in% c("survival", "cumhaz")) {
    if (missing(times)) {
      stop(sprintf("`type = \"%s\"` needs `times`.", type), call. = FALSE)
    }
    times <- check_times(times)
    out <- aft_curves(object, x, times, type, z)
  } else {
    if (type == "expectation") {
      if (missing(tau)) {
        stop("`type = \"expectation\"` needs `tau`.", call. = FALSE)
      }
      check_number(tau, "tau", min = 0)
    } else {
      tau <- Inf
    }
    out <- aft_means(object, x, tau, z)
    if (type == "mean") {
      out$tau <- NULL
    }
  }
  rownames(out) <- NULL
  out
}

# The full parameter vector, in the order of vcov(): beta, log(scale), threshold.
aft_theta <- function(object) {
  c(
    object$coefficients,
    if (object$dist != "exponential") log(object$scale),
    if (object$has_threshold) object$threshold
  )
}

aft_unpack <- function(object, theta) {
  p <- length(object$coefficients)
  list(
    beta = theta[seq_len(p)],
    sigma = if (object$dist == "exponential") 1 else exp(theta[p + 1L]),
    gamma = if (object$has_threshold) theta[length(theta)] else 0
  )
}

# Quantiles: log(q - gamma) = eta + sigma * w_p, with the delta method on that scale. A threshold
# adds its own (time-scale) term to the variance of q.
aft_quantiles <- function(object, x, p, z) {
  family <- aft_family(object$dist)
  v <- object$var
  k <- nrow(v)
  n_coef <- length(object$coefficients)
  w_p <- family$quantile(p)
  rows <- lapply(seq_len(nrow(x)), function(i) {
    eta <- sum(x[i, ] * object$coefficients)
    log_q <- eta + object$scale * w_p
    q <- exp(log_q)
    # Gradient of q itself in the parameters.
    grad <- matrix(0, length(p), k)
    grad[, seq_len(n_coef)] <- q %o% x[i, ]
    if (object$dist != "exponential") {
      grad[, n_coef + 1L] <- q * object$scale * w_p
    }
    if (object$has_threshold) {
      grad[, k] <- 1
    }
    estimate <- object$threshold + q
    std_error <- sqrt(pmax(rowSums((grad %*% v) * grad), 0))
    se_log <- std_error / estimate
    data.frame(
      subject = i,
      p = p,
      estimate = estimate,
      std_error = std_error,
      conf_low = estimate * exp(-z * se_log),
      conf_high = estimate * exp(z * se_log)
    )
  })
  do.call(rbind, rows)
}

# Survival and cumulative hazard curves, with limits built on the scale of z.
aft_curves <- function(object, x, times, type, z_alpha) {
  family <- aft_family(object$dist)
  v <- object$var
  k <- nrow(v)
  n_coef <- length(object$coefficients)
  sigma <- object$scale
  rows <- lapply(seq_len(nrow(x)), function(i) {
    eta <- sum(x[i, ] * object$coefficients)
    shifted <- times - object$threshold
    live <- shifted > 0
    zs <- ifelse(live, (log(pmax(shifted, .Machine$double.xmin)) - eta) / sigma, -Inf)
    grad <- matrix(0, length(times), k)
    grad[, seq_len(n_coef)] <- -outer(rep(1 / sigma, length(times)), x[i, ])
    if (object$dist != "exponential") {
      grad[, n_coef + 1L] <- -zs
    }
    if (object$has_threshold) {
      grad[, k] <- -1 / (pmax(shifted, .Machine$double.xmin) * sigma)
    }
    grad[!live, ] <- 0
    se_z <- sqrt(pmax(rowSums((grad %*% v) * grad), 0))
    survival <- family$survival(zs)
    low <- family$survival(zs + z_alpha * se_z)
    high <- family$survival(zs - z_alpha * se_z)
    std_error <- ifelse(live, family$density(zs) * se_z, 0)
    if (type == "cumhaz") {
      estimate <- -log(survival)
      out_low <- -log(high)
      out_high <- -log(low)
      std_error <- std_error / survival
    } else {
      estimate <- survival
      out_low <- low
      out_high <- high
    }
    data.frame(
      subject = i,
      time = times,
      estimate = estimate,
      std_error = std_error,
      conf_low = out_low,
      conf_high = out_high
    )
  })
  do.call(rbind, rows)
}

# Means restricted to tau (tau = Inf gives the mean), with the delta method through a numerical
# gradient in the parameters.
aft_means <- function(object, x, tau, z) {
  family <- aft_family(object$dist)
  theta <- aft_theta(object)
  v <- object$var
  mean_at <- function(theta, xi) {
    par <- aft_unpack(object, theta)
    eta <- sum(xi * par$beta)
    if (!is.finite(tau)) {
      return(par$gamma + exp(eta) * family$mean(par$sigma))
    }
    if (tau <= par$gamma) {
      return(tau)
    }
    s_at <- function(t) family$survival((log(t) - eta) / par$sigma)
    area <- stats::integrate(
      s_at, 0, tau - par$gamma, rel.tol = 1e-12, abs.tol = 0, subdivisions = 1000L
    )$value
    par$gamma + area
  }
  rows <- lapply(seq_len(nrow(x)), function(i) {
    xi <- x[i, ]
    estimate <- mean_at(theta, xi)
    grad <- vapply(seq_along(theta), function(j) {
      h <- 1e-5 * max(abs(theta[j]), 1)
      up <- theta
      down <- theta
      up[j] <- up[j] + h
      down[j] <- down[j] - h
      (mean_at(up, xi) - mean_at(down, xi)) / (2 * h)
    }, numeric(1))
    std_error <- sqrt(max(drop(grad %*% v %*% grad), 0))
    se_log <- std_error / estimate
    data.frame(
      subject = i,
      tau = tau,
      estimate = estimate,
      std_error = std_error,
      conf_low = estimate * exp(-z * se_log),
      conf_high = estimate * exp(z * se_log)
    )
  })
  do.call(rbind, rows)
}

# -- residuals -------------------------------------------------------------------------------------

#' Residuals of an accelerated failure time model
#'
#' `residuals()` gives one of several kinds of residuals for each row the model was fit to.
#'
#' - `"response"`: the observed time minus the time predicted by the location,
#'   \eqn{t - \gamma - \exp(\eta)}.
#' - `"cox_snell"`: \eqn{-\log S(t)}, the fitted cumulative hazard at the observed time.
#' - `"martingale"`: the event indicator minus the Cox-Snell residual.
#' - `"deviance"`: the signed square root of twice the gap between the row's log-likelihood and
#'   its largest possible value.
#' - `"score"`: each row's contribution to the gradient of the log-likelihood in the parameters,
#'   one column per parameter (the coefficients, the log scale, and the threshold).
#' - `"dfbeta"`: the approximate change in each parameter when the row is left out, the score
#'   residuals times the variance of the parameters.
#' - `"dfbetas"`: dfbeta divided by the standard errors of the parameters.
#'
#' With case weights, the score, dfbeta, and dfbetas residuals include the weights.
#'
#' @param object An [aft()] fit.
#' @param type The kind of residual. The default is `"martingale"`.
#' @param ... Unused.
#'
#' @return A numeric vector with one value per row, or for `"score"`, `"dfbeta"`, and `"dfbetas"` a
#'   matrix with one row per row of the data and one column per parameter.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- aft(Surv(time, status) ~ age + sex, data = lung)
#' residuals(fit, type = "deviance")
#' head(residuals(fit, type = "dfbeta"))
#' @name aft_residuals
#' @export
residuals.aft <- function(object,
                          type = c(
                            "martingale", "response", "cox_snell", "deviance", "score", "dfbeta",
                            "dfbetas"
                          ),
                          ...) {
  type <- match.arg(type)
  family <- aft_family(object$dist)
  eta <- object$linear_predictors
  if (type == "response") {
    return(object$time - object$threshold - exp(eta))
  }
  shifted <- object$time - object$threshold
  z <- (log(shifted) - eta) / object$scale
  is_event <- object$event == 1
  cox_snell <- -log(family$survival(z))
  if (type == "cox_snell") {
    return(cox_snell)
  }
  if (type == "martingale") {
    return(object$event - cox_snell)
  }
  if (type == "deviance") {
    pieces <- aft_pieces(family, z, is_event)
    best <- family$event(0)$g
    d2 <- ifelse(is_event, 2 * (best - pieces$g), -2 * pieces$g)
    return(ifelse(is_event, sign(z), 1) * sqrt(pmax(d2, 0)))
  }

  pieces <- aft_pieces(family, z, is_event)
  sigma <- object$scale
  w <- object$weights
  score <- object$x * (w * -pieces$d1 / sigma)
  if (object$dist != "exponential") {
    score <- cbind(score, w * (-pieces$d1 * z - is_event))
  }
  if (object$has_threshold) {
    score <- cbind(score, w * (pieces$d1 / sigma - is_event) * (-1 / shifted))
  }
  colnames(score) <- rownames(object$var)
  if (type == "score") {
    return(score)
  }
  dfbeta <- score %*% object$var
  if (type == "dfbeta") {
    return(dfbeta)
  }
  sweep(dfbeta, 2L, sqrt(diag(object$var)), "/")
}
