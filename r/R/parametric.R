# Parametric survival distributions without covariates: parametric(), its methods, and
# compare_distributions(). The fits use the location-scale engine of aft(). The numeric fixtures
# (parametric_*) are replayed by tests/testthat/test-parametric.R.

#' Parametric survival distributions
#'
#' `parametric()` fits a Weibull, exponential, log-normal, or log-logistic distribution to
#' right-censored survival times by maximum likelihood. It gives one fit for the whole sample, or
#' one per group when the right-hand side of the formula names grouping variables.
#'
#' Each distribution is the intercept-only [aft()] model, \eqn{\log T = \mu + \sigma \varepsilon},
#' reported in its usual parameters:
#'
#' - Weibull: `shape` \eqn{1 / \sigma} and `scale` \eqn{\exp(\mu)}, with
#'   \eqn{S(t) = \exp(-(t / \mathrm{scale})^{\mathrm{shape}})}.
#' - Exponential: `rate` \eqn{\exp(-\mu)}, with \eqn{S(t) = \exp(-\mathrm{rate} \cdot t)}.
#' - Log-normal: `mu` and `sigma`, the mean and standard deviation of \eqn{\log T}.
#' - Log-logistic: `alpha` \eqn{\exp(\mu)} (the median) and `beta` \eqn{1 / \sigma} (the shape),
#'   with \eqn{S(t) = 1 / (1 + (t / \alpha)^\beta)}.
#'
#' Standard errors come from the delta method. Confidence limits for the positive parameters are
#' built on the log scale, and those for the log-normal `mu` on the plain scale.
#'
#' @inheritParams aft
#' @param formula A formula with a right-censored [Surv()] response on the left and either `1` or
#'   the grouping variables on the right, such as `Surv(time, status) ~ 1` or
#'   `Surv(time, status) ~ sex`.
#'
#' @return A `parametric` object. Use [as.data.frame()] for the parameter table, [predict()] for
#'   the survival, cumulative hazard, hazard, and density at given times, [quantile()],
#'   [median()], [mean()], and [rmst()] for summaries of the survival time, and [logLik()],
#'   [AIC()], and [BIC()] to compare distributions (see also [compare_distributions()]).
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- parametric(Surv(time, status) ~ 1, data = lung)
#' fit
#' predict(fit, times = c(100, 365))
#' median(fit)
#'
#' # One log-normal fit per group
#' parametric(Surv(time, status) ~ sex, data = lung, dist = "lognormal")
#' @export
parametric <- function(formula,
                       data,
                       weights,
                       subset,
                       na.action = stats::na.omit,
                       dist = c("weibull", "exponential", "lognormal", "loglogistic"),
                       conf_level = 0.95) {
  dist <- match.arg(dist)
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`parametric()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  groups <- group_keys(frame$vars, nrow(frame$mf))
  response <- frame$response
  fits <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    x <- matrix(1, sum(rows), 1L, dimnames = list(NULL, "(Intercept)"))
    aft_engine(x, response[rows], frame$weights[rows], dist, FALSE, 50, 1e-10, "parametric()")
  })
  structure(
    list(
      fits = fits,
      groups = groups$keys,
      dist = dist,
      conf_level = conf_level,
      n = nrow(frame$mf),
      n_event = sum(vapply(fits, function(f) f$n_event, numeric(1))),
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "parametric"
  )
}

# The usual parameters of one fit: the estimates, the standard errors of their logs (or of the
# value itself, for the log-normal mu), and whether the limits are built on the log scale.
parametric_params <- function(fit) {
  v <- fit$var
  mu <- fit$coefficients[[1L]]
  se_mu <- sqrt(v[1L, 1L])
  se_s <- if (fit$dist == "exponential") NA_real_ else sqrt(v[2L, 2L])
  sigma <- fit$scale
  switch(fit$dist,
    weibull = list(
      param = c("shape", "scale"), estimate = c(1 / sigma, exp(mu)), se = c(se_s, se_mu),
      log_scale = c(TRUE, TRUE)
    ),
    exponential = list(param = "rate", estimate = exp(-mu), se = se_mu, log_scale = TRUE),
    lognormal = list(
      param = c("mu", "sigma"), estimate = c(mu, sigma), se = c(se_mu, se_s),
      log_scale = c(FALSE, TRUE)
    ),
    loglogistic = list(
      param = c("alpha", "beta"), estimate = c(exp(mu), 1 / sigma), se = c(se_mu, se_s),
      log_scale = c(TRUE, TRUE)
    )
  )
}

#' @export
as.data.frame.parametric <- function(x, row.names = NULL, optional = FALSE, ...,
                                     conf_level = x$conf_level) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  tables <- lapply(x$fits, function(fit) {
    params <- parametric_params(fit)
    est <- params$estimate
    se_log <- params$se
    data.frame(
      param = params$param,
      estimate = est,
      std_error = ifelse(params$log_scale, est * se_log, se_log),
      conf_low = ifelse(params$log_scale, est * exp(-z * se_log), est - z * se_log),
      conf_high = ifelse(params$log_scale, est * exp(z * se_log), est + z * se_log),
      stringsAsFactors = FALSE
    )
  })
  bind_groups(x$groups, tables)
}

#' @export
logLik.parametric <- function(object, ...) {
  structure(
    sum(vapply(object$fits, function(f) f$loglik[2L], numeric(1))),
    df = sum(vapply(object$fits, function(f) nrow(f$var), integer(1))),
    nobs = object$n,
    class = "logLik"
  )
}

#' @export
nobs.parametric <- function(object, ...) {
  object$n
}

#' @export
print.parametric <- function(x, digits = getOption("digits"), ...) {
  cat("<parametric> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%s distribution. n = %d, events = %s\n\n",
    dist_label(x$dist), x$n, format(x$n_event)
  ))
  table <- as.data.frame(x)
  if (nrow(x$groups) == 1L) {
    table$strata <- NULL
  }
  print(table, digits = max(3L, digits - 3L), row.names = FALSE)
  loglik <- stats::logLik(x)
  cat(sprintf(
    "\nLog-likelihood: %s, AIC: %s\n",
    format(as.numeric(loglik), digits = max(6L, digits - 4L)),
    format(stats::AIC(x), digits = max(6L, digits - 4L))
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

#' Survival, hazard, and summaries of a parametric fit
#'
#' `predict()` gives the survival, cumulative hazard, hazard, or density of each fitted
#' distribution at given times. `quantile()` gives the times by which given proportions of subjects
#' have had the event, `median()` the 0.5 quantile, `mean()` the mean survival time, and [rmst()]
#' the restricted mean survival time up to `tau`.
#'
#' Standard errors use the delta method. Limits for survival and the cumulative hazard are built on
#' the scale of the standardized log time, so survival stays within 0 and 1, and limits for the
#' hazard, the density, quantiles, and means on the log scale.
#'
#' @param object,x A [parametric()] fit.
#' @param times The times to evaluate at.
#' @param type What to give: `"survival"` (the default), `"cumhaz"`, `"hazard"`, or `"density"`.
#' @param probs Proportions of subjects who have had the event, between 0 and 1 (exclusive).
#' @param tau The end of the window for the restricted mean.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param na.rm Unused. Present for compatibility with the generic.
#' @param ... Unused.
#'
#' @return A data frame with one row per group (and per time or proportion): `strata`, the
#'   grouping variables, then `time` for `predict()`, `prob` and `time` for `quantile()` (`median()`
#'   omits `prob`), or `tau` for `rmst()`, then `estimate` (not for `quantile()`), `std_error`,
#'   `conf_low`, and `conf_high`. The layout is the same for one group (`strata` is `"all"`) as
#'   for several.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
#' )
#' fit <- parametric(Surv(time, status) ~ 1, data = lung, dist = "loglogistic")
#' predict(fit, times = c(100, 365, 730), type = "hazard")
#' quantile(fit, probs = c(0.25, 0.5, 0.75))
#' mean(fit)
#' rmst(fit, tau = 730)
#' @name parametric_predict
#' @export
predict.parametric <- function(object,
                               times,
                               type = c("survival", "cumhaz", "hazard", "density"),
                               conf_level = object$conf_level,
                               ...) {
  type <- match.arg(type)
  check_conf_level(conf_level)
  times <- check_times(times)
  z <- z_value(conf_level)
  x <- matrix(1, 1L, 1L, dimnames = list(NULL, "(Intercept)"))
  tables <- lapply(object$fits, function(fit) {
    if (type %in% c("survival", "cumhaz")) {
      out <- aft_curves(fit, x, times, type, z)
      out$subject <- NULL
      return(out)
    }
    family <- aft_family(fit$dist)
    value <- function(theta) {
      par <- aft_unpack(fit, theta)
      zs <- (log(times) - par$beta[[1L]]) / par$sigma
      density <- family$density(zs) / (par$sigma * times)
      if (type == "density") density else density / family$survival(zs)
    }
    theta <- aft_theta(fit)
    estimate <- value(theta)
    grad <- vapply(seq_along(theta), function(j) {
      h <- 1e-6 * max(abs(theta[j]), 1)
      up <- theta
      down <- theta
      up[j] <- up[j] + h
      down[j] <- down[j] - h
      (value(up) - value(down)) / (2 * h)
    }, numeric(length(times)))
    grad <- matrix(grad, nrow = length(times))
    std_error <- sqrt(pmax(rowSums((grad %*% fit$var) * grad), 0))
    se_log <- std_error / estimate
    data.frame(
      time = times,
      estimate = estimate,
      std_error = std_error,
      conf_low = estimate * exp(-z * se_log),
      conf_high = estimate * exp(z * se_log)
    )
  })
  bind_groups(object$groups, tables)
}

#' @rdname parametric_predict
#' @export
quantile.parametric <- function(x, probs = 0.5, conf_level = x$conf_level, ...) {
  if (!is.numeric(probs) || anyNA(probs) || any(probs <= 0 | probs >= 1)) {
    stop("`probs` must be numbers between 0 and 1 (exclusive).", call. = FALSE)
  }
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  design <- matrix(1, 1L, 1L, dimnames = list(NULL, "(Intercept)"))
  tables <- lapply(x$fits, function(fit) {
    out <- aft_quantiles(fit, design, probs, z)
    data.frame(
      prob = probs,
      time = out$estimate,
      std_error = out$std_error,
      conf_low = out$conf_low,
      conf_high = out$conf_high
    )
  })
  bind_groups(x$groups, tables)
}

#' @rdname parametric_predict
#' @export
median.parametric <- function(x, na.rm = FALSE, ...) {
  out <- stats::quantile(x, probs = 0.5, ...)
  out$prob <- NULL
  out
}

#' @rdname parametric_predict
#' @export
mean.parametric <- function(x, conf_level = x$conf_level, ...) {
  parametric_means(x, Inf, conf_level)
}

#' @rdname parametric_predict
#' @export
rmst.parametric <- function(object, tau, conf_level = object$conf_level, ...) {
  check_number(tau, "tau", min = 0)
  parametric_means(object, tau, conf_level)
}

parametric_means <- function(object, tau, conf_level) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  design <- matrix(1, 1L, 1L, dimnames = list(NULL, "(Intercept)"))
  tables <- lapply(object$fits, function(fit) {
    out <- aft_means(fit, design, tau, z)
    out$subject <- NULL
    if (!is.finite(tau)) {
      out$tau <- NULL
    }
    out
  })
  bind_groups(object$groups, tables)
}

#' Compare parametric distributions
#'
#' `compare_distributions()` fits each of several distributions with [parametric()] and ranks them
#' by AIC, lowest (best) first. AIC penalizes the log-likelihood by twice the number of parameters,
#' and BIC by the number of parameters times the log of the number of observations, so BIC favors
#' simpler distributions more strongly.
#'
#' @inheritParams parametric
#' @param dists The distributions to compare. The default is all four: `"weibull"`,
#'   `"exponential"`, `"lognormal"`, and `"loglogistic"`.
#'
#' @return A data frame with one row per distribution, sorted by AIC: `dist`, `n_params`, `loglik`,
#'   `aic`, and `bic`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
#' )
#' compare_distributions(Surv(time, status) ~ 1, data = lung)
#' @export
compare_distributions <- function(formula,
                                  data,
                                  weights,
                                  subset,
                                  na.action = stats::na.omit,
                                  dists = c("weibull", "exponential", "lognormal", "loglogistic")) {
  dists <- unique(vapply(dists, function(d) {
    match.arg(d, c("weibull", "exponential", "lognormal", "loglogistic"))
  }, character(1)))
  env <- parent.frame()
  call <- match.call()
  call[[1L]] <- quote(greenwood::parametric)
  call$dists <- NULL
  rows <- lapply(dists, function(d) {
    call$dist <- d
    fit <- eval(call, env)
    loglik <- stats::logLik(fit)
    data.frame(
      dist = d,
      n_params = attr(loglik, "df"),
      loglik = as.numeric(loglik),
      aic = stats::AIC(fit),
      bic = stats::BIC(fit),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out <- out[order(out$aic), ]
  rownames(out) <- NULL
  out
}
