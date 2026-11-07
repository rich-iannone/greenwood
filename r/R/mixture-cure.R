# The mixture cure model: mixture_cure() and its methods. tests/testthat/test-mixture-cure.R checks
# it against the cure_ph_e1684 fixture and Greenwood's Python values.

#' Mixture cure model
#'
#' `mixture_cure()` fits a mixture cure model, in which some subjects will never have the event.
#' A subject is uncured (susceptible) with probability \eqn{\pi(z) = \mathrm{expit}(z^\top
#' \gamma)}, the incidence model, and the uncured follow a proportional hazards model,
#' \eqn{S_u(t \mid x) = S_0(t)^{\exp(x^\top \beta)}}, the latency model. The population survival
#' is
#' \deqn{S(t \mid x, z) = 1 - \pi(z) + \pi(z) S_u(t \mid x),}
#' which levels off at the cured fraction \eqn{1 - \pi(z)} rather than dropping to 0.
#'
#' The model is fit by the EM algorithm. Each step finds the probability \eqn{w_i} that a subject
#' is uncured (1 for an event, and for a censored subject \eqn{\pi S_u / (1 - \pi + \pi S_u)}).
#' Then it refits the logistic incidence model to the \eqn{w_i}, the Cox latency model with
#' \eqn{\log w_i} as an offset (Breslow ties), and the Breslow baseline survival with the
#' \eqn{w_i} in the risk sets. The baseline is 0 after the last event time, so subjects censored
#' later are taken as cured. The steps stop when the summed squared changes in \eqn{\gamma},
#' \eqn{\beta}, and the baseline survival at each subject's time fall to `tol` or below.
#'
#' The model has no closed-form standard errors. They come from refitting to `n_boot` resamples,
#' drawn separately from the events and the censored rows so that each resample keeps the
#' number of events. The resamples use R's random number generator (use [set.seed()] to make
#' them reproducible).
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the latency
#'   covariates on the right, such as `Surv(time, status) ~ treatment`. Factors are coded with
#'   treatment contrasts. The latency model has no intercept (the baseline survival takes its
#'   place).
#' @param cure A one-sided formula with the incidence covariates, such as `~ treatment`. The
#'   incidence model always has an intercept, and `~ 1` gives one cured fraction for everyone.
#' @param data A data frame holding the variables in `formula` and `cure`.
#' @param weights Optional case weights, a column of `data` or a numeric vector. They weight the
#'   incidence and latency fits and the baseline survival, so whole-number weights act as copies.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values in either formula. The default
#'   drops them.
#' @param max_iter The largest number of EM iterations. The default is `50`.
#' @param tol The convergence tolerance on the summed squared changes. The default is `1e-7`.
#' @param n_boot The number of bootstrap resamples for the standard errors. The default is `100`.
#'   `NULL` gives none, and the standard errors are `NA`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return A `mixture_cure` object. Use [coef()] or [as.data.frame()] for the coefficients of
#'   the two models and [predict()] for the probability of being uncured and the survival curves.
#'   The object also holds `cure_coefficients`, `latency_coefficients`, `baseline` (a data frame
#'   of the baseline survival of the uncured, `time` and `survival`, at the event times),
#'   `iterations`, `converged`, `n`, `n_event`, and with `n_boot`, `boot` (the coefficients of
#'   each resample, one row each).
#'
#' @examples
#' melanoma <- data.frame(
#'   time = c(0.3, 5.6, 5.6, 1.5, 6.4, 0.8, 6.7, 3.3, 0.6, 0.7, 3.6, 7.4, 1.8, 1.1, 1, 4.3, 3.7, 3,
#'     1, 2.2, 1.1, 7.7, 5.4, 3.7, 1.4, 0.7, 8, 1.9, 2.2, 8, 1.2, 4.2, 5.6, 1.9, 0.2, 7.7, 0.5, 0.1,
#'     1.1, 7.1),
#'   status = c(1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
#'     1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0),
#'   treatment = rep(c(0, 1), times = 20)
#' )
#'
#' # Bootstrap standard errors, from resamples drawn with R's random number generator
#' set.seed(1)
#' fit <- mixture_cure(Surv(time, status) ~ treatment, cure = ~ treatment, data = melanoma)
#' fit
#'
#' # The probability of being uncured, and the survival curves, of each group
#' groups <- data.frame(treatment = c(0, 1))
#' predict(fit, newdata = groups, type = "uncured")
#' predict(fit, newdata = groups, type = "survival", times = c(1, 3, 6))
#' @export
mixture_cure <- function(formula,
                         cure,
                         data,
                         weights,
                         subset,
                         na.action = stats::na.omit,
                         max_iter = 50,
                         tol = 1e-7,
                         n_boot = 100,
                         conf_level = 0.95) {
  if (missing(cure) || !inherits(cure, "formula") || length(cure) != 2L) {
    stop(
      "`cure` must be a one-sided formula with the incidence covariates, such as `~ x`.",
      call. = FALSE
    )
  }
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  if (tol <= 0) {
    stop("`tol` must be a single number above 0.", call. = FALSE)
  }
  if (!is.null(n_boot)) {
    check_number(n_boot, "n_boot", min = 1)
  }
  check_conf_level(conf_level)
  data_arg <- if (missing(data)) NULL else data
  parts <- split_formula(formula, data_arg)
  if (length(parts$strata) > 0L || length(parts$cluster) > 0L) {
    stop("`mixture_cure()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(parts$offset) || !is.null(attr(stats::terms(cure), "offset"))) {
    stop("`mixture_cure()` doesn't support `offset()` terms.", call. = FALSE)
  }

  # One model frame for both formulas, so that a row missing a variable of either is dropped.
  latency_rhs <- stats::reformulate(
    if (length(parts$terms) == 0L) "1" else parts$terms,
    intercept = TRUE, env = parts$env
  )
  variables <- unique(c(all.vars(latency_rhs), all.vars(cure)))
  combined <- stats::reformulate(
    if (length(variables) == 0L) "1" else variables,
    response = parts$response, env = parts$env
  )
  frame <- gw_model_frame(match.call(expand.dots = FALSE), parent.frame(), combined, data_arg)
  parts_y <- surv_parts(frame$response, "mixture_cure()", types = "right")
  time <- parts_y$stop
  status <- as.numeric(parts_y$event == 1)
  w <- frame$weights
  if (!any(status == 1 & w > 0)) {
    stop("`mixture_cure()` needs at least one event.", call. = FALSE)
  }
  sub_design <- function(rhs) {
    mf <- stats::model.frame(rhs, data = frame$mf, drop.unused.levels = TRUE)
    covariate_design(list(mf = mf))
  }
  latency <- sub_design(latency_rhs)
  incidence <- sub_design(cure)
  x <- latency$x
  z <- cbind("(Intercept)" = 1, incidence$x)

  fit <- cure_em(time, status, x, z, w, max_iter, tol)
  if (!fit$converged) {
    warning(
      sprintf("`mixture_cure()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }
  gamma <- stats::setNames(fit$gamma, colnames(z))
  beta <- stats::setNames(fit$beta, colnames(x))
  estimate <- c(cure = gamma, latency = beta)

  boot <- NULL
  std_error <- stats::setNames(rep(NA_real_, length(estimate)), names(estimate))
  if (!is.null(n_boot)) {
    events <- which(status == 1)
    censored <- which(status == 0)
    boot <- matrix(NA_real_, n_boot, length(estimate), dimnames = list(NULL, names(estimate)))
    for (b in seq_len(n_boot)) {
      rows <- c(
        events[sample.int(length(events), length(events), replace = TRUE)],
        censored[sample.int(length(censored), length(censored), replace = TRUE)]
      )
      refit <- tryCatch(
        suppressWarnings(cure_em(
          time[rows], status[rows], x[rows, , drop = FALSE], z[rows, , drop = FALSE], w[rows],
          max_iter, tol
        )),
        error = function(e) NULL
      )
      if (!is.null(refit)) {
        boot[b, ] <- c(refit$gamma, refit$beta)
      }
    }
    # Resamples in which a fit fails, or diverges, are left out.
    boot <- boot[stats::complete.cases(boot) & apply(is.finite(boot), 1L, all), , drop = FALSE]
    if (nrow(boot) >= 2L) {
      std_error[] <- apply(boot, 2L, stats::sd)
    }
  }

  structure(
    list(
      coefficients = estimate,
      cure_coefficients = gamma,
      latency_coefficients = beta,
      std_error = std_error,
      boot = boot,
      conf_level = conf_level,
      baseline = data.frame(time = fit$baseline_time, survival = fit$baseline_survival),
      iterations = fit$iterations,
      converged = fit$converged,
      n = length(time),
      n_event = as.integer(sum(status)),
      x = x,
      z = z,
      latency_design = latency[c("terms", "xlevels", "contrasts")],
      cure_design = incidence[c("terms", "xlevels", "contrasts")],
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      cure = cure,
      call = match.call()
    ),
    class = "mixture_cure"
  )
}

# The EM fit. `w` holds the case weights, and the rows with weight 0 are left out.
cure_em <- function(time, status, x, z, w, max_iter, tol) {
  keep <- w > 0
  time <- time[keep]
  status <- status[keep]
  x <- x[keep, , drop = FALSE]
  z <- z[keep, , drop = FALSE]
  w <- w[keep]

  uncured <- status
  gamma <- cure_logistic(z, uncured, w, numeric(ncol(z)))
  beta <- cure_cox(x, time, status, uncured, w, numeric(ncol(x)))
  baseline <- cure_breslow(time, status, x, beta, uncured, w)
  s <- cure_baseline_at(baseline, time)
  converged <- FALSE
  iterations <- as.integer(max_iter)
  for (it in seq_len(max_iter)) {
    pi <- stats::plogis(drop(z %*% gamma))
    s_u <- s^exp(linear_predictor(x, beta))
    uncured <- status + (1 - status) * pi * s_u / (1 - pi + pi * s_u)
    gamma_new <- cure_logistic(z, uncured, w, gamma)
    beta_new <- cure_cox(x, time, status, uncured, w, beta)
    # The baseline is updated with the previous step's beta, as in the reference algorithm.
    baseline <- cure_breslow(time, status, x, beta, uncured, w)
    s_new <- cure_baseline_at(baseline, time)
    change <- sum((gamma_new - gamma)^2) + sum((beta_new - beta)^2) + sum(w * (s_new - s)^2)
    gamma <- gamma_new
    beta <- beta_new
    s <- s_new
    if (change <= tol) {
      converged <- TRUE
      iterations <- it
      break
    }
  }
  list(
    gamma = gamma, beta = beta, baseline_time = baseline$time,
    baseline_survival = baseline$survival, iterations = iterations, converged = converged
  )
}

# Newton-Raphson for a concave log-likelihood, halving steps that lower it. `objective` gives
# the log-likelihood, its gradient, and the information matrix.
cure_newton <- function(objective, start, max_iter = 100L, tol = 1e-10) {
  par <- start
  if (length(par) == 0L) {
    return(par)
  }
  current <- objective(par)
  for (it in seq_len(max_iter)) {
    step <- tryCatch(
      solve(current$info, current$grad, tol = 0),
      error = function(e) NULL
    )
    if (is.null(step) || any(!is.finite(step))) {
      stop("The information matrix is singular, so the model can't be fit.", call. = FALSE)
    }
    for (halving in 0:30) {
      candidate <- objective(par + step)
      if (is.finite(candidate$loglik) &&
        candidate$loglik >= current$loglik - 1e-12 * abs(current$loglik)) {
        break
      }
      step <- step / 2
    }
    par <- par + step
    current <- candidate
    if (max(abs(step)) < tol) {
      break
    }
  }
  par
}

# The logistic incidence model fit to the probabilities `y` of being uncured, with case weights
# `w`. With `y` between 0 and 1, it's the quasi-binomial fit.
cure_logistic <- function(z, y, w, start) {
  objective <- function(gamma) {
    eta <- drop(z %*% gamma)
    p <- stats::plogis(eta)
    list(
      loglik = sum(w * (y * stats::plogis(eta, log.p = TRUE) +
        (1 - y) * stats::plogis(-eta, log.p = TRUE))),
      grad = drop(crossprod(z, w * (y - p))),
      info = crossprod(z * (w * p * (1 - p)), z)
    )
  }
  cure_newton(objective, start)
}

# The Cox latency model with offset log(`uncured`), Breslow ties, and case weights `w`, fit to the
# rows with `uncured` above 0.
cure_cox <- function(x, time, status, uncured, w, start) {
  if (ncol(x) == 0L) {
    return(start)
  }
  keep <- uncured > 0
  order <- order(time[keep])
  x <- x[keep, , drop = FALSE][order, , drop = FALSE]
  time <- time[keep][order]
  dead <- (status * w)[keep][order]
  offset <- log(uncured[keep][order])
  w <- w[keep][order]
  p <- ncol(x)
  death_times <- unique(time[dead > 0])
  first <- findInterval(death_times, time, left.open = TRUE) + 1L
  d <- tabulate_sum(match(time[dead > 0], death_times), dead[dead > 0], length(death_times))
  from_end <- function(v) apply(as.matrix(v), 2L, function(col) rev(cumsum(rev(col))))
  pairs <- x[, rep(seq_len(p), times = p), drop = FALSE] * x[, rep(seq_len(p), each = p),
    drop = FALSE
  ]
  objective <- function(beta) {
    eta <- linear_predictor(x, beta) + offset
    top <- max(eta)
    r <- w * exp(eta - top)
    s0 <- rev(cumsum(rev(r)))[first]
    s1 <- from_end(r * x)[first, , drop = FALSE]
    s2 <- from_end(r * pairs)[first, , drop = FALSE]
    mean_x <- s1 / s0
    info <- matrix(colSums(d * (s2 / s0)), p, p) - crossprod(mean_x * sqrt(d))
    list(
      loglik = sum(dead * eta) - sum(d * (log(s0) + top)),
      grad = colSums(dead * x) - colSums(d * mean_x),
      info = info
    )
  }
  cure_newton(objective, start)
}

# The Breslow baseline survival of the uncured at the event times. Each event time's hazard is
# the weighted number of events over the weighted risk at or after it.
cure_breslow <- function(time, status, x, beta, uncured, w) {
  dead <- status * w
  death_times <- sort(unique(time[dead > 0]))
  d <- tabulate_sum(match(time[dead > 0], death_times), dead[dead > 0], length(death_times))
  risk <- weight_at_or_above(time, w * uncured * exp(linear_predictor(x, beta)), death_times)
  list(time = death_times, survival = exp(-cumsum(d / risk)))
}

# The baseline survival at `times`: 1 before the first event time, and 0 after the last.
cure_baseline_at <- function(baseline, times) {
  out <- step_at(baseline$time, baseline$survival, times, 1)
  out[times > max(baseline$time)] <- 0
  out
}

#' @export
coef.mixture_cure <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.mixture_cure <- function(object, ...) {
  if (is.null(object$boot) || nrow(object$boot) < 2L) {
    stop("The variance needs a fit with `n_boot`.", call. = FALSE)
  }
  stats::cov(object$boot)
}

#' @export
nobs.mixture_cure <- function(object, ...) {
  object$n
}

#' Predictions and coefficients of a mixture cure model
#'
#' `predict()` gives, for each row of `newdata`, the population survival curve
#' \eqn{1 - \pi + \pi S_u(t)} (`type = "survival"`, the default), the probability \eqn{\pi} of
#' being uncured (`type = "uncured"`), or the linear predictor of the latency model
#' (`type = "lp"`). The survival curves level off at the cured fraction \eqn{1 - \pi} after the
#' last event time.
#'
#' `as.data.frame()` gives the coefficients of both models, with bootstrap standard errors and
#' Wald tests and limits when the model was fit with `n_boot` (`NA` otherwise).
#'
#' @param object,x A [mixture_cure()] fit.
#' @param newdata A data frame with the covariates of both models for the subjects to predict
#'   for. The default is the data the model was fit to.
#' @param type What to predict: `"survival"` (the default), `"uncured"`, or `"lp"`.
#' @param times For `"survival"`, the times to read the curves at.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `type = "uncured"` and `type = "lp"`, a numeric vector with one value per row. For
#'   `type = "survival"`, a data frame with the columns `subject` (the row of `newdata`), `time`,
#'   and `estimate`. `as.data.frame()` gives a data frame with the columns `submodel` (`"cure"` or
#'   `"latency"`), `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, and
#'   `conf_high`.
#'
#' @examples
#' melanoma <- data.frame(
#'   time = c(0.3, 5.6, 5.6, 1.5, 6.4, 0.8, 6.7, 3.3, 0.6, 0.7, 3.6, 7.4, 1.8, 1.1, 1, 4.3, 3.7, 3,
#'     1, 2.2, 1.1, 7.7, 5.4, 3.7, 1.4, 0.7, 8, 1.9, 2.2, 8, 1.2, 4.2, 5.6, 1.9, 0.2, 7.7, 0.5, 0.1,
#'     1.1, 7.1),
#'   status = c(1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0,
#'     1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0),
#'   treatment = rep(c(0, 1), times = 20)
#' )
#' fit <- mixture_cure(Surv(time, status) ~ treatment, cure = ~ treatment, data = melanoma,
#'   n_boot = NULL
#' )
#' groups <- data.frame(treatment = c(0, 1))
#' predict(fit, newdata = groups, type = "uncured")
#' predict(fit, newdata = groups, type = "survival", times = c(1, 3, 6, 9))
#' as.data.frame(fit)
#' @name mixture_cure_predict
#' @export
predict.mixture_cure <- function(object,
                                 newdata,
                                 type = c("survival", "uncured", "lp"),
                                 times,
                                 ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    x <- object$x
    z <- object$z
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    new_design <- function(design, names) {
      design$coefficients <- stats::setNames(numeric(length(names)), names)
      cox_new_design(design, newdata)
    }
    x <- new_design(object$latency_design, names(object$latency_coefficients))
    z <- cbind(1, new_design(object$cure_design, names(object$cure_coefficients)[-1L]))
  }
  lp <- unname(linear_predictor(x, object$latency_coefficients))
  if (type == "lp") {
    return(lp)
  }
  pi <- unname(stats::plogis(drop(z %*% object$cure_coefficients)))
  if (type == "uncured") {
    return(pi)
  }
  if (missing(times)) {
    stop("`type = \"survival\"` needs `times`.", call. = FALSE)
  }
  times <- check_times(times)
  s0 <- cure_baseline_at(object$baseline, times)
  rows <- lapply(seq_along(lp), function(i) {
    data.frame(subject = i, time = times, estimate = 1 - pi[i] + pi[i] * s0^exp(lp[i]))
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname mixture_cure_predict
#' @export
as.data.frame.mixture_cure <- function(x,
                                       row.names = NULL,
                                       optional = FALSE,
                                       ...,
                                       conf_level = x$conf_level) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  estimate <- unname(x$coefficients)
  std_error <- unname(x$std_error)
  statistic <- estimate / std_error
  data.frame(
    submodel = rep(c("cure", "latency"),
      c(length(x$cure_coefficients), length(x$latency_coefficients))
    ),
    term = c(names(x$cure_coefficients), names(x$latency_coefficients)),
    estimate = estimate,
    std_error = std_error,
    statistic = statistic,
    p_value = 2 * stats::pnorm(-abs(statistic)),
    conf_low = estimate - z * std_error,
    conf_high = estimate + z * std_error,
    stringsAsFactors = FALSE
  )
}

#' @export
print.mixture_cure <- function(x, digits = getOption("digits"), ...) {
  cat(
    "<mixture_cure> ", deparse_formula(x$formula), ", cure = ", deparse_formula(x$cure), "\n",
    sep = ""
  )
  cat(sprintf(
    "n = %d, events = %d, EM iterations = %d, converged = %s%s\n",
    x$n, x$n_event, x$iterations, if (x$converged) "yes" else "no",
    if (is.null(x$boot)) "" else sprintf(", bootstrap resamples = %d", nrow(x$boot))
  ))
  table <- as.data.frame(x)
  show <- function(rows) {
    coefs <- table[rows, c("estimate", "std_error", "statistic", "p_value")]
    rownames(coefs) <- table$term[rows]
    print(coefs, digits = max(3L, digits - 3L))
  }
  cat("\nIncidence (log odds of being uncured):\n")
  show(table$submodel == "cure")
  cat("\nLatency (log hazard ratios among the uncured):\n")
  if (length(x$latency_coefficients) == 0L) {
    cat("No covariates.\n")
  } else {
    show(table$submodel == "latency")
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}
