# Buckley-James regression: buckley_james() and its methods. No R fixture covers it, so
# tests/testthat/test-buckley-james.R checks it against Greenwood's Python values.

#' Buckley-James regression
#'
#' `buckley_james()` fits the accelerated failure time model \eqn{\log T = x^\top \beta +
#' \varepsilon} without assuming a distribution for the errors \eqn{\varepsilon}. Starting from
#' the least squares fit of the log times, it repeats two steps until \eqn{\beta} settles:
#'
#' 1. The residual \eqn{e_i = \log T_i - x_i^\top \beta} of a censored subject is only known to
#'    exceed its observed value. It's replaced by its expected value given that, under the
#'    Kaplan-Meier curve of the residuals.
#' 2. \eqn{\beta} is refit by least squares to these filled-in log times.
#'
#' The largest residual is taken as an event when the residual curve is fit, so the curve drops to
#' 0 and the expected values are defined. The iterations sometimes settle into a cycle between two
#' values rather than a single one. When the estimates two steps apart agree to within `tol`, the
#' average of the last two is taken as converged.
#'
#' With no censoring, the fit is the least squares fit of the log times, after one step. The model
#' has no closed-form standard errors. With `n_boot`, they come from refitting to `n_boot`
#' resamples of the rows, drawn with R's random number generator (use [set.seed()] to make them
#' reproducible).
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the
#'   covariates on the right, such as `Surv(time, status) ~ age + sex`. Factors are coded with
#'   treatment contrasts. The model always has an intercept.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector. They weight the
#'   residual curve and the least squares fits.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param max_iter The largest number of iterations. The default is `200`.
#' @param tol The convergence tolerance on the largest change in a coefficient. The default is
#'   `1e-6`.
#' @param n_boot The number of bootstrap resamples for the standard errors. The default, `NULL`,
#'   gives none, and the standard errors are `NA`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return A `buckley_james` object. Use [coef()] or [as.data.frame()] for the coefficients and
#'   [predict()] for the linear predictors and survival curves. The object also holds
#'   `std_error`, `iterations`, `converged`, `n`, `n_event`, and with `n_boot`, `boot` (the
#'   coefficients of each resample, one row each).
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
#' fit <- buckley_james(Surv(time, status) ~ age + sex, data = lung)
#' fit
#'
#' # Bootstrap standard errors
#' set.seed(1)
#' booted <- buckley_james(Surv(time, status) ~ age + sex, data = lung, n_boot = 50)
#' as.data.frame(booted)
#' @export
buckley_james <- function(formula,
                          data,
                          weights,
                          subset,
                          na.action = stats::na.omit,
                          max_iter = 200,
                          tol = 1e-6,
                          n_boot = NULL,
                          conf_level = 0.95) {
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  if (tol <= 0) {
    stop("`tol` must be a single number above 0.", call. = FALSE)
  }
  if (!is.null(n_boot)) {
    check_number(n_boot, "n_boot", min = 1)
  }
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`buckley_james()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`buckley_james()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "buckley_james()", types = "right")
  time <- parts$stop
  event <- parts$event == 1
  if (any(time <= 0)) {
    stop("`buckley_james()` needs positive times.", call. = FALSE)
  }
  if (!any(event)) {
    stop("`buckley_james()` needs at least one event.", call. = FALSE)
  }
  design <- covariate_design(frame)
  x <- cbind("(Intercept)" = 1, design$x)
  w <- frame$weights
  log_time <- log(time)

  fit <- bj_fit(log_time, event, x, w, tol, max_iter)
  if (!fit$converged) {
    warning(
      sprintf("`buckley_james()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }
  beta <- stats::setNames(fit$beta, colnames(x))

  boot <- NULL
  std_error <- stats::setNames(rep(NA_real_, ncol(x)), colnames(x))
  if (!is.null(n_boot)) {
    n <- length(time)
    boot <- matrix(NA_real_, n_boot, ncol(x), dimnames = list(NULL, colnames(x)))
    for (b in seq_len(n_boot)) {
      rows <- sample.int(n, n, replace = TRUE)
      if (any(event[rows] & w[rows] > 0)) {
        boot[b, ] <- bj_fit(log_time[rows], event[rows], x[rows, , drop = FALSE], w[rows], tol,
          max_iter
        )$beta
      }
    }
    # Resamples with no events, or in which the covariates are collinear, are left out.
    boot <- boot[stats::complete.cases(boot), , drop = FALSE]
    if (nrow(boot) >= 2L) {
      std_error[] <- apply(boot, 2L, stats::sd)
    }
  }

  structure(
    list(
      coefficients = beta,
      std_error = std_error,
      boot = boot,
      conf_level = conf_level,
      iterations = fit$iterations,
      converged = fit$converged,
      residual_curve = bj_residual_curve(log_time - drop(x %*% beta), event, w),
      n = length(time),
      n_event = sum(event),
      x = design$x,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "buckley_james"
  )
}

# The Buckley-James iterations from the least squares fit. Settling into a cycle between two
# values counts as converged, at their average.
bj_fit <- function(log_time, event, x, w, tol, max_iter) {
  sw <- sqrt(w)
  q <- qr(sw * x)
  if (q$rank < ncol(x)) {
    return(list(beta = rep(NA_real_, ncol(x)), iterations = 0L, converged = FALSE))
  }
  solve_ls <- function(y) drop(qr.coef(q, sw * y))
  beta <- solve_ls(log_time)
  previous <- NULL
  converged <- FALSE
  iterations <- 0L
  for (it in seq_len(max_iter)) {
    iterations <- it
    beta_new <- solve_ls(bj_pseudo_response(log_time, event, x, w, beta))
    if (max(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      converged <- TRUE
      break
    }
    if (it >= 3L && max(abs(beta_new - previous)) < tol) {
      beta <- (beta_new + beta) / 2
      converged <- TRUE
      break
    }
    previous <- beta
    beta <- beta_new
  }
  list(beta = beta, iterations = iterations, converged = converged)
}

# The Kaplan-Meier curve of the residuals, shifted to start above 0, with the largest taken as an
# event so that the curve drops to 0.
bj_residual_curve <- function(residual, event, w) {
  shift <- 1 - min(residual)
  shifted <- residual + shift
  top <- which.max(shifted)
  event[top] <- TRUE
  counts <- count_events(rep(-Inf, length(shifted)), shifted, as.numeric(event), w)
  survival <- cumprod(ifelse(counts$n_risk > 0, 1 - counts$n_event / counts$n_risk, 1))
  list(
    time = counts$time, survival = survival, shift = shift, tau = shifted[top], shifted = shifted
  )
}

# The filled-in log times: a censored subject's log time plus the expected residual life of its
# residual s, the area under the residual curve from s to the largest residual divided by S(s).
bj_pseudo_response <- function(log_time, event, x, w, beta) {
  curve <- bj_residual_curve(log_time - drop(x %*% beta), event, w)
  knots <- c(0, curve$time)
  heights <- c(1, curve$survival)
  area_at <- function(u) {
    j <- findInterval(u, knots)
    c(0, cumsum(heights[-length(heights)] * diff(knots)))[j] + heights[j] * (u - knots[j])
  }
  s <- curve$shifted
  fill <- which(!event & s < curve$tau)
  out <- log_time
  if (length(fill) > 0L) {
    at_s <- step_at(curve$time, curve$survival, s[fill], 1)
    out[fill] <- log_time[fill] + (area_at(curve$tau) - area_at(s[fill])) / at_s
  }
  out
}

#' @export
coef.buckley_james <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.buckley_james <- function(object, ...) {
  if (is.null(object$boot) || nrow(object$boot) < 2L) {
    stop("The variance needs a fit with `n_boot`.", call. = FALSE)
  }
  stats::cov(object$boot)
}

#' @export
nobs.buckley_james <- function(object, ...) {
  object$n
}

#' Predictions and coefficients of a Buckley-James regression
#'
#' `predict()` gives, for each row of `newdata`, the survival curve (`type = "survival"`, the
#' default) or the linear predictor, the predicted location of the log time (`type = "lp"`). With
#' no distribution assumed for the errors, the survival curve is the Kaplan-Meier curve of the
#' fitted residuals, moved by the subject's linear predictor:
#' \eqn{S(t \mid x) = \hat S_e(\log t - x^\top \hat\beta)}. It's a step function, only as
#' reliable as the number of residuals behind it.
#'
#' `as.data.frame()` gives the coefficients, with bootstrap standard errors and Wald tests and
#' limits when the model was fit with `n_boot` (`NA` otherwise).
#'
#' @param object,x A [buckley_james()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"survival"` (the default) or `"lp"`.
#' @param times For `"survival"`, the times to read the curves at.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `type = "lp"`, a numeric vector with one value per row. For `type = "survival"`, a
#'   data frame with the columns `subject` (the row of `newdata`), `time`, and `estimate`.
#'   `as.data.frame()` gives a data frame with the columns `term`, `estimate`, `std_error`,
#'   `statistic`, `p_value`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- buckley_james(Surv(time, status) ~ age + sex, data = lung)
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(100, 300, 500))
#' predict(fit, newdata = newdata, type = "lp")
#' @name buckley_james_predict
#' @export
predict.buckley_james <- function(object, newdata, type = c("survival", "lp"), times, ...) {
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
  if (type == "lp") {
    return(lp)
  }
  if (missing(times)) {
    stop("`type = \"survival\"` needs `times`.", call. = FALSE)
  }
  times <- check_times(times)
  curve <- object$residual_curve
  rows <- lapply(seq_along(lp), function(i) {
    shifted <- log(times) - lp[i] + curve$shift
    estimate <- step_at(curve$time, curve$survival, shifted, 1)
    data.frame(subject = i, time = times, estimate = estimate)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname buckley_james_predict
#' @export
as.data.frame.buckley_james <- function(x,
                                        row.names = NULL,
                                        optional = FALSE,
                                        ...,
                                        conf_level = x$conf_level) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  estimate <- x$coefficients
  std_error <- x$std_error
  statistic <- estimate / std_error
  data.frame(
    term = names(estimate),
    estimate = unname(estimate),
    std_error = unname(std_error),
    statistic = unname(statistic),
    p_value = unname(2 * stats::pnorm(-abs(statistic))),
    conf_low = unname(estimate - z * std_error),
    conf_high = unname(estimate + z * std_error),
    stringsAsFactors = FALSE
  )
}

#' @export
print.buckley_james <- function(x, digits = getOption("digits"), ...) {
  cat("<buckley_james> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "n = %d, events = %d, iterations = %d, converged = %s%s\n\n",
    x$n, x$n_event, x$iterations, if (x$converged) "yes" else "no",
    if (is.null(x$boot)) "" else sprintf(", bootstrap resamples = %d", nrow(x$boot))
  ))
  table <- as.data.frame(x)
  coefs <- table[c("estimate", "std_error", "statistic", "p_value")]
  rownames(coefs) <- table$term
  print(coefs, digits = max(3L, digits - 3L))
  cat_dropped(x$n_dropped)
  invisible(x)
}
