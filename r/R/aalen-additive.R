# Aalen's additive hazards model: aalen_additive() and its methods. The fits in fixtures/r/
# (aalen_lung_age_sex, aalen_veteran) are replayed by tests/testthat/test-aalen-additive.R.

#' Aalen's additive hazards model
#'
#' `aalen_additive()` fits Aalen's additive hazards model, in which the hazard is a sum of
#' covariate effects that change over time:
#' \deqn{h(t \mid x) = \beta_0(t) + \beta_1(t) x_1 + \cdots + \beta_p(t) x_p.}
#' Nothing is assumed about the shape of the \eqn{\beta_j(t)}, so the model suits effects that
#' grow, fade, or change sign over follow-up. What's estimated is the cumulative regression
#' function \eqn{B_j(t) = \int_0^t \beta_j(s) ds} of each term.
#'
#' At each event time, the increments \eqn{dB(t)} are the least squares fit of the event to the
#' covariates of the subjects at risk, weighted by the case weights. The cumulative coefficients
#' add these up over time. Event times with fewer than `nmin` subjects at risk (by weight), or at
#' which the covariates of those at risk are collinear, are left out, since the fit is unstable
#' there.
#'
#' The table gives a test of each term. The statistic is a weighted sum of its increments,
#' \eqn{U_j = \sum_t w_j(t) dB_j(t)}, with variance \eqn{\sum_t w_j(t)^2 dB_j(t)^2}. With
#' `test = "aalen"` (the default), \eqn{w_j(t)} is the number at risk divided by the variance
#' factor of the term in the regression at \eqn{t}. With `test = "nrisk"`, it's the number at
#' risk. `estimate` is \eqn{U_j} scaled to the size of an increment: divided by the sum of the
#' weights for `"aalen"`, and by the number of events used for `"nrisk"`. `slope` is the slope of
#' a weighted least squares line through the origin fitted to the cumulative weighted increments,
#' an average of \eqn{\beta_j(t)} over time. A chi-square test of all the covariates together
#' takes the covariance of the statistics into account.
#'
#' @param formula A formula with a right-censored or counting-process [Surv()] response on the
#'   left and the covariates on the right, such as `Surv(time, status) ~ age + sex`. Factors are
#'   coded with treatment contrasts. The model always has an intercept, \eqn{\beta_0(t)}.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param nmin The smallest number at risk (the sum of the weights of those at risk) for an event
#'   time to be used. The default is three times the number of covariate columns.
#' @param test The weights of the tests: `"aalen"` (the default) or `"nrisk"`.
#' @param qrtol The tolerance for deciding that the covariates at risk are collinear. The default
#'   is `1e-7`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return An `aalen_additive` object. Use [as.data.frame()] for the table of tests,
#'   [cumulative_coefficients()] for \eqn{B_j(t)} with its confidence limits, and [predict()] for
#'   survival and cumulative hazard curves. [coef()] and [vcov()] give the scaled statistics
#'   (`estimate` in the table) and their variance.
#'
#'   The object also holds `times`, `n_risk`, and `increments` (one row per event used, so tied
#'   events have a row each), `test_statistic` and `test_var` (the unscaled statistics and their
#'   variance), `wald_test` (the chi-square test of the covariates: `statistic`, `df`, `p_value`),
#'   `n`, `n_event`, and `n_times` (the number of distinct event times used).
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
#' fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
#' fit
#'
#' # The cumulative effect of sex over time
#' coefs <- cumulative_coefficients(fit)
#' coefs[coefs$term == "sex", ]
#'
#' # Survival curves for two new subjects
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(100, 200))
#' @export
aalen_additive <- function(formula,
                           data,
                           weights,
                           subset,
                           na.action = stats::na.omit,
                           nmin,
                           test = c("aalen", "nrisk"),
                           qrtol = 1e-7,
                           conf_level = 0.95) {
  test <- match.arg(test)
  check_number(qrtol, "qrtol", min = 0)
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop("`aalen_additive()` doesn't support `strata()` or `cluster()` terms.", call. = FALSE)
  }
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`aalen_additive()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "aalen_additive()")
  design <- covariate_design(frame)
  x <- design$x
  p <- ncol(x)
  if (p == 0L) {
    stop("`aalen_additive()` needs at least one covariate.", call. = FALSE)
  }
  if (missing(nmin)) {
    nmin <- 3 * p
  }
  check_number(nmin, "nmin", min = 0)
  event <- parts$event
  if (sum(event) == 0) {
    stop("`aalen_additive()` needs at least one event.", call. = FALSE)
  }
  fit <- aa_increments(parts$start, parts$stop, event, frame$weights, x, nmin, qrtol)
  if (length(fit$times) == 0L) {
    stop(
      "No event time has `nmin` or more at risk with covariates that aren't collinear.",
      call. = FALSE
    )
  }
  terms <- c("Intercept", colnames(x))
  colnames(fit$increments) <- terms

  tw <- if (test == "aalen") fit$tweight else matrix(fit$n_risk, length(fit$times), p + 1L)
  tx <- tw * fit$increments
  statistic <- colSums(tx)
  test_var <- crossprod(tx)
  scale <- if (test == "aalen") colSums(tw) else rep(length(fit$times), p + 1L)
  slope <- colSums(apply(tx, 2L, cumsum) * fit$times) / colSums(tw * fit$times^2)
  covariates <- -1L
  chisq <- tryCatch(
    drop(statistic[covariates] %*% solve(test_var[covariates, covariates], statistic[covariates])),
    error = function(e) NA_real_
  )
  names(statistic) <- names(slope) <- terms
  dimnames(test_var) <- list(terms, terms)
  var <- test_var / outer(scale, scale)

  structure(
    list(
      coefficients = statistic / scale,
      var = var,
      slope = slope,
      test_statistic = statistic,
      test_var = test_var,
      wald_test = list(
        statistic = chisq, df = p, p_value = stats::pchisq(chisq, p, lower.tail = FALSE)
      ),
      test = test,
      times = fit$times,
      n_risk = fit$n_risk,
      n_event_at = fit$n_event_at,
      increments = fit$increments,
      tweight = fit$tweight,
      nmin = nmin,
      n = length(event),
      n_event = sum(event),
      n_times = length(unique(fit$times)),
      conf_level = conf_level,
      x = x,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "aalen_additive"
  )
}

# The least squares increments at each event, in time order (tied events in row order). At each
# event time, the regression of the events on the covariates of those at risk has slopes
# V^{-1} w_i (x_i - m) / N for event i, where N is the weight at risk, m the weighted means, and V
# the weighted covariance, and an intercept of w_i / N - m' slopes. `tweight` holds Aalen's test
# weights: N / (1 + m' V^{-1} m) for the intercept and N / diag(V^{-1}) for the covariates.
aa_increments <- function(start, stop, event, w, x, nmin, qrtol) {
  p <- ncol(x)
  deaths <- which(event == 1)
  deaths <- deaths[order(stop[deaths])]
  times <- unique(stop[deaths])
  pieces <- vector("list", length(times))
  for (k in seq_along(times)) {
    s <- times[k]
    at_risk <- start < s & stop >= s
    wr <- w[at_risk]
    n_risk <- sum(wr)
    if (n_risk < nmin || n_risk <= 0) {
      next
    }
    xr <- x[at_risk, , drop = FALSE]
    means <- colSums(wr * xr) / n_risk
    centred <- xr - rep(means, each = nrow(xr))
    v <- crossprod(centred * sqrt(wr)) / n_risk
    q <- qr(v, tol = qrtol)
    if (q$rank < p) {
      next
    }
    vmat <- qr.coef(q, diag(p))
    who <- deaths[stop[deaths] == s]
    xd <- x[who, , drop = FALSE] - rep(means, each = length(who))
    slopes <- t(qr.coef(q, t(w[who] * xd))) / n_risk
    slopes <- matrix(slopes, nrow = length(who))
    b0 <- w[who] / n_risk - drop(slopes %*% means)
    twt <- n_risk / c(1 + drop(means %*% vmat %*% means), diag(vmat))
    pieces[[k]] <- list(
      times = rep(s, length(who)),
      n_risk = rep(n_risk, length(who)),
      n_event_at = rep(sum(w[who]), length(who)),
      increments = cbind(b0, slopes),
      tweight = matrix(twt, length(who), p + 1L, byrow = TRUE)
    )
  }
  pieces <- pieces[!vapply(pieces, is.null, logical(1))]
  list(
    times = unlist(lapply(pieces, `[[`, "times")),
    n_risk = unlist(lapply(pieces, `[[`, "n_risk")),
    n_event_at = unlist(lapply(pieces, `[[`, "n_event_at")),
    increments = do.call(rbind, lapply(pieces, `[[`, "increments")),
    tweight = do.call(rbind, lapply(pieces, `[[`, "tweight"))
  )
}

#' @export
coef.aalen_additive <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.aalen_additive <- function(object, ...) {
  object$var
}

#' @export
nobs.aalen_additive <- function(object, ...) {
  object$n
}

#' Table of an Aalen additive hazards model
#'
#' `as.data.frame()` gives the tests of an [aalen_additive()] fit, one row per term.
#'
#' @param x An [aalen_additive()] fit.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#'
#' @return A data frame with the columns `term`, `slope`, `estimate` (the scaled statistic),
#'   `std_error`, `statistic` (the z statistic), `p_value`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
#' as.data.frame(fit)
#' @name aalen_additive_table
#' @export
as.data.frame.aalen_additive <- function(x,
                                         row.names = NULL,
                                         optional = FALSE,
                                         ...,
                                         conf_level = x$conf_level) {
  check_conf_level(conf_level)
  estimate <- x$coefficients
  std_error <- sqrt(diag(x$var))
  z <- z_value(conf_level)
  statistic <- x$test_statistic / sqrt(diag(x$test_var))
  data.frame(
    term = names(estimate),
    slope = unname(x$slope),
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
print.aalen_additive <- function(x, digits = getOption("digits"), ...) {
  cat("<aalen_additive> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%s test weights. n = %d, events = %s, event times used = %d\n\n",
    if (x$test == "aalen") "Aalen" else "Number-at-risk",
    x$n, format(x$n_event), x$n_times
  ))
  table <- as.data.frame(x)
  coefs <- table[c("slope", "estimate", "std_error", "statistic", "p_value")]
  rownames(coefs) <- table$term
  print(coefs, digits = max(3L, digits - 3L))
  short <- max(3L, digits - 4L)
  cat(sprintf(
    "\nChi-square test of the covariates: %s on %d df, p = %s\n",
    format(x$wald_test$statistic, digits = short),
    as.integer(x$wald_test$df),
    format_p(x$wald_test$p_value)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

#' Cumulative regression coefficients of a model
#'
#' `cumulative_coefficients()` gives the cumulative regression coefficients of a model whose
#' effects change over time, at every distinct time, with standard errors and confidence limits.
#'
#' @param object A fitted model.
#' @param ... Passed on to methods.
#'
#' @return A data frame. See the methods for the columns.
#'
#' @export
cumulative_coefficients <- function(object, ...) {
  UseMethod("cumulative_coefficients")
}

#' @describeIn cumulative_coefficients For an [aalen_additive()] fit, \eqn{B_j(t)} for the
#'   intercept and each covariate, at each distinct event time used. The standard errors are the
#'   square roots of the summed squared increments, and the limits are
#'   \eqn{B_j(t) \pm z} standard errors. The table has the columns `term`, `time`, `n_risk`,
#'   `n_event`, `estimate`, `std_error`, `conf_low`, and `conf_high`, one block of rows per term.
#'
#' @param conf_level The confidence level. The default is the level the model was fit with.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
#' cumulative_coefficients(fit)
#' @export
cumulative_coefficients.aalen_additive <- function(object, conf_level = object$conf_level, ...) {
  check_conf_level(conf_level)
  z <- z_value(conf_level)
  # The last row at each time holds the sums after all the events there.
  last <- !duplicated(object$times, fromLast = TRUE)
  estimate <- apply(object$increments, 2L, cumsum)[last, , drop = FALSE]
  std_error <- sqrt(apply(object$increments^2, 2L, cumsum))[last, , drop = FALSE]
  terms <- colnames(object$increments)
  k <- sum(last)
  data.frame(
    term = rep(terms, each = k),
    time = rep(object$times[last], length(terms)),
    n_risk = rep(object$n_risk[last], length(terms)),
    n_event = rep(object$n_event_at[last], length(terms)),
    estimate = as.vector(estimate),
    std_error = as.vector(std_error),
    conf_low = as.vector(estimate - z * std_error),
    conf_high = as.vector(estimate + z * std_error),
    stringsAsFactors = FALSE
  )
}

#' Predictions from an Aalen additive hazards model
#'
#' `predict()` gives, for each row of `newdata` (or of the data the model was fit to), the
#' cumulative hazard \eqn{H(t \mid x) = B_0(t) + \sum_j B_j(t) x_j} or the survival
#' \eqn{\exp(-H(t \mid x))}. An additive model can give a cumulative hazard that falls below 0 for
#' some covariate values, so it's floored at 0.
#'
#' The standard error of the cumulative hazard is \eqn{\sqrt{x^\top \Sigma(t) x}}, where
#' \eqn{\Sigma(t)} sums the outer products of the increments up to \eqn{t}. Its limits are
#' \eqn{H \pm z} standard errors, floored at 0, and the survival limits follow from them.
#'
#' @param object An [aalen_additive()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"survival"` (the default) or `"cumhaz"`.
#' @param times The times to read the curves at. The default is the distinct event times used in
#'   the fit.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return A data frame with one row per subject and time: `subject` (the row of `newdata`),
#'   `time`, `estimate`, `std_error` (of the cumulative hazard), `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "cumhaz", times = c(100, 200, 300))
#' @name aalen_additive_predict
#' @export
predict.aalen_additive <- function(object,
                                   newdata,
                                   type = c("survival", "cumhaz"),
                                   times,
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
  times <- if (missing(times)) unique(object$times) else check_times(times)
  z <- z_value(conf_level)
  # The cumulative sums after the last event at or before each time, with zeros before the first.
  at <- findInterval(times, object$times)
  cumulative <- rbind(0, apply(object$increments, 2L, cumsum))[at + 1L, , drop = FALSE]
  p1 <- ncol(object$increments)
  outer_inc <- object$increments[, rep(seq_len(p1), p1), drop = FALSE] *
    object$increments[, rep(seq_len(p1), each = p1), drop = FALSE]
  sigma <- rbind(0, apply(outer_inc, 2L, cumsum))[at + 1L, , drop = FALSE]
  rows <- lapply(seq_len(nrow(x)), function(i) {
    xi <- c(1, x[i, ])
    cumhaz <- drop(cumulative %*% xi)
    std_error <- sqrt(pmax(drop(sigma %*% as.vector(outer(xi, xi))), 0))
    estimate <- pmax(cumhaz, 0)
    low <- pmax(cumhaz - z * std_error, 0)
    high <- pmax(cumhaz + z * std_error, 0)
    if (type == "survival") {
      out <- list(estimate = exp(-estimate), conf_low = exp(-high), conf_high = exp(-low))
    } else {
      out <- list(estimate = estimate, conf_low = low, conf_high = high)
    }
    data.frame(
      subject = i, time = times, estimate = out$estimate, std_error = std_error,
      conf_low = out$conf_low, conf_high = out$conf_high
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
