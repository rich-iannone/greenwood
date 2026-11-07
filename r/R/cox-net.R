# Elastic-net penalized Cox regression: cox_net(), cv_coxnet(), and their methods. Tested by
# tests/testthat/test-cox-net.R, against cox_ph() for an unpenalized fit, the optimality
# conditions of the penalized fit, and Greenwood's Python values.

#' Elastic-net penalized Cox regression
#'
#' `cox_net()` fits the Cox proportional hazards model with an elastic-net penalty, for when there
#' are many covariates for the number of events, or covariates that are strongly correlated. The
#' coefficients minimize the negative log partial likelihood, divided by the total weight
#' \eqn{W} (the number of rows without case weights), plus the penalty
#' \deqn{\lambda \left(\alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2\right),}
#' with \eqn{\lambda} the `penalizer` and \eqn{\alpha} the `l1_ratio`. `l1_ratio = 1` is the lasso,
#' which sets some coefficients to exactly 0 and so selects covariates. `l1_ratio = 0` is ridge
#' regression, which shrinks all of the coefficients toward 0 without removing any. Values in
#' between mix the two. Tied event times are handled with Breslow's method, and `penalizer = 0`
#' gives the unpenalized Cox fit with Breslow ties.
#'
#' With `standardize = TRUE` (the default), each covariate is penalized on the scale of its
#' standard deviation, so the penalty treats all of them alike whatever their units. The
#' coefficients are reported on the scale of the covariates either way. The fit uses accelerated
#' proximal gradient descent (FISTA) from \eqn{\beta = 0}, with a backtracking line search, and
#' stops once the relative change in the coefficients is below `tol`.
#'
#' Penalized coefficients are biased toward 0 by design, so no standard errors or p-values are
#' given. Use [cv_coxnet()] to choose the `penalizer`. The effective degrees of freedom are the
#' number of nonzero coefficients for the lasso, and otherwise the trace of
#' \eqn{I_A (I_A + W \lambda (1 - \alpha))^{-1}}, with \eqn{I_A} the information matrix of the
#' standardized nonzero coefficients. [logLik()] uses them, so [AIC()] and [BIC()] work, with
#' [BIC()] using the number of events.
#'
#' @param formula A formula with a right-censored or counting-process [Surv()] response on the
#'   left and the covariates on the right, such as `Surv(time, status) ~ age + sex`. Factors are
#'   coded with treatment contrasts. Collinear covariates are allowed.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param penalizer The penalty strength \eqn{\lambda}, a non-negative number. The default is
#'   `0.1`.
#' @param l1_ratio The elastic-net mixing \eqn{\alpha}, between 0 (ridge) and 1 (lasso). The
#'   default is `0.5`.
#' @param standardize Whether to penalize each covariate on the scale of its standard deviation.
#'   The default is `TRUE`.
#' @param max_iter The largest number of iterations. The default is `1000`.
#' @param tol The convergence tolerance for the relative change in the coefficients. The default
#'   is `1e-7`.
#'
#' @return A `cox_net` object. Use [coef()] or [as.data.frame()] for the coefficients and
#'   [predict()] for linear predictors, risk scores, and survival curves. The object also holds
#'   `loglik` (the log partial likelihood at the estimate), `effective_df`, `n_nonzero`, `center`
#'   and `scale` (each covariate's weighted mean and the scale it was penalized on), `n`,
#'   `n_event`, `iterations`, and `converged`.
#'
#' @seealso [cv_coxnet()] to choose the penalizer, and [cox_ph()] for the unpenalized model.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
#'   ecog = c(1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1)
#' )
#'
#' # The lasso, which sets some coefficients to 0
#' fit <- cox_net(Surv(time, status) ~ age + sex + ecog, data = lung, penalizer = 0.1,
#'   l1_ratio = 1
#' )
#' fit
#'
#' # Ridge regression shrinks every coefficient
#' ridge <- cox_net(Surv(time, status) ~ age + sex + ecog, data = lung, penalizer = 0.1,
#'   l1_ratio = 0
#' )
#' as.data.frame(ridge)
#'
#' # Survival curves for new subjects
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2), ecog = c(0, 2))
#' predict(ridge, newdata = newdata, type = "survival", times = c(200, 400))
#' @export
cox_net <- function(formula,
                    data,
                    weights,
                    subset,
                    na.action = stats::na.omit,
                    penalizer = 0.1,
                    l1_ratio = 0.5,
                    standardize = TRUE,
                    max_iter = 1000,
                    tol = 1e-7) {
  check_coxnet_args(l1_ratio, standardize, max_iter, tol)
  check_number(penalizer, "penalizer", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- coxnet_input(frame, formula, if (missing(data)) NULL else data, "cox_net()")

  fit <- coxnet_fit(
    input$x, input$start, input$stop, input$event, input$w, penalizer, l1_ratio, standardize,
    max_iter, tol
  )
  if (!fit$converged) {
    warning(
      sprintf("`cox_net()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }
  names_x <- colnames(input$x)
  beta <- stats::setNames(fit$beta, names_x)
  setup <- coxnet_setup(input$start, input$stop, input$event, input$w)
  xc <- sweep(input$x, 2L, fit$center)
  cumhaz <- cumsum(coxnet_hazard(setup, linear_predictor(xc, beta)))

  structure(
    list(
      coefficients = beta,
      penalizer = penalizer,
      l1_ratio = l1_ratio,
      standardize = standardize,
      loglik = fit$loglik,
      effective_df = fit$effective_df,
      n_nonzero = sum(beta != 0),
      center = stats::setNames(fit$center, names_x),
      scale = stats::setNames(fit$scale, names_x),
      baseline = data.frame(time = setup$times, cumhaz = cumhaz),
      times = sort(unique(input$stop)),
      n = nrow(input$x),
      n_event = sum(input$event),
      iterations = fit$iterations,
      converged = fit$converged,
      x = input$x,
      terms = input$terms,
      xlevels = input$xlevels,
      contrasts = input$contrasts,
      has_weights = frame$has_weights,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "cox_net"
  )
}

check_coxnet_args <- function(l1_ratio, standardize, max_iter, tol) {
  if (!is.numeric(l1_ratio) || length(l1_ratio) != 1L || is.na(l1_ratio) || l1_ratio < 0 ||
    l1_ratio > 1) {
    stop("`l1_ratio` must be a single number between 0 and 1.", call. = FALSE)
  }
  check_flag(standardize, "standardize")
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
}

# The design matrix and the response of a cox_net() or cv_coxnet() fit.
coxnet_input <- function(frame, formula, data, fn) {
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop(sprintf("`%s` doesn't support `strata()` or `cluster()` terms.", fn), call. = FALSE)
  }
  if (!is.null(split_formula(formula, data)$offset)) {
    stop(sprintf("`%s` doesn't support `offset()` terms.", fn), call. = FALSE)
  }
  parts <- surv_parts(frame$response, fn)
  design <- covariate_design(frame, check_rank = FALSE)
  if (ncol(design$x) == 0L) {
    stop(sprintf("`%s` needs at least one covariate.", fn), call. = FALSE)
  }
  if (sum(parts$event[frame$weights > 0]) == 0) {
    stop(sprintf("`%s` needs at least one event.", fn), call. = FALSE)
  }
  c(design, list(start = parts$start, stop = parts$stop, event = parts$event, w = frame$weights))
}

# -- fitting -------------------------------------------------------------------------------------

# What the Breslow partial likelihood needs that doesn't change with beta: the distinct event
# times and the weight of the events at each, the order of the rows by stop and start time, and
# which event times each row is at risk for (those after its start, up to its stop).
coxnet_setup <- function(start, stop, event, w) {
  dead <- event == 1
  times <- sort(unique(stop[dead]))
  counting <- any(is.finite(start))
  o_stop <- order(stop)
  o_start <- order(start)
  list(
    times = times,
    d = tabulate_sum(match(stop[dead], times), w[dead], length(times)),
    w = w,
    w_event = w * dead,
    counting = counting,
    o_stop = o_stop,
    o_start = o_start,
    stop_before = findInterval(times, stop[o_stop], left.open = TRUE),
    start_before = findInterval(times, start[o_start], left.open = TRUE),
    upto_stop = findInterval(stop, times),
    upto_start = findInterval(start, times)
  )
}

# The sum of the risk scores `r` over the rows at risk at each event time.
coxnet_s0 <- function(setup, r) {
  if (setup$counting) {
    # At risk at t: start < t, less those with stop < t.
    c(0, cumsum(r[setup$o_start]))[setup$start_before + 1L] -
      c(0, cumsum(r[setup$o_stop]))[setup$stop_before + 1L]
  } else {
    rev(cumsum(rev(r[setup$o_stop])))[setup$stop_before + 1L]
  }
}

# The Breslow hazard increments at the event times for linear predictors `eta`.
coxnet_hazard <- function(setup, eta) {
  shift <- max(eta)
  setup$d / coxnet_s0(setup, setup$w * exp(eta - shift)) * exp(-shift)
}

# The log partial likelihood (Breslow ties) at linear predictors `eta`, and with `gradient = TRUE`
# its gradient with respect to the coefficients of `x`.
coxnet_loglik <- function(setup, x, eta, gradient = FALSE) {
  shift <- max(eta)
  r <- setup$w * exp(eta - shift)
  s0 <- coxnet_s0(setup, r)
  loglik <- sum(setup$w_event * eta) - sum(setup$d * (log(s0) + shift))
  if (!gradient) {
    return(list(loglik = loglik))
  }
  cumulative <- c(0, cumsum(setup$d / s0))
  expected <- r * (cumulative[setup$upto_stop + 1L] - cumulative[setup$upto_start + 1L])
  list(loglik = loglik, gradient = drop(crossprod(x, setup$w_event - expected)))
}

soft_threshold <- function(v, threshold) {
  sign(v) * pmax(abs(v) - threshold, 0)
}

# The elastic-net fit. The covariates are centered at their weighted means and divided by their
# weighted standard deviations (or by 1), and FISTA minimizes
# -loglik / W + lambda (alpha |b|_1 + (1 - alpha) / 2 |b|^2) on that scale, from b = 0.
coxnet_fit <- function(x, start, stop, event, w, penalizer, l1_ratio, standardize, max_iter, tol) {
  use <- w > 0
  x <- x[use, , drop = FALSE]
  start <- start[use]
  stop <- stop[use]
  event <- event[use]
  w <- w[use]
  p <- ncol(x)
  total <- sum(w)
  standardized <- coxnet_standardize(x, w, standardize)
  xs <- standardized$x
  setup <- coxnet_setup(start, stop, event, w)
  ridge <- penalizer * (1 - l1_ratio)
  lasso <- penalizer * l1_ratio

  smooth <- function(b, gradient = FALSE) {
    terms <- coxnet_loglik(setup, xs, drop(xs %*% b), gradient)
    out <- list(value = -terms$loglik / total + 0.5 * ridge * sum(b * b))
    if (gradient) {
      out$gradient <- -terms$gradient / total + ridge * b
    }
    out
  }

  fit <- fista(smooth, p, lasso, max_iter, tol)
  beta <- fit$beta

  list(
    beta = beta / standardized$scale,
    center = standardized$center,
    scale = standardized$scale,
    loglik = coxnet_loglik(setup, xs, drop(xs %*% beta))$loglik,
    effective_df = coxnet_effective_df(beta, xs, start, stop, event, w, ridge * total, ridge),
    iterations = fit$iterations,
    converged = fit$converged
  )
}

# Accelerated proximal gradient descent (FISTA) for the elastic net, from b = 0. `smooth(b,
# gradient)` gives the smooth part of the objective (and its gradient), and the lasso part is
# `lasso * |b|_1`. The step is halved until the quadratic bound at the momentum point holds, and
# the iterations stop once the relative change in the coefficients is below `tol`.
fista <- function(smooth, p, lasso, max_iter, tol) {
  beta <- numeric(p)
  momentum <- beta
  t_acc <- 1
  step <- 1
  converged <- FALSE
  iterations <- 0L
  while (iterations < max_iter) {
    iterations <- iterations + 1L
    at <- smooth(momentum, gradient = TRUE)
    repeat {
      candidate <- soft_threshold(momentum - step * at$gradient, step * lasso)
      diff <- candidate - momentum
      bound <- at$value + sum(at$gradient * diff) + sum(diff * diff) / (2 * step)
      if (smooth(candidate)$value <= bound) {
        break
      }
      step <- step / 2
      if (step < 1e-12) {
        break
      }
    }
    t_next <- (1 + sqrt(1 + 4 * t_acc^2)) / 2
    momentum <- candidate + ((t_acc - 1) / t_next) * (candidate - beta)
    change <- sqrt(sum((candidate - beta)^2)) / (sqrt(sum(beta^2)) + tol)
    beta <- candidate
    t_acc <- t_next
    if (change < tol) {
      converged <- TRUE
      break
    }
  }

  list(beta = beta, iterations = iterations, converged = converged)
}

# The covariates centered at their weighted means and divided by their weighted standard
# deviations (with the total weight as the divisor), or by 1. A constant covariate keeps a scale
# of 1.
coxnet_standardize <- function(x, w, standardize) {
  total <- sum(w)
  center <- colSums(x * w) / total
  xc <- sweep(x, 2L, center)
  scale <- rep(1, ncol(x))
  if (standardize) {
    scale <- sqrt(colSums(w * xc^2) / total)
    scale[!is.finite(scale) | scale <= 0] <- 1
  }
  list(x = sweep(xc, 2L, scale, "/"), center = center, scale = scale)
}

# The effective degrees of freedom: the number of nonzero coefficients without a ridge penalty,
# and otherwise trace(I_A (I_A + W lambda (1 - alpha))^-1) over the nonzero coefficients.
coxnet_effective_df <- function(beta, xs, start, stop, event, w, ridge_total, ridge) {
  active <- beta != 0
  if (ridge == 0 || !any(active)) {
    return(sum(active))
  }
  xa <- xs[, active, drop = FALSE]
  setup <- cox_setup(xa, start, stop, event, w, rep(1L, nrow(xa)), efron = FALSE)
  info <- cox_terms(beta[active], setup)$info
  sum(diag(solve(info + diag(ridge_total, sum(active)), info)))
}

# -- methods -------------------------------------------------------------------------------------

#' @export
coef.cox_net <- function(object, ...) {
  object$coefficients
}

#' @export
nobs.cox_net <- function(object, ...) {
  object$n_event
}

#' @export
logLik.cox_net <- function(object, ...) {
  structure(object$loglik, df = object$effective_df, nobs = object$n_event, class = "logLik")
}

#' Predictions and coefficients of an elastic-net Cox model
#'
#' `predict()` gives the linear predictor, the risk score, or the survival curve of each row of
#' `newdata` (or of the data the model was fit to). The linear predictor is
#' \eqn{(x - \bar{x})^\top \beta}, centered at the weighted means of the covariates in the fitted
#' data, and the risk score is its exponential. The survival curve is
#' \eqn{\exp(-H_0(t) \exp(\eta))}, with \eqn{H_0} Breslow's estimate of the baseline cumulative
#' hazard (for the centered linear predictor \eqn{\eta}) at the penalized coefficients.
#' `as.data.frame()` gives the coefficients and hazard ratios, with the coefficients set to 0 by
#' the lasso kept in the table.
#'
#' @param object,x A [cox_net()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"lp"` (the default), `"risk"`, or `"survival"`.
#' @param times For `"survival"`, the times to read the curves at. The default is every distinct
#'   time in the fitted data.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `"lp"` and `"risk"`, a numeric vector with one value per row of `newdata`. For
#'   `"survival"`, a data frame with one row per subject and time: `subject` (the row of
#'   `newdata`), `time`, and `estimate`. `as.data.frame()` gives a data frame with the columns
#'   `term`, `estimate`, and `hazard_ratio`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#' fit <- cox_net(Surv(time, status) ~ age + sex, data = lung, penalizer = 0.05)
#' as.data.frame(fit)
#' predict(fit, type = "risk")
#' predict(fit, newdata = data.frame(age = 60, sex = 2), type = "survival", times = c(200, 400))
#' @name cox_net_predict
#' @export
predict.cox_net <- function(object, newdata, type = c("lp", "risk", "survival"), times, ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    x <- object$x
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    x <- cox_new_design(object, newdata)
  }
  lp <- linear_predictor(sweep(x, 2L, object$center), object$coefficients)
  if (type == "lp") {
    return(unname(lp))
  }
  if (type == "risk") {
    return(unname(exp(lp)))
  }
  times <- if (missing(times)) object$times else check_times(times)
  cumhaz <- step_at(object$baseline$time, object$baseline$cumhaz, times, 0)
  data.frame(
    subject = rep(seq_along(lp), each = length(times)),
    time = rep(times, times = length(lp)),
    estimate = exp(-rep(exp(unname(lp)), each = length(times)) * cumhaz)
  )
}

#' @rdname cox_net_predict
#' @export
as.data.frame.cox_net <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(
    term = names(x$coefficients),
    estimate = unname(x$coefficients),
    hazard_ratio = unname(exp(x$coefficients)),
    stringsAsFactors = FALSE
  )
}

#' @export
print.cox_net <- function(x, digits = getOption("digits"), ...) {
  cat("<cox_net> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "Elastic net, penalizer = %s, l1_ratio = %s. n = %d, events = %s\n\n",
    format(x$penalizer), format(x$l1_ratio), x$n, format(x$n_event)
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(estimate = table$estimate, hazard_ratio = table$hazard_ratio,
    row.names = table$term
  )
  print(coefs, digits = max(3L, digits - 3L))
  short <- max(3L, digits - 4L)
  cat(sprintf(
    "\nNonzero coefficients: %d of %d. Effective df = %s, log partial likelihood = %s\n",
    x$n_nonzero, length(x$coefficients), format(x$effective_df, digits = short),
    format(x$loglik, digits = short + 2L)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- cross-validation ----------------------------------------------------------------------------

#' Choose the penalizer of an elastic-net Cox model by cross-validation
#'
#' `cv_coxnet()` fits [cox_net()] at each of a sequence of penalizers, scoring each on held-out
#' folds, so the penalizer can be chosen for how well the model predicts for new subjects. The
#' rows are split into `k` folds. At each penalizer, the model is fit to all but one fold and
#' scored on the fold left out, for every fold. The score is Harrell's concordance of the linear
#' predictor (`"concordance"`, higher is better, see [concordance_index()]), or the integrated
#' Brier score over `times` (`"brier"`, lower is better, see [integrated_brier_score()]).
#'
#' Without `penalizers`, the sequence runs from \eqn{\lambda_{max}} down to `eps` times it, in
#' `n_penalizers` steps equally spaced on the log scale. \eqn{\lambda_{max}} is the smallest
#' lasso penalizer that sets every coefficient to 0,
#' \eqn{\max_j |U_j(0)| / (W \max(\alpha, 0.01))}, with \eqn{U(0)} the score of the log partial
#' likelihood of the standardized covariates at \eqn{\beta = 0} and \eqn{W} the total weight.
#'
#' The best penalizer has the best mean score over the folds. The one-standard-error penalizer is
#' the largest one whose mean score is within one standard deviation (over the folds) of the
#' best, giving a simpler model that predicts about as well. Refit [cox_net()] at the penalizer
#' chosen.
#'
#' With `stratified = TRUE` (the default), the folds are drawn separately among the subjects with
#' and without an event, so every fold gets about the same share of events. Folds whose training
#' or held-out rows have no events are left out of the scores. The folds are drawn from R's random
#' number generator, so call [set.seed()] first for results that can be repeated.
#'
#' @inheritParams cox_net
#' @param formula A formula with a right-censored [Surv()] response on the left and the
#'   covariates on the right, such as `Surv(time, status) ~ age + sex`.
#' @param l1_ratio The elastic-net mixing \eqn{\alpha}, between 0 (ridge) and 1 (lasso), the same
#'   for every penalizer. The default is `1`.
#' @param penalizers The penalizers to try. The default, `NULL`, gives the sequence described
#'   above.
#' @param n_penalizers The number of penalizers in the default sequence. The default is `50`.
#' @param eps The ratio of the smallest penalizer in the default sequence to the largest. The
#'   default is `1e-3`.
#' @param k The number of folds, at least 2 and at most the number of rows. The default is `5`.
#' @param metric The score: `"concordance"` (the default) or `"brier"`.
#' @param times At least two times, for the `"brier"` metric.
#' @param stratified Draw the folds within each event status. The default is `TRUE`.
#'
#' @return A `cv_coxnet` object. `path` is a data frame with one row per penalizer, from the
#'   largest to the smallest: `penalizer`, `mean_score` and `std_score` (the mean and standard
#'   deviation of the scores over the folds), and `n_nonzero` (the mean number of nonzero
#'   coefficients over the folds). `scores` has one row per penalizer and fold: `penalizer`,
#'   `fold`, `score`, and `n_nonzero`. `best_penalizer`, `best_score`, `penalizer_1se`, and
#'   `score_1se` give the choices, and `folds` the fold of each row used. [as.data.frame()] gives
#'   `path`.
#'
#' @examples
#' set.seed(4)
#' n <- 60
#' sim <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n))
#' sim$time <- round(rexp(n, rate = 0.1 * exp(0.8 * sim$x1 - 0.5 * sim$x2)), 1) + 0.1
#' sim$status <- rbinom(n, 1, 0.8)
#'
#' set.seed(1)
#' cv <- cv_coxnet(Surv(time, status) ~ x1 + x2 + x3 + x4, data = sim, n_penalizers = 10, k = 3)
#' cv
#' as.data.frame(cv)
#'
#' # Refit at the penalizer chosen
#' cox_net(Surv(time, status) ~ x1 + x2 + x3 + x4, data = sim, penalizer = cv$best_penalizer,
#'   l1_ratio = 1
#' )
#' @export
cv_coxnet <- function(formula,
                      data,
                      weights,
                      subset,
                      na.action = stats::na.omit,
                      l1_ratio = 1,
                      penalizers = NULL,
                      n_penalizers = 50,
                      eps = 1e-3,
                      k = 5,
                      metric = c("concordance", "brier"),
                      times = NULL,
                      standardize = TRUE,
                      max_iter = 1000,
                      tol = 1e-7,
                      stratified = TRUE) {
  metric <- match.arg(metric)
  check_coxnet_args(l1_ratio, standardize, max_iter, tol)
  check_flag(stratified, "stratified")
  check_number(k, "k", min = 2)
  if (k != round(k)) {
    stop("`k` must be a whole number.", call. = FALSE)
  }
  k <- as.integer(k)
  if (metric == "brier" &&
    (is.null(times) || !is.numeric(times) || length(times) < 2L || anyNA(times))) {
    stop("The \"brier\" metric needs at least two `times`.", call. = FALSE)
  }
  if (!is.null(penalizers) &&
    (!is.numeric(penalizers) || length(penalizers) == 0L || anyNA(penalizers) ||
      any(!is.finite(penalizers)) || any(penalizers < 0))) {
    stop("`penalizers` must be one or more non-negative numbers.", call. = FALSE)
  }
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  input <- coxnet_input(frame, formula, if (missing(data)) NULL else data, "cv_coxnet()")
  response <- frame$response
  if (attr(response, "type") != "right") {
    stop("`cv_coxnet()` needs right-censored data.", call. = FALSE)
  }
  x <- input$x
  w <- input$w
  event <- input$event
  n <- nrow(x)
  if (k > n) {
    stop(sprintf("`k` must be at most the number of rows used (%d).", n), call. = FALSE)
  }

  if (is.null(penalizers)) {
    check_number(n_penalizers, "n_penalizers", min = 1)
    check_number(eps, "eps", min = 0)
    if (eps <= 0 || eps > 1) {
      stop("`eps` must be a number above 0 and at most 1.", call. = FALSE)
    }
    use <- w > 0
    xs <- coxnet_standardize(x[use, , drop = FALSE], w[use], standardize)$x
    setup <- coxnet_setup(input$start[use], input$stop[use], event[use], w[use])
    score <- coxnet_loglik(setup, xs, numeric(nrow(xs)), gradient = TRUE)$gradient
    lambda_max <- max(abs(score)) / (sum(w) * max(l1_ratio, 1e-2))
    penalizers <- exp(seq(log(lambda_max), log(eps * lambda_max), length.out = n_penalizers))
  } else {
    penalizers <- sort(as.numeric(penalizers), decreasing = TRUE)
  }

  n_events <- sum(event)
  if (n_events < 2L * k) {
    warning(
      sprintf(
        paste(
          "There are %d events for %d folds, fewer than two per fold.",
          "Some folds may have too few events to fit or score."
        ),
        n_events, k
      ),
      call. = FALSE
    )
  }
  folds <- cv_folds(event, k, stratified)

  not_converged <- 0L
  grid <- expand.grid(fold = seq_len(k), index = seq_along(penalizers))
  results <- lapply(seq_len(nrow(grid)), function(r) {
    i <- grid$fold[r]
    train <- folds != i & w > 0
    test <- folds == i
    if (sum(event[train]) == 0 || sum(event[test]) == 0) {
      return(c(NA_real_, NA_real_))
    }
    fit <- coxnet_fit(
      x[train, , drop = FALSE], input$start[train], input$stop[train], event[train], w[train],
      penalizers[grid$index[r]], l1_ratio, standardize, max_iter, tol
    )
    if (!fit$converged) {
      not_converged <<- not_converged + 1L
    }
    lp <- linear_predictor(sweep(x[test, , drop = FALSE], 2L, fit$center), fit$beta)
    score <- if (metric == "concordance") {
      concordance_index(response[test], risk = lp)
    } else {
      setup <- coxnet_setup(input$start[train], input$stop[train], event[train], w[train])
      eta <- linear_predictor(sweep(x[train, , drop = FALSE], 2L, fit$center), fit$beta)
      h0 <- step_at(setup$times, cumsum(coxnet_hazard(setup, eta)), times, 0)
      probs <- exp(-outer(exp(lp), h0))
      integrated_brier_score(response[test], survival_prob = probs, times = times)
    }
    c(score, sum(fit$beta != 0))
  })
  if (not_converged > 0L) {
    warning(
      sprintf(
        "%d of the fits didn't converge in %d iterations.", not_converged, as.integer(max_iter)
      ),
      call. = FALSE
    )
  }
  results <- do.call(rbind, results)
  scores <- data.frame(
    penalizer = penalizers[grid$index],
    fold = grid$fold,
    score = results[, 1L],
    n_nonzero = results[, 2L]
  )

  index <- factor(grid$index, levels = seq_along(penalizers))
  summarize <- function(values, f) {
    vapply(split(values, index), function(v) {
      v <- v[!is.na(v)]
      if (length(v) == 0L) NA_real_ else f(v)
    }, numeric(1))
  }
  path <- data.frame(
    penalizer = penalizers,
    mean_score = unname(summarize(scores$score, mean)),
    std_score = unname(summarize(scores$score, function(v) {
      if (length(v) > 1L) stats::sd(v) else 0
    })),
    n_nonzero = unname(summarize(scores$n_nonzero, mean))
  )

  higher <- metric == "concordance"
  best <- if (higher) which.max(path$mean_score) else which.min(path$mean_score)
  if (length(best) == 0L) {
    stop("No fold could be scored. Use fewer folds.", call. = FALSE)
  }
  limit <- path$std_score[best]
  within <- if (higher) {
    which(path$mean_score >= path$mean_score[best] - limit)
  } else {
    which(path$mean_score <= path$mean_score[best] + limit)
  }
  one_se <- within[which.max(path$penalizer[within])]

  structure(
    list(
      path = path,
      scores = scores,
      best_penalizer = path$penalizer[best],
      best_score = path$mean_score[best],
      penalizer_1se = path$penalizer[one_se],
      score_1se = path$mean_score[one_se],
      metric = metric,
      l1_ratio = l1_ratio,
      k = k,
      folds = folds,
      n = n,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "cv_coxnet"
  )
}

#' @export
as.data.frame.cv_coxnet <- function(x, ...) {
  x$path
}

#' @export
print.cv_coxnet <- function(x, digits = getOption("digits"), ...) {
  short <- max(3L, digits - 3L)
  cat("<cv_coxnet> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%d-fold cross-validation, l1_ratio = %s, %s (%s is better)\n",
    x$k, format(x$l1_ratio), x$metric, if (x$metric == "concordance") "higher" else "lower"
  ))
  cat(sprintf(
    "%d penalizers from %s to %s\n\n",
    nrow(x$path), format(min(x$path$penalizer), digits = short),
    format(max(x$path$penalizer), digits = short)
  ))
  cat(sprintf(
    "Best penalizer:  %s (mean %s %s)\n",
    format(x$best_penalizer, digits = short), x$metric, format(x$best_score, digits = short)
  ))
  cat(sprintf(
    "1-SE penalizer:  %s (mean %s %s)\n",
    format(x$penalizer_1se, digits = short), x$metric, format(x$score_1se, digits = short)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}
