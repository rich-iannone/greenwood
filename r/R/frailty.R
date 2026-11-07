# Shared frailty for cox_ph(): a random effect per cluster that multiplies the hazard of every
# subject in the cluster. Tested by tests/testthat/test-frailty.R, against Greenwood's Python
# values and by recovering a simulated frailty variance.

# The frailty distribution of a cox_ph() call, from the `frailty` argument or the formula's
# `frailty()` term, after checking that the rest of the call can be fit with it. NULL for none.
cox_frailty_kind <- function(frailty, frame, parts, ties, robust, theta, max_iter) {
  if (!is.null(frailty) &&
    (!is.character(frailty) || length(frailty) != 1L || !frailty %in% c("gamma", "lognormal"))) {
    stop("`frailty` must be \"gamma\" or \"lognormal\".", call. = FALSE)
  }
  from_formula <- frame$frailty_distribution
  if (!is.null(frailty) && !is.null(from_formula) && frailty != from_formula) {
    stop(
      sprintf(
        "The formula asks for a %s frailty, but `frailty = \"%s\"` was also given. Use one.",
        from_formula, frailty
      ),
      call. = FALSE
    )
  }
  if (is.null(frailty)) {
    frailty <- from_formula
  }
  check_number(max_iter, "frailty_max_iter", min = 1)
  if (is.null(frailty)) {
    if (!is.null(frame$frailty_cluster)) {
      stop("`frailty_cluster` needs `frailty = \"gamma\"` or `frailty = \"lognormal\"`.", call. = FALSE)
    }
    return(NULL)
  }
  if (is.null(frame$frailty_cluster)) {
    stop(
      sprintf(
        "`frailty = \"%s\"` needs the clusters, as a `frailty()` term or as `frailty_cluster`.",
        frailty
      ),
      call. = FALSE
    )
  }
  if (!is.numeric(theta) || length(theta) != 1L || !is.finite(theta) || theta <= 0) {
    stop("`frailty_theta` must be a single positive number.", call. = FALSE)
  }
  shared <- sprintf("A shared %s frailty", frailty)
  if (parts$type != "right") {
    stop(sprintf("%s needs right-censored data.", shared), call. = FALSE)
  }
  if (ties != "breslow") {
    stop(sprintf("%s needs `ties = \"breslow\"`.", shared), call. = FALSE)
  }
  if (!is.null(frame$strata)) {
    stop(sprintf("%s can't be combined with `strata()` terms.", shared), call. = FALSE)
  }
  if (robust || !is.null(frame$cluster)) {
    stop(sprintf("%s can't be combined with a robust or clustered variance.", shared), call. = FALSE)
  }
  frailty
}

# Gamma frailty, with mean 1 and variance theta, by alternating three steps. The coefficients are
# fit with each cluster's current frailty multiplying its risk scores. Each cluster's frailty is
# then its posterior mean (d + 1/theta) / (H + 1/theta), with d its events and H its expected
# events under the fit (the Breslow cumulative hazard at each row's time times the row's risk
# score). theta is the value on a log grid from 1e-8 to 1e3 that maximizes the marginal likelihood
# of the clusters' event counts. The steps repeat until the coefficients and theta settle.
cox_gamma_frailty <- function(setup, index, k, theta, frailty_max_iter, max_iter, tol) {
  d <- tabulate_sum(index, setup$w * setup$event, k)
  grid <- exp(seq(log(1e-8), log(1e3), length.out = 80L))
  beta <- numeric(ncol(setup$x))
  effects <- rep(1, k)
  converged <- FALSE
  for (iteration in seq_len(frailty_max_iter)) {
    setup$offset <- log(effects[index])
    fit <- cox_newton(setup, max_iter, tol)
    piece <- cox_terms(fit$beta, setup, detail = TRUE)$detail[[1L]]
    cumhaz <- step_at(piece$times, cumsum(piece$hazard), setup$stop, 0)
    eta <- drop(setup$x %*% fit$beta)
    expected <- tabulate_sum(index, cumhaz * exp(eta) * setup$w, k)
    effects <- (d + 1 / theta) / (expected + 1 / theta)
    previous <- theta
    theta <- grid[which.max(vapply(grid, gamma_profile, numeric(1), d = d, h = expected))]
    change <- max(abs(fit$beta - beta))
    beta <- fit$beta
    if (change <= tol && abs(theta - previous) <= tol * (abs(theta) + tol)) {
      converged <- TRUE
      break
    }
  }
  statistic <- max(0, 2 * (gamma_profile(theta, d, expected) - gamma_profile(1e-5, d, expected)))
  list(
    fit = fit,
    offset = log(effects[index]),
    theta = theta,
    effects = effects,
    test = frailty_chisq(statistic),
    iterations = iteration,
    converged = converged
  )
}

# The log marginal likelihood of the clusters' event counts d, given their expected counts h, with
# the frailties integrated out over a gamma distribution of variance theta.
gamma_profile <- function(theta, d, h) {
  inv <- 1 / theta
  sum(lgamma(d + inv) - lgamma(inv) + inv * log(inv) - (d + inv) * log(inv + h))
}

# Lognormal frailty: the log frailties u are normal with variance sigma2 (reported as theta). The
# coefficients and u are fit together by maximizing the partial likelihood with a cluster
# indicator per u, penalized by -sum(u^2) / (2 sigma2). sigma2 is then updated to
# (sum(u^2) + trace(H_uu^-1)) / K, with H_uu the penalized information of u and K the number of
# clusters, and the steps repeat until both settle. The variance of the coefficients is the
# inverse of their information with u integrated out (a Schur complement). The test of sigma2 = 0
# compares the Laplace approximation of the marginal likelihood with the fit without frailty.
cox_lognormal_frailty <- function(setup, index, k, sigma2, frailty_max_iter, max_iter, tol) {
  x <- setup$x
  n <- nrow(x)
  p <- ncol(x)
  indicators <- matrix(0, n, k)
  indicators[cbind(seq_len(n), index)] <- 1
  joint <- cox_setup(
    cbind(x, indicators), setup$start, setup$stop, setup$event, setup$w, rep(1L, n),
    efron = FALSE
  )
  u_at <- p + seq_len(k)
  penalized <- function(theta, sigma2) {
    terms <- cox_terms(theta, joint)
    u <- theta[u_at]
    terms$loglik <- terms$loglik - 0.5 / sigma2 * sum(u^2)
    terms$score[u_at] <- terms$score[u_at] - u / sigma2
    terms$info[u_at, u_at] <- terms$info[u_at, u_at] + diag(1 / sigma2, k)
    terms
  }

  theta <- numeric(p + k)
  converged <- FALSE
  for (iteration in seq_len(frailty_max_iter)) {
    theta_before <- theta
    sigma2_before <- sigma2
    # Newton-Raphson from the last estimate, halving the step whenever it lowers the penalized
    # likelihood, until the penalized likelihood stops changing.
    loglik <- -Inf
    inner <- 0L
    inner_converged <- FALSE
    while (inner < max_iter) {
      inner <- inner + 1L
      current <- penalized(theta, sigma2)
      step <- drop(cox_solve(current$info, current$score))
      for (halving in 0:20) {
        candidate <- penalized(theta + step, sigma2)$loglik
        if (isTRUE(candidate >= current$loglik - 1e-12) || halving == 20L) {
          break
        }
        step <- step / 2
      }
      inner_converged <- abs(candidate - loglik) <= tol * (abs(candidate) + tol)
      theta <- theta + step
      loglik <- candidate
      if (inner_converged) {
        break
      }
    }
    info <- penalized(theta, sigma2)$info
    u <- theta[u_at]
    h_uu <- info[u_at, u_at, drop = FALSE]
    trace <- tryCatch(sum(diag(solve(h_uu))), error = function(e) k * sigma2)
    sigma2 <- max((sum(u^2) + trace) / k, 1e-8)
    if (max(abs(theta - theta_before)) <= tol &&
      abs(sigma2 - sigma2_before) <= tol * (abs(sigma2) + tol)) {
      converged <- TRUE
      break
    }
  }

  beta_at <- seq_len(p)
  h_bu <- info[beta_at, u_at, drop = FALSE]
  marginal_info <- info[beta_at, beta_at, drop = FALSE] - h_bu %*% solve(h_uu, t(h_bu))
  null <- cox_terms(numeric(p + k), joint)
  score_null <- null$score[beta_at]
  laplace <- loglik - 0.5 * k * log(sigma2) -
    0.5 * as.numeric(determinant(h_uu, logarithm = TRUE)$modulus)
  plain <- cox_newton(setup, max_iter, tol)
  list(
    fit = list(
      beta = theta[beta_at],
      loglik_null = null$loglik,
      loglik = loglik,
      info = marginal_info,
      score_null = sum(score_null * cox_solve(null$info[beta_at, beta_at, drop = FALSE], score_null)),
      iterations = inner,
      converged = inner_converged
    ),
    offset = u[index],
    theta = sigma2,
    effects = u,
    test = frailty_chisq(max(0, 2 * (laplace - plain$loglik))),
    iterations = iteration,
    converged = converged
  )
}

# The likelihood ratio test of no frailty. The variance 0 is on the boundary of its range, so the
# statistic follows an even mixture of 0 and a chi-squared on 1 df under the null.
frailty_chisq <- function(statistic) {
  list(
    statistic = statistic,
    df = 1L,
    p_value = 0.5 * stats::pchisq(statistic, 1, lower.tail = FALSE)
  )
}

#' Test for frailty in a Cox model
#'
#' `frailty_test()` tests whether the frailty variance of a [cox_ph()] fit with a shared frailty is
#' 0, which would mean no extra variation between the clusters beyond what the covariates explain.
#' It's a likelihood ratio test. For a gamma frailty it compares the marginal likelihood of the
#' clusters' event counts at the estimated variance with that at a variance near 0. For a lognormal
#' frailty it compares the Laplace approximation of the marginal likelihood with the likelihood of
#' the model without frailty. A variance of 0 is the smallest possible, so the p-value is half the
#' chi-squared (1 df) tail probability.
#'
#' @param fit A [cox_ph()] fit with a `frailty()` term or a `frailty` argument.
#'
#' @return A data frame with one row: `distribution`, `theta` (the estimated variance), `statistic`,
#'   `df`, and `p_value`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   inst = c(3, 3, 3, 5, 1, 12, 7, 11, 1, 7, 6, 12, 2, 16, 11, 21, 12, 1, 22, 16)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + frailty(inst), data = lung, ties = "breslow")
#' frailty_test(fit)
#' @export
frailty_test <- function(fit) {
  if (!inherits(fit, "cox_ph")) {
    stop("`fit` must be a `cox_ph()` fit.", call. = FALSE)
  }
  if (is.null(fit$frailty)) {
    stop("`fit` has no frailty. Fit it with a `frailty()` term or a `frailty` argument.", call. = FALSE)
  }
  test <- fit$frailty$test
  data.frame(
    distribution = fit$frailty$distribution,
    theta = fit$frailty$theta,
    statistic = test$statistic,
    df = test$df,
    p_value = test$p_value,
    stringsAsFactors = FALSE
  )
}
