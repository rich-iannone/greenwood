# The Cox proportional hazards model: cox_ph() and its methods. The numeric fixtures in
# fixtures/r/ (cox_*) are replayed by tests/testthat/test-cox-ph.R.

#' Cox proportional hazards regression
#'
#' `cox_ph()` fits the Cox proportional hazards model, in which the hazard of a subject with
#' covariates \eqn{x} is \eqn{h_0(t) \exp(x^\top \beta)}. The baseline hazard \eqn{h_0} is left
#' unspecified, and the coefficients \eqn{\beta} are estimated by maximizing the partial
#' likelihood with Newton-Raphson iterations (halving the step whenever it lowers the likelihood).
#' Each coefficient is a log hazard ratio: the change in the log hazard for a one-unit increase in
#' its covariate.
#'
#' Tied event times are handled with Efron's approximation by default, or with Breslow's. With
#' case weights, Efron's approximation uses the mean weight of the tied events at each time.
#'
#' A `strata()` term gives each stratum its own baseline hazard while the coefficients are shared,
#' as in `Surv(time, status) ~ age + strata(sex)`. Counting-process data,
#' `Surv(start, stop, event)`, handle delayed entry and time-varying covariates: each row is an
#' interval over which a subject's covariates are constant, and the subject is at risk only within
#' it.
#'
#' With `robust = TRUE`, or a `cluster` (given as an argument or as a `cluster()` term), the
#' variance is the robust sandwich estimate built from the score residuals, summed within each
#' cluster. That variance is used for the standard errors, the confidence limits, and the Wald
#' test. The likelihood ratio and score tests are unaffected.
#'
#' A `frailty()` term, as in `Surv(time, status) ~ age + frailty(inst)`, adds a shared frailty: a
#' random effect for each cluster that multiplies the hazard of all its subjects, for clusters
#' whose subjects are more alike than the covariates explain. `frailty(inst)` is a gamma frailty,
#' with mean 1 and variance \eqn{\theta}. `frailty(inst, distribution = "gaussian")` (or
#' `frailty.gaussian(inst)`) is a lognormal frailty, whose logarithm is normal with mean 0 and
#' variance \eqn{\sigma^2}. The `frailty` and `frailty_cluster` arguments do the same as the term.
#' The gamma fit alternates between fitting the coefficients with the clusters' current frailties,
#' setting each frailty to its posterior mean, and choosing \eqn{\theta} on a grid of 80 values
#' from 1e-8 to 1e3 to maximize the marginal likelihood of the clusters' event counts. The
#' lognormal fit maximizes the partial likelihood over the coefficients and the log frailties
#' together, penalized by their normal density, and updates \eqn{\sigma^2} from the log frailties
#' and their variance. [frailty_test()] tests whether the variance is 0. A shared frailty needs
#' right-censored data and `ties = "breslow"`, and can't be combined with strata or a robust
#' variance. Predictions are for a subject with a frailty of 1, and the residuals and the baseline
#' hazard include the fitted frailties.
#'
#' @param formula A formula with a [Surv()] response on the left and the covariates on the right,
#'   such as `Surv(time, status) ~ age + sex`. Factors are coded with treatment contrasts. A
#'   `strata()` term stratifies the baseline hazard, a `cluster()` term gives the clusters, and a
#'   `frailty()` term adds a shared frailty.
#' @param data A data frame holding the variables in `formula`.
#' @param weights Optional case weights, a column of `data` or a numeric vector.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param ties The handling of tied event times: `"efron"` (the default) or `"breslow"`.
#' @param robust Use the robust (sandwich) variance. The default is `FALSE`, or `TRUE` when there
#'   are clusters.
#' @param cluster Optional cluster identifiers, a column of `data` or a vector, for observations
#'   that are correlated within clusters. Implies `robust = TRUE`.
#' @param frailty The distribution of a shared frailty: `"gamma"` or `"lognormal"`. The default,
#'   `NULL`, gives none, unless the formula has a `frailty()` term.
#' @param frailty_cluster The clusters of a shared frailty, a column of `data` or a vector, when
#'   they're not given by a `frailty()` term.
#' @param frailty_theta The starting value of the frailty variance, above 0. The default is `0.5`.
#' @param frailty_max_iter The largest number of rounds of the frailty fit. The default is `30`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param max_iter The largest number of Newton-Raphson iterations. The default is `30`.
#' @param tol The convergence tolerance: iterations stop once the log partial likelihood changes
#'   by less than `tol` times its size. The default is `1e-9`.
#'
#' @return A `cox_ph` object. Use [as.data.frame()] for the coefficient table, [coef()] and
#'   [vcov()] for the estimates and their variance, [predict()] for linear predictors, risk scores,
#'   and survival curves, [residuals()] for diagnostics, [baseline_hazard()] for the baseline
#'   cumulative hazard, and [cox_zph()] to check proportional hazards. [logLik()], [AIC()], and
#'   [nobs()] work as usual.
#'
#'   The object also holds `loglik` (the log partial likelihood at \eqn{\beta = 0} and at the
#'   estimate), the global tests `lr_test`, `wald_test`, and `score_test` (each with `statistic`,
#'   `df`, and `p_value`), `concordance` and its standard error `concordance_se`, `n`, `n_event`,
#'   `iterations`, and `converged`. With a shared frailty, `frailty` holds the `distribution`, the
#'   `cluster` variable, `theta` (the variance), `effects` (each cluster's frailty for a gamma
#'   frailty, or its log frailty for a lognormal one, named by cluster), the variance `test`, and
#'   the number of `iterations` of the frailty fit and whether it `converged`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#'
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' fit
#'
#' # Hazard ratios with confidence limits
#' as.data.frame(fit, exponentiate = TRUE)
#'
#' # Survival curves for new subjects
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(200, 400))
#' @export
cox_ph <- function(formula,
                   data,
                   weights,
                   subset,
                   na.action = stats::na.omit,
                   ties = c("efron", "breslow"),
                   robust = FALSE,
                   cluster,
                   frailty = NULL,
                   frailty_cluster,
                   frailty_theta = 0.5,
                   frailty_max_iter = 30,
                   conf_level = 0.95,
                   max_iter = 30,
                   tol = 1e-9) {
  ties <- match.arg(ties)
  check_flag(robust, "robust")
  check_conf_level(conf_level)
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)

  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data,
    frailty = TRUE
  )
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop("`cox_ph()` doesn't support `offset()` terms.", call. = FALSE)
  }
  parts <- surv_parts(frame$response, "cox_ph()")
  design <- cox_design_matrix(frame)
  x <- design$x
  w <- frame$weights
  n <- nrow(x)
  if (sum(parts$event) == 0) {
    stop("`cox_ph()` needs at least one event.", call. = FALSE)
  }

  frailty <- cox_frailty_kind(frailty, frame, parts, ties, robust, frailty_theta, frailty_max_iter)

  strata <- group_keys(frame$strata, n)
  means <- colSums(x * w) / sum(w)
  xc <- sweep(x, 2L, means)
  setup <- cox_setup(xc, parts$start, parts$stop, parts$event, w, strata$index, ties == "efron")
  frailty_fit <- NULL
  if (is.null(frailty)) {
    fit <- cox_newton(setup, max_iter, tol)
  } else {
    groups <- frame$frailty_cluster
    levels <- unique(groups)
    index <- match(groups, levels)
    frailty_fit <- if (frailty == "gamma") {
      cox_gamma_frailty(setup, index, length(levels), frailty_theta, frailty_max_iter, max_iter, tol)
    } else {
      cox_lognormal_frailty(
        setup, index, length(levels), frailty_theta, frailty_max_iter, max_iter, tol
      )
    }
    fit <- frailty_fit$fit
    setup$offset <- frailty_fit$offset
    frailty_fit <- list(
      distribution = frailty,
      cluster = if (is.null(frame$frailty_name)) {
        deparse1(match.call()$frailty_cluster)
      } else {
        frame$frailty_name
      },
      theta = frailty_fit$theta,
      effects = stats::setNames(frailty_fit$effects, as.character(levels)),
      test = frailty_fit$test,
      iterations = frailty_fit$iterations,
      converged = frailty_fit$converged
    )
  }
  if (!fit$converged) {
    warning(
      sprintf("`cox_ph()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }

  p <- ncol(x)
  beta <- fit$beta
  names(beta) <- colnames(x)
  naive_var <- cox_solve(fit$info, diag(p))
  dimnames(naive_var) <- list(colnames(x), colnames(x))

  robust <- robust || !is.null(frame$cluster)
  object <- list(
    coefficients = beta,
    var = naive_var,
    naive_var = NULL,
    loglik = c(fit$loglik_null, fit$loglik),
    lr_test = NULL,
    wald_test = NULL,
    score_test = NULL,
    concordance = NA_real_,
    concordance_se = NA_real_,
    linear_predictors = linear_predictor(xc, beta),
    means = means,
    n = n,
    n_event = sum(parts$event),
    iterations = fit$iterations,
    converged = fit$converged,
    ties = ties,
    robust = robust,
    conf_level = conf_level,
    strata = strata$keys,
    strata_names = names(frame$strata),
    terms = design$terms,
    assign = design$assign,
    xlevels = design$xlevels,
    contrasts = design$contrasts,
    setup = setup,
    stratum = strata$index,
    weights = w,
    has_weights = frame$has_weights,
    cluster = frame$cluster,
    frailty = frailty_fit,
    n_dropped = frame$n_dropped,
    na_action = frame$na_action,
    formula = formula,
    call = match.call()
  )
  class(object) <- "cox_ph"

  clusters <- frame$cluster
  if (robust) {
    influence <- cox_score_residuals(object) * w
    if (!is.null(clusters)) {
      influence <- rowsum(influence, match(clusters, unique(clusters)), reorder = FALSE)
    }
    dfbeta <- influence %*% naive_var
    object$naive_var <- naive_var
    object$var <- crossprod(dfbeta)
    dimnames(object$var) <- dimnames(naive_var)
  }

  wald <- drop(beta %*% cox_solve(object$var, beta))
  object$lr_test <- cox_chisq(2 * (fit$loglik - fit$loglik_null), p)
  object$wald_test <- cox_chisq(wald, p)
  object$score_test <- cox_chisq(fit$score_null, p)

  concordance <- cox_concordance(
    object$linear_predictors, parts$start, parts$stop, parts$event, w, strata$index, clusters
  )
  object$concordance <- concordance$estimate
  object$concordance_se <- concordance$std_error
  object
}

# The design matrix (without an intercept column) and what's needed to rebuild it for new data.
cox_design_matrix <- function(frame, fn = "cox_ph()") {
  no_covariates <- sprintf(
    "`%s` needs at least one covariate on the right-hand side of `formula`.", fn
  )
  if (length(frame$terms) == 0L) {
    stop(no_covariates, call. = FALSE)
  }
  tt <- attr(frame$mf, "terms")
  labels <- attr(tt, "term.labels")
  drop <- which(!labels %in% frame$terms)
  if (length(drop) > 0L) {
    tt <- stats::drop.terms(tt, dropx = drop, keep.response = TRUE)
  }
  mm <- stats::model.matrix(tt, frame$mf)
  assign <- attr(mm, "assign")
  contrasts <- attr(mm, "contrasts")
  keep <- assign != 0L
  x <- mm[, keep, drop = FALSE]
  if (ncol(x) == 0L) {
    stop(no_covariates, call. = FALSE)
  }
  if (qr(cbind(1, x))$rank < ncol(x) + 1L) {
    stop(
      "The covariates are collinear (or one is constant), so the model can't be fit.",
      call. = FALSE
    )
  }
  storage.mode(x) <- "double"
  rownames(x) <- NULL
  list(
    x = x,
    terms = stats::delete.response(tt),
    assign = assign[keep],
    xlevels = stats::.getXlevels(tt, frame$mf),
    contrasts = contrasts
  )
}

# The design matrix of `newdata`, with the columns and coding of the fit.
cox_new_design <- function(object, newdata) {
  mf <- stats::model.frame(
    object$terms, newdata, na.action = stats::na.pass, xlev = object$xlevels
  )
  mm <- stats::model.matrix(object$terms, mf, contrasts.arg = object$contrasts)
  x <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  missing_columns <- setdiff(names(object$coefficients), colnames(x))
  if (length(missing_columns) > 0L) {
    stop(
      sprintf("`newdata` doesn't give the column(s) %s.", paste(missing_columns, collapse = ", ")),
      call. = FALSE
    )
  }
  x <- x[, names(object$coefficients), drop = FALSE]
  rownames(x) <- NULL
  x
}

# The stratum (a row of `object$strata`) of each row of `newdata`.
cox_new_strata <- function(object, newdata) {
  if (is.null(object$strata_names)) {
    return(rep(1L, nrow(newdata)))
  }
  env <- environment(object$formula)
  values <- lapply(object$strata_names, function(name) eval(str2lang(name), newdata, env))
  names(values) <- object$strata_names
  labels <- strata_labels(as.data.frame(values, check.names = FALSE, stringsAsFactors = FALSE))
  index <- match(labels, object$strata$strata)
  if (anyNA(index)) {
    stop(
      sprintf(
        "`newdata` has strata that the model wasn't fit to: %s.",
        paste(unique(labels[is.na(index)]), collapse = "; ")
      ),
      call. = FALSE
    )
  }
  index
}

# -- partial likelihood --------------------------------------------------------------------------

# The parts of the partial likelihood that don't change with beta: the products of covariate
# pairs, and per stratum the rows, the distinct event times, and the rows with an event at each.
cox_setup <- function(x, start, stop, event, w, stratum, efron, offset = numeric(nrow(x))) {
  p <- ncol(x)
  layers <- lapply(sort(unique(stratum)), function(s) {
    rows <- which(stratum == s)
    dead <- rows[event[rows] == 1]
    times <- sort(unique(stop[dead]))
    at <- match(stop[dead], times)
    list(
      rows = rows,
      dead = dead,
      times = times,
      at = at,
      m = tabulate(at, length(times)),
      wd = tabulate_sum(at, w[dead], length(times))
    )
  })
  list(
    x = x,
    xx = x[, rep(seq_len(p), p), drop = FALSE] * x[, rep(seq_len(p), each = p), drop = FALSE],
    start = start,
    stop = stop,
    event = event,
    w = w,
    layers = layers,
    efron = efron,
    counting = any(is.finite(start)),
    offset = offset
  )
}

# Column sums of `values` over the rows at risk (start < t <= stop) at each of `times`.
sums_at_risk <- function(start, stop, values, times, counting) {
  above <- function(at) {
    o <- order(at)
    cumulative <- rbind(0, matrix(apply(values[o, , drop = FALSE], 2L, cumsum), nrow = length(o)))
    below <- findInterval(times, at[o], left.open = TRUE)
    total <- cumulative[nrow(cumulative), ]
    sweep(-cumulative[below + 1L, , drop = FALSE], 2L, total, "+")
  }
  out <- above(stop)
  if (counting) {
    out <- out - above(start)
  }
  out
}

# The log partial likelihood, its gradient, and its information matrix at `beta`. With
# `detail = TRUE`, also the per-time pieces used by the residuals, the baseline hazard, and the
# proportional-hazards test, for each stratum.
#
# At an event time with m tied events, Efron's approximation sums over j = 0, ..., m - 1 with the
# risk-set sums reduced by the fraction j / m of the tied events' sums, each term weighted by the
# mean event weight. Breslow's is the same with a single term (m = 1). The setup's `offset` (the
# log frailty of each row) multiplies the risk scores but isn't part of the events' own terms.
cox_terms <- function(beta, setup, detail = FALSE) {
  x <- setup$x
  p <- ncol(x)
  eta <- drop(x %*% beta)
  risk <- setup$w * exp(eta + setup$offset)
  values <- cbind(risk, risk * x, risk * setup$xx)
  i1 <- 1L + seq_len(p)
  i2 <- 1L + p + seq_len(p * p)
  pairs_a <- rep(seq_len(p), p)
  pairs_b <- rep(seq_len(p), each = p)

  loglik <- 0
  score <- numeric(p)
  info <- numeric(p * p)
  details <- vector("list", length(setup$layers))
  for (s in seq_along(setup$layers)) {
    layer <- setup$layers[[s]]
    n_times <- length(layer$times)
    if (n_times == 0L) {
      next
    }
    rows <- layer$rows
    at_risk <- sums_at_risk(
      setup$start[rows], setup$stop[rows], values[rows, , drop = FALSE], layer$times,
      setup$counting
    )
    dead <- layer$dead
    in_deaths <- rowsum(values[dead, , drop = FALSE], layer$at, reorder = TRUE)
    w_dead <- setup$w[dead]
    loglik <- loglik + sum(w_dead * eta[dead])
    score <- score + colSums(w_dead * x[dead, , drop = FALSE])

    m <- if (setup$efron) layer$m else rep(1L, n_times)
    s0 <- at_risk[, 1L]
    s1 <- at_risk[, i1, drop = FALSE]
    s2 <- at_risk[, i2, drop = FALSE]
    d0 <- in_deaths[, 1L]
    d1 <- in_deaths[, i1, drop = FALSE]
    d2 <- in_deaths[, i2, drop = FALSE]

    info_k <- matrix(0, n_times, p * p)
    if (detail) {
      hazard <- numeric(n_times)
      hazard2 <- numeric(n_times)
      x_hazard <- matrix(0, n_times, p)
      e_hazard <- numeric(n_times)
      e_x_hazard <- matrix(0, n_times, p)
      x_mean <- matrix(0, n_times, p)
    }
    for (j in seq_len(max(m)) - 1L) {
      on <- m > j
      f <- j / m[on]
      mean_w <- layer$wd[on] / m[on]
      denom <- s0[on] - f * d0[on]
      a <- (s1[on, , drop = FALSE] - f * d1[on, , drop = FALSE]) / denom
      loglik <- loglik - sum(mean_w * log(denom))
      score <- score - colSums(mean_w * a)
      info_k[on, ] <- info_k[on, , drop = FALSE] + mean_w *
        ((s2[on, , drop = FALSE] - f * d2[on, , drop = FALSE]) / denom -
          a[, pairs_a, drop = FALSE] * a[, pairs_b, drop = FALSE])
      if (detail) {
        hazard[on] <- hazard[on] + mean_w / denom
        hazard2[on] <- hazard2[on] + mean_w / denom^2
        x_hazard[on, ] <- x_hazard[on, , drop = FALSE] + mean_w * a / denom
        e_hazard[on] <- e_hazard[on] + (1 - f) * mean_w / denom
        e_x_hazard[on, ] <- e_x_hazard[on, , drop = FALSE] + (1 - f) * mean_w * a / denom
        x_mean[on, ] <- x_mean[on, , drop = FALSE] + a / m[on]
      }
    }
    info <- info + colSums(info_k)
    if (detail) {
      details[[s]] <- list(
        times = layer$times,
        n_event = layer$wd,
        hazard = hazard,
        hazard2 = hazard2,
        x_hazard = x_hazard,
        e_hazard = e_hazard,
        e_x_hazard = e_x_hazard,
        x_mean = x_mean,
        info = info_k
      )
    }
  }
  out <- list(loglik = loglik, score = score, info = matrix(info, p, p))
  if (detail) {
    out$detail <- details
  }
  out
}

# Solve `a %*% x = b` for a symmetric positive-definite `a`.
cox_solve <- function(a, b) {
  factor <- tryCatch(chol(a), error = function(e) NULL)
  if (is.null(factor)) {
    stop(
      "The information matrix is singular, so the model can't be fit. Check for collinear covariates.",
      call. = FALSE
    )
  }
  backsolve(factor, forwardsolve(t(factor), b))
}

# Newton-Raphson from beta = 0, halving the step whenever it lowers the log partial likelihood.
cox_newton <- function(setup, max_iter, tol) {
  p <- ncol(setup$x)
  beta <- numeric(p)
  current <- cox_terms(beta, setup)
  loglik_null <- current$loglik
  score_null <- sum(current$score * cox_solve(current$info, current$score))
  converged <- FALSE
  iterations <- 0L
  while (iterations < max_iter) {
    iterations <- iterations + 1L
    step <- drop(cox_solve(current$info, current$score))
    halvings <- 0L
    repeat {
      candidate <- cox_terms(beta + step, setup)
      if (is.finite(candidate$loglik) && candidate$loglik >= current$loglik - 1e-12 * abs(current$loglik)) {
        break
      }
      halvings <- halvings + 1L
      if (halvings > 30L) {
        break
      }
      step <- step / 2
    }
    change <- abs(candidate$loglik - current$loglik)
    beta <- beta + step
    current <- candidate
    if (change <= tol * (abs(current$loglik) + tol)) {
      converged <- TRUE
      break
    }
  }
  list(
    beta = beta,
    loglik_null = loglik_null,
    loglik = current$loglik,
    info = current$info,
    score_null = score_null,
    iterations = iterations,
    converged = converged
  )
}

cox_chisq <- function(statistic, df) {
  list(statistic = statistic, df = df, p_value = stats::pchisq(statistic, df, lower.tail = FALSE))
}

# -- per-row pieces --------------------------------------------------------------------------------

# Each row's cumulative hazard contribution A (its expected number of events divided by its risk
# score) and the matching covariate-weighted sum B, over the event times in (start, stop] of its
# stratum. A row with an event uses Efron's reduced terms at its own event time.
cox_row_sums <- function(object, detail) {
  setup <- object$setup
  n <- nrow(setup$x)
  p <- ncol(setup$x)
  a_sum <- numeric(n)
  b_sum <- matrix(0, n, p)
  x_mean <- matrix(0, n, p)
  for (s in seq_along(setup$layers)) {
    layer <- setup$layers[[s]]
    piece <- detail[[s]]
    if (is.null(piece)) {
      next
    }
    rows <- layer$rows
    cumulative <- function(value, query) {
      step_at(piece$times, cumsum(value), query, 0)
    }
    a_sum[rows] <- cumulative(piece$hazard, setup$stop[rows]) -
      cumulative(piece$hazard, setup$start[rows])
    for (k in seq_len(p)) {
      b_sum[rows, k] <- cumulative(piece$x_hazard[, k], setup$stop[rows]) -
        cumulative(piece$x_hazard[, k], setup$start[rows])
    }
    dead <- layer$dead
    a_sum[dead] <- a_sum[dead] - piece$hazard[layer$at] + piece$e_hazard[layer$at]
    b_sum[dead, ] <- b_sum[dead, , drop = FALSE] - piece$x_hazard[layer$at, , drop = FALSE] +
      piece$e_x_hazard[layer$at, , drop = FALSE]
    x_mean[dead, ] <- piece$x_mean[layer$at, , drop = FALSE]
  }
  list(a = a_sum, b = b_sum, x_mean = x_mean)
}

cox_detail <- function(object) {
  cox_terms(object$coefficients, object$setup, detail = TRUE)$detail
}

# Score residuals, one row per observation (unweighted).
cox_score_residuals <- function(object, detail = cox_detail(object)) {
  sums <- cox_row_sums(object, detail)
  x <- object$setup$x
  r <- exp(object$linear_predictors + object$setup$offset)
  out <- object$setup$event * (x - sums$x_mean) - r * (x * sums$a - sums$b)
  colnames(out) <- names(object$coefficients)
  out
}

cox_martingale <- function(object, detail = cox_detail(object)) {
  sums <- cox_row_sums(object, detail)
  object$setup$event - exp(object$linear_predictors + object$setup$offset) * sums$a
}

# Schoenfeld residuals, one row per event, ordered by stratum and time.
cox_schoenfeld <- function(object, detail = cox_detail(object)) {
  setup <- object$setup
  rows <- list()
  times <- list()
  strata <- list()
  for (s in seq_along(setup$layers)) {
    layer <- setup$layers[[s]]
    if (length(layer$dead) == 0L) {
      next
    }
    o <- order(setup$stop[layer$dead])
    dead <- layer$dead[o]
    residual <- setup$x[dead, , drop = FALSE] - detail[[s]]$x_mean[layer$at[o], , drop = FALSE]
    rows[[length(rows) + 1L]] <- residual
    times[[length(times) + 1L]] <- setup$stop[dead]
    strata[[length(strata) + 1L]] <- rep(s, length(dead))
  }
  out <- do.call(rbind, rows)
  colnames(out) <- names(object$coefficients)
  list(residuals = out, time = unlist(times), stratum = unlist(strata), row = unlist(lapply(
    setup$layers, function(layer) layer$dead[order(setup$stop[layer$dead])]
  )))
}

# -- concordance -----------------------------------------------------------------------------------

# Harrell's concordance of the linear predictor, with its infinitesimal-jackknife standard error.
# A pair is comparable when one member has an event at a time the other is still at risk (a member
# censored at that time counts as at risk); pairs with events at the same time are not compared.
# A higher linear predictor for the member with the event is concordant, and ties in the linear
# predictor count half. Pairs are weighted by the product of their case weights.
cox_concordance <- function(lp, start, stop, event, w, stratum, cluster = NULL) {
  n <- length(lp)
  influence <- matrix(0, n, 3L)
  counts <- numeric(3L)
  counting <- any(is.finite(start))
  for (s in unique(stratum)) {
    rows <- which(stratum == s)
    dead_rows <- rows[event[rows] == 1]
    for (t in sort(unique(stop[dead_rows]))) {
      dead <- dead_rows[stop[dead_rows] == t]
      later <- rows[(stop[rows] > t | (stop[rows] == t & event[rows] == 0)) &
        (!counting | start[rows] < t)]
      if (length(later) == 0L) {
        next
      }
      diff <- outer(lp[dead], lp[later], "-")
      kinds <- list(diff > 0, diff < 0, diff == 0)
      for (k in 1:3) {
        pair <- kinds[[k]] * outer(w[dead], w[later])
        counts[k] <- counts[k] + sum(pair)
        influence[dead, k] <- influence[dead, k] + drop(kinds[[k]] %*% w[later])
        influence[later, k] <- influence[later, k] + drop(w[dead] %*% kinds[[k]])
      }
    }
  }
  pairs <- sum(counts)
  if (pairs == 0) {
    return(list(estimate = NA_real_, std_error = NA_real_))
  }
  somers <- (counts[1L] - counts[2L]) / pairs
  dfbeta <- w * ((influence[, 1L] - influence[, 2L]) / pairs - somers / pairs * rowSums(influence))
  if (!is.null(cluster)) {
    dfbeta <- rowsum(dfbeta, match(cluster, unique(cluster)), reorder = FALSE)
  }
  list(estimate = (somers + 1) / 2, std_error = sqrt(sum(dfbeta^2) / 4))
}

# -- methods ---------------------------------------------------------------------------------------

#' @export
coef.cox_ph <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.cox_ph <- function(object, ...) {
  object$var
}

#' @export
nobs.cox_ph <- function(object, ...) {
  object$n_event
}

#' @export
logLik.cox_ph <- function(object, ...) {
  structure(
    object$loglik[2L],
    df = length(object$coefficients),
    nobs = object$n_event,
    class = "logLik"
  )
}

#' Coefficient table of a Cox model
#'
#' `as.data.frame()` gives one row per coefficient: the estimate (a log hazard ratio, or a hazard
#' ratio with `exponentiate = TRUE`), its standard error, the Wald statistic, the two-sided
#' p-value, and the confidence limits.
#'
#' @param x A [cox_ph()] fit.
#' @param row.names,optional Unused, for compatibility with the generic.
#' @param exponentiate Give hazard ratios instead of log hazard ratios. The standard error, the
#'   statistic, and the p-value are those of the log hazard ratio either way. The default is
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
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' as.data.frame(fit)
#' as.data.frame(fit, exponentiate = TRUE, conf_level = 0.9)
#' @name cox_ph_table
#' @export
as.data.frame.cox_ph <- function(x,
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
print.cox_ph <- function(x, digits = getOption("digits"), ...) {
  cat("<", class(x)[1L], "> ", deparse_formula(x$formula), "\n", sep = "")
  if (!is.null(x$cause)) {
    cat(sprintf("Cause \"%s\", with the other causes taken as censoring.\n", x$cause))
  }
  cat(sprintf(
    "%s ties. n = %d, events = %s%s\n\n",
    if (x$ties == "efron") "Efron" else "Breslow",
    x$n,
    format(x$n_event),
    if (is.null(x$strata_names)) "" else sprintf(", strata = %d", nrow(x$strata))
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(
    estimate = table$estimate,
    hazard_ratio = exp(table$estimate),
    std_error = table$std_error,
    statistic = table$statistic,
    p_value = table$p_value
  )
  rownames(coefs) <- table$term
  if (x$robust) {
    names(coefs)[3L] <- "robust_se"
  }
  print(coefs, digits = max(3L, digits - 3L))
  short <- max(3L, digits - 4L)
  test_line <- function(label, test) {
    cat(sprintf(
      "%s %s on %d df, p = %s\n",
      formatC(label, width = -22L),
      format(test$statistic, digits = short),
      as.integer(test$df),
      format_p(test$p_value)
    ))
  }
  cat("\n")
  test_line("Likelihood ratio test:", x$lr_test)
  test_line(if (x$robust) "Robust Wald test:" else "Wald test:", x$wald_test)
  test_line("Score test:", x$score_test)
  if (!is.null(x$frailty)) {
    name <- if (x$frailty$distribution == "gamma") "theta" else "sigma2"
    cat(sprintf(
      "\nShared %s frailty by %s: %s = %s\n",
      x$frailty$distribution, x$frailty$cluster, name, format(x$frailty$theta, digits = short)
    ))
    cat(sprintf(
      "Frailty variance test (%s = 0): LR = %s, p = %s\n",
      name, format(x$frailty$test$statistic, digits = short), format_p(x$frailty$test$p_value)
    ))
  }
  cat(sprintf(
    "Concordance: %s (std. error %s)\n",
    format(x$concordance, digits = short),
    format(x$concordance_se, digits = short)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- predictions -----------------------------------------------------------------------------------

#' Predictions from a Cox model
#'
#' `predict()` gives the linear predictor, the risk score, or the survival curve or cumulative
#' hazard of each row of `newdata` (or of the data the model was fit to).
#'
#' The linear predictor is \eqn{(x - \bar{x})^\top \beta}, centered at the weighted means
#' \eqn{\bar{x}} of the covariates in the fitted data, and the risk score is its exponential. The
#' cumulative hazard of a subject is the baseline cumulative hazard of its stratum (Breslow's
#' estimator, or Efron's when the model uses Efron's ties) times \eqn{\exp(x^\top \beta)}, and its
#' survival is \eqn{\exp(-H(t))}. The standard error of the cumulative hazard accounts for the
#' uncertainty in both the baseline hazard and \eqn{\beta}. Confidence limits are built on the log
#' scale of the cumulative hazard and of survival, with the upper survival limit capped at 1.
#'
#' With `conditional_after`, the curve is conditional on surviving to that time: the cumulative
#' hazard counts only the event times at or after it, so survival is
#' \eqn{S(t) / S(c^-)} for \eqn{t \ge c}, and 1 before.
#'
#' With `trajectory`, the prediction is for one subject whose covariates change over time, such as
#' the measurements of a model fit to the episodes from [split_episodes()]. Each row of
#' `trajectory` gives the covariates over an interval \eqn{(a_k, b_k]}, and the cumulative hazard
#' adds up the baseline hazard over each interval times the risk score of its covariates,
#' \eqn{H(t) = \sum_k \exp(x_k^\top \beta) (H_0(\min(t, b_k)) - H_0(a_k))} over the intervals
#' that started before \eqn{t}. No hazard builds up before the first interval or in a gap between
#' two, and the last interval holds on past its end, with a warning when `times` go beyond it. The
#' standard error accounts for the uncertainty in the baseline hazard and in \eqn{\beta} across
#' all the intervals. A stratified model takes the stratifying variables from `trajectory`, and
#' they can change from one interval to the next.
#'
#' The `"quantile"` of a subject's survival curve for a proportion \eqn{p} is the first time at
#' which survival drops to \eqn{1 - p} or below, and its confidence limits are the same times for
#' the lower and upper confidence curves. It's `NA` when the curve never drops that far. The
#' `"expectation"` is the restricted mean survival time, the area under the survival curve from 0
#' to `tau`, and its confidence limits are the areas under the confidence curves.
#'
#' @param object A [cox_ph()] fit.
#' @param newdata A data frame with the covariates (and any stratifying variables) of the subjects
#'   to predict for. The default is the data the model was fit to.
#' @param type What to predict: `"lp"` (the default), `"risk"`, `"survival"`, `"cumhaz"`,
#'   `"quantile"`, or `"expectation"`.
#' @param times For `"survival"` and `"cumhaz"`, the times to read the curves at. The default is
#'   every distinct time in the fitted data.
#' @param conditional_after For `"survival"` and `"cumhaz"`, an optional time to condition on
#'   surviving to.
#' @param trajectory A data frame with the path of covariates of one subject, in place of
#'   `newdata`: one row per interval, in time order and not overlapping, with its start `tstart`,
#'   its end `tstop`, and the covariates (and any stratifying variables) over it. Only for
#'   `"survival"` (the default with `trajectory`) and `"cumhaz"`, and not with
#'   `conditional_after`.
#' @param p For `"quantile"`, the proportions of subjects who have had the event, between 0 and 1.
#'   The default, `0.5`, gives the median.
#' @param tau For `"expectation"`, the end of the window for the restricted mean.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#' @param ... Unused.
#'
#' @return For `"lp"` and `"risk"`, a numeric vector with one value per row of `newdata`. For
#'   `"survival"` and `"cumhaz"`, a data frame with one row per subject and time: `subject` (the
#'   row of `newdata`), `strata`, `time`, `estimate`, `std_error`, `conf_low`, and `conf_high`.
#'   With `trajectory`, `subject` is 1 and `strata` is the stratum of the interval in effect at
#'   each time. For `"quantile"`, a data frame with one row per subject and proportion:
#'   `subject`, `strata`, `p`, `estimate`, `conf_low`, and `conf_high`. For `"expectation"`, one
#'   row per subject: `subject`, `strata`, `tau`, `estimate`, `conf_low`, and `conf_high`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' predict(fit, type = "risk")
#'
#' newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
#' predict(fit, newdata = newdata, type = "survival", times = c(200, 400))
#'
#' # Survival beyond day 400 for subjects alive at day 200
#' predict(
#'   fit,
#'   newdata = newdata,
#'   type = "survival",
#'   times = c(300, 400),
#'   conditional_after = 200
#' )
#'
#' # Median survival time, and the mean survival time over two years
#' predict(fit, newdata = newdata, type = "quantile")
#' predict(fit, newdata = newdata, type = "expectation", tau = 730)
#'
#' # Survival for a subject whose age is updated at day 300
#' path <- data.frame(tstart = c(0, 300), tstop = c(300, 800), age = c(60, 61), sex = 1)
#' predict(fit, trajectory = path, times = c(200, 400, 800))
#' @name cox_ph_predict
#' @export
predict.cox_ph <- function(object,
                           newdata,
                           type = c("lp", "risk", "survival", "cumhaz", "quantile", "expectation"),
                           times,
                           conditional_after = NULL,
                           trajectory = NULL,
                           p = 0.5,
                           tau,
                           conf_level = object$conf_level,
                           ...) {
  if (!is.null(trajectory)) {
    type <- if (missing(type)) "survival" else match.arg(type)
    if (!type %in% c("survival", "cumhaz")) {
      stop("With `trajectory`, `type` must be \"survival\" or \"cumhaz\".", call. = FALSE)
    }
    if (!missing(newdata)) {
      stop("Give one of `newdata` and `trajectory`, not both.", call. = FALSE)
    }
    if (!is.null(conditional_after)) {
      stop("`conditional_after` can't be used with `trajectory`.", call. = FALSE)
    }
    check_conf_level(conf_level)
    return(cox_trajectory(object, trajectory, type, times, z_value(conf_level)))
  }
  type <- match.arg(type)
  check_conf_level(conf_level)
  if (type == "quantile" && (!is.numeric(p) || anyNA(p) || any(p < 0 | p > 1))) {
    stop("`p` must be numbers between 0 and 1.", call. = FALSE)
  }
  if (type == "expectation") {
    if (missing(tau)) {
      stop("`type = \"expectation\"` needs `tau`.", call. = FALSE)
    }
    check_number(tau, "tau", min = 0)
  }
  if (missing(newdata)) {
    x <- object$setup$x
    stratum <- object$stratum
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    x <- sweep(cox_new_design(object, newdata), 2L, object$means)
    stratum <- if (type %in% c("lp", "risk")) NULL else cox_new_strata(object, newdata)
  }
  lp <- linear_predictor(x, object$coefficients)
  if (type == "lp") {
    return(lp)
  }
  if (type == "risk") {
    return(exp(lp))
  }

  detail <- cox_detail(object)
  z <- z_value(conf_level)
  if (type %in% c("quantile", "expectation")) {
    return(cox_summary_times(object, detail, x, stratum, type, p, tau, z))
  }
  if (missing(times)) {
    times <- sort(unique(object$setup$stop))
  }
  times <- check_times(times)
  if (!is.null(conditional_after)) {
    check_number(conditional_after, "conditional_after")
  }
  tables <- lapply(seq_len(nrow(x)), function(i) {
    curve <- cox_curve(object, detail, stratum[i], x[i, ], times, conditional_after)
    cbind(
      data.frame(subject = i, strata = object$strata$strata[stratum[i]], time = times),
      cox_curve_frame(curve, type, z)
    )
  })
  out <- do.call(rbind, tables)
  rownames(out) <- NULL
  out
}

# The estimate, standard error, and confidence limits of a cumulative hazard curve from
# cox_curve(), or of the survival curve it gives.
cox_curve_frame <- function(curve, type, z) {
  if (type == "cumhaz") {
    factor <- ifelse(curve$cumhaz > 0, exp(z * curve$std_error / curve$cumhaz), 1)
    return(data.frame(
      estimate = curve$cumhaz,
      std_error = curve$std_error,
      conf_low = curve$cumhaz / factor,
      conf_high = curve$cumhaz * factor
    ))
  }
  survival <- exp(-curve$cumhaz)
  data.frame(
    estimate = survival,
    std_error = survival * curve$std_error,
    conf_low = survival * exp(-z * curve$std_error),
    conf_high = pmin(survival * exp(z * curve$std_error), 1)
  )
}

# The survival or cumulative hazard of one subject whose covariates follow `trajectory`, a data
# frame with one row per interval (tstart, tstop] and the covariates over it. The cumulative
# hazard adds each interval's share of the baseline hazard of its stratum times its risk score,
# and the last interval holds on past its end.
cox_trajectory <- function(object, trajectory, type, times, z) {
  if (!is.data.frame(trajectory)) {
    stop("`trajectory` must be a data frame.", call. = FALSE)
  }
  absent <- setdiff(c("tstart", "tstop"), names(trajectory))
  if (length(absent) > 0L) {
    stop(
      sprintf(
        "`trajectory` needs the columns `tstart` and `tstop`, and doesn't have %s.",
        paste0("`", absent, "`", collapse = " or ")
      ),
      call. = FALSE
    )
  }
  tstart <- trajectory$tstart
  tstop <- trajectory$tstop
  if (nrow(trajectory) == 0L || !is.numeric(tstart) || !is.numeric(tstop) || anyNA(tstart) ||
    anyNA(tstop)) {
    stop("`tstart` and `tstop` in `trajectory` must be numbers with no missing values.",
      call. = FALSE
    )
  }
  if (any(tstart < 0) || any(tstart >= tstop)) {
    stop("Every interval of `trajectory` must have 0 <= tstart < tstop.", call. = FALSE)
  }
  k <- length(tstart)
  if (k > 1L && any(tstart[-1L] < tstop[-k])) {
    stop("The intervals of `trajectory` must be in time order and must not overlap.",
      call. = FALSE
    )
  }
  x <- sweep(cox_new_design(object, trajectory), 2L, object$means)
  if (anyNA(x)) {
    stop("`trajectory` has missing covariate values.", call. = FALSE)
  }
  stratum <- cox_new_strata(object, trajectory)
  if (missing(times)) {
    times <- sort(unique(object$setup$stop))
  }
  times <- check_times(times)
  beyond <- sum(times > tstop[k])
  if (beyond > 0L) {
    warning(
      sprintf(
        paste(
          "`trajectory` ends at %s, so %d of `times` %s carried past it with the covariates of",
          "its last interval."
        ),
        format(tstop[k]), beyond, if (beyond == 1L) "is" else "are"
      ),
      call. = FALSE
    )
  }
  tstop[k] <- Inf

  detail <- cox_detail(object)
  p <- ncol(x)
  n_times <- length(times)
  cumhaz <- numeric(n_times)
  var0 <- numeric(n_times)
  q <- matrix(0, n_times, p)
  for (i in seq_len(k)) {
    piece <- detail[[stratum[i]]]
    if (is.null(piece)) {
      next
    }
    # Each sum over the event times of the stratum in (tstart, min(t, tstop)].
    upper <- findInterval(pmin(pmax(times, tstart[i]), tstop[i]), piece$times)
    lower <- findInterval(tstart[i], piece$times)
    over <- function(value) {
      total <- c(0, cumsum(value))
      total[upper + 1L] - total[lower + 1L]
    }
    risk <- exp(sum(x[i, ] * object$coefficients))
    hazard <- over(piece$hazard)
    x_hazard <- vapply(seq_len(p), function(j) over(piece$x_hazard[, j]), numeric(n_times))
    cumhaz <- cumhaz + risk * hazard
    var0 <- var0 + risk^2 * over(piece$hazard2)
    q <- q + risk * (outer(hazard, x[i, ]) - matrix(x_hazard, nrow = n_times))
  }
  v <- if (is.null(object$naive_var)) object$var else object$naive_var
  variance <- var0 + rowSums((q %*% v) * q)
  curve <- list(cumhaz = cumhaz, std_error = sqrt(pmax(variance, 0)))
  # The stratum at each time is the one of the interval in effect then.
  current <- pmax(findInterval(times, trajectory$tstart, left.open = TRUE), 1L)
  out <- cbind(
    data.frame(subject = 1L, strata = object$strata$strata[stratum[current]], time = times),
    cox_curve_frame(curve, type, z)
  )
  rownames(out) <- NULL
  out
}

# Quantiles or restricted means of the survival curves of the rows of `x`, with limits from the
# confidence curves. The curves only change at the event times of the stratum.
cox_summary_times <- function(object, detail, x, stratum, type, p, tau, z) {
  tables <- lapply(seq_len(nrow(x)), function(i) {
    times <- if (is.null(detail[[stratum[i]]])) numeric(0) else detail[[stratum[i]]]$times
    curve <- cox_curve(object, detail, stratum[i], x[i, ], times)
    survival <- exp(-curve$cumhaz)
    low <- survival * exp(-z * curve$std_error)
    high <- pmin(survival * exp(z * curve$std_error), 1)
    if (type == "quantile") {
      at <- function(band) vapply(1 - p, function(l) km_crossing(times, band, l), numeric(1))
      out <- data.frame(p = p, estimate = at(survival), conf_low = at(low), conf_high = at(high))
    } else {
      area <- function(band) {
        widths <- diff(pmin(c(0, times, tau), tau))
        sum(c(1, band) * widths)
      }
      out <- data.frame(tau = tau, estimate = area(survival), conf_low = area(low),
        conf_high = area(high))
    }
    cbind(data.frame(subject = i, strata = object$strata$strata[stratum[i]]), out)
  })
  out <- do.call(rbind, tables)
  rownames(out) <- NULL
  out
}

# The cumulative hazard and its standard error at `times` for a subject with centered covariates
# `xc` in stratum `s`. The variance is the baseline term plus d' V d, with d the derivative of the
# cumulative hazard with respect to beta.
cox_curve <- function(object, detail, s, xc, times, conditional_after = NULL) {
  piece <- detail[[s]]
  p <- length(xc)
  if (is.null(piece)) {
    return(list(cumhaz = numeric(length(times)), std_error = numeric(length(times))))
  }
  hazard <- piece$hazard
  hazard2 <- piece$hazard2
  x_hazard <- piece$x_hazard
  if (!is.null(conditional_after)) {
    before <- piece$times < conditional_after
    hazard[before] <- 0
    hazard2[before] <- 0
    x_hazard[before, ] <- 0
  }
  at <- findInterval(times, piece$times)
  cumulative <- function(value) c(0, cumsum(value))[at + 1L]
  cumhaz0 <- cumulative(hazard)
  var0 <- cumulative(hazard2)
  x_cum <- vapply(seq_len(p), function(k) cumulative(x_hazard[, k]), numeric(length(times)))
  x_cum <- matrix(x_cum, nrow = length(times))
  d <- outer(cumhaz0, xc) - x_cum
  risk <- exp(sum(xc * object$coefficients))
  v <- if (is.null(object$naive_var)) object$var else object$naive_var
  variance <- (var0 + rowSums((d %*% v) * d)) * risk^2
  list(cumhaz = cumhaz0 * risk, std_error = sqrt(pmax(variance, 0)))
}

#' Baseline hazard of a model
#'
#' `baseline_hazard()` gives the baseline hazard of a fitted model, with its standard error and
#' confidence limits. For a [cox_ph()] fit, it's the cumulative hazard \eqn{H_0(t)} at every
#' distinct time in each stratum. For a [piecewise_exponential()] fit, it's the hazard in each
#' interval.
#'
#' @param object A fitted model.
#' @param ... Passed on to methods.
#'
#' @return A data frame. See the methods for the columns.
#'
#' @export
baseline_hazard <- function(object, ...) {
  UseMethod("baseline_hazard")
}

#' @describeIn baseline_hazard For a [cox_ph()] fit, the cumulative hazard of a subject with every
#'   covariate at 0 (or at its mean, with `centered = TRUE`), by Breslow's estimator (Efron's when
#'   the model uses Efron's ties). Survival is \eqn{\exp(-H_0(t))}, and the confidence limits of
#'   the cumulative hazard map onto those of survival, so log-scale limits for \eqn{H_0} are
#'   log-log limits for survival. The table has the columns `strata`, the stratifying variables,
#'   `time`, `n_risk`, `n_event`, `estimate`, `std_error`, `conf_low`, and `conf_high`, with the
#'   same layout for one stratum (`strata` is `"all"`) as for several.
#'
#' @param centered Give the cumulative hazard at the means of the covariates instead of at 0. The
#'   default is `FALSE`.
#' @param conf_type The scale for confidence intervals: `"log"` (the default) or `"plain"`, with
#'   the lower limit floored at 0.
#' @param conf_level The confidence level. The default is the level the model was fit with.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' baseline_hazard(fit, centered = TRUE)
#' @export
baseline_hazard.cox_ph <- function(object,
                                   centered = FALSE,
                                   conf_type = c("log", "plain"),
                                   conf_level = object$conf_level,
                                   ...) {
  check_flag(centered, "centered")
  conf_type <- match.arg(conf_type)
  check_conf_level(conf_level)
  detail <- cox_detail(object)
  setup <- object$setup
  xc <- if (centered) numeric(length(object$means)) else -object$means
  z <- z_value(conf_level)
  tables <- lapply(seq_along(setup$layers), function(s) {
    rows <- setup$layers[[s]]$rows
    counts <- count_events(setup$start[rows], setup$stop[rows], setup$event[rows], setup$w[rows])
    curve <- cox_curve(object, detail, s, xc, counts$time)
    estimate <- curve$cumhaz
    se <- curve$std_error
    if (conf_type == "plain") {
      low <- pmax(estimate - z * se, 0)
      high <- estimate + z * se
    } else {
      factor <- ifelse(estimate > 0, exp(z * se / estimate), 1)
      low <- estimate / factor
      high <- estimate * factor
    }
    data.frame(
      time = counts$time,
      n_risk = counts$n_risk,
      n_event = counts$n_event,
      estimate = estimate,
      std_error = se,
      conf_low = low,
      conf_high = high
    )
  })
  bind_groups(object$strata, tables)
}

# -- residuals -------------------------------------------------------------------------------------

#' Residuals of a Cox model
#'
#' `residuals()` gives the diagnostic residuals of a [cox_ph()] fit.
#'
#' * `"martingale"`: the event indicator minus the expected number of events (the subject's
#'   cumulative hazard over its follow-up). One value per row.
#' * `"deviance"`: martingale residuals transformed to be more symmetric about 0,
#'   \eqn{\mathrm{sign}(m) \sqrt{-2 (m + \delta \log(\delta - m))}}. One value per row.
#' * `"score"`: each row's contribution to the score at the estimate. One column per coefficient.
#' * `"schoenfeld"`: at each event, the covariates minus their risk-weighted mean over the risk
#'   set. One row per event, ordered by stratum and time, with the event times as row names.
#' * `"scaledsch"`: Schoenfeld residuals scaled by the number of events times the variance, plus
#'   the coefficients. Plotted against time, each column traces the coefficient's drift over time.
#' * `"dfbeta"`: the approximate change in the coefficients when each row is left out, the
#'   weighted score residuals times the variance.
#' * `"dfbetas"`: `"dfbeta"` divided by the standard errors.
#' * `"leverage"`: the quadratic form of each row's score residual in the variance, a measure of
#'   its influence on the fit. One value per row.
#'
#' The variance used is the model-based one, even for a fit with a robust variance.
#'
#' @param object A [cox_ph()] fit.
#' @param type The type of residual. The default is `"martingale"`.
#' @param ... Unused.
#'
#' @return A numeric vector for `"martingale"`, `"deviance"`, and `"leverage"`, and a matrix with
#'   one column per coefficient for the other types.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' residuals(fit)
#' residuals(fit, type = "dfbeta")
#' @name cox_ph_residuals
#' @export
residuals.cox_ph <- function(object,
                             type = c(
                               "martingale", "deviance", "score", "schoenfeld", "scaledsch",
                               "dfbeta", "dfbetas", "leverage"
                             ),
                             ...) {
  type <- match.arg(type)
  detail <- cox_detail(object)
  v <- if (is.null(object$naive_var)) object$var else object$naive_var
  event <- object$setup$event
  if (type %in% c("martingale", "deviance")) {
    m <- cox_martingale(object, detail)
    if (type == "martingale") {
      return(m)
    }
    log_term <- ifelse(event > 0, event * log(pmax(event - m, .Machine$double.xmin)), 0)
    return(sign(m) * sqrt(-2 * (m + log_term)))
  }
  if (type %in% c("schoenfeld", "scaledsch")) {
    sch <- cox_schoenfeld(object, detail)
    out <- sch$residuals
    if (type == "scaledsch") {
      out <- out %*% v * nrow(out) + rep(object$coefficients, each = nrow(out))
    }
    colnames(out) <- names(object$coefficients)
    rownames(out) <- format(sch$time, trim = TRUE)
    return(out)
  }
  score <- cox_score_residuals(object, detail)
  if (type == "score") {
    return(score)
  }
  if (type == "leverage") {
    return(rowSums((score %*% v) * score))
  }
  out <- (score * object$weights) %*% v
  if (type == "dfbetas") {
    out <- sweep(out, 2L, sqrt(diag(v)), "/")
  }
  colnames(out) <- names(object$coefficients)
  out
}

# -- proportional hazards ------------------------------------------------------------------------

#' Test the proportional hazards assumption of a Cox model
#'
#' `cox_zph()` tests whether each coefficient of a [cox_ph()] fit is constant over time, the
#' assumption behind proportional hazards (Grambsch and Therneau, 1994). For each term it is the
#' score test of adding an interaction between the term and a function \eqn{g(t)} of time, and the
#' global test adds all of them at once. A small p-value suggests the term's effect changes over
#' time.
#'
#' The time function is set by `transform`: `"km"` (the default) uses \eqn{1 - \hat{S}(t^-)}, one
#' minus the Kaplan-Meier estimate of the pooled sample just before each time, `"rank"` uses the
#' ranks of the times, `"identity"` the times themselves, and `"log"` their logarithms. The
#' function is centered at its mean over the events.
#'
#' With `breaks`, the follow-up is also split into windows at those times, and the tests are
#' repeated within each window that holds at least 2 events. This shows where in time a departure
#' from proportional hazards arises.
#'
#' @param fit A [cox_ph()] fit.
#' @param transform The function of time: `"km"` (the default), `"rank"`, `"identity"`, or
#'   `"log"`.
#' @param breaks Optional times at which to split the follow-up into windows.
#'
#' @return A `cox_zph` object. Its `table` has one row per term and a final `"GLOBAL"` row, with
#'   the columns `term`, `statistic` (a chi-squared statistic), `df`, and `p_value`. It also holds
#'   `time` and `x` (the event times and their transformed values) and `y` (the scaled Schoenfeld
#'   residuals) for plotting each term's coefficient over time, and `transform`. With `breaks`,
#'   `windows` is a data frame with the columns `window`, `start`, `end`, `n_event`, `term`,
#'   `statistic`, `df`, and `p_value`. [as.data.frame()] gives the table.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' cox_zph(fit)
#' cox_zph(fit, transform = "identity", breaks = 300)
#' @export
cox_zph <- function(fit, transform = c("km", "rank", "identity", "log"), breaks = NULL) {
  if (!inherits(fit, "cox_ph")) {
    stop("`fit` must be a `cox_ph()` fit.", call. = FALSE)
  }
  transform <- match.arg(transform)
  if (!is.null(breaks)) {
    breaks <- sort(unique(check_times(breaks, "breaks")))
  }
  setup <- fit$setup
  detail <- cox_detail(fit)
  sch <- cox_schoenfeld(fit, detail)

  stop_time <- setup$stop
  values <- switch(transform,
    identity = stop_time,
    log = log(stop_time),
    rank = rank(stop_time),
    km = {
      counts <- count_events(setup$start, stop_time, setup$event, rep(1, length(stop_time)))
      km <- cumprod(ifelse(counts$n_risk > 0, 1 - counts$n_event / counts$n_risk, 1))
      1 - c(1, km)[findInterval(stop_time, counts$time, left.open = TRUE) + 1L]
    }
  )
  g <- values[sch$row]

  # The information of each event (its event time's information, shared out among tied events),
  # so a window of events adds up the right pieces.
  info <- do.call(rbind, lapply(seq_along(setup$layers), function(s) {
    layer <- setup$layers[[s]]
    if (length(layer$dead) == 0L) {
      return(NULL)
    }
    at <- layer$at[order(setup$stop[layer$dead])]
    detail[[s]]$info[at, , drop = FALSE] / layer$m[at]
  }))
  weighted <- sch$residuals * setup$w[sch$row]

  table <- zph_table(fit, weighted, info, g - mean(g))
  v <- if (is.null(fit$naive_var)) fit$var else fit$naive_var
  scaled <- sch$residuals %*% v * nrow(sch$residuals) +
    rep(fit$coefficients, each = nrow(sch$residuals))
  colnames(scaled) <- names(fit$coefficients)

  windows <- NULL
  if (!is.null(breaks)) {
    edges <- c(-Inf, breaks, Inf)
    windows <- do.call(rbind, lapply(seq_len(length(edges) - 1L), function(k) {
      inside <- sch$time > edges[k] & sch$time <= edges[k + 1L]
      if (sum(inside) < 2L) {
        return(NULL)
      }
      g_in <- g[inside]
      result <- zph_table(
        fit, weighted[inside, , drop = FALSE], info[inside, , drop = FALSE], g_in - mean(g_in)
      )
      start <- if (k == 1L) 0 else edges[k]
      cbind(
        data.frame(
          window = sprintf("(%s, %s]", format(start), format(edges[k + 1L])),
          start = start,
          end = edges[k + 1L],
          n_event = sum(inside),
          stringsAsFactors = FALSE
        ),
        result
      )
    }))
    rownames(windows) <- NULL
  }

  structure(
    list(
      table = table,
      time = sch$time,
      x = g,
      y = scaled,
      transform = transform,
      windows = windows,
      formula = fit$formula
    ),
    class = "cox_zph"
  )
}

# The per-term and global score tests for adding g(t) interactions, from the weighted Schoenfeld
# residuals and per-event information of a set of events, and the centered time function.
zph_table <- function(fit, residuals, info, g) {
  p <- ncol(residuals)
  u <- c(numeric(p), colSums(residuals * g))
  sum_info <- function(weight) matrix(colSums(info * weight), p, p)
  i0 <- sum_info(1)
  i1 <- sum_info(g)
  i2 <- sum_info(g^2)
  imat <- rbind(cbind(i0, i1), cbind(i1, i2))
  terms <- attr(fit$terms, "term.labels")
  test <- function(index) {
    keep <- c(seq_len(p), p + index)
    sum(u[keep] * solve(imat[keep, keep, drop = FALSE], u[keep]))
  }
  statistic <- c(
    vapply(seq_along(terms), function(k) test(which(fit$assign == k)), numeric(1)),
    test(seq_len(p))
  )
  df <- c(as.integer(tabulate(fit$assign, length(terms))), p)
  data.frame(
    term = c(terms, "GLOBAL"),
    statistic = statistic,
    df = df,
    p_value = stats::pchisq(statistic, df, lower.tail = FALSE),
    stringsAsFactors = FALSE
  )
}

#' @export
as.data.frame.cox_zph <- function(x, ...) {
  x$table
}

#' @export
print.cox_zph <- function(x, digits = getOption("digits"), ...) {
  cat("<cox_zph> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf("Transform: %s\n\n", x$transform))
  table <- x$table[-1L]
  rownames(table) <- x$table$term
  print(table, digits = max(3L, digits - 3L))
  if (!is.null(x$windows)) {
    cat("\nBy window:\n")
    print(x$windows[c("window", "n_event", "term", "statistic", "df", "p_value")],
      digits = max(3L, digits - 3L), row.names = FALSE
    )
  }
  invisible(x)
}
