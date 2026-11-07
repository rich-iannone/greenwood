# Competing risks: aalen_johansen(), grays_test(), and fine_gray(). The numeric fixtures
# (cif_mgus2, finegray_mgus2_pcm, grays_test_*) are replayed by tests/testthat/test-competing.R.

# -- shared helpers ------------------------------------------------------------------------------

# The exit times, the cause of each row (0 for censored, k for the k-th state), and the state
# names of a competing-risks response.
cr_response <- function(frame, fn) {
  type <- attr(frame$response, "type")
  if (identical(type, "mcounting")) {
    stop(sprintf("`%s` doesn't support delayed entry (counting-process data) yet.", fn),
      call. = FALSE
    )
  }
  if (!identical(type, "mright")) {
    stop(
      sprintf(
        paste0(
          "`%s` needs a competing-risks response, `Surv(time, event)` with `event` a factor ",
          "whose first level is censoring."
        ),
        fn
      ),
      call. = FALSE
    )
  }
  y <- unclass(frame$response)
  list(
    time = unname(y[, "time"]),
    status = as.integer(y[, "status"]),
    states = attr(frame$response, "states")
  )
}

# The cause code (1..k) of `cause`, given as a state name or as a code.
cr_cause <- function(cause, states) {
  if (is.character(cause) && length(cause) == 1L && cause %in% states) {
    return(match(cause, states))
  }
  if (is.numeric(cause) && length(cause) == 1L && !is.na(cause) && cause %in% seq_along(states)) {
    return(as.integer(cause))
  }
  stop(
    sprintf(
      "`cause` must be one of the states (%s) or a number from 1 to %d.",
      paste0("\"", states, "\"", collapse = ", "), length(states)
    ),
    call. = FALSE
  )
}

cr_no_specials <- function(frame, fn) {
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop(sprintf("`%s` doesn't support `strata()` or `cluster()` terms.", fn), call. = FALSE)
  }
}

# -- Aalen-Johansen --------------------------------------------------------------------------------

#' Cumulative incidence with competing risks
#'
#' `aalen_johansen()` estimates the cumulative incidence function of each cause when subjects can
#' have one of several kinds of event. The cumulative incidence \eqn{F_k(t)} is the probability of
#' having an event of cause \eqn{k} by time \eqn{t}. Unlike one minus a Kaplan-Meier curve that
#' treats the other causes as censoring, it accounts for subjects who can no longer have the event
#' because another one happened first, so the incidences of all causes add up to the probability
#' of any event.
#'
#' At each distinct time \eqn{t_j}, with \eqn{n_j} subjects at risk, \eqn{d_{kj}} events of cause
#' \eqn{k}, and \eqn{\hat S(t_{j-1})} the Kaplan-Meier estimate of being free of any event just
#' before, the estimate is
#' \deqn{\hat F_k(t) = \sum_{t_j \le t} \hat S(t_{j-1}) \frac{d_{kj}}{n_j}.}
#' The standard error is the delta-method one. Confidence limits are built on the plain scale (the
#' default), the log scale, or the log-log scale, and kept within 0 and 1. The log and log-log
#' limits are missing where the estimate is 0 or 1.
#'
#' @param formula A formula with a competing-risks [Surv()] response on the left and either `1` or
#'   the grouping variables on the right. The response is `Surv(time, event)` with `event` a
#'   factor: its first level means censored and each other level is a cause, as in
#'   `Surv(time, factor(status, 0:2, c("censor", "pcm", "death"))) ~ sex`.
#' @param data A data frame holding the variables in `formula`.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param conf_type The scale of the confidence limits: `"plain"` (the default), `"log"`, or
#'   `"log-log"`.
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#'
#' @return An `aalen_johansen` object. [as.data.frame()] gives the estimates with one row per
#'   group, cause, and distinct time: `strata`, the grouping variables, `cause`, `time`, `n_risk`,
#'   `estimate`, `std_error`, `conf_low`, and `conf_high`. The layout is the same for one group
#'   (`strata` is `"all"`) as for several. [predict()] reads the curves at given times.
#'
#' @seealso [grays_test()] to compare cumulative incidence between groups, [fine_gray()] to model
#'   it, and [multi_state()] for more general state structures.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1),
#'   sex = rep(c("F", "M"), times = 6)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' fit <- aalen_johansen(Surv(time, event) ~ 1, data = mgus)
#' fit
#' as.data.frame(fit)
#'
#' # One set of curves per sex, read at given times
#' by_sex <- aalen_johansen(Surv(time, event) ~ sex, data = mgus, conf_type = "log-log")
#' predict(by_sex, times = c(50, 100))
#' @export
aalen_johansen <- function(formula,
                           data,
                           subset,
                           na.action = stats::na.omit,
                           conf_type = c("plain", "log", "log-log"),
                           conf_level = 0.95) {
  conf_type <- match.arg(conf_type)
  check_conf_level(conf_level)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  cr_no_specials(frame, "aalen_johansen()")
  cr <- cr_response(frame, "aalen_johansen()")
  groups <- group_keys(frame$vars, length(cr$time))
  z <- z_value(conf_level)
  curves <- lapply(seq_len(nrow(groups$keys)), function(g) {
    rows <- groups$index == g
    aj_curves(cr$time[rows], cr$status[rows], length(cr$states), z, conf_type)
  })
  n_event <- vapply(seq_along(cr$states), function(k) sum(cr$status == k), numeric(1))
  names(n_event) <- cr$states
  structure(
    list(
      curves = curves,
      groups = groups$keys,
      states = cr$states,
      n = length(cr$time),
      n_event = n_event,
      conf_type = conf_type,
      conf_level = conf_level,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "aalen_johansen"
  )
}

# The cumulative incidence of each cause in one group: a list with one data frame per cause.
aj_curves <- function(time, status, n_causes, z, conf_type) {
  times <- sort(unique(time))
  at <- match(time, times)
  n_risk <- weight_at_or_above(time, rep(1, length(time)), times)
  d_any <- tabulate_sum(at, as.numeric(status > 0), length(times))
  surv <- cumprod(1 - d_any / n_risk)
  surv_left <- c(1, surv[-length(surv)])
  # The pieces of the delta-method variance that are the same for every cause.
  a <- ifelse(n_risk > d_any, d_any / (n_risk * (n_risk - d_any)), 0)
  lapply(seq_len(n_causes), function(k) {
    d_k <- tabulate_sum(at, as.numeric(status == k), length(times))
    cif <- cumsum(surv_left * d_k / n_risk)
    b <- surv_left^2 * (n_risk - d_k) / n_risk * d_k / n_risk^2
    c_k <- surv_left * d_k / n_risk^2
    var <- cif^2 * cumsum(a) - 2 * cif * cumsum(a * cif) + cumsum(a * cif^2) + cumsum(b) -
      2 * cif * cumsum(c_k) + 2 * cumsum(c_k * cif)
    se <- sqrt(pmax(var, 0))
    limits <- cif_limits(cif, se, conf_type, z)
    data.frame(
      time = times,
      n_risk = n_risk,
      estimate = cif,
      std_error = se,
      conf_low = limits$low,
      conf_high = limits$high
    )
  })
}

# Confidence limits for a cumulative incidence, kept within 0 and 1. The plain limits are 0 where
# the estimate is 0, the others are missing there, and all are missing where it's 1.
cif_limits <- function(cif, se, conf_type, z) {
  if (conf_type == "plain") {
    low <- cif - z * se
    high <- cif + z * se
    low[cif <= 0] <- 0
    high[cif <= 0] <- 0
  } else if (conf_type == "log") {
    low <- cif * exp(-z * se / cif)
    high <- cif * exp(z * se / cif)
  } else {
    log_f <- log(cif)
    theta <- log(-log_f)
    se_theta <- se / (cif * abs(log_f))
    low <- exp(-exp(theta + z * se_theta))
    high <- exp(-exp(theta - z * se_theta))
  }
  low <- pmin(pmax(low, 0), 1)
  high <- pmin(pmax(high, 0), 1)
  low[cif >= 1] <- NaN
  high[cif >= 1] <- NaN
  list(low = low, high = high)
}

aj_table <- function(object, curves_of) {
  tables <- lapply(seq_along(object$curves), function(g) {
    do.call(rbind, lapply(seq_along(object$states), function(k) {
      out <- curves_of(object$curves[[g]][[k]])
      cbind(data.frame(cause = object$states[k], stringsAsFactors = FALSE), out)
    }))
  })
  bind_groups(object$groups, tables)
}

#' @export
as.data.frame.aalen_johansen <- function(x, row.names = NULL, optional = FALSE, ...) {
  aj_table(x, identity)
}

#' Cumulative incidence at given times
#'
#' `predict()` reads the cumulative incidence curves of an [aalen_johansen()] fit at given times.
#' Each curve is a right-continuous step function that is 0 before the first event.
#'
#' @param object An [aalen_johansen()] fit.
#' @param times The times to read the curves at.
#' @param ... Unused.
#'
#' @return A data frame with one row per group, cause, and time: `strata`, the grouping variables,
#'   `cause`, `time`, `estimate`, `std_error`, `conf_low`, and `conf_high`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' fit <- aalen_johansen(Surv(time, event) ~ 1, data = mgus)
#' predict(fit, times = c(25, 100, 200))
#' @name aalen_johansen_predict
#' @export
predict.aalen_johansen <- function(object, times, ...) {
  times <- check_times(times)
  start <- cif_limits(0, 0, object$conf_type, z_value(object$conf_level))
  aj_table(object, function(curve) {
    data.frame(
      time = times,
      estimate = step_at(curve$time, curve$estimate, times, 0),
      std_error = step_at(curve$time, curve$std_error, times, 0),
      conf_low = step_at(curve$time, curve$conf_low, times, start$low),
      conf_high = step_at(curve$time, curve$conf_high, times, start$high)
    )
  })
}

#' @export
print.aalen_johansen <- function(x, digits = getOption("digits"), ...) {
  cat("<aalen_johansen> ", deparse_formula(x$formula), "\n", sep = "")
  counts <- paste(sprintf("%s = %s", x$states, format(x$n_event)), collapse = ", ")
  cat(sprintf("n = %d. Events: %s\n\n", x$n, counts))
  table <- aj_table(x, function(curve) {
    last <- nrow(curve)
    curve[last, c("time", "estimate", "std_error", "conf_low", "conf_high")]
  })
  if (nrow(x$groups) == 1L) {
    table$strata <- NULL
  }
  cat(sprintf("Cumulative incidence at the last time (%s limits):\n", x$conf_type))
  print(table, digits = max(3L, digits - 3L), row.names = FALSE)
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- Gray's test -----------------------------------------------------------------------------------

#' Gray's test for differences in cumulative incidence
#'
#' `grays_test()` tests whether the cumulative incidence of one cause differs between groups when
#' there are competing risks. It's the competing-risks counterpart of [logrank_test()]: where the
#' log-rank test compares hazards, Gray's test compares the subdistribution hazards, which
#' determine the cumulative incidence. Subjects who had a competing event stay in the risk set,
#' down-weighted by how likely they would have been to remain uncensored.
#'
#' The statistic compares the observed events of the cause in each group with the number expected
#' if the groups shared one cumulative incidence curve, and has a chi-square distribution with one
#' fewer degrees of freedom than there are groups. The variance is Gray's (1988).
#'
#' @inheritParams aalen_johansen
#' @param formula A formula with a competing-risks [Surv()] response on the left and the grouping
#'   variables on the right (see [aalen_johansen()]).
#' @param cause The cause to compare, as a state name or as a number (`1` for the first cause, the
#'   default).
#'
#' @return A `greenwood_test` object with the `statistic`, the degrees of freedom `df`, the
#'   `p_value`, the `method`, and `groups`, a data frame with one row per group: `group`, the
#'   grouping variables, `n`, and the `observed` and `expected` events of the cause.
#'
#' @references Gray, R. J. (1988). A class of K-sample tests for comparing the cumulative incidence
#'   of a competing risk. *The Annals of Statistics*, 16(3), 1141-1154.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2),
#'   sex = rep(c("F", "M"), times = 7)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' grays_test(Surv(time, event) ~ sex, data = mgus, cause = "pcm")
#' @export
grays_test <- function(formula, data, subset, na.action = stats::na.omit, cause = 1) {
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  cr_no_specials(frame, "grays_test()")
  if (is.null(frame$vars)) {
    stop("`grays_test()` needs grouping variables on the right-hand side of `formula`.",
      call. = FALSE
    )
  }
  cr <- cr_response(frame, "grays_test()")
  target <- cr_cause(cause, cr$states)
  groups <- group_keys(frame$vars, length(cr$time))
  if (nrow(groups$keys) < 2L) {
    stop("`grays_test()` needs at least two groups.", call. = FALSE)
  }
  if (!any(cr$status == target)) {
    stop(sprintf("There are no events of cause \"%s\".", cr$states[target]), call. = FALSE)
  }
  test <- gray_statistic(cr$time, cr$status, groups$index, nrow(groups$keys), target)
  input <- list(stratified = FALSE, keys = groups$keys, group = groups$index)
  method <- sprintf("Gray's test (cause = \"%s\")", cr$states[target])
  lr_result(input, frame, test$statistic, test$df, method, test$observed, test$expected)
}

# Gray's statistic with rho = 0, following the algorithm of the cmprsk package. Rows are visited in
# order of time, one block of tied times at a time. `group` numbers the groups 1..n_groups.
gray_statistic <- function(time, status, group, n_groups, target) {
  ng <- n_groups
  ng1 <- ng - 1L
  o <- order(time)
  y <- time[o]
  m <- ifelse(status[o] == target, 1L, ifelse(status[o] > 0, 2L, 0L))
  ig <- group[o]
  rs <- tabulate(ig, ng)

  s <- numeric(ng1)
  v <- matrix(0, ng1, ng1)
  f1m <- numeric(ng)
  f1 <- numeric(ng)
  skmm <- rep(1, ng)
  skm <- rep(1, ng)
  v3 <- numeric(ng)
  v2 <- matrix(0, ng1, ng)
  c_mat <- matrix(0, ng, ng)
  fm <- 0
  observed <- numeric(ng)
  expected <- numeric(ng)

  blocks <- split(seq_along(y), factor(y, levels = unique(y)))
  for (rows in blocks) {
    d <- matrix(0, 3L, ng)
    for (i in rows) {
      d[m[i] + 1L, ig[i]] <- d[m[i] + 1L, ig[i]] + 1
    }
    leaving <- tabulate(ig[rows], ng)
    nd1 <- sum(d[2L, ])
    nd2 <- sum(d[3L, ])
    if (nd1 == 0 && nd2 == 0) {
      rs <- rs - leaving
      next
    }
    live <- rs > 0
    td <- d[2L, ] + d[3L, ]
    skm[live] <- skmm[live] * (rs[live] - td[live]) / rs[live]
    f1[live] <- f1m[live] + skmm[live] * d[2L, live] / rs[live]
    tr <- sum(rs[live] / skmm[live])
    tq <- sum(rs[live] * (1 - f1m[live]) / skmm[live])
    f <- if (tr > 0) fm + nd1 / tr else fm

    t1 <- ifelse(live, rs / skmm, 0)
    a_mat <- diag(t1, ng) - outer(t1, t1) / tr
    if (1 - fm != 0) {
      c_mat <- c_mat + a_mat * nd1 / (tr * (1 - fm))
    }

    e <- if (tq > 0) ifelse(live, nd1 * rs * (1 - f1m) / (skmm * tq), 0) else numeric(ng)
    s <- s + (d[2L, ] - e)[seq_len(ng1)]
    observed <- observed + d[2L, ]
    expected <- expected + e

    if (nd1 > 0) {
      for (k in which(live)) {
        t4 <- if (skm[k] > 0) 1 - (1 - f) / skm[k] else 1
        t5 <- if (tr * skmm[k] > 1) 1 - (nd1 - 1) / (tr * skmm[k] - 1) else 1
        t3 <- t5 * skmm[k] * nd1 / (tr * rs[k])
        v3[k] <- v3[k] + t4^2 * t3
        u <- a_mat[seq_len(ng1), k] - t4 * c_mat[seq_len(ng1), k]
        v2[, k] <- v2[, k] + u * t4 * t3
        v <- v + outer(u, u) * t3
      }
    }
    if (nd2 > 0) {
      for (k in which(skm > 0 & d[3L, ] > 0)) {
        t4 <- (1 - f) / skm[k]
        t5 <- if (rs[k] > 1) 1 - (d[3L, k] - 1) / (rs[k] - 1) else 1
        t3 <- t5 * skmm[k]^2 * d[3L, k] / rs[k]^2
        v3[k] <- v3[k] + t4^2 * t3
        u <- t4 * c_mat[seq_len(ng1), k]
        v2[, k] <- v2[, k] - u * t4 * t3
        v <- v + outer(u, u) * t3
      }
    }
    rs <- rs - leaving
    fm <- f
    f1m <- f1
    skmm <- skm
  }

  c1 <- c_mat[seq_len(ng1), , drop = FALSE]
  v <- v + c1 %*% (v3 * t(c1)) + c1 %*% t(v2) + v2 %*% t(c1)
  list(statistic = sum(s * solve(v, s)), df = ng1, observed = observed, expected = expected)
}

# -- Fine-Gray -------------------------------------------------------------------------------------

#' Fine-Gray regression for competing risks
#'
#' `fine_gray()` models the cumulative incidence of one cause with the proportional
#' subdistribution hazards model of Fine and Gray. The subdistribution hazard of a subject with
#' covariates \eqn{x} is \eqn{h_0(t) \exp(x^\top \beta)}, and the cumulative incidence is
#' \deqn{F(t \mid x) = 1 - \exp\{-H_0(t) \exp(x^\top \beta)\}.}
#' Each coefficient is a log subdistribution hazard ratio. A positive one means a higher
#' cumulative incidence of the cause.
#'
#' The coefficients maximize a weighted partial likelihood with Breslow's handling of ties.
#' Subjects who had a competing event stay in the risk set after it, weighted by the
#' Kaplan-Meier estimate of remaining uncensored at each later time divided by that at their own
#' event time. Events of either kind are taken to happen just before any censoring tied with them.
#' The standard errors are the robust (sandwich) ones, and the model-based variance is kept as
#' `naive_var`.
#'
#' @inheritParams aalen_johansen
#' @param formula A formula with a competing-risks [Surv()] response on the left (see
#'   [aalen_johansen()]) and the covariates on the right, such as
#'   `Surv(time, event) ~ age + sex`. Factors are coded with treatment contrasts.
#' @param cause The cause to model, as a state name or as a number (`1` for the first cause, the
#'   default).
#' @param max_iter The largest number of Newton-Raphson iterations. The default is `30`.
#' @param tol The convergence tolerance: iterations stop once the log partial likelihood changes
#'   by less than `tol` times its size. The default is `1e-9`.
#'
#' @return A `fine_gray` object. Use [as.data.frame()] for the coefficient table, [coef()] and
#'   [vcov()] for the estimates and their robust variance, and [predict()] for linear predictors,
#'   relative subdistribution hazards, and cumulative incidence. The object also holds `naive_var`,
#'   `loglik` (at \eqn{\beta = 0} and at the estimate), `baseline` (the baseline cumulative
#'   subdistribution hazard at the event times of the cause, for covariates at their means), `n`,
#'   `n_event`, and `cause`.
#'
#' @references Fine, J. P. and Gray, R. J. (1999). A proportional hazards model for the
#'   subdistribution of a competing risk. *Journal of the American Statistical Association*,
#'   94(446), 496-509.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' fit <- fine_gray(Surv(time, event) ~ age, data = mgus, cause = "pcm")
#' fit
#' as.data.frame(fit, exponentiate = TRUE)
#'
#' # Cumulative incidence of pcm for two new subjects
#' predict(fit, newdata = data.frame(age = c(60, 75)), type = "cif", times = c(100, 200))
#' @export
fine_gray <- function(formula,
                      data,
                      cause = 1,
                      subset,
                      na.action = stats::na.omit,
                      conf_level = 0.95,
                      max_iter = 30,
                      tol = 1e-9) {
  check_conf_level(conf_level)
  check_number(max_iter, "max_iter", min = 1)
  check_number(tol, "tol", min = 0)
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  cr_no_specials(frame, "fine_gray()")
  cr <- cr_response(frame, "fine_gray()")
  target <- cr_cause(cause, cr$states)
  design <- cox_design_matrix(frame, "fine_gray()")
  if (!any(cr$status == target)) {
    stop(sprintf("There are no events of cause \"%s\".", cr$states[target]), call. = FALSE)
  }
  x <- design$x
  means <- colMeans(x)
  xc <- sweep(x, 2L, means)
  setup <- fg_setup(cr$time, cr$status, target)

  fit <- fg_newton(xc, setup, max_iter, tol)
  if (!fit$converged) {
    warning(
      sprintf("`fine_gray()` didn't converge in %d iterations.", as.integer(max_iter)),
      call. = FALSE
    )
  }
  beta <- fit$beta
  names(beta) <- colnames(x)
  final <- fg_terms(beta, xc, setup, detail = TRUE)
  naive_var <- cox_solve(final$info, diag(length(beta)))
  scores <- fg_scores(beta, xc, setup, final)
  var <- naive_var %*% crossprod(scores) %*% naive_var
  dimnames(naive_var) <- list(colnames(x), colnames(x))
  dimnames(var) <- dimnames(naive_var)

  structure(
    list(
      coefficients = beta,
      var = var,
      naive_var = naive_var,
      loglik = c(fit$loglik_null, final$loglik),
      wald_test = cox_chisq(drop(beta %*% cox_solve(var, beta)), length(beta)),
      baseline = data.frame(time = setup$times, cumhaz = cumsum(setup$d / final$s0)),
      linear_predictors = linear_predictor(xc, beta),
      means = means,
      n = nrow(x),
      n_event = sum(setup$d),
      cause = cr$states[target],
      states = cr$states,
      iterations = fit$iterations,
      converged = fit$converged,
      conf_level = conf_level,
      terms = design$terms,
      assign = design$assign,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "fine_gray"
  )
}

# What the weighted risk sets need and that doesn't change with beta: the censoring
# Kaplan-Meier just before each subject's time and each event time of the cause, the event times
# and counts, and the subjects with a competing event.
fg_setup <- function(time, status, target) {
  g_before <- censoring_before(time, status == 0)
  is_event <- status == target
  times <- sort(unique(time[is_event]))
  list(
    time = time,
    is_event = is_event,
    event_index = match(time[is_event], times),
    competing = status != target & status != 0,
    g_subject = g_before(time),
    g_times = g_before(times),
    times = times,
    d = tabulate(match(time[is_event], times), length(times))
  )
}

# Column sums of `values` (a matrix with one row per subject) over the weighted risk set of each
# event time t: weight 1 for subjects with time >= t, and G(t-) / G(T-) for those with a competing
# event at T < t.
fg_risk_sums <- function(values, setup) {
  values <- as.matrix(values)
  cumulative <- function(rows, scale) {
    o <- rows[order(setup$time[rows])]
    sums <- rbind(0, apply(values[o, , drop = FALSE] / scale[o], 2L, cumsum))
    before <- findInterval(setup$times, setup$time[o], left.open = TRUE)
    sums[before + 1L, , drop = FALSE]
  }
  everyone <- seq_along(setup$time)
  above <- sweep(-cumulative(everyone, rep(1, length(everyone))), 2L, colSums(values), "+")
  competing <- which(setup$competing)
  if (length(competing) == 0L) {
    return(above)
  }
  above + setup$g_times * cumulative(competing, setup$g_subject)
}

# The log partial likelihood and, with `detail`, its gradient and information and the risk-set
# sums. `x` is centered.
fg_terms <- function(beta, x, setup, detail = FALSE) {
  lp <- drop(x %*% beta)
  r <- exp(lp)
  s0 <- drop(fg_risk_sums(r, setup))
  loglik <- sum(lp[setup$is_event]) - sum(setup$d * log(s0))
  if (!detail) {
    return(list(loglik = loglik))
  }
  p <- ncol(x)
  s1 <- fg_risk_sums(x * r, setup)
  s2 <- fg_risk_sums(x[, rep(seq_len(p), p), drop = FALSE] * x[, rep(seq_len(p), each = p)] * r,
    setup
  )
  xbar <- s1 / s0
  gradient <- colSums(x[setup$is_event, , drop = FALSE]) - colSums(setup$d * xbar)
  info <- matrix(colSums(setup$d * s2 / s0), p, p) - crossprod(xbar, setup$d * xbar)
  list(loglik = loglik, gradient = gradient, info = info, s0 = s0, xbar = xbar, r = r)
}

fg_newton <- function(x, setup, max_iter, tol) {
  beta <- numeric(ncol(x))
  current <- fg_terms(beta, x, setup, detail = TRUE)
  loglik_null <- current$loglik
  converged <- FALSE
  iterations <- 0L
  for (iter in seq_len(max_iter)) {
    iterations <- iter
    step <- cox_solve(current$info, current$gradient)
    candidate <- beta + step
    new <- fg_terms(candidate, x, setup)
    halvings <- 0L
    while (!is.finite(new$loglik) || new$loglik < current$loglik - 1e-12 * abs(current$loglik)) {
      halvings <- halvings + 1L
      if (halvings > 20L) {
        break
      }
      step <- step / 2
      candidate <- beta + step
      new <- fg_terms(candidate, x, setup)
    }
    change <- abs(new$loglik - current$loglik)
    beta <- candidate
    current <- fg_terms(beta, x, setup, detail = TRUE)
    if (change <= tol * (abs(current$loglik) + tol)) {
      converged <- TRUE
      break
    }
  }
  list(beta = beta, loglik_null = loglik_null, converged = converged, iterations = iterations)
}

# Each subject's score residuals: x - xbar at their own event of the cause, minus the weighted
# compensator over the risk sets they're in.
fg_scores <- function(beta, x, setup, terms) {
  dl <- setup$d / terms$s0
  xbar <- terms$xbar
  n_times <- length(setup$times)
  scores <- matrix(0, nrow(x), ncol(x))
  ev <- which(setup$is_event)
  scores[ev, ] <- x[ev, , drop = FALSE] - xbar[setup$event_index, , drop = FALSE]

  # Event times at or before each subject's time, where the subject has weight 1.
  k <- findInterval(setup$time, setup$times)
  at <- function(cumulative) {
    rbind(0, cumulative)[k + 1L, , drop = FALSE]
  }
  a <- cumsum(dl)
  b <- apply(xbar * dl, 2L, cumsum)
  b <- matrix(b, nrow = n_times)
  own <- x * c(0, a)[k + 1L] - at(b)

  # Later event times, where a subject with a competing event has weight G(t-) / G(T-).
  ga <- cumsum(setup$g_times * dl)
  gb <- matrix(apply(xbar * (setup$g_times * dl), 2L, cumsum), nrow = n_times)
  later_a <- ga[n_times] - c(0, ga)[k + 1L]
  later_b <- sweep(-at(gb), 2L, gb[n_times, ], "+")
  after <- (x * later_a - later_b) * (setup$competing / setup$g_subject)

  scores - terms$r * (own + after)
}

#' @export
coef.fine_gray <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.fine_gray <- function(object, ...) {
  object$var
}

#' @export
nobs.fine_gray <- function(object, ...) {
  object$n
}

#' @export
as.data.frame.fine_gray <- function(x,
                                    row.names = NULL,
                                    optional = FALSE,
                                    ...,
                                    exponentiate = FALSE,
                                    conf_level = x$conf_level) {
  as.data.frame.cox_ph(x, exponentiate = exponentiate, conf_level = conf_level)
}

#' @export
print.fine_gray <- function(x, digits = getOption("digits"), ...) {
  cat("<fine_gray> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "Subdistribution hazards for cause \"%s\". n = %d, events = %s\n\n",
    x$cause, x$n, format(x$n_event)
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(
    estimate = table$estimate,
    hazard_ratio = exp(table$estimate),
    robust_se = table$std_error,
    statistic = table$statistic,
    p_value = table$p_value
  )
  rownames(coefs) <- table$term
  print(coefs, digits = max(3L, digits - 3L))
  cat(sprintf(
    "\nRobust Wald test: %s on %d df, p = %s\n",
    format(x$wald_test$statistic, digits = max(3L, digits - 4L)),
    as.integer(x$wald_test$df),
    format_p(x$wald_test$p_value)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

#' Predictions from a Fine-Gray model
#'
#' `predict()` gives the linear predictor, the relative subdistribution hazard, or the cumulative
#' incidence of the modeled cause for new subjects (or for the subjects the model was fit to).
#' The linear predictor is centered at the covariate means of the fit, and the cumulative
#' incidence is \eqn{1 - \exp\{-H_0(t) \exp(\eta)\}}, with \eqn{H_0} the Breslow estimate of the
#' baseline cumulative subdistribution hazard.
#'
#' @param object A [fine_gray()] fit.
#' @param newdata A data frame of covariates. The default uses the data the model was fit to.
#' @param type What to give: `"lp"` (the default), `"risk"` (\eqn{\exp(\eta)}), or `"cif"`.
#' @param times For `type = "cif"`, the times to evaluate at. The default is the event times of
#'   the cause.
#' @param ... Unused.
#'
#' @return For `"lp"` and `"risk"`, a numeric vector with one value per subject. For `"cif"`, a
#'   data frame with one row per subject and time: `subject` (the row of `newdata`), `time`, and
#'   `estimate`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' fit <- fine_gray(Surv(time, event) ~ age, data = mgus, cause = "pcm")
#' predict(fit, newdata = data.frame(age = c(60, 75)), type = "risk")
#' predict(fit, newdata = data.frame(age = c(60, 75)), type = "cif", times = c(100, 200))
#' @name fine_gray_predict
#' @export
predict.fine_gray <- function(object, newdata, type = c("lp", "risk", "cif"), times, ...) {
  type <- match.arg(type)
  lp <- if (missing(newdata)) {
    object$linear_predictors
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    linear_predictor(
      sweep(cox_new_design(object, newdata), 2L, object$means), object$coefficients
    )
  }
  if (type == "lp") {
    return(lp)
  }
  if (type == "risk") {
    return(exp(lp))
  }
  base <- object$baseline
  times <- if (missing(times)) base$time else check_times(times)
  cumhaz <- step_at(base$time, base$cumhaz, times, 0)
  data.frame(
    subject = rep(seq_along(lp), each = length(times)),
    time = rep(times, times = length(lp)),
    estimate = 1 - exp(-rep(cumhaz, times = length(lp)) * rep(exp(lp), each = length(times)))
  )
}
