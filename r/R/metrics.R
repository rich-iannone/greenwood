# Prediction-performance metrics: concordance, Brier scores, time-dependent AUC, and calibration,
# for survival and for competing-risks predictions. The numeric fixtures (brier_lung, td_auc_lung,
# concordance_ipcw_lung, concordance_ci_ij, and calibration_incidence_mgus2) are replayed by
# tests/testthat/test-metrics.R.

# -- shared pieces -----------------------------------------------------------------------------

# The times and status codes of a Surv() response passed to a metric. Survival metrics take
# right-censored data (status 0 or 1). Incidence metrics take competing-risks data (status 0 for
# censored and 1 to k for the causes, with the cause names in `states`).
metric_response <- function(response, fn, competing = FALSE) {
  if (!inherits(response, "Surv")) {
    stop(sprintf("`response` of `%s` must be a `Surv()` object.", fn), call. = FALSE)
  }
  type <- attr(response, "type")
  wanted <- if (competing) "mright" else "right"
  if (!identical(type, wanted)) {
    what <- if (competing) {
      "competing-risks data, `Surv(time, event)` with `event` a factor whose first level is censoring"
    } else {
      "right-censored data"
    }
    stop(sprintf("`%s` needs %s, not \"%s\".", fn, what, type), call. = FALSE)
  }
  y <- unclass(response)
  if (anyNA(y)) {
    stop(sprintf("`response` of `%s` has missing values.", fn), call. = FALSE)
  }
  list(time = unname(y[, "time"]), status = unname(y[, "status"]), states = attr(response, "states"))
}

# A vector of scores, one per subject.
metric_scores <- function(x, n, name) {
  if (!is.numeric(x) || length(x) != n || anyNA(x)) {
    stop(sprintf("`%s` must be %d numbers, one per subject, with no missing values.", name, n),
      call. = FALSE
    )
  }
  as.numeric(x)
}

# A subjects-by-times matrix of predicted probabilities.
metric_probs <- function(x, n, times, name) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (is.null(dim(x)) && length(times) == 1L) {
    x <- matrix(x, ncol = 1L)
  }
  if (!is.numeric(x) || !identical(dim(x), c(n, length(times))) || anyNA(x)) {
    stop(
      sprintf(
        "`%s` must be a matrix with one row per subject and one column per time (%d by %d).",
        name, n, length(times)
      ),
      call. = FALSE
    )
  }
  unname(x)
}

# The Kaplan-Meier estimate of the censoring distribution, G, as a function giving G(t-), the
# value just before t. Events tied with a censoring are taken to come first, so they're out of its
# risk set.
censoring_before <- function(time, censored) {
  steps <- censoring_steps(time, censored)
  if (length(steps$time) == 0L) {
    return(function(t) rep(1, length(t)))
  }
  function(t) {
    i <- findInterval(t, steps$time, left.open = TRUE)
    ifelse(i > 0, steps$g[pmax(i, 1L)], 1)
  }
}

# The censoring times and the value of G just after each, with events tied with a censoring
# taken to come first.
censoring_steps <- function(time, censored) {
  censor_times <- sort(unique(time[censored]))
  censored_at <- tabulate_sum(
    match(time[censored], censor_times), rep(1, sum(censored)), length(censor_times)
  )
  later <- length(time) - findInterval(censor_times, sort(time))
  list(time = censor_times, g = cumprod(1 - censored_at / (later + censored_at)))
}

# The number of later scores below each case's score, counting ties as one half.
concordant_count <- function(case_score, later_scores) {
  sum(later_scores < case_score) + 0.5 * sum(later_scores == case_score)
}

# The trapezoidal average of `values` over `times`.
time_average <- function(times, values) {
  sum(diff(times) * (values[-1L] + values[-length(values)]) / 2) / (times[length(times)] - times[1L])
}

check_metric_times <- function(times, at_least = 1L) {
  times <- check_times(times)
  if (length(times) < at_least) {
    stop(sprintf("`times` must have at least %d values.", at_least), call. = FALSE)
  }
  times
}

# -- concordance -------------------------------------------------------------------------------

#' Concordance of risk scores with survival
#'
#' `concordance_index()` gives Harrell's C, the proportion of comparable pairs of subjects in which
#' the subject with the higher risk score has the event first. A pair is comparable when one
#' subject has the event while the other is still under observation (a later time, or censored at
#' the same time). Ties in the score count as one half, and pairs with the same event time are
#' left out. A C of 0.5 is no better than chance and 1 is perfect ranking.
#'
#' `concordance_index_ipcw()` gives Uno's C, which weights each case by
#' \eqn{1 / G(T_i-)^2}, where \eqn{G} is the Kaplan-Meier estimate of the censoring distribution.
#' Under independent censoring it is consistent for the concordance probability, which Harrell's C
#' isn't when censoring is heavy. Only cases with an event at or before `tau` count, and they're
#' compared with subjects whose time is strictly later.
#'
#' @param response A right-censored [Surv()] object.
#' @param risk A risk score for each subject, such as a Cox model's linear predictor. Higher
#'   scores mean earlier events.
#' @param tau The truncation time for `concordance_index_ipcw()`. The default is the last event
#'   time.
#'
#' @return A single number.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age, data = lung)
#' death <- Surv(lung$time, lung$status)
#'
#' concordance_index(death, risk = predict(fit))
#' concordance_index_ipcw(death, risk = predict(fit), tau = 700)
#' @export
concordance_index <- function(response, risk) {
  y <- metric_response(response, "concordance_index()")
  risk <- metric_scores(risk, length(y$time), "risk")
  event <- y$status == 1
  concordant <- 0
  comparable <- 0
  for (i in which(event)) {
    later <- y$time > y$time[i] | (y$time == y$time[i] & !event)
    comparable <- comparable + sum(later)
    concordant <- concordant + concordant_count(risk[i], risk[later])
  }
  if (comparable == 0) {
    stop("There are no comparable pairs.", call. = FALSE)
  }
  concordant / comparable
}

#' @rdname concordance_index
#' @export
concordance_index_ipcw <- function(response, risk, tau = NULL) {
  input <- concordance_input(response, NULL, tau, "concordance_index_ipcw()")
  weighted_concordance(input, metric_scores(risk, input$n, "risk"))$estimate
}

# The cases and comparison sets of Uno's weighted concordance. Without `cause` the response is
# right-censored and the cases are the events at or before `tau`. With `cause` it is
# competing-risks data, the cases are the events of the cause, and the events of the other causes
# are `competing`.
concordance_input <- function(response, cause, tau, fn) {
  y <- metric_response(response, fn, competing = !is.null(cause))
  if (is.null(cause)) {
    is_cause <- y$status == 1
    if (!any(is_cause)) {
      stop("There are no events.", call. = FALSE)
    }
  } else {
    cause <- cr_cause(cause, y$states)
    is_cause <- y$status == cause
    if (!any(is_cause)) {
      stop(sprintf("There are no events of cause \"%s\".", y$states[cause]), call. = FALSE)
    }
  }
  if (is.null(tau)) {
    tau <- max(y$time[is_cause])
  }
  check_number(tau, "tau")
  cases <- which(is_cause & y$time <= tau)
  if (length(cases) == 0L) {
    what <- if (is.null(cause)) "events" else "events of the cause"
    stop(sprintf("There are no %s at or before `tau` = %s.", what, format(tau)), call. = FALSE)
  }
  list(
    n = length(y$time),
    time = y$time,
    cases = cases,
    competing = y$status > 0 & !is_cause,
    censored = y$status == 0
  )
}

# Uno's weighted concordance of `score` over the cases of `input` (from concordance_input()). Each
# case i is compared with every subject whose time is later, with weight 1 / G(T_i-)^2, and with
# every competing event at or before its own time, with weight 1 / (G(T_i-) G(T_j-)). G is the
# Kaplan-Meier curve of the censored rows, and cases with G(T_i-) = 0 are left out.
#
# With `influence`, it also gives each subject's infinitesimal-jackknife influence `phi`, the
# derivative of the estimate in the subject's case weight with the censoring weights held fixed,
# so that the variance is sum(phi^2). Upweighting either member of a pair scales that pair, so
# both members get its contribution.
weighted_concordance <- function(input, score, influence = FALSE) {
  time <- input$time
  competing <- input$competing
  g_all <- censoring_before(time, input$censored)(time)
  cases <- input$cases[g_all[input$cases] > 0]
  pairs <- function(i) {
    later <- which(time > time[i])
    earlier <- which(competing & time <= time[i] & g_all > 0)
    j <- c(later, earlier)
    list(
      j = j,
      w = c(rep(1 / g_all[i]^2, length(later)), 1 / (g_all[i] * g_all[earlier])),
      pair = (score[i] > score[j]) + 0.5 * (score[i] == score[j])
    )
  }
  numerator <- 0
  denominator <- 0
  for (i in cases) {
    p <- pairs(i)
    numerator <- numerator + sum(p$w * p$pair)
    denominator <- denominator + sum(p$w)
  }
  if (denominator == 0) {
    stop("There are no comparable pairs.", call. = FALSE)
  }
  estimate <- numerator / denominator
  if (!influence) {
    return(list(estimate = estimate))
  }
  phi <- numeric(length(time))
  for (i in cases) {
    p <- pairs(i)
    d <- p$w * (p$pair - estimate) / denominator
    phi[i] <- phi[i] + sum(d)
    phi[p$j] <- phi[p$j] + d
  }
  list(estimate = estimate, phi = phi)
}

#' Concordance with a confidence interval, and comparing two risk scores
#'
#' `concordance_index_ci()` gives Uno's weighted concordance (see [concordance_index_ipcw()]), or
#' with `cause` the competing-risks concordance of [concordance_index_incidence()], with a standard
#' error from the infinitesimal jackknife. Each subject's influence is the derivative of the
#' estimate in that subject's case weight, with the censoring weights held fixed, and the variance
#' is the sum of the squared influences. The confidence interval is built on the logit scale (the
#' default), the log scale, or the original scale, then transformed back and kept within 0 and 1.
#'
#' `concordance_index_compare()` tests whether two risk scores for the same subjects have the same
#' concordance. The standard error of the difference comes from the difference of the two scores'
#' influences, which accounts for the correlation between the estimates. The test is a Wald test,
#' and the confidence interval is on the original scale.
#'
#' @param response A right-censored [Surv()] object or, with `cause`, a competing-risks one,
#'   `Surv(time, event)` with `event` a factor whose first level means censored.
#' @param risk A risk score for each subject, such as a Cox model's linear predictor or, with
#'   `cause`, a predicted cumulative incidence. Higher scores mean earlier events.
#' @param risk_a,risk_b The two risk scores compared by `concordance_index_compare()`.
#' @param cause For competing risks, the cause, as a level of `event` or its number (1 for the
#'   first cause). The default, `NULL`, is the concordance for right-censored data.
#' @param tau The truncation time. The default is the last event time (of the cause).
#' @param conf_level The confidence level, between 0 and 1. The default is `0.95`.
#' @param transform The scale of the confidence interval: `"logit"` (the default), `"log"`, or
#'   `"identity"`.
#'
#' @return `concordance_index_ci()` gives a one-row data frame with `estimate`, `std_error`,
#'   `conf_low`, `conf_high`, and `n` (the number of subjects). `concordance_index_compare()` gives
#'   a one-row data frame with `difference` (the concordance of `risk_a` minus that of `risk_b`),
#'   `std_error`, `conf_low`, `conf_high`, `statistic`, and the two-sided `p_value`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)
#' )
#' one <- cox_ph(Surv(time, status) ~ age, data = lung)
#' two <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' death <- Surv(lung$time, lung$status)
#'
#' concordance_index_ci(death, risk = predict(two))
#' concordance_index_compare(death, risk_a = predict(two), risk_b = predict(one))
#' @export
concordance_index_ci <- function(response,
                                 risk,
                                 cause = NULL,
                                 tau = NULL,
                                 conf_level = 0.95,
                                 transform = c("logit", "log", "identity")) {
  transform <- match.arg(transform)
  check_conf_level(conf_level)
  input <- concordance_input(response, cause, tau, "concordance_index_ci()")
  fit <- weighted_concordance(input, metric_scores(risk, input$n, "risk"), influence = TRUE)
  estimate <- fit$estimate
  std_error <- sqrt(sum(fit$phi^2))
  z <- z_value(conf_level)
  limits <- switch(transform,
    logit = stats::plogis(stats::qlogis(estimate) + c(-1, 1) * z * std_error /
      (estimate * (1 - estimate))),
    log = estimate * exp(c(-1, 1) * z * std_error / estimate),
    identity = estimate + c(-1, 1) * z * std_error
  )
  limits <- pmin(pmax(limits, 0), 1)
  data.frame(
    estimate = estimate,
    std_error = std_error,
    conf_low = limits[1L],
    conf_high = limits[2L],
    n = input$n
  )
}

#' @rdname concordance_index_ci
#' @export
concordance_index_compare <- function(response,
                                      risk_a,
                                      risk_b,
                                      cause = NULL,
                                      tau = NULL,
                                      conf_level = 0.95) {
  check_conf_level(conf_level)
  input <- concordance_input(response, cause, tau, "concordance_index_compare()")
  a <- weighted_concordance(input, metric_scores(risk_a, input$n, "risk_a"), influence = TRUE)
  b <- weighted_concordance(input, metric_scores(risk_b, input$n, "risk_b"), influence = TRUE)
  difference <- a$estimate - b$estimate
  std_error <- sqrt(sum((a$phi - b$phi)^2))
  statistic <- if (std_error > 0) difference / std_error else 0
  z <- z_value(conf_level)
  data.frame(
    difference = difference,
    std_error = std_error,
    conf_low = difference - z * std_error,
    conf_high = difference + z * std_error,
    statistic = statistic,
    p_value = 2 * stats::pnorm(-abs(statistic))
  )
}

# -- Brier score -------------------------------------------------------------------------------

#' Brier score of predicted survival probabilities
#'
#' `brier_score()` gives the inverse-probability-of-censoring weighted (Graf) Brier score at each
#' time, the mean squared difference between the predicted survival probability and the observed
#' status. At time \eqn{t}, a subject with an event by \eqn{t} contributes
#' \eqn{\hat S_i(t)^2 / G(T_i-)} and one still under observation after \eqn{t} contributes
#' \eqn{(1 - \hat S_i(t))^2 / G(t-)}, where \eqn{G} is the Kaplan-Meier estimate of the censoring
#' distribution. Subjects censored by \eqn{t} contribute 0, and the sum is divided by the number
#' of subjects. Lower is better. Predicting 0.5 for everyone gives 0.25.
#'
#' `integrated_brier_score()` averages the scores over the times with the trapezoidal rule.
#'
#' @param response A right-censored [Surv()] object.
#' @param survival_prob The predicted survival probabilities, a matrix (or data frame) with one
#'   row per subject and one column per time. With one time, a vector is accepted.
#' @param times The times, in the order of the columns of `survival_prob`.
#'   `integrated_brier_score()` needs at least two, in increasing order.
#'
#' @return `brier_score()` gives a vector with one score per time. `integrated_brier_score()` gives
#'   a single number.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age, data = lung)
#' times <- c(200, 300, 400)
#'
#' # One row per subject and one column per time
#' p <- predict(fit, newdata = lung, type = "survival", times = times)
#' probs <- matrix(p$estimate, ncol = length(times), byrow = TRUE)
#'
#' death <- Surv(lung$time, lung$status)
#' brier_score(death, survival_prob = probs, times = times)
#' integrated_brier_score(death, survival_prob = probs, times = times)
#' @export
brier_score <- function(response, survival_prob, times) {
  y <- metric_response(response, "brier_score()")
  times <- check_metric_times(times)
  probs <- metric_probs(survival_prob, length(y$time), times, "survival_prob")
  event <- y$status == 1
  g_before <- censoring_before(y$time, !event)
  g_exit <- g_before(y$time)
  vapply(seq_along(times), function(j) {
    t <- times[j]
    s <- probs[, j]
    dead <- y$time <= t & event
    alive <- y$time > t
    contribution <- ifelse(dead, s^2 / g_exit, ifelse(alive, (1 - s)^2 / g_before(t), 0))
    mean(contribution)
  }, numeric(1))
}

#' @rdname brier_score
#' @export
integrated_brier_score <- function(response, survival_prob, times) {
  times <- check_metric_times(times, at_least = 2L)
  time_average(times, brier_score(response, survival_prob, times))
}

# -- time-dependent AUC ------------------------------------------------------------------------

#' Time-dependent AUC of a risk marker
#'
#' `time_dependent_auc()` gives the cumulative-dynamic AUC of Uno and others (2011) at each time.
#' At time \eqn{t} the cases are the subjects with an event by \eqn{t} and the controls are those
#' still under observation after \eqn{t}. The AUC is the proportion of case-control pairs in which
#' the case has the higher marker, with ties counted as one half and each case weighted by
#' \eqn{1 / G(T_i-)^2}, where \eqn{G} is the Kaplan-Meier estimate of the censoring distribution.
#' It is `NA` at a time with no cases or no controls.
#'
#' `integrated_auc()` averages the AUC over the times with the trapezoidal rule, after dropping
#' the times where it is `NA`.
#'
#' @param response A right-censored [Surv()] object.
#' @param marker A risk score for each subject. Higher values should mean earlier events, so negate
#'   a predicted survival probability before passing it.
#' @param times The times. `integrated_auc()` needs at least two, in increasing order.
#'
#' @return `time_dependent_auc()` gives a vector with one AUC per time. `integrated_auc()` gives a
#'   single number.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age, data = lung)
#' death <- Surv(lung$time, lung$status)
#'
#' time_dependent_auc(death, marker = predict(fit), times = c(200, 300, 400))
#' integrated_auc(death, marker = predict(fit), times = c(200, 300, 400))
#' @export
time_dependent_auc <- function(response, marker, times) {
  y <- metric_response(response, "time_dependent_auc()")
  marker <- metric_scores(marker, length(y$time), "marker")
  times <- check_metric_times(times)
  event <- y$status == 1
  g_before <- censoring_before(y$time, !event)
  vapply(times, function(t) {
    cases <- which(y$time <= t & event)
    controls <- marker[y$time > t]
    if (length(cases) == 0L || length(controls) == 0L) {
      return(NA_real_)
    }
    g <- g_before(y$time[cases])
    w <- ifelse(g > 0, 1 / g^2, 0)
    concordant <- vapply(marker[cases], concordant_count, numeric(1), later_scores = controls)
    denominator <- sum(w) * length(controls)
    if (denominator > 0) sum(w * concordant) / denominator else NA_real_
  }, numeric(1))
}

#' @rdname time_dependent_auc
#' @export
integrated_auc <- function(response, marker, times) {
  times <- check_metric_times(times, at_least = 2L)
  auc <- time_dependent_auc(response, marker, times)
  keep <- !is.na(auc)
  if (sum(keep) < 2L) {
    stop("There are fewer than two times with an AUC.", call. = FALSE)
  }
  time_average(times[keep], auc[keep])
}

# -- calibration -------------------------------------------------------------------------------

#' Calibration of predicted survival probabilities
#'
#' `calibration()` checks predicted survival probabilities at one time against what was observed.
#' The subjects are split into `n_bins` groups at the quantiles of their predictions, and in each
#' group the mean prediction is set beside the Kaplan-Meier estimate at `time`, with its
#' confidence interval on the log scale. In a well-calibrated model the two are close. Bins with
#' no subjects are dropped, so tied predictions can give fewer rows.
#'
#' @param response A right-censored or counting-process [Surv()] object.
#' @param predicted The predicted survival probability at `time` for each subject.
#' @param time The time at which the predictions are checked.
#' @param n_bins The number of groups, at least 2. The default is 10.
#' @param conf_level The confidence level of the observed interval. The default is `0.95`.
#'
#' @return A data frame with one row per bin: `bin`, `n`, `predicted` (the mean prediction),
#'   `observed`, `observed_lower`, and `observed_upper`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age, data = lung)
#' p <- predict(fit, newdata = lung, type = "survival", times = 300)
#'
#' calibration(Surv(lung$time, lung$status), predicted = p$estimate, time = 300, n_bins = 3)
#' @export
calibration <- function(response, predicted, time, n_bins = 10, conf_level = 0.95) {
  if (!inherits(response, "Surv")) {
    stop("`response` of `calibration()` must be a `Surv()` object.", call. = FALSE)
  }
  parts <- surv_parts(response, "calibration()")
  if (anyNA(unclass(response))) {
    stop("`response` of `calibration()` has missing values.", call. = FALSE)
  }
  n <- length(parts$stop)
  predicted <- metric_scores(predicted, n, "predicted")
  check_number(time, "time")
  check_number(n_bins, "n_bins", min = 2)
  check_conf_level(conf_level)
  z <- z_value(conf_level)

  edges <- stats::quantile(predicted, probs = seq(0, 1, length.out = n_bins + 1L), names = FALSE)
  edges[c(1L, n_bins + 1L)] <- c(-Inf, Inf)
  bin <- pmin(pmax(findInterval(predicted, edges), 1L), n_bins)
  rows <- lapply(sort(unique(bin)), function(b) {
    members <- bin == b
    curve <- km_curve(
      parts$start[members], parts$stop[members], parts$event[members], rep(1, sum(members)),
      "log", z
    )
    data.frame(
      n = sum(members),
      predicted = mean(predicted[members]),
      observed = step_at(curve$time, curve$estimate, time, 1),
      observed_lower = step_at(curve$time, curve$conf_low, time, 1),
      observed_upper = step_at(curve$time, curve$conf_high, time, 1)
    )
  })
  out <- cbind(bin = seq_along(rows), do.call(rbind, rows))
  rownames(out) <- NULL
  out
}

# -- competing risks ---------------------------------------------------------------------------

#' Prediction metrics for competing risks
#'
#' These metrics judge predicted cumulative incidences of one cause, with the other causes as
#' competing events. Each uses the Kaplan-Meier estimate \eqn{G} of the censoring distribution,
#' with an event of any cause ending follow-up.
#'
#' `brier_score_incidence()` gives the weighted Brier score at each time, the mean of
#' \eqn{w_i (y_i - \hat F_i(t))^2} where \eqn{y_i} is 1 for an event of the cause by \eqn{t}. The
#' weight is \eqn{1 / G(T_i-)} for an event of any cause by \eqn{t}, \eqn{1 / G(t-)} for a subject
#' still under observation after \eqn{t}, and 0 for one censored by \eqn{t}.
#' `integrated_brier_score_incidence()` averages it over the times with the trapezoidal rule.
#'
#' `concordance_index_incidence()` gives a weighted concordance for one cause. A case with an event
#' of the cause at \eqn{T_i \le \tau} is compared with every subject whose time is later (with
#' weight \eqn{1 / G(T_i-)^2}) and with every subject who had a competing event at or before
#' \eqn{T_i} (with weight \eqn{1 / (G(T_i-) G(T_j-))}). Higher predicted incidence should go with
#' the earlier event, and ties count as one half.
#'
#' `calibration_incidence()` gives, at each time, the absolute difference between the mean
#' predicted incidence and the Aalen-Johansen estimate for the whole sample.
#'
#' `accuracy_in_time()` takes the predicted incidence of every cause. At each time it predicts
#' each subject's outcome as the most likely of being event-free (one minus the sum of the
#' incidences) and each cause, and gives the proportion predicted correctly among the subjects not
#' censored by that time.
#'
#' @param response A competing-risks [Surv()] object, `Surv(time, event)` with `event` a factor
#'   whose first level means censored and whose other levels are the causes.
#' @param incidence_prob The predicted cumulative incidence of `cause`. For
#'   `concordance_index_incidence()`, one value per subject (such as the incidence at a fixed
#'   time). Otherwise a matrix with one row per subject and one column per time.
#' @param incidence_probs For `accuracy_in_time()`, an array of predicted cumulative incidences
#'   with dimensions subjects, causes (in the order of the levels), and times.
#' @param times The times, in the order of the columns of the predictions. The integrated score
#'   needs at least two, in increasing order.
#' @param cause The cause, as a level of `event` or its number (1 for the first cause).
#' @param tau The truncation time for `concordance_index_incidence()`. The default is the last
#'   event time of the cause.
#'
#' @return A vector with one value per time, or a single number from
#'   `integrated_brier_score_incidence()` and `concordance_index_incidence()`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' outcome <- Surv(mgus$time, mgus$event)
#'
#' # The same predicted incidence for everyone: the Aalen-Johansen estimate
#' times <- c(50, 100, 150)
#' aj <- predict(aalen_johansen(Surv(time, event) ~ 1, data = mgus), times = times)
#' pcm <- aj$estimate[aj$cause == "pcm"]
#' probs <- matrix(pcm, nrow = nrow(mgus), ncol = length(times), byrow = TRUE)
#'
#' brier_score_incidence(outcome, incidence_prob = probs, times = times, cause = "pcm")
#' calibration_incidence(outcome, incidence_prob = probs, times = times, cause = "pcm")
#'
#' # A score that rises with age
#' concordance_index_incidence(outcome, incidence_prob = mgus$age / 100, cause = "death")
#' @name incidence_metrics
NULL

#' @rdname incidence_metrics
#' @export
brier_score_incidence <- function(response, incidence_prob, times, cause) {
  y <- metric_response(response, "brier_score_incidence()", competing = TRUE)
  cause <- cr_cause(cause, y$states)
  times <- check_metric_times(times)
  probs <- metric_probs(incidence_prob, length(y$time), times, "incidence_prob")
  any_event <- y$status > 0
  g_before <- censoring_before(y$time, !any_event)
  g_exit <- g_before(y$time)
  vapply(seq_along(times), function(j) {
    t <- times[j]
    g_t <- g_before(t)
    observed <- as.numeric(y$status == cause & y$time <= t)
    w <- ifelse(
      any_event & y$time <= t,
      ifelse(g_exit > 0, 1 / g_exit, 0),
      ifelse(y$time > t & g_t > 0, 1 / g_t, 0)
    )
    mean(w * (observed - probs[, j])^2)
  }, numeric(1))
}

#' @rdname incidence_metrics
#' @export
integrated_brier_score_incidence <- function(response, incidence_prob, times, cause) {
  times <- check_metric_times(times, at_least = 2L)
  time_average(times, brier_score_incidence(response, incidence_prob, times, cause))
}

#' @rdname incidence_metrics
#' @export
concordance_index_incidence <- function(response, incidence_prob, cause, tau = NULL) {
  if (missing(cause) || is.null(cause)) {
    stop("`concordance_index_incidence()` needs a `cause`.", call. = FALSE)
  }
  input <- concordance_input(response, cause, tau, "concordance_index_incidence()")
  weighted_concordance(input, metric_scores(incidence_prob, input$n, "incidence_prob"))$estimate
}

#' @rdname incidence_metrics
#' @export
calibration_incidence <- function(response, incidence_prob, times, cause) {
  y <- metric_response(response, "calibration_incidence()", competing = TRUE)
  cause <- cr_cause(cause, y$states)
  times <- check_metric_times(times)
  probs <- metric_probs(incidence_prob, length(y$time), times, "incidence_prob")
  curve <- aj_curves(y$time, y$status, length(y$states), z_value(0.95), "plain")[[cause]]
  abs(colMeans(probs) - step_at(curve$time, curve$estimate, times, 0))
}

#' @rdname incidence_metrics
#' @export
accuracy_in_time <- function(response, incidence_probs, times) {
  y <- metric_response(response, "accuracy_in_time()", competing = TRUE)
  times <- check_metric_times(times)
  n <- length(y$time)
  k <- length(y$states)
  if (!is.numeric(incidence_probs) || !identical(dim(incidence_probs), c(n, k, length(times))) ||
    anyNA(incidence_probs)) {
    stop(
      sprintf(
        "`incidence_probs` must be an array with dimensions %d (subjects), %d (causes), and %d (times).",
        n, k, length(times)
      ),
      call. = FALSE
    )
  }
  vapply(seq_along(times), function(j) {
    t <- times[j]
    keep <- !(y$status == 0 & y$time <= t)
    if (!any(keep)) {
      return(NA_real_)
    }
    cif <- matrix(incidence_probs[keep, , j], ncol = k)
    predicted <- max.col(cbind(1 - rowSums(cif), cif), ties.method = "first") - 1L
    observed <- ifelse(y$time[keep] <= t, y$status[keep], 0)
    mean(predicted == observed)
  }, numeric(1))
}

#' Score predicted cumulative incidences of one cause
#'
#' `score_cr()` gives the main measures of a competing-risks prediction in one table: the
#' concordance at `tau`, with its confidence interval (see [concordance_index_ci()]), the Brier
#' score at each time, and the integrated Brier score (see [brier_score_incidence()]). The
#' concordance uses the predictions at the last of `times` that is at or before `tau` (or the first
#' time, if all are later).
#'
#' @inheritParams incidence_metrics
#' @param times The times, in increasing order, of the columns of `incidence_prob`. At least two.
#' @param tau The truncation time of the concordance. The default is the last of `times`.
#' @param conf_level The confidence level of the concordance's interval. The default is `0.95`.
#'
#' @return A data frame with one row per measure: `metric` (`"concordance"`, then
#'   `"brier(t)"` for each time `t`, then `"integrated_brier"`), `estimate`, and, for the
#'   concordance only, `std_error`, `conf_low`, and `conf_high`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' fit <- fine_gray(Surv(time, event) ~ age, data = mgus, cause = "pcm")
#' times <- c(50, 100, 150)
#' p <- predict(fit, newdata = mgus, type = "cif", times = times)
#' probs <- matrix(p$estimate, ncol = length(times), byrow = TRUE)
#'
#' score_cr(Surv(mgus$time, mgus$event), incidence_prob = probs, cause = "pcm", times = times)
#' @export
score_cr <- function(response, incidence_prob, cause, times, tau = NULL, conf_level = 0.95) {
  times <- check_metric_times(times, at_least = 2L)
  if (is.unsorted(times, strictly = TRUE)) {
    stop("`times` must be in increasing order.", call. = FALSE)
  }
  y <- metric_response(response, "score_cr()", competing = TRUE)
  probs <- metric_probs(incidence_prob, length(y$time), times, "incidence_prob")
  if (is.null(tau)) {
    tau <- times[length(times)]
  }
  check_number(tau, "tau")
  at_tau <- max(findInterval(tau, times), 1L)
  concordance <- concordance_index_ci(response, probs[, at_tau],
    cause = cause, tau = tau, conf_level = conf_level
  )
  brier <- brier_score_incidence(response, probs, times, cause = cause)
  data.frame(
    metric = c("concordance", sprintf("brier(%s)", format(times, trim = TRUE)), "integrated_brier"),
    estimate = c(concordance$estimate, brier, time_average(times, brier)),
    std_error = c(concordance$std_error, rep(NA_real_, length(times) + 1L)),
    conf_low = c(concordance$conf_low, rep(NA_real_, length(times) + 1L)),
    conf_high = c(concordance$conf_high, rep(NA_real_, length(times) + 1L))
  )
}
