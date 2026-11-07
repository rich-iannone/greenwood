# No R fixture covers the piecewise exponential model, so the fits are checked against glm() on
# the split data, against the exponential aft(), and against Greenwood's Python values.

lung_events <- function() {
  lung <- survival::lung
  lung$dead <- lung$status - 1
  lung
}

test_that("the fits match Python", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = c(180, 365))
  expect_equal(unname(coef(fit)), c(0.01579012916434935, -0.5077425600240448), tolerance = 1e-10)
  expect_equal(unname(sqrt(diag(vcov(fit)))), c(0.0091750050447928, 0.16717203241253098),
    tolerance = 1e-10
  )
  expect_equal(
    baseline_hazard(fit)$log_hazard, c(-6.64607690124296, -6.112316993760453, -6.035743551645541),
    tolerance = 1e-10
  )
  expect_equal(fit$loglik, -1149.5972960399226, tolerance = 1e-10)
  expect_equal(AIC(fit), 2309.1945920798453, tolerance = 1e-10)
  expect_equal(BIC(fit), 2326.3413202246174, tolerance = 1e-10)
  expect_equal(fit$lr_test$statistic, 13.459809700590085, tolerance = 1e-10)
  expect_equal(fit$loglik_null, -1156.3272008902177, tolerance = 1e-10)
  survival <- predict(fit, newdata = lung[1:2, ], times = c(30, 180, 365, 1000))
  expect_equal(survival$estimate, c(
    0.9273168194559066, 0.6358707119082115, 0.28756290667203854, 0.015193336198346829,
    0.9336636015794155, 0.6624339319383035, 0.3218538483539507, 0.022182775749351444
  ), tolerance = 1e-10)

  chosen <- piecewise_exponential(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
  expect_identical(chosen$breaks, c(153.75, 267.5, 432))
  expect_equal(unname(coef(chosen)), c(
    0.01002401698507643, -0.558073546115009, 0.46700930460369683
  ), tolerance = 1e-10)
  expect_equal(chosen$loglik, -1130.6995301960496, tolerance = 1e-10)
  expect_identical(c(chosen$n, chosen$n_dropped), c(227L, 1L))

  bic <- piecewise_exponential(Surv(time, status) ~ age + sex,
    data = lung, knot_strategy = "bic", max_knots = 5
  )
  expect_identical(bic$breaks, 267)
  expect_equal(unname(coef(bic)), c(0.01605986170357297, -0.5047298284101508), tolerance = 1e-10)
})

test_that("the fit is the Poisson glm() of the split data", {
  skip_if_not_installed("survival")
  lung <- lung_events()
  lung <- lung[!is.na(lung$ph.ecog), ]
  breaks <- c(100, 250, 500)
  fit <- piecewise_exponential(Surv(time, dead) ~ age + sex + ph.ecog, data = lung, breaks = breaks)
  split <- survival::survSplit(Surv(time, dead) ~ age + sex + ph.ecog,
    data = lung, cut = breaks, episode = "interval"
  )
  split$exposure <- split$time - split$tstart
  reference <- stats::glm(dead ~ 0 + factor(interval) + age + sex + ph.ecog + offset(log(exposure)),
    family = stats::poisson(), data = split, control = stats::glm.control(epsilon = 1e-14)
  )
  expect_equal(unname(coef(fit)), unname(coef(reference)[5:7]), tolerance = 1e-9)
  expect_equal(unname(fit$log_hazard), unname(coef(reference)[1:4]), tolerance = 1e-9)
  expect_equal(unname(fit$full_var), unname(vcov(reference)), tolerance = 1e-7)
  # The Poisson log-likelihood also has the events' log exposures (and log(y!), which is 0).
  events <- split$dead == 1
  expect_equal(
    fit$loglik + sum(log(split$exposure[events])), as.numeric(logLik(reference)),
    tolerance = 1e-10
  )
  expect_identical(attr(logLik(fit), "df"), 7L)
  expect_identical(nobs(fit), 227L)
  expect_identical(fit$n_dropped, 0L)
  table <- as.data.frame(fit)
  expect_named(table, c(
    "term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"
  ))
  expect_identical(table$term, c("age", "sex", "ph.ecog"))
  expect_equal(table$std_error, unname(sqrt(diag(vcov(reference)))[5:7]), tolerance = 1e-7)
  expect_equal(table$conf_high, table$estimate + stats::qnorm(0.975) * table$std_error)

  # The likelihood ratio test compares with the model of the intervals alone.
  null <- stats::glm(dead ~ 0 + factor(interval) + offset(log(exposure)),
    family = stats::poisson(), data = split
  )
  expect_equal(fit$lr_test$statistic, 2 * as.numeric(logLik(reference) - logLik(null)),
    tolerance = 1e-8
  )
  expect_identical(fit$lr_test$df, 3L)
  expect_equal(fit$lr_test$p_value, stats::pchisq(fit$lr_test$statistic, 3, lower.tail = FALSE))
})

test_that("with no breaks, it's the exponential model", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = numeric(0))
  exponential <- aft(Surv(time, status) ~ age + sex, data = lung, dist = "exponential")
  expect_equal(fit$loglik, as.numeric(logLik(exponential)), tolerance = 1e-8)
  expect_equal(unname(coef(fit)), -unname(coef(exponential)[-1L]), tolerance = 1e-6)
  expect_equal(unname(fit$log_hazard), -unname(coef(exponential)[1L]), tolerance = 1e-6)
  expect_identical(length(fit$log_hazard), 1L)
  expect_output(print(fit), "1 interval. n = 228", fixed = TRUE)
})

test_that("splitting the follow-up into counting-process rows doesn't change the fit", {
  skip_if_not_installed("survival")
  lung <- lung_events()
  right <- piecewise_exponential(Surv(time, dead) ~ age + sex, data = lung, breaks = c(180, 365))
  split <- survival::survSplit(Surv(time, dead) ~ age + sex, data = lung, cut = c(90, 300, 600))
  counting <- piecewise_exponential(Surv(tstart, time, dead) ~ age + sex,
    data = split, breaks = c(180, 365)
  )
  expect_equal(coef(counting), coef(right), tolerance = 1e-10)
  expect_equal(counting$loglik, right$loglik, tolerance = 1e-10)
  expect_equal(baseline_hazard(counting), baseline_hazard(right), tolerance = 1e-10)

  # Late entry leaves out the time before entry.
  late <- data.frame(start = c(0, 50, 0, 20), stop = c(100, 150, 80, 200), event = c(1, 0, 1, 1))
  fit <- piecewise_exponential(Surv(start, stop, event) ~ 1, data = late, breaks = 100)
  # Exposure in (0, 100]: 100 + 50 + 80 + 80 = 310, with 2 events. In (100, Inf): 50 + 100.
  expect_equal(baseline_hazard(fit)$hazard, c(2 / 310, 1 / 150), tolerance = 1e-10)
})

test_that("weights act as copies of rows", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$k <- rep(1:3, length.out = nrow(lung))
  weighted <- piecewise_exponential(Surv(time, status) ~ age + sex,
    data = lung, weights = k, breaks = c(180, 365)
  )
  copies <- lung[rep(seq_len(nrow(lung)), lung$k), ]
  copied <- piecewise_exponential(Surv(time, status) ~ age + sex,
    data = copies, breaks = c(180, 365)
  )
  expect_equal(coef(weighted), coef(copied), tolerance = 1e-10)
  expect_equal(vcov(weighted), vcov(copied), tolerance = 1e-10)
  expect_equal(weighted$loglik, copied$loglik, tolerance = 1e-10)
  expect_true(weighted$has_weights)
})

test_that("the breaks chosen have the lowest AIC or BIC among the candidates", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  event_times <- sort(unique(lung$time[lung$status == 2]))
  candidates <- lapply(0:4, function(k) {
    if (k == 0) {
      return(numeric(0))
    }
    stats::quantile(event_times, seq_len(k) / (k + 1), names = FALSE)
  })
  fits <- lapply(candidates, function(b) piecewise_exponential(f, data = lung, breaks = b))
  aic <- vapply(fits, AIC, numeric(1))
  bic <- vapply(fits, BIC, numeric(1))
  by_aic <- piecewise_exponential(f, data = lung, max_knots = 4)
  by_bic <- piecewise_exponential(f, data = lung, max_knots = 4, knot_strategy = "bic")
  expect_equal(by_aic$breaks, candidates[[which.min(aic)]])
  expect_equal(by_bic$breaks, candidates[[which.min(bic)]])
  expect_identical(by_aic$knot_strategy, "aic")
  expect_null(fits[[1]]$knot_strategy)
  expect_output(print(by_bic), "breaks chosen by BIC", fixed = TRUE)

  none <- piecewise_exponential(f, data = lung, max_knots = 0)
  expect_identical(none$breaks, numeric(0))
  few <- data.frame(time = c(1, 2, 2, 3, 4), status = c(1, 1, 1, 0, 0), x = c(1, 0, 1, 0, 1))
  expect_identical(piecewise_exponential(Surv(time, status) ~ x, data = few)$breaks, numeric(0))
})

test_that("predictions are the piecewise cumulative hazard", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = c(180, 365))
  newdata <- data.frame(age = c(50, 70), sex = c(2, 1))
  lp <- predict(fit, newdata = newdata, type = "lp")
  expect_equal(lp, drop(as.matrix(newdata) %*% coef(fit)))
  expect_equal(predict(fit, newdata = newdata, type = "risk"), exp(lp))
  expect_identical(length(predict(fit, type = "lp")), 228L)

  hazard <- baseline_hazard(fit)$hazard
  times <- c(0, 100, 180, 200, 400, 1000)
  spent <- cbind(pmin(times, 180), pmax(pmin(times, 365) - 180, 0), pmax(times - 365, 0))
  cumhaz <- predict(fit, newdata = newdata, type = "cumhaz", times = times)
  expect_named(cumhaz, c("subject", "time", "estimate", "std_error", "conf_low", "conf_high"))
  expect_equal(cumhaz$estimate, c(outer(drop(spent %*% hazard), exp(lp))), tolerance = 1e-12)
  survival <- predict(fit, newdata = newdata, times = times)
  expect_equal(survival$estimate, exp(-cumhaz$estimate), tolerance = 1e-12)
  expect_identical(survival[survival$time == 0, "estimate"], c(1, 1))
  expect_identical(cumhaz[cumhaz$time == 0, "std_error"], c(0, 0))

  # Within the first interval, log H = log(lambda_1) + log(t) + x'beta, a linear function of the
  # parameters.
  gradient <- c(1, 0, 0, 50, 2)
  se <- sqrt(drop(gradient %*% fit$full_var %*% gradient))
  first <- cumhaz[cumhaz$subject == 1 & cumhaz$time == 100, ]
  expect_equal(first$std_error, first$estimate * se, tolerance = 1e-10)
  z <- stats::qnorm(0.975)
  expect_equal(first$conf_low, first$estimate * exp(-z * se), tolerance = 1e-10)
  s <- survival[survival$subject == 1 & survival$time == 100, ]
  expect_equal(s$conf_low, exp(-first$conf_high), tolerance = 1e-12)
  expect_equal(s$conf_high, exp(-first$conf_low), tolerance = 1e-12)
  narrow <- predict(fit, newdata = newdata, times = times, conf_level = 0.8)
  expect_true(all(narrow$conf_low[-c(1, 7)] > survival$conf_low[-c(1, 7)]))

  table <- baseline_hazard(fit)
  expect_named(table, c(
    "start", "stop", "hazard", "log_hazard", "std_error", "conf_low", "conf_high"
  ))
  expect_identical(table$start, c(0, 180, 365))
  expect_identical(table$stop, c(180, 365, Inf))
  expect_equal(table$conf_high, exp(table$log_hazard + z * table$std_error))

  lung$group <- factor(ifelse(lung$sex == 1, "m", "f"), levels = c("m", "f"))
  by_factor <- piecewise_exponential(Surv(time, status) ~ age + group,
    data = lung, breaks = c(180, 365)
  )
  expect_named(coef(by_factor), c("age", "groupf"))
  expect_equal(
    predict(by_factor, newdata = data.frame(age = 50, group = "f"), times = 300),
    predict(fit, newdata = data.frame(age = 50, sex = 2), times = 300),
    tolerance = 1e-8
  )
})

test_that("print shows the coefficients, the baseline hazard, and the tests", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- piecewise_exponential(Surv(time, status) ~ age + sex, data = lung, breaks = c(180, 365))
  expect_output(print(fit), "3 intervals. n = 228, events = 165", fixed = TRUE)
  expect_output(print(fit), "(180, 365]", fixed = TRUE)
  expect_output(print(fit), "(365, Inf)", fixed = TRUE)
  expect_output(print(fit), "Likelihood ratio test: 13.5 on 2 df", fixed = TRUE)
  intercept_only <- piecewise_exponential(Surv(time, status) ~ 1, data = lung, breaks = 365)
  expect_length(coef(intercept_only), 0L)
  expect_identical(intercept_only$lr_test$statistic, 0)
  expect_output(print(intercept_only), "Baseline hazard")
  expect_identical(predict(intercept_only, type = "lp"), numeric(228))
  dropped <- piecewise_exponential(Surv(time, status) ~ ph.ecog, data = lung, breaks = 365)
  expect_output(print(dropped), "1 row was dropped")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  expect_error(piecewise_exponential(f, data = lung, breaks = c(0, 100)), "`breaks`")
  expect_error(piecewise_exponential(f, data = lung, breaks = "a"), "`breaks`")
  expect_error(piecewise_exponential(f, data = lung, breaks = 2000), "No one is followed")
  expect_warning(
    piecewise_exponential(f, data = lung, breaks = c(1, 2)),
    "interval (0, 1] has no events", fixed = TRUE
  )
  expect_error(piecewise_exponential(f, data = lung, knot_strategy = "cv"), "should be one of")
  expect_error(piecewise_exponential(f, data = lung, max_knots = 1.5), "whole number")
  expect_error(piecewise_exponential(f, data = lung, max_knots = -1), "`max_knots`")
  expect_error(piecewise_exponential(f, data = lung, conf_level = 1), "conf_level")
  expect_error(piecewise_exponential(f, data = lung, tol = 0), "`tol`")
  expect_warning(
    piecewise_exponential(f, data = lung, breaks = 365, max_iter = 1),
    "didn't converge in 1 iterations"
  )
  expect_error(piecewise_exponential(Surv(time, status * 0) ~ age, data = lung), "one event")
  expect_error(piecewise_exponential(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(piecewise_exponential(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(
    piecewise_exponential(Surv(time, status) ~ x,
      data = data.frame(time = 0:2, status = 1, x = c(1, 2, 4))
    ),
    "positive times"
  )
  expect_error(
    piecewise_exponential(Surv(time, time2, type = "interval2") ~ 1,
      data = data.frame(time = c(1, 2), time2 = c(2, 3))
    ),
    "right-censored or counting-process"
  )
  fit <- piecewise_exponential(f, data = lung, breaks = 365)
  expect_error(predict(fit), "needs `times`")
  expect_error(predict(fit, type = "cumhaz"), "needs `times`")
  expect_error(predict(fit, times = -1), "negative")
  expect_error(predict(fit, times = "a"), "`times`")
  expect_error(predict(fit, newdata = list(age = 1), type = "lp"), "data frame")
  expect_error(predict(fit, newdata = data.frame(age = 1), type = "lp"), "sex")
  expect_error(predict(fit, type = "hazard"), "should be one of")
  expect_error(baseline_hazard(fit, conf_level = 2), "conf_level")
})
