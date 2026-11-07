# Replays the parametric distribution fixtures in fixtures/r/ (parametric_*), and checks the
# grouped fits, the derived summaries, and compare_distributions().

TOL <- 1e-9
DISTS <- c("weibull", "exponential", "lognormal", "loglogistic")

test_that("the fits on lung match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  for (dist in DISTS) {
    expected <- read_fixture(paste0("parametric_", dist))
    fit <- parametric(Surv(time, status) ~ 1, data = survival::lung, dist = dist)
    one <- fit$fits[[1L]]
    # survreg() stops at a looser tolerance for the log-normal, so its estimates are a little off.
    tol <- if (dist == "lognormal") 1e-7 else TOL
    expect_identical(fit$n, as.integer(expected$n))
    expect_equal(fit$n_event, expected$nevent)
    expect_equal(unname(one$coefficients), expected$mu, tolerance = tol)
    expect_equal(sqrt(one$var[1, 1]), expected$mu_se, tolerance = tol)
    expect_equal(one$scale, expected$scale, tolerance = tol)
    expect_equal(as.numeric(logLik(fit)), expected$loglik, tolerance = TOL)
    if (dist != "exponential") {
      expect_equal(sqrt(one$var[2, 2]), expected$log_scale_se, tolerance = tol)
    }

    q <- quantile(fit, probs = r_numbers(expected$pred_p))
    expect_identical(
      names(q), c("strata", "prob", "time", "std_error", "conf_low", "conf_high")
    )
    expect_equal(q$time, r_numbers(expected$pred_quantile), tolerance = tol)
    s <- predict(fit, times = r_numbers(expected$pred_times))
    expect_identical(
      names(s), c("strata", "time", "estimate", "std_error", "conf_low", "conf_high")
    )
    expect_equal(s$estimate, r_numbers(expected$pred_survival), tolerance = tol)
  }
})

test_that("the parameters are the usual ones, with delta-method errors", {
  fit <- parametric(Surv(time, status) ~ 1, data = small_cox)
  one <- fit$fits[[1L]]
  mu <- unname(one$coefficients)
  sigma <- one$scale
  table <- as.data.frame(fit)
  expect_identical(
    names(table), c("strata", "param", "estimate", "std_error", "conf_low", "conf_high")
  )
  expect_identical(table$param, c("shape", "scale"))
  expect_equal(table$estimate, c(1 / sigma, exp(mu)))
  expect_equal(table$std_error, c(sqrt(one$var[2, 2]) / sigma, exp(mu) * sqrt(one$var[1, 1])))
  z <- stats::qnorm(0.975)
  expect_equal(table$conf_low, table$estimate * exp(-z * table$std_error / table$estimate))

  # The Weibull survival and hazard in their usual parameters.
  shape <- table$estimate[1]
  scale <- table$estimate[2]
  times <- c(50, 200, 500)
  expect_equal(predict(fit, times = times)$estimate, exp(-(times / scale)^shape))
  expect_equal(predict(fit, times = times, type = "cumhaz")$estimate, (times / scale)^shape)
  hazard <- predict(fit, times = times, type = "hazard")
  expect_equal(hazard$estimate, (shape / scale) * (times / scale)^(shape - 1))
  expect_true(all(hazard$conf_low < hazard$estimate & hazard$estimate < hazard$conf_high))
  expect_equal(
    predict(fit, times = times, type = "density")$estimate,
    stats::dweibull(times, shape = shape, scale = scale)
  )
  expect_equal(median(fit)$time, scale * log(2)^(1 / shape))
  expect_equal(mean(fit)$estimate, scale * gamma(1 + 1 / shape))
  # The area under the Weibull curve up to tau, through the incomplete gamma function.
  area <- scale * gamma(1 + 1 / shape) * stats::pgamma((365 / scale)^shape, 1 / shape)
  expect_equal(rmst(fit, tau = 365)$estimate, area, tolerance = 1e-10)

  exponential <- as.data.frame(parametric(Surv(time, status) ~ 1, data = small_cox,
    dist = "exponential"
  ))
  expect_identical(exponential$param, "rate")
  expect_equal(exponential$estimate, sum(small_cox$status == 2) / sum(small_cox$time))

  lognormal <- as.data.frame(parametric(Surv(time, status) ~ 1, data = small_cox,
    dist = "lognormal"
  ))
  expect_identical(lognormal$param, c("mu", "sigma"))
  expect_equal(lognormal$conf_low[1], lognormal$estimate[1] - z * lognormal$std_error[1])

  loglogistic <- parametric(Surv(time, status) ~ 1, data = small_cox, dist = "loglogistic")
  params <- as.data.frame(loglogistic)
  expect_identical(params$param, c("alpha", "beta"))
  expect_equal(median(loglogistic)$time, params$estimate[1])
})

test_that("grouped fits are one fit per group", {
  skip_if_not_installed("survival")
  fit <- parametric(Surv(time, status) ~ sex, data = survival::lung, dist = "lognormal")
  table <- as.data.frame(fit)
  expect_identical(table$strata, rep(c("sex=1", "sex=2"), each = 2))
  expect_identical(table$sex, rep(c(1, 2), each = 2))
  for (s in 1:2) {
    one <- parametric(Surv(time, status) ~ 1, data = survival::lung[survival::lung$sex == s, ],
      dist = "lognormal"
    )
    expect_equal(table$estimate[table$sex == s], as.data.frame(one)$estimate)
  }
  expect_identical(attr(logLik(fit), "df"), 4L)
  q <- quantile(fit, probs = c(0.25, 0.5))
  expect_identical(nrow(q), 4L)
  expect_identical(names(median(fit)), c("strata", "sex", "time", "std_error", "conf_low", "conf_high"))
  expect_identical(names(rmst(fit, tau = 365)), c(
    "strata", "sex", "tau", "estimate", "std_error", "conf_low", "conf_high"
  ))
})

test_that("compare_distributions() ranks by AIC", {
  table <- compare_distributions(Surv(time, status) ~ 1, data = small_cox)
  expect_identical(names(table), c("dist", "n_params", "loglik", "aic", "bic"))
  expect_setequal(table$dist, DISTS)
  expect_false(is.unsorted(table$aic))
  for (i in seq_len(nrow(table))) {
    fit <- parametric(Surv(time, status) ~ 1, data = small_cox, dist = table$dist[i])
    expect_equal(table$loglik[i], as.numeric(logLik(fit)))
    expect_equal(table$aic[i], -2 * table$loglik[i] + 2 * table$n_params[i])
    expect_equal(table$bic[i], -2 * table$loglik[i] + log(30) * table$n_params[i])
  }
  expect_identical(table$n_params[table$dist == "exponential"], 1L)

  # The data are found from inside a function too.
  compare <- function(d) {
    compare_distributions(Surv(time, status) ~ 1, data = d, dists = c("weibull", "lognormal"))
  }
  expect_identical(sort(compare(small_cox[1:20, ])$dist), c("lognormal", "weibull"))
  expect_error(compare_distributions(Surv(time, status) ~ 1, data = small_cox, dists = "gamma"))
})

test_that("bad input is rejected", {
  expect_error(
    parametric(Surv(time, status) ~ strata(sex), data = small_cox),
    "doesn't support `strata()`", fixed = TRUE
  )
  expect_error(parametric(Surv(time, status) ~ 1, data = small_cox, dist = "gamma"))
  fit <- parametric(Surv(time, status) ~ 1, data = small_cox)
  expect_error(quantile(fit, probs = 1), "`probs`")
  expect_error(predict(fit, times = "a"), "`times`")
  expect_error(rmst(fit, tau = -1), "`tau`")
})

test_that("the printout is stable", {
  expect_snapshot(print(parametric(Surv(time, status) ~ 1, data = small_cox)))
  expect_snapshot(print(parametric(Surv(time, status) ~ sex, data = small_cox, dist = "lognormal")))
})
