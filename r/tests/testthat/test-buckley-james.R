# No R fixture covers Buckley-James regression, so the fits are checked against Greenwood's
# Python values, and against least squares when nothing is censored.

test_that("the fits match Python", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- buckley_james(Surv(time, status) ~ age + sex, data = lung)
  expect_equal(unname(coef(fit)), c(6.379093242992207, -0.02306412131297025, 0.49119928713684136),
    tolerance = 1e-10
  )
  expect_named(coef(fit), c("(Intercept)", "age", "sex"))
  expect_identical(fit$iterations, 13L)
  expect_true(fit$converged)
  expect_identical(c(fit$n, fit$n_event), c(228L, 165L))
  survival <- predict(fit, newdata = lung[1:3, ], times = c(30, 180, 365, 730, 2000))
  expect_named(survival, c("subject", "time", "estimate"))
  expect_identical(survival$subject, rep(1:3, each = 5))
  expect_equal(survival$estimate, c(
    0.9429824561403509, 0.5404950952005622, 0.24009872847731917, 0.02653844882879742, 0,
    0.9561403508771931, 0.6049147232507077, 0.27860918761481035, 0.059711509864794184, 0,
    0.9649122807017544, 0.7165067792708385, 0.4338100373357785, 0.1286094058626336, 0
  ), tolerance = 1e-10)

  more <- buckley_james(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = lung)
  expect_equal(unname(coef(more)), c(
    6.5832843713249698, -0.021033849492812999, 0.4928112321841544, -0.34001705985595015,
    0.004630144615887944
  ), tolerance = 1e-10)
  expect_identical(more$iterations, 14L)
  expect_identical(more$n_dropped, 15L)
  expect_equal(
    predict(more, newdata = lung[2:3, ], times = c(30, 180, 365, 730, 2000))$estimate,
    c(
      0.9671361502347416, 0.7700376975680401, 0.4979325735342948, 0.13727537498726455, 0,
      0.9812206572769953, 0.8478068431876117, 0.5755347009265954, 0.2472885267171605,
      0.014381229760570574
    ),
    tolerance = 1e-10
  )

  veteran <- survival::veteran
  vet <- buckley_james(Surv(time, status) ~ karno + age + trt, data = veteran)
  expect_equal(unname(coef(vet)), c(
    1.3299625653969256, 0.04068300944495725, 0.01126281143251391, -0.14188320591157436
  ), tolerance = 1e-10)
  expect_identical(vet$iterations, 6L)
  tight <- buckley_james(Surv(time, status) ~ karno + age + trt,
    data = veteran, max_iter = 1000, tol = 1e-9
  )
  expect_equal(unname(coef(tight)), c(
    1.3299625633038825, 0.04068300974282636, 0.01126281093480533, -0.14188319419991624
  ), tolerance = 1e-10)
  expect_identical(tight$iterations, 9L)
})

test_that("with no censoring, the fit is least squares after one step", {
  set.seed(3)
  n <- 200
  data <- data.frame(age = stats::rnorm(n, 60, 10), sex = stats::rbinom(n, 1, 0.5))
  data$time <- exp(6 + 0.01 * data$age + 0.3 * data$sex + stats::rnorm(n, 0, 0.5))
  data$status <- 1
  fit <- buckley_james(Surv(time, status) ~ age + sex, data = data)
  expect_identical(fit$iterations, 1L)
  expect_true(fit$converged)
  expect_equal(coef(fit), coef(stats::lm(log(time) ~ age + sex, data = data)), tolerance = 1e-10)

  # Weighted, it's weighted least squares.
  data$w <- stats::runif(n, 0.5, 2)
  weighted <- buckley_james(Surv(time, status) ~ age + sex, data = data, weights = w)
  reference <- stats::lm(log(time) ~ age + sex, data = data, weights = w)
  expect_equal(coef(weighted), coef(reference), tolerance = 1e-10)
})

test_that("the filled-in log times are the conditional means under the residual curve", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- buckley_james(Surv(time, status) ~ age + sex, data = lung)
  x <- cbind(1, lung$age, lung$sex)
  event <- lung$status == 2
  log_time <- log(lung$time)
  filled <- bj_pseudo_response(log_time, event, x, rep(1, nrow(lung)), coef(fit))
  expect_identical(filled[event], log_time[event])
  expect_true(all(filled[!event] >= log_time[!event]))
  # At the fixed point, refitting to the filled-in log times gives the coefficients back.
  refit <- coef(stats::lm(filled ~ age + sex, data = lung))
  expect_equal(unname(refit), unname(coef(fit)), tolerance = 1e-5)

  # The conditional mean of one censored residual, by a direct sum over the curve.
  residual <- log_time - drop(x %*% coef(fit))
  shifted <- residual + 1 - min(residual)
  i <- which(!event)[1]
  tail_event <- event | seq_along(event) == which.max(shifted)
  km <- survival::survfit(survival::Surv(shifted, tail_event) ~ 1)
  times <- c(km$time, max(shifted))
  heights <- c(km$surv, 0)
  from <- pmax(times[-length(times)], shifted[i])
  to <- times[-1L]
  inside <- to > from
  area <- sum(heights[-length(heights)][inside] * (to - from)[inside])
  s_i <- km$surv[findInterval(shifted[i], km$time)]
  expect_equal(filled[i], log_time[i] + area / s_i, tolerance = 1e-12)
})

test_that("weights act as copies of rows", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$k <- rep(1:3, length.out = nrow(lung))
  weighted <- buckley_james(Surv(time, status) ~ age + sex, data = lung, weights = k)
  copies <- lung[rep(seq_len(nrow(lung)), lung$k), ]
  copied <- buckley_james(Surv(time, status) ~ age + sex, data = copies)
  expect_equal(coef(weighted), coef(copied), tolerance = 1e-10)
  expect_identical(weighted$iterations, copied$iterations)
  newdata <- data.frame(age = 60, sex = 2)
  expect_equal(
    predict(weighted, newdata = newdata, times = c(100, 300, 600)),
    predict(copied, newdata = newdata, times = c(100, 300, 600)),
    tolerance = 1e-10
  )
})

test_that("bootstrap standard errors come from the resamples", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  fit <- buckley_james(f, data = lung)
  table <- as.data.frame(fit)
  expect_named(table, c(
    "term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"
  ))
  expect_true(all(is.na(table$std_error)))
  expect_true(all(is.na(table$p_value)))
  expect_error(vcov(fit), "`n_boot`")
  expect_null(fit$boot)

  set.seed(10)
  booted <- buckley_james(f, data = lung, n_boot = 20)
  expect_identical(coef(booted), coef(fit))
  expect_identical(dim(booted$boot), c(20L, 3L))
  expect_equal(booted$std_error, apply(booted$boot, 2L, stats::sd))
  expect_equal(sqrt(diag(vcov(booted))), booted$std_error)
  table <- as.data.frame(booted)
  expect_true(all(table$std_error > 0))
  expect_equal(table$statistic, unname(coef(booted) / booted$std_error))
  expect_equal(table$conf_high, unname(coef(booted) + stats::qnorm(0.975) * booted$std_error))
  narrow <- as.data.frame(booted, conf_level = 0.8)
  expect_true(all(narrow$conf_high < table$conf_high))
  # The resamples are drawn with R's random number generator.
  set.seed(10)
  again <- buckley_james(f, data = lung, n_boot = 20)
  expect_identical(again$boot, booted$boot)
  expect_output(print(booted), "bootstrap resamples = 20", fixed = TRUE)
  # A check on the size of the bootstrap standard error of age.
  expect_lt(booted$std_error[["age"]], 0.05)
})

test_that("predictions, factors, and print work", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- buckley_james(Surv(time, status) ~ age + sex, data = lung)
  newdata <- data.frame(age = c(50, 70), sex = c(2, 1))
  lp <- predict(fit, newdata = newdata, type = "lp")
  expect_equal(lp, drop(cbind(1, as.matrix(newdata)) %*% coef(fit)))
  expect_identical(length(predict(fit, type = "lp")), 228L)
  survival <- predict(fit, newdata = newdata, times = seq(10, 1000, by = 10))
  for (i in 1:2) {
    s <- survival$estimate[survival$subject == i]
    expect_true(all(diff(s) <= 0))
    expect_true(all(s >= 0 & s <= 1))
  }
  # A longer predicted log time moves the curve to the right.
  expect_true(all(
    survival$estimate[survival$subject == 1] >= survival$estimate[survival$subject == 2]
  ))
  expect_identical(predict(fit, newdata = newdata[1, ], times = 0)$estimate, 1)

  lung$group <- factor(ifelse(lung$sex == 1, "m", "f"), levels = c("m", "f"))
  by_factor <- buckley_james(Surv(time, status) ~ age + group, data = lung)
  expect_named(coef(by_factor), c("(Intercept)", "age", "groupf"))
  expect_equal(unname(coef(by_factor)[-1L]), unname(coef(fit)[-1L]), tolerance = 1e-10)
  expect_equal(
    predict(by_factor, newdata = data.frame(age = 50, group = "f"), type = "lp"),
    predict(fit, newdata = data.frame(age = 50, sex = 2), type = "lp"),
    tolerance = 1e-10
  )

  expect_identical(nobs(fit), 228L)
  expect_output(print(fit), "n = 228, events = 165, iterations = 13, converged = yes", fixed = TRUE)
  expect_warning(
    short <- buckley_james(Surv(time, status) ~ age + sex, data = lung, max_iter = 1),
    "didn't converge in 1 iterations"
  )
  expect_false(short$converged)
  expect_output(print(short), "converged = no", fixed = TRUE)
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  expect_error(buckley_james(f, data = lung, max_iter = 0), "`max_iter`")
  expect_error(buckley_james(f, data = lung, tol = 0), "`tol`")
  expect_error(buckley_james(f, data = lung, n_boot = 0), "`n_boot`")
  expect_error(buckley_james(f, data = lung, conf_level = 2), "conf_level")
  expect_error(buckley_james(Surv(time, status * 0) ~ age, data = lung), "at least one event")
  expect_error(buckley_james(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(buckley_james(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(
    buckley_james(Surv(time, status) ~ age, data = data.frame(time = 0:2, status = 1, age = 1:3)),
    "positive times"
  )
  heart <- survival::heart
  expect_error(buckley_james(Surv(start, stop, event) ~ age, data = heart), "right-censored")
  fit <- buckley_james(f, data = lung)
  expect_error(predict(fit, newdata = lung[1:2, ]), "needs `times`")
  expect_error(predict(fit, newdata = list(age = 1), type = "lp"), "data frame")
  expect_error(predict(fit, newdata = data.frame(age = 1), type = "lp"), "sex")
  expect_error(predict(fit, times = "a"), "`times`")
  expect_error(predict(fit, type = "hazard"), "should be one of")
})
