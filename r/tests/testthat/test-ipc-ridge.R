# Replays the ipcridge_lung fixture in fixtures/r/, written from survival::survfit() for the
# censoring distribution and from lm.wfit() on an augmented design for the ridge. Weighted fits,
# which the fixture doesn't cover, are checked against survfit() and lm() directly.

lung_complete <- function() {
  lung <- survival::lung
  lung[stats::complete.cases(lung[c("time", "status", "age", "sex")]), ]
}

test_that("the censoring distribution and IPC weights reproduce the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("ipcridge_lung")
  lung <- lung_complete()
  censoring <- censoring_distribution(Surv(lung$time, lung$status))
  expect_equal(
    predict(censoring, times = r_numbers(expected$censoring_times)),
    r_numbers(expected$censoring_surv),
    tolerance = 1e-10
  )
  expect_equal(weights(censoring), r_numbers(expected$ipc_weights), tolerance = 1e-10)
  expect_identical(censoring$n, 228L)
  expect_identical(censoring$n_censored, 63L)
})

test_that("the ridge reproduces the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("ipcridge_lung")
  lung <- lung_complete()
  alpha <- r_numbers(expected$ridge_alpha)
  fit <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = alpha)
  expect_equal(unname(coef(fit)[-1L]), r_numbers(expected$ridge_coef), tolerance = 1e-10)
  expect_equal(unname(coef(fit)[1L]), r_numbers(expected$ridge_intercept), tolerance = 1e-10)
  expect_equal(predict(fit), r_numbers(expected$ridge_lp), tolerance = 1e-10)
  expect_equal(predict(fit, newdata = lung), predict(fit))
  expect_identical(c(fit$n, fit$n_event), as.integer(c(expected$n, expected$n_event)))
  expect_equal(fit$ipc_weights, r_numbers(expected$ipc_weights), tolerance = 1e-10)
})

test_that("the censoring distribution is the Kaplan-Meier curve of the censorings", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  set.seed(2)
  w <- stats::runif(nrow(lung), 0.5, 2)
  censoring <- censoring_distribution(Surv(lung$time, lung$status), weights = w)
  reference <- survival::survfit(survival::Surv(lung$time, lung$status == 1) ~ 1, weights = w)
  times <- c(0, 5, 100, 365, 365.5, 1022, 2000)
  expect_equal(
    predict(censoring, times = times),
    summary(reference, times = times, extend = TRUE)$surv,
    tolerance = 1e-12
  )
  curve <- as.data.frame(censoring)
  expect_named(curve, c("time", "survival"))
  expect_identical(curve$time, sort(unique(lung$time[lung$status == 1])))

  # G(t-) is the curve just before t, and the weights are 1 / G(T-) for the events.
  t <- curve$time[3]
  expect_identical(predict(censoring, times = t, left = TRUE), curve$survival[2])
  expect_identical(predict(censoring, times = t), curve$survival[3])
  ipc <- weights(censoring)
  events <- lung$status == 2
  expect_identical(ipc[!events], rep(0, sum(!events)))
  expect_equal(ipc[events], 1 / predict(censoring, times = lung$time[events], left = TRUE))
  late <- weights(censoring, tau = 365)
  expect_identical(late[lung$time > 365], rep(0, sum(lung$time > 365)))
  expect_identical(late[lung$time <= 365], ipc[lung$time <= 365])

  # Without censoring, the curve stays at 1.
  none <- censoring_distribution(Surv(c(1, 2, 3), c(1, 1, 1)))
  expect_identical(weights(none), c(1, 1, 1))
  expect_identical(predict(none, times = 10), 1)
  expect_output(print(none), "censoring times = 0, G after the last = 1", fixed = TRUE)
  expect_output(print(censoring), "n = 228, censored = 63, censoring times = ", fixed = TRUE)
})

test_that("alpha = 0 is IPC-weighted least squares, and the penalty is as documented", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  unpenalized <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = 0)
  lung$ipc <- unpenalized$ipc_weights
  events <- lung[lung$status == 2, ]
  reference <- stats::lm(log(time) ~ age + sex, data = events, weights = ipc)
  expect_equal(coef(unpenalized), coef(reference), tolerance = 1e-10)

  # Without standardizing, the penalty is alpha times the sum of the squared coefficients.
  raw <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = 50, standardize = FALSE)
  x <- cbind(1, events$age, events$sex)
  augmented <- rbind(x, cbind(0, diag(sqrt(50), 2)))
  fit <- stats::lm.wfit(augmented, c(log(events$time), 0, 0), c(events$ipc, 1, 1))
  expect_equal(unname(coef(raw)), unname(fit$coefficients), tolerance = 1e-10)
  expect_identical(unname(raw$scale), c(1, 1))

  standardized <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = 50)
  expect_equal(unname(standardized$scale), c(stats::sd(events$age), stats::sd(events$sex)))
  norms <- vapply(c(0.1, 10, 1000), function(alpha) {
    sqrt(sum(coef(ipc_ridge(Surv(time, status) ~ age + sex, data = lung, alpha = alpha))[-1]^2))
  }, numeric(1))
  expect_true(all(diff(norms) < 0))
})

test_that("weights act as copies of rows", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  lung$k <- rep(1:3, length.out = nrow(lung))
  weighted <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, weights = k, alpha = 2)
  copies <- lung[rep(seq_len(nrow(lung)), lung$k), ]
  copied <- ipc_ridge(Surv(time, status) ~ age + sex, data = copies, alpha = 2)
  expect_equal(coef(weighted), coef(copied), tolerance = 1e-12)
  expect_equal(weighted$scale, copied$scale, tolerance = 1e-12)
  expect_true(weighted$has_weights)

  # A weight of 0 leaves the row out.
  lung$zero <- as.numeric(seq_len(nrow(lung)) != 5)
  zeroed <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung, weights = zero)
  dropped <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung[-5, ])
  expect_equal(coef(zeroed), coef(dropped), tolerance = 1e-12)
})

test_that("factors, predictions, the table, and print work", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- ipc_ridge(Surv(time, status) ~ age + sex, data = lung)
  lung$group <- factor(ifelse(lung$sex == 1, "m", "f"), levels = c("m", "f"))
  by_factor <- ipc_ridge(Surv(time, status) ~ age + group, data = lung)
  expect_named(coef(by_factor), c("(Intercept)", "age", "groupf"))
  # Coding sex as 1 and 2 rather than 0 and 1 only moves the intercept.
  expect_equal(unname(coef(by_factor)[-1L]), unname(coef(fit)[-1L]))
  expect_equal(unname(coef(by_factor)[1L]), unname(coef(fit)[1L] + coef(fit)[["sex"]]))
  expect_equal(
    predict(by_factor, newdata = data.frame(age = c(50, 70), group = c("f", "m"))),
    predict(fit, newdata = data.frame(age = c(50, 70), sex = c(2, 1)))
  )

  newdata <- data.frame(age = 60, sex = 2)
  lp <- predict(fit, newdata = newdata)
  expect_equal(lp, sum(coef(fit) * c(1, 60, 2)))
  expect_equal(predict(fit, newdata = newdata, type = "response"), exp(lp))
  expect_identical(length(predict(fit)), nobs(fit))

  table <- as.data.frame(fit)
  expect_named(table, c("term", "estimate"))
  expect_identical(table$term, c("(Intercept)", "age", "sex"))
  expect_identical(table$estimate, unname(coef(fit)))

  expect_output(print(fit), "IPC-weighted ridge, alpha = 1. n = 228, events = 165", fixed = TRUE)
  dropped <- ipc_ridge(Surv(time, status) ~ ph.ecog, data = lung)
  expect_identical(dropped$n_dropped, 1L)
  expect_output(print(dropped), "1 row was dropped")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  expect_error(ipc_ridge(f, data = lung, alpha = -1), "`alpha`")
  expect_error(ipc_ridge(f, data = lung, standardize = NA), "`standardize`")
  expect_error(ipc_ridge(Surv(time, status) ~ 1, data = lung), "at least one covariate")
  expect_error(ipc_ridge(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(ipc_ridge(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(ipc_ridge(Surv(time, status * 0) ~ age, data = lung), "at least one event")
  expect_error(
    ipc_ridge(Surv(time, status) ~ age, data = data.frame(time = 0:2, status = 1, age = 1:3)),
    "positive event times"
  )
  heart <- survival::heart
  expect_error(ipc_ridge(Surv(start, stop, event) ~ age, data = heart), "right-censored")
  fit <- ipc_ridge(f, data = lung)
  expect_error(predict(fit, newdata = list(age = 1)), "data frame")
  expect_error(predict(fit, newdata = data.frame(age = 1)), "sex")
  expect_error(predict(fit, type = "survival"), "should be one of")

  expect_error(censoring_distribution(c(1, 2)), "`Surv()`", fixed = TRUE)
  expect_error(censoring_distribution(Surv(c(1, 2), c(1, 0)), weights = 1), "2 non-negative")
  expect_error(censoring_distribution(Surv(c(1, 2), c(1, 0)), weights = c(1, -1)), "non-negative")
  censoring <- censoring_distribution(Surv(c(1, 2), c(1, 0)))
  expect_error(predict(censoring, times = "a"), "`times`")
  expect_error(predict(censoring, times = 1, left = NA), "`left`")
  expect_error(weights(censoring, tau = NA), "`tau`")
})
