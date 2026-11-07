# Replays the cumhaz and cumhaz_var fields of the Kaplan-Meier fixtures in fixtures/r/.

TOL <- 1e-9

expect_na_matches <- function(curve, expected, label) {
  expect_equal(curve$time, r_numbers(expected$time), tolerance = TOL, label = paste(label, "time"))
  expect_equal(curve$estimate, r_numbers(expected$cumhaz), tolerance = TOL, label = paste(label, "cumhaz"))
  expect_equal(
    curve$std_error^2,
    r_numbers(expected$cumhaz_var),
    tolerance = TOL,
    label = paste(label, "variance")
  )
}

test_that("the cumulative hazard on lung and veteran matches the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  lung <- nelson_aalen(Surv(time, status) ~ 1, data = survival::lung)
  expect_na_matches(as.data.frame(lung), read_fixture("km_lung_overall")$overall, "lung")
  veteran <- nelson_aalen(Surv(time, status) ~ 1, data = survival::veteran)
  expect_na_matches(as.data.frame(veteran), read_fixture("km_veteran_overall")$overall, "veteran")
})

test_that("curves by sex on lung match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("km_lung_by_sex")
  curves <- as.data.frame(nelson_aalen(Surv(time, status) ~ sex, data = survival::lung))
  for (sex in names(expected)) {
    expect_na_matches(curves[curves$sex == as.numeric(sex), ], expected[[sex]], paste("sex", sex))
  }
})

test_that("confidence limits are on the log or plain scale", {
  z <- qnorm(0.975)
  log_scale <- as.data.frame(nelson_aalen(Surv(time, status) ~ 1, data = small))
  expect_equal(log_scale$conf_low, log_scale$estimate * exp(-z * log_scale$std_error / log_scale$estimate))
  expect_equal(log_scale$conf_high, log_scale$estimate * exp(z * log_scale$std_error / log_scale$estimate))

  plain <- as.data.frame(nelson_aalen(Surv(time, status) ~ 1, data = small, conf_type = "plain", conf_level = 0.9))
  z <- qnorm(0.95)
  expect_equal(plain$conf_low, pmax(plain$estimate - z * plain$std_error, 0))
  expect_equal(plain$conf_high, plain$estimate + z * plain$std_error)
  expect_true(any(plain$conf_low == 0))
})

test_that("a curve sums d / n and its variance d / n^2", {
  fit <- as.data.frame(nelson_aalen(Surv(time, status) ~ 1, data = small))
  table <- event_table(Surv(time, status) ~ 1, data = small)
  expect_equal(fit$estimate, cumsum(table$n_event / table$n_risk))
  expect_equal(fit$std_error, sqrt(cumsum(table$n_event / table$n_risk^2)))
  expect_named(fit, c("strata", "time", "n_risk", "n_event", "estimate", "std_error", "conf_low", "conf_high"))
})

test_that("integer weights match replicated rows, and delayed entry uses start times", {
  w <- c(2, 1, 3, 1, 1, 2, 1, 1, 2, 1)
  weighted <- nelson_aalen(Surv(time, status) ~ sex, data = small, weights = w)
  replicated <- nelson_aalen(Surv(time, status) ~ sex, data = small[rep(seq_along(w), w), ])
  expect_equal(as.data.frame(weighted), as.data.frame(replicated))

  entry <- transform(small, start = c(0, 0, 100, 0, 200, 0, 0, 50, 0, 0))
  fit <- as.data.frame(nelson_aalen(Surv(start, time, status) ~ 1, data = entry))
  km <- as.data.frame(kaplan_meier(Surv(start, time, status) ~ 1, data = entry))
  expect_identical(fit$n_risk, km$n_risk)
})

test_that("predict() reads the step function", {
  fit <- nelson_aalen(Surv(time, status) ~ sex, data = small)
  curve <- as.data.frame(fit)
  pred <- predict(fit, times = c(0, 306, 5000))
  expect_named(pred, c("strata", "sex", "time", "estimate", "std_error", "conf_low", "conf_high"))
  first <- curve[curve$sex == 1, ]
  expect_identical(pred$estimate[1:3], c(0, first$estimate[first$time == 306], max(first$estimate)))
  expect_identical(pred$std_error[1], 0)
})

test_that("one curve and several give the same layout", {
  one <- as.data.frame(nelson_aalen(Surv(time, status) ~ 1, data = small))
  two <- as.data.frame(nelson_aalen(Surv(time, status) ~ sex, data = small))
  expect_identical(names(one), setdiff(names(two), "sex"))
  expect_identical(unique(one$strata), "all")
})

test_that("invalid input is rejected", {
  expect_error(nelson_aalen(Surv(time, status) ~ 1, data = small, conf_type = "log-log"))
  expect_error(nelson_aalen(Surv(time, status) ~ 1, data = small, conf_level = 2), "between 0 and 1")
  expect_error(nelson_aalen(time ~ 1, data = small), "`Surv\\(\\)` response")
})

test_that("a fit prints a summary per group", {
  expect_snapshot({
    nelson_aalen(Surv(time, status) ~ 1, data = small)
    nelson_aalen(Surv(time, status) ~ sex, data = small)
  })
})
