# influence_diagnostics() and smooth_hr(). The reference values are the Python package's, for the
# same fit of Surv(time, status) ~ age + sex to the lung data.

lung_fit <- function() cox_ph(Surv(time, status) ~ age + sex, data = lung)

test_that("influence_diagnostics() gathers the residuals and the likelihood displacement", {
  fit <- lung_fit()
  out <- influence_diagnostics(fit)
  expect_s3_class(out, "data.frame")
  expect_named(out, c(
    "leverage", "martingale", "deviance", "dfbeta_age", "dfbeta_sex", "dfbetas_age",
    "dfbetas_sex", "ld"
  ))
  expect_identical(nrow(out), 228L)
  expect_equal(out$martingale, residuals(fit, type = "martingale"))
  expect_equal(out$deviance, residuals(fit, type = "deviance"))
  expect_equal(out$leverage, residuals(fit, type = "leverage"))
  expect_equal(unname(out$dfbetas_sex), unname(residuals(fit, type = "dfbetas")[, "sex"]))
  expect_equal(
    unlist(out[3, ]),
    c(
      leverage = 0.069462, martingale = -3.129819, deviance = -2.501927, dfbeta_age = 0.001917,
      dfbeta_sex = 0.029004, dfbetas_age = 0.207895, dfbetas_sex = 0.173202, ld = 0.069462
    ),
    tolerance = 1e-5
  )
  expect_identical(order(out$ld, decreasing = TRUE)[1:3], c(6L, 85L, 38L))
  # Without weights, the likelihood displacement is the leverage.
  expect_equal(out$ld, out$leverage)
})

test_that("influence_diagnostics() uses the model-based variance of a robust fit", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung, robust = TRUE)
  out <- influence_diagnostics(fit)
  expect_equal(out$ld, influence_diagnostics(lung_fit())$ld)
  expect_error(influence_diagnostics(kaplan_meier(Surv(time, status) ~ 1, data = lung)), "cox_ph")
})

test_that("smooth_hr() matches the Python package's curve", {
  curve <- smooth_hr(lung_fit(), term = "age")
  expect_s3_class(curve, "smooth_hr")
  expect_equal(curve$reference, 62.44736842105263, tolerance = 1e-12)
  expect_equal(curve$knots, 63)
  expect_equal(curve$adjustment, c(sex = 1.394736842105263), tolerance = 1e-12)
  at <- c(1, 51, 101, 151, 200)
  expect_equal(
    curve$grid[at], c(39, 49.80402010050251, 60.608040201005025, 71.41206030150754, 82),
    tolerance = 1e-12
  )
  expect_equal(
    curve$log_hr[at],
    c(
      -1.2506619349745183, 0.07073230228652772, 0.01909091139333111, 0.20444681794456154,
      1.0641151678997662
    ),
    tolerance = 1e-9
  )
  expect_equal(
    curve$log_hr_lower[at],
    c(
      -3.0923787757044532, -0.4608238175241319, -0.0473812269979644, -0.1815950163650272,
      -0.3281246426047284
    ),
    tolerance = 1e-9
  )
  expect_equal(
    curve$log_hr_upper[at],
    c(
      0.5910549057554233, 0.6022884220971914, 0.0855630497846269, 0.5904886522541540,
      2.4563549784042533
    ),
    tolerance = 1e-9
  )
  expect_equal(curve$hr, exp(curve$log_hr))
  expect_equal(curve$hr_lower, exp(curve$log_hr_lower))
  expect_equal(curve$log_hr[which.min(abs(curve$grid - curve$reference))], 0, tolerance = 0.01)

  curve <- smooth_hr(lung_fit(), term = "age", df = 6, n_grid = 11, reference = 60)
  expect_equal(curve$grid, seq(39, 82, by = 4.3))
  expect_equal(curve$knots, c(56, 63, 69))
  expect_equal(
    curve$log_hr,
    c(
      -1.2544616992364164, -0.425939889804948, 0.05735253167022969, 0.24206047681771525,
      0.17483597792615901, -0.016588646900727158, 0.062040461231998856, 0.3197448287297915,
      0.2843430974926144, 0.5420736535547035, 1.9357200709748867
    ),
    tolerance = 1e-9
  )
})

test_that("smooth_hr()'s limits don't depend on which basis column is dropped", {
  fit <- lung_fit()
  curve <- smooth_hr(fit, term = "age", n_grid = 25)
  knots <- c(rep(39, 4), 63, rep(82, 4))
  basis <- bspline_basis(lung$age, knots)
  expect_equal(rowSums(basis), rep(1, nrow(lung)))
  # The same model, without the last basis column instead of the first.
  kept <- basis[, -5]
  refit <- cox_ph(Surv(time, status) ~ kept + sex, data = lung)
  diff <- bspline_basis(curve$grid, knots)[, -5] -
    rep(bspline_basis(curve$reference, knots)[, -5], each = 25)
  se <- sqrt(rowSums((diff %*% refit$var[1:4, 1:4]) * diff))
  expect_equal(curve$log_hr, drop(diff %*% refit$coefficients[1:4]), tolerance = 1e-8)
  expect_equal(curve$log_hr_upper - curve$log_hr, stats::qnorm(0.975) * se, tolerance = 1e-8)
})

test_that("bspline_basis() is a cubic B-spline basis", {
  knots <- c(rep(0, 4), 0.3, 0.6, rep(1, 4))
  x <- c(0, 0.1, 0.3, 0.45, 0.99, 1)
  basis <- bspline_basis(x, knots)
  expect_identical(dim(basis), c(6L, 6L))
  expect_equal(rowSums(basis), rep(1, 6))
  expect_true(all(basis >= 0))
  expect_equal(basis[1, ], c(1, 0, 0, 0, 0, 0))
  expect_equal(basis[6, ], c(0, 0, 0, 0, 0, 1))
})

test_that("smooth_hr() keeps the strata, weights, entry times, and clusters", {
  lung2 <- lung
  lung2$w <- rep(c(1, 2), length.out = nrow(lung2))
  lung2$entry <- pmin(lung2$time / 4, 20)
  stratified <- cox_ph(Surv(time, status) ~ age + strata(sex), data = lung2)
  curve <- smooth_hr(stratified, term = "age", n_grid = 5)
  expect_length(curve$adjustment, 0L)
  expect_true(all(is.finite(curve$log_hr_upper)))

  weighted <- cox_ph(Surv(time, status) ~ age + sex, data = lung2, weights = w)
  curve <- smooth_hr(weighted, term = "age", n_grid = 5)
  expect_equal(curve$reference, sum(lung2$w * lung2$age) / sum(lung2$w))
  expect_false(isTRUE(all.equal(curve$log_hr, smooth_hr(lung_fit(), "age", n_grid = 5)$log_hr)))

  counting <- cox_ph(Surv(entry, time, status) ~ age + sex, data = lung2)
  expect_true(all(is.finite(smooth_hr(counting, term = "age", n_grid = 5)$log_hr)))

  clustered <- cox_ph(Surv(time, status) ~ age + sex, data = lung2, cluster = inst)
  plain <- smooth_hr(lung_fit(), term = "age", n_grid = 5)
  robust <- smooth_hr(clustered, term = "age", n_grid = 5)
  expect_false(isTRUE(all.equal(robust$log_hr_upper, plain$log_hr_upper)))
})

test_that("smooth_hr() has a table, a print method, and checks its arguments", {
  curve <- smooth_hr(lung_fit(), term = "age", n_grid = 10)
  table <- as.data.frame(curve)
  expect_named(table, c("age", "log_hr", "conf_low", "conf_high"))
  expect_identical(nrow(table), 10L)
  table <- as.data.frame(curve, scale = "hr")
  expect_named(table, c("age", "hr", "conf_low", "conf_high"))
  expect_equal(table$conf_high, curve$hr_upper)
  printed <- capture.output(print(curve))
  expect_match(printed[1], "Smooth hazard ratio for age")
  expect_true(any(grepl("6 of 10 grid points", printed)))

  fit <- lung_fit()
  expect_error(smooth_hr(fit, term = "weight"), "must be one of")
  expect_error(smooth_hr(fit, term = "age", df = 2), "df")
  expect_error(smooth_hr(kaplan_meier(Surv(time, status) ~ 1, data = lung), "age"), "cox_ph")
  frail <- cox_ph(
    Surv(time, status) ~ age, data = lung, ties = "breslow", frailty = "gamma",
    frailty_cluster = inst
  )
  expect_error(smooth_hr(frail, term = "age"), "frailty")
})
