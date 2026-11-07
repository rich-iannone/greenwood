# Replays the Turnbull fixture in fixtures/r/ (turnbull_veteran_km_equivalence) and checks the
# interval-censored behavior of turnbull() directly.

visits <- data.frame(
  lower = c(0, 4, 7, 0, 3, 5),
  upper = c(4, NA, 7, 2.5, 6, 5)
)

test_that("with no interval censoring, the estimate matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("turnbull_veteran_km_equivalence")
  fit <- turnbull(Surv(time, status) ~ 1, data = survival::veteran)
  p <- predict(fit, times = r_numbers(expected$time))
  # The reference EM stops at a looser tolerance than this one.
  expect_equal(p$estimate, r_numbers(expected$surv), tolerance = 1e-6)
})

test_that("with exact and right-censored times, the estimate is Kaplan-Meier", {
  data <- data.frame(time = 1:5, status = c(1, 1, 0, 1, 0))
  fit <- turnbull(Surv(time, status) ~ 1, data = data)
  km <- kaplan_meier(Surv(time, status) ~ 1, data = data)
  times <- c(0.5, 1:5, 6)
  expect_equal(predict(fit, times = times)$estimate, predict(km, times = times)$estimate,
    tolerance = 1e-8
  )
  expect_equal(rmst(fit, tau = 4)$estimate, rmst(km, tau = 4)$estimate, tolerance = 1e-8)
  expect_equal(rmrl(fit, s = 2, tau = 4)$estimate, rmrl(km, s = 2, tau = 4)$estimate,
    tolerance = 1e-8
  )

  exact <- turnbull(Surv(time, status) ~ 1, data = data.frame(time = 1:3, status = 1))
  table <- as.data.frame(exact)
  expect_equal(table$estimate, c(2, 1, 0) / 3, tolerance = 1e-10)
  expect_identical(table$interval_low, table$interval_high)
})

test_that("mixed censoring gives a proper distribution", {
  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
  table <- as.data.frame(fit)
  expect_identical(names(table), c("strata", "interval_low", "interval_high", "prob_mass", "estimate"))
  expect_identical(unique(table$strata), "all")
  expect_equal(sum(table$prob_mass), 1, tolerance = 1e-6)
  expect_true(all(diff(table$estimate) <= 1e-9))
  expect_equal(table$estimate[nrow(table)], 0, tolerance = 1e-6)
  expect_true(fit$curves[[1]]$converged)

  # The exact deaths at 5 and 7 are point intervals.
  points <- table$interval_high[table$interval_low == table$interval_high]
  expect_true(all(c(5, 7) %in% points))
  # The masses match the reference implementation.
  expect_equal(table$prob_mass[c(1, 5, 8)], c(1 / 3, 4 / 9, 2 / 9), tolerance = 1e-8)
})

test_that("a left-censored observation puts its mass before its bound", {
  data <- data.frame(time = c(3, 10), code = c(2, 0))
  fit <- turnbull(Surv(time, time, code, type = "interval") ~ 1, data = data)
  expect_true(all(as.data.frame(fit)$interval_high <= 3))

  # type = "left" codes status 0 as left-censored.
  left <- turnbull(Surv(time, status, type = "left") ~ 1,
    data = data.frame(time = c(2, 3, 5), status = c(1, 0, 1))
  )
  expect_true(all(as.data.frame(left)$interval_high <= 5))
  expect_equal(sum(as.data.frame(left)$prob_mass), 1, tolerance = 1e-8)
})

test_that("quantiles report the bounds of an ambiguous crossing", {
  wide <- turnbull(Surv(lower, upper, type = "interval2") ~ 1,
    data = data.frame(lower = c(0, 0), upper = c(10, 10))
  )
  q <- quantile(wide, probs = 0.5)
  expect_identical(names(q), c("strata", "prob", "time", "time_low", "time_high"))
  expect_true(is.na(q$time))
  expect_identical(c(q$time_low, q$time_high), c(0, 10))

  exact <- turnbull(Surv(time, status) ~ 1, data = data.frame(time = 1:4, status = 1))
  med <- median(exact)
  expect_identical(names(med), c("strata", "time", "time_low", "time_high"))
  expect_identical(c(med$time, med$time_low, med$time_high), c(2, 2, 2))

  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
  q <- quantile(fit, probs = c(0.1, 0.5, 0.9))
  expect_identical(nrow(q), 3L)
  expect_identical(q$time, c(NA, 5, 7))
  expect_identical(q$time_low, c(0, 5, 7))
  expect_identical(q$time_high, c(2.5, 5, 7))

  never <- turnbull(Surv(time, status) ~ 1, data = data.frame(time = 1:2, status = c(1, 0)))
  q <- quantile(never, probs = 0.99)
  expect_true(all(is.na(c(q$time, q$time_low, q$time_high))))
  expect_error(quantile(fit, probs = 2), "`probs`")
})

test_that("predict() is NA strictly inside an ambiguous interval", {
  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
  p <- predict(fit, times = c(1, 2.5, 5, 7))
  expect_identical(names(p), c("strata", "time", "estimate"))
  expect_true(is.na(p$estimate[1]))
  expect_false(anyNA(p$estimate[-1]))
  expect_equal(p$estimate[2:3], c(2 / 3, 2 / 9), tolerance = 1e-8)

  early <- turnbull(Surv(time, status) ~ 1, data = data.frame(time = 2:3, status = 1))
  expect_identical(predict(early, times = 0)$estimate, 1)
})

test_that("restricted means place the mass at the upper ends", {
  wide <- turnbull(Surv(lower, upper, type = "interval2") ~ 1,
    data = data.frame(lower = c(0, 0), upper = c(10, 10))
  )
  expect_equal(rmst(wide, tau = 10)$estimate, 10)
  expect_equal(rmst(wide, tau = 5)$estimate, 5)

  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
  expect_equal(rmrl(fit, s = 0, tau = 7)$estimate, rmst(fit, tau = 7)$estimate)
  expect_equal(rmst(fit, tau = 7)$estimate, 2.5 + 2.5 * 2 / 3 + 2 * 2 / 9, tolerance = 1e-8)
  expect_identical(names(rmrl(fit, s = 4, tau = 7)), c("strata", "s", "tau", "estimate"))

  done <- turnbull(Surv(time, status) ~ 1, data = data.frame(time = 1:2, status = 1))
  expect_true(is.na(rmrl(done, s = 2, tau = 5)$estimate))
  expect_error(rmrl(done, s = 5, tau = 4), "`tau`")
  expect_error(rmrl(done, s = -1, tau = 4), "`s`")
})

test_that("groups are fit separately", {
  data <- data.frame(time = c(1, 2, 1, 2), status = 1, arm = c("a", "a", "b", "b"))
  fit <- turnbull(Surv(time, status) ~ arm, data = data)
  expect_identical(rmst(fit, tau = 2)$strata, c("arm=a", "arm=b"))
  expect_identical(names(median(fit)), c("strata", "arm", "time", "time_low", "time_high"))
  expect_identical(unique(as.data.frame(fit)$arm), c("a", "b"))

  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ group,
    data = rbind(transform(visits, group = "x"), transform(visits, group = "y", upper = upper + 1))
  )
  alone <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = transform(visits,
    upper = upper + 1
  ))
  table <- as.data.frame(fit)
  expect_equal(table$prob_mass[table$group == "y"], as.data.frame(alone)$prob_mass)
})

test_that("weights act like repeated rows", {
  lower <- c(0, 4, 0, 3)
  upper <- c(4, NA, 2.5, 6)
  repeated <- turnbull(Surv(lower, upper, type = "interval2") ~ 1,
    data = data.frame(lower = c(lower, 0), upper = c(upper, 2.5))
  )
  weighted <- turnbull(Surv(lower, upper, type = "interval2") ~ 1,
    data = data.frame(lower = lower, upper = upper, w = c(1, 1, 2, 1)), weights = w
  )
  expect_equal(as.data.frame(weighted)$estimate, as.data.frame(repeated)$estimate,
    tolerance = 1e-6
  )
})

test_that("the EM increases the likelihood and converges", {
  support <- tb_support(c(0, 4, 7, 0, 3, 5), c(4, Inf, 7, 2.5, 6, 5))
  a <- support$alpha * 1
  loglik <- vapply(1:30, function(k) {
    s <- tb_em(support$alpha, rep(1, 6), tol = 1e-14, max_iter = k)$s
    sum(log(drop(a %*% s)))
  }, numeric(1))
  expect_true(all(diff(loglik) >= -1e-10))

  em <- tb_em(support$alpha, rep(1, 6), tol = 1e-10, max_iter = 10000)
  expect_true(em$converged)
  expect_lt(em$n_iter, 10000)
  expect_equal(sum(em$s), 1)

  capped <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits, max_iter = 2)
  expect_false(capped$curves[[1]]$converged)
  expect_identical(capped$curves[[1]]$n_iter, 2L)
})

test_that("bad input is rejected", {
  counting <- data.frame(start = c(0, 1), stop = c(2, 3), status = c(1, 0))
  expect_error(turnbull(Surv(start, stop, status) ~ 1, data = counting), "delayed entry")
  expect_error(turnbull(Surv(time, event) ~ 1, data = small_cr), "multi-state")
  expect_error(turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits, tol = 0), "`tol`")
  expect_error(turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits, max_iter = 0),
    "`max_iter`"
  )
  expect_error(
    turnbull(Surv(lower, upper, type = "interval2") ~ strata(lower), data = visits),
    "`strata()`",
    fixed = TRUE
  )
  fit <- turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)
  expect_error(predict(fit, times = "a"), "`times`")
})

test_that("the printout is stable", {
  expect_snapshot(print(turnbull(Surv(lower, upper, type = "interval2") ~ 1, data = visits)))
})
