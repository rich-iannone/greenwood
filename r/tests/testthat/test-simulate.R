# Checks the layout of simulate_competing_risks() and recovers its parameters with the package's
# own estimators.

test_that("the observed data follow from the latent times", {
  set.seed(1)
  sim <- simulate_competing_risks(n = 200, n_causes = 3, n_covariates = 2)
  expect_named(sim, c("time", "event", "x1", "x2"))
  expect_identical(levels(sim$event), c("censor", "cause1", "cause2", "cause3"))
  latent <- attr(sim, "latent_times")
  censoring <- attr(sim, "censoring_time")
  expect_identical(dim(latent), c(200L, 3L))
  expect_identical(colnames(latent), c("cause1", "cause2", "cause3"))
  expect_identical(sim$time, pmin(apply(latent, 1L, min), censoring))
  first <- apply(latent, 1L, which.min)
  censored <- censoring < apply(latent, 1L, min)
  expect_identical(as.integer(sim$event) - 1L, ifelse(censored, 0L, first))
  expect_identical(dim(attr(sim, "coef")), c(2L, 3L))
  expect_true(all(abs(attr(sim, "coef")) <= 0.5))

  set.seed(1)
  again <- simulate_competing_risks(n = 200, n_causes = 3, n_covariates = 2)
  expect_identical(again, sim)

  # Without censoring, every subject has an event.
  plain <- simulate_competing_risks(n = 50, censoring_scale = NULL, states = c("relapse", "death"))
  expect_false(any(plain$event == "censor"))
  expect_identical(attr(plain, "censoring_time"), rep(Inf, 50))
  expect_null(attr(plain, "coef"))
  expect_named(plain, c("time", "event"))
  expect_identical(sum(simulate_competing_risks(n = 50, censoring_scale = 0)$event == "censor"), 0L)
})

test_that("the estimators recover the simulation's parameters", {
  set.seed(2)
  beta <- cbind(c(0.5, -0.3), c(0, 0.4))
  sim <- simulate_competing_risks(n = 4000,
    shapes = c(1.2, 1), n_covariates = 2, coef = beta, states = c("relapse", "death")
  )
  # Each cause-specific Cox model recovers that cause's effects.
  relapse <- cox_ph(Surv(time, event == "relapse") ~ x1 + x2, data = sim)
  death <- cox_ph(Surv(time, event == "death") ~ x1 + x2, data = sim)
  # The standard errors are about 0.03.
  expect_lt(max(abs(coef(relapse) - beta[, 1])), 0.1)
  expect_lt(max(abs(coef(death) - beta[, 2])), 0.1)

  # With exponential times, the censored share is close to 1 / (1 + censoring_scale).
  set.seed(3)
  exponential <- simulate_competing_risks(n = 20000, shapes = c(1, 1), censoring_scale = 1.5)
  expect_equal(mean(exponential$event == "censor"), 0.4, tolerance = 0.02)

  # One cause without censoring is a Weibull sample.
  set.seed(4)
  weibull <- simulate_competing_risks(n = 4000,
    n_causes = 1, shapes = 2, scales = 5, censoring_scale = NULL
  )
  fit <- aft(Surv(time, rep(1, 4000)) ~ 1, data = weibull, dist = "weibull")
  expect_equal(exp(unname(coef(fit))[1]), 5, tolerance = 0.02)
  expect_equal(1 / fit$scale, 2, tolerance = 0.05)
})

test_that("bad input is rejected", {
  expect_error(simulate_competing_risks(n = 0), "`n`")
  expect_error(simulate_competing_risks(n = 10.5), "`n`")
  expect_error(simulate_competing_risks(n_causes = 0), "`n_causes`")
  expect_error(simulate_competing_risks(shapes = c(1, -1)), "`shapes`")
  expect_error(simulate_competing_risks(scales = 1), "`scales`")
  expect_error(simulate_competing_risks(states = "a"), "`states`")
  expect_error(simulate_competing_risks(states = c("censor", "death")), "distinct")
  expect_error(simulate_competing_risks(states = c("death", "death")), "distinct")
  expect_error(simulate_competing_risks(censoring_scale = -1), "`censoring_scale`")
  expect_error(simulate_competing_risks(n_covariates = 2, coef = matrix(0, 3, 2)), "2 by 2")
})
