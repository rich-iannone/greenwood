# Checks bootstrap() against the closed-form estimates and against the bootstrap standard errors
# of the Python package. The two packages draw different resamples, so the standard errors agree
# only to within the Monte Carlo error.

lung_data <- function() survival::lung

test_that("the estimates and standard errors match the Python package", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  # Python's values with 4000 resamples.
  cases <- list(
    list(statistic = "median", estimate = 310, se = 23.28331222478965),
    list(statistic = "survival", times = 365, estimate = 0.40924162446006446,
      se = 0.03549821498836867),
    list(statistic = "quantile", p = 0.25, estimate = 170, se = 12.088339895447055),
    list(statistic = "rmst", tau = 365, estimate = 263.22186648200665, se = 7.826934568882382)
  )
  for (case in cases) {
    set.seed(10)
    result <- bootstrap(Surv(time, status) ~ 1,
      data = lung, statistic = case$statistic, tau = case$tau, p = case$p, times = case$times,
      n_boot = 1000
    )
    expect_equal(result$estimate, case$estimate, tolerance = 1e-12)
    expect_equal(result$std_error, case$se, tolerance = 0.1)
  }

  diffs <- list(
    list(statistic = "median_diff", estimate = -156, se = 61.721595473844964),
    list(statistic = "rmst_diff", tau = 365, estimate = -55.97032433997978,
      se = 14.794917448549482),
    list(statistic = "survival_diff", times = 365, estimate = -0.19037519554652693,
      se = 0.07352039901473938)
  )
  for (case in diffs) {
    set.seed(20)
    result <- bootstrap(Surv(time, status) ~ sex,
      data = lung, statistic = case$statistic, tau = case$tau, times = case$times, n_boot = 1000
    )
    expect_equal(result$estimate, case$estimate, tolerance = 1e-12)
    expect_equal(result$std_error, case$se, tolerance = 0.15)
    expect_identical(result$groups, c("sex=1", "sex=2"))
  }
})

test_that("the estimates are the Kaplan-Meier summaries", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  fit <- kaplan_meier(Surv(time, status) ~ sex, data = lung)
  set.seed(1)
  result <- bootstrap(Surv(time, status) ~ sex,
    data = lung, statistic = "rmst_diff", tau = 500, n_boot = 20
  )
  expect_equal(result$estimate, diff(rev(rmst(fit, tau = 500)$estimate)))
  medians <- median(fit)$time
  set.seed(1)
  result <- bootstrap(Surv(time, status) ~ sex, data = lung, statistic = "median_diff", n_boot = 20)
  expect_equal(result$estimate, medians[1] - medians[2])

  # The bootstrap standard error of S(365) is close to Greenwood's.
  set.seed(2)
  result <- bootstrap(Surv(time, status) ~ 1,
    data = lung, statistic = "survival", times = 365, n_boot = 1000
  )
  greenwood <- summary(kaplan_meier(Surv(time, status) ~ 1, data = lung), times = 365)$std_error
  expect_equal(result$std_error, greenwood, tolerance = 0.1)
})

test_that("the intervals are read from the replicates", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  set.seed(3)
  result <- bootstrap(Surv(time, status) ~ 1, data = lung, statistic = "rmst", tau = 365, n_boot = 200)
  expect_length(result$distribution, 200L)
  expect_equal(result$std_error, stats::sd(result$distribution))
  expect_equal(c(result$conf_low, result$conf_high),
    unname(stats::quantile(result$distribution, c(0.025, 0.975)))
  )

  set.seed(3)
  normal <- bootstrap(Surv(time, status) ~ 1,
    data = lung, statistic = "rmst", tau = 365, n_boot = 200, conf_level = 0.9, ci_type = "normal"
  )
  expect_identical(normal$distribution, result$distribution)
  expect_equal(normal$conf_high - normal$estimate, stats::qnorm(0.95) * normal$std_error)
  expect_equal(normal$estimate - normal$conf_low, stats::qnorm(0.95) * normal$std_error)

  # The BCa limits, from the bias correction and the jackknife acceleration.
  set.seed(3)
  bca <- bootstrap(Surv(time, status) ~ 1,
    data = lung, statistic = "rmst", tau = 365, n_boot = 200, ci_type = "bca"
  )
  n <- nrow(lung)
  jackknife <- vapply(seq_len(n), function(i) {
    rmst(kaplan_meier(Surv(time, status) ~ 1, data = lung[-i, ]), tau = 365)$estimate
  }, numeric(1))
  d <- mean(jackknife) - jackknife
  a <- sum(d^3) / (6 * sum(d^2)^1.5)
  z0 <- stats::qnorm(mean(bca$distribution < bca$estimate))
  z <- stats::qnorm(c(0.025, 0.975))
  probs <- stats::pnorm(z0 + (z0 + z) / (1 - a * (z0 + z)))
  expect_equal(c(bca$conf_low, bca$conf_high), unname(stats::quantile(bca$distribution, probs)))

  # Python's BCa limits for S(365) with 4000 resamples are 0.3414 and 0.4811.
  set.seed(4)
  bca <- bootstrap(Surv(time, status) ~ 1,
    data = lung, statistic = "survival", times = 365, n_boot = 1000, ci_type = "bca"
  )
  expect_equal(c(bca$conf_low, bca$conf_high), c(0.34139571703736393, 0.4811191488100924),
    tolerance = 0.03
  )
})

test_that("resampling follows the random seed and keeps the group sizes", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  set.seed(5)
  first <- bootstrap(Surv(time, status) ~ 1, data = lung, statistic = "median", n_boot = 50)
  set.seed(5)
  second <- bootstrap(Surv(time, status) ~ 1, data = lung, statistic = "median", n_boot = 50)
  expect_identical(first, second)

  sizes <- function(fit) vapply(fit$curves, function(curve) curve$n_risk[1L], numeric(1))
  seen <- list()
  record <- function(fit) {
    seen[[length(seen) + 1L]] <<- sizes(fit)
    0
  }
  set.seed(6)
  bootstrap(Surv(time, status) ~ sex, data = lung, statistic = record, n_boot = 10)
  # A custom statistic resamples the whole sample, so the group sizes vary.
  expect_true(length(unique(lapply(seen[-1L], identity))) > 1L)

  # Whole-number weights act as copies of the rows, so doubling them changes nothing.
  lung$w <- 2
  set.seed(7)
  weighted <- bootstrap(Surv(time, status) ~ sex,
    data = lung, statistic = "survival_diff", times = 365, weights = w, n_boot = 30
  )
  set.seed(7)
  plain <- bootstrap(Surv(time, status) ~ sex,
    data = lung, statistic = "survival_diff", times = 365, n_boot = 30
  )
  expect_equal(weighted$distribution, plain$distribution)
})

test_that("a custom statistic gets a Kaplan-Meier fit", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  half_year <- function(fit) predict(fit, times = 182.5)$estimate
  set.seed(8)
  custom <- bootstrap(Surv(time, status) ~ 1, data = lung, statistic = half_year, n_boot = 50)
  set.seed(8)
  builtin <- bootstrap(Surv(time, status) ~ 1,
    data = lung, statistic = "survival", times = 182.5, n_boot = 50
  )
  expect_identical(custom$statistic, "custom")
  expect_equal(custom$distribution, builtin$distribution)
  expect_equal(custom$estimate, builtin$estimate)

  # A replicate that fails is missing and left out. This one fails when the resample leaves out
  # the earliest time, 5 days.
  earliest <- function(fit) if (min(fit$curves[[1L]]$time) > 5) stop("no early time") else 1
  set.seed(8)
  failed <- bootstrap(Surv(time, status) ~ 1, data = lung, statistic = earliest, n_boot = 20)
  expect_true(anyNA(failed$distribution))
  expect_true(all(failed$distribution == 1, na.rm = TRUE))
  expect_identical(failed$std_error, 0)
  # On the original data, the error isn't caught.
  expect_error(
    bootstrap(Surv(time, status) ~ 1, data = lung, statistic = function(fit) stop("no fit")),
    "no fit"
  )
})

test_that("the result prints and converts to a data frame", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  set.seed(9)
  result <- bootstrap(Surv(time, status) ~ sex,
    data = lung, statistic = "rmst_diff", tau = 365, n_boot = 20
  )
  frame <- as.data.frame(result)
  expect_named(frame, c("estimate", "std_error", "conf_low", "conf_high"))
  expect_identical(nrow(frame), 1L)
  expect_output(print(result), "sex=1 minus sex=2")
  expect_output(print(result), "20 resamples")

  # Medians the curve never reaches are missing, and the printout counts them.
  short <- data.frame(time = c(1, 2, 3, 4, 5, 6), status = c(1, 0, 0, 0, 0, 1))
  set.seed(9)
  result <- bootstrap(Surv(time, status) ~ 1, data = short, statistic = "median", n_boot = 50)
  expect_true(anyNA(result$distribution))
  expect_output(print(result), "couldn't be computed")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- lung_data()
  km <- Surv(time, status) ~ 1
  by_sex <- Surv(time, status) ~ sex
  expect_error(bootstrap(km, data = lung), "needs a `statistic`")
  expect_error(bootstrap(km, data = lung, statistic = "mean"), "one of")
  expect_error(bootstrap(km, data = lung, statistic = "rmst"), "needs `tau`")
  expect_error(bootstrap(km, data = lung, statistic = "quantile"), "needs `p`")
  expect_error(bootstrap(km, data = lung, statistic = "quantile", p = 1.5), "`p`")
  expect_error(bootstrap(km, data = lung, statistic = "survival"), "needs `times`")
  expect_error(bootstrap(km, data = lung, statistic = "median_diff"), "grouping variable")
  expect_error(bootstrap(by_sex, data = lung, statistic = "median"), "median_diff")
  expect_error(bootstrap(by_sex, data = lung, statistic = "quantile", p = 0.5), "no grouping")
  expect_error(
    bootstrap(Surv(time, status) ~ ph.ecog, data = lung, statistic = "median_diff"),
    "two groups"
  )
  expect_error(bootstrap(km, data = lung, statistic = "median", n_boot = 0), "`n_boot`")
  expect_error(bootstrap(km, data = lung, statistic = "median", n_boot = 2.5), "whole number")
  expect_error(bootstrap(km, data = lung, statistic = "median", ci_type = "basic"), "should be one of")
  expect_error(bootstrap(km, data = lung, statistic = function(fit) 1:2), "single number")
  expect_error(
    bootstrap(Surv(time, status) ~ strata(sex), data = lung, statistic = "median"),
    "strata"
  )
})
