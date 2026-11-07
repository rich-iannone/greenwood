# Replays the log-rank fixtures in fixtures/r/ (survdiff() results and a direct computation of the
# MaxCombo statistics in the generator).

TOL <- 1e-9

expect_groups_match <- function(result, expected) {
  expect_identical(result$groups$group, paste0(names(result$groups)[2], "=", r_strings(expected$groups)))
  expect_equal(result$groups$n, r_numbers(expected$n))
  expect_equal(result$groups$observed, r_numbers(expected$obs), tolerance = TOL)
  expect_equal(result$groups$expected, r_numbers(expected$exp), tolerance = TOL)
}

test_that("the log-rank test on lung matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("logrank_lung_sex")
  result <- logrank_test(Surv(time, status) ~ sex, data = survival::lung)
  expect_groups_match(result, expected)
  expect_equal(result$statistic, expected$chisq, tolerance = TOL)
  expect_identical(result$df, 1L)
  expect_equal(result$p_value, expected$p, tolerance = TOL)
  expect_identical(result$method, "Log-rank test")
})

test_that("the G-rho test matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("grho_lung_sex_rho1")
  result <- logrank_test(Surv(time, status) ~ sex, data = survival::lung, rho = 1)
  expect_groups_match(result, expected)
  expect_equal(result$statistic, expected$chisq, tolerance = TOL)
  expect_equal(result$p_value, expected$p, tolerance = TOL)
  expect_identical(result$method, "G-rho test (rho = 1, gamma = 0)")
})

test_that("a test across four groups matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("logrank_veteran_celltype")
  result <- logrank_test(Surv(time, status) ~ celltype, data = survival::veteran)
  expect_groups_match(result, expected)
  expect_equal(result$statistic, expected$chisq, tolerance = TOL)
  expect_identical(result$df, 3L)
  expect_equal(result$p_value, expected$p, tolerance = TOL)
})

test_that("the stratified test matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("logrank_stratified_lung_sex_ecog")
  result <- logrank_test(Surv(time, status) ~ sex + strata(ph.ecog), data = survival::lung)
  expect_equal(result$statistic, expected$chisq, tolerance = TOL)
  expect_identical(result$df, as.integer(expected$df))
  expect_equal(result$p_value, expected$p, tolerance = TOL)
  expect_identical(result$method, "Stratified log-rank test")
  expect_identical(result$n_dropped, 1L)
})

test_that("pairwise tests match the fixture for every correction", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("pairwise_logrank_veteran")
  key <- function(a, b) paste(pmin(a, b), pmax(a, b))
  want <- key(r_strings(expected$group1), r_strings(expected$group2))
  for (correction in c("holm", "bh", "bonferroni")) {
    result <- pairwise_logrank_test(
      Surv(time, status) ~ celltype,
      data = survival::veteran,
      correction = correction
    )
    expect_named(result, c("group1", "group2", "statistic", "p_value", "p_adjusted"))
    rows <- match(want, key(result$group1, result$group2))
    expect_false(anyNA(rows))
    expect_equal(result$p_value[rows], r_numbers(expected$p_value), tolerance = TOL)
    expect_equal(result$p_adjusted[rows], r_numbers(expected[[correction]]), tolerance = TOL)
  }
})

test_that("a pairwise test uses only the rows of the two groups", {
  three <- transform(small, grade = rep(c(0, 1, 2), length.out = 10))
  pairs <- pairwise_logrank_test(Surv(time, status) ~ grade, data = three, correction = "none")
  expect_identical(pairs$group1, c("0", "0", "1"))
  expect_identical(pairs$group2, c("1", "2", "2"))
  direct <- logrank_test(Surv(time, status) ~ grade, data = three, subset = grade %in% c(0, 2))
  expect_equal(pairs$statistic[2], direct$statistic)
  expect_identical(pairs$p_adjusted, pairs$p_value)
})

test_that("the MaxCombo statistics match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("maxcombo_lung_sex")
  result <- maxcombo_test(Surv(time, status) ~ sex, data = survival::lung)
  expect_equal(result$z$z, unname(r_numbers(expected$z_statistics)), tolerance = TOL)
  correlation <- do.call(rbind, lapply(expected$correlation, r_numbers))
  expect_equal(unname(result$correlation), correlation, tolerance = TOL)
  expect_equal(result$statistic, expected$statistic, tolerance = TOL)
  expect_true(result$p_value > 0 && result$p_value < 2 * pnorm(-expected$statistic) * 4)
})

test_that("MaxCombo with one weight function is the log-rank test", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  one <- maxcombo_test(Surv(time, status) ~ sex, data = lung, rho = 1, gamma = 0)
  grho <- logrank_test(Surv(time, status) ~ sex, data = lung, rho = 1)
  expect_equal(one$statistic^2, grho$statistic)
  expect_equal(one$p_value, grho$p_value)
  expect_error(maxcombo_test(Surv(time, status) ~ ph.ecog, data = lung), "exactly two groups")
  expect_error(maxcombo_test(Surv(time, status) ~ sex, data = lung, rho = c(0, 1), gamma = 0), "same length")
})

test_that("the multivariate normal probability is accurate", {
  # For equicorrelated statistics the probability is a one-dimensional integral.
  for (case in list(c(rho = 0.6, k = 4, c = 2.1), c(rho = 0.9, k = 3, c = 2.8), c(rho = 0.2, k = 2, c = 1.5))) {
    rho <- case[["rho"]]
    k <- case[["k"]]
    cut <- case[["c"]]
    corr <- matrix(rho, k, k)
    diag(corr) <- 1
    exact <- integrate(function(u) {
      dnorm(u) * (pnorm((cut - sqrt(rho) * u) / sqrt(1 - rho)) -
        pnorm((-cut - sqrt(rho) * u) / sqrt(1 - rho)))^k
    }, -Inf, Inf, rel.tol = 1e-12)$value
    expect_equal(pmvnorm_box(cut, corr), exact, tolerance = 1e-4)
  }
  expect_equal(pmvnorm_box(1.96, diag(2)), (2 * pnorm(1.96) - 1)^2, tolerance = 1e-5)
  expect_equal(pmvnorm_box(1.96, matrix(1, 2, 2)), 2 * pnorm(1.96) - 1, tolerance = 1e-5)
})

test_that("the trend test on lung matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("trend_lung_ecog")
  fits <- list(
    linear = trend_test(Surv(time, status) ~ ph.ecog, data = survival::lung),
    quadratic = trend_test(Surv(time, status) ~ ph.ecog, data = survival::lung,
      scores = c(0, 1, 4, 9)
    ),
    peto = trend_test(Surv(time, status) ~ ph.ecog, data = survival::lung, rho = 1),
    stratified = trend_test(Surv(time, status) ~ ph.ecog + strata(sex), data = survival::lung)
  )
  for (case in names(fits)) {
    fit <- fits[[case]]
    want <- expected[[case]]
    expect_equal(fit$groups$score, r_numbers(want$scores))
    expect_equal(fit$groups$expected, r_numbers(want$expected), tolerance = TOL)
    expect_equal(fit$statistic, want$statistic, tolerance = TOL)
    expect_equal(fit$p_value, want$p, tolerance = 1e-8)
  }
})

test_that("the trend test is the Cox score test without ties", {
  skip_if_not_installed("survival")
  set.seed(1)
  n <- 80
  data <- data.frame(
    time = rexp(n),
    status = rbinom(n, 1, 0.8),
    dose = sample(c(0, 1, 2, 3), n, replace = TRUE),
    site = sample(c("a", "b"), n, replace = TRUE)
  )
  result <- trend_test(Surv(time, status) ~ dose, data = data)
  cox <- survival::coxph(survival::Surv(time, status) ~ dose, data = data)
  expect_equal(result$statistic, unname(cox$score), tolerance = 1e-10)
  expect_identical(result$groups$score, c(0, 1, 2, 3))

  scores <- c(0, 1, 4, 9)
  custom <- trend_test(Surv(time, status) ~ dose, data = data, scores = scores)
  cox <- survival::coxph(survival::Surv(time, status) ~ I(dose^2), data = data)
  expect_equal(custom$statistic, unname(cox$score), tolerance = 1e-10)

  stratified <- trend_test(Surv(time, status) ~ dose + strata(site), data = data)
  # coxph() finds strata() by name, so the formula is evaluated where survival's is visible.
  cox <- eval(
    quote(coxph(Surv(time, status) ~ dose + strata(site), data = data)),
    list(data = data),
    asNamespace("survival")
  )
  expect_equal(stratified$statistic, unname(cox$score), tolerance = 1e-10)
  expect_identical(stratified$method, "Stratified log-rank test for trend")

  # Scores can be shifted, scaled, or named.
  expect_equal(trend_test(Surv(time, status) ~ dose, data = data, scores = c(5, 7, 9, 11))$statistic, result$statistic)
  named <- trend_test(Surv(time, status) ~ dose, data = data, scores = c("3" = 9, "0" = 0, "1" = 1, "2" = 4))
  expect_equal(named$statistic, custom$statistic)
})

test_that("the trend test with two groups is the log-rank test", {
  trend <- trend_test(Surv(time, status) ~ sex, data = small)
  logrank <- logrank_test(Surv(time, status) ~ sex, data = small)
  expect_equal(trend$statistic, logrank$statistic)
  expect_error(trend_test(Surv(time, status) ~ sex + ecog, data = small), "one grouping variable")
  expect_error(trend_test(Surv(time, status) ~ sex, data = small, scores = 1:3), "one score per group")
  expect_error(trend_test(Surv(time, status) ~ sex, data = small, scores = c(a = 1, b = 2)), "no score for")
})

test_that("weights, subsets, and counting-process data are supported", {
  w <- c(2, 1, 3, 1, 1, 2, 1, 1, 2, 1)
  weighted <- logrank_test(Surv(time, status) ~ sex, data = small, weights = w)
  replicated <- logrank_test(Surv(time, status) ~ sex, data = small[rep(seq_along(w), w), ])
  expect_equal(weighted$statistic, replicated$statistic)
  expect_equal(weighted$groups$observed, replicated$groups$observed)

  entry <- transform(small, start = 0)
  counting <- logrank_test(Surv(start, time, status) ~ sex, data = entry)
  expect_equal(counting$statistic, logrank_test(Surv(time, status) ~ sex, data = small)$statistic)
})

test_that("tests check their input", {
  expect_error(logrank_test(Surv(time, status) ~ 1, data = small), "grouping variables")
  expect_error(logrank_test(Surv(time, status) ~ sex, data = small, subset = sex == 1), "at least 2 groups")
  expect_error(logrank_test(Surv(time, status * 0) ~ sex, data = small), "at least one event")
  expect_error(logrank_test(Surv(time, status) ~ sex, data = small, rho = -1), "no less than 0")
  expect_error(pairwise_logrank_test(Surv(time, status) ~ sex, data = small, correction = "fdr"))
})

test_that("results print and convert to data frames", {
  result <- logrank_test(Surv(time, status) ~ sex, data = small)
  expect_named(as.data.frame(result), c("method", "statistic", "df", "p_value"))
  combo <- maxcombo_test(Surv(time, status) ~ sex, data = small)
  expect_named(as.data.frame(combo), c("method", "statistic", "p_value"))
  expect_snapshot({
    result
    logrank_test(Surv(time, status) ~ sex + strata(ecog), data = small, rho = 1)
    trend_test(Surv(time, status) ~ ecog, data = small)
    combo
  })
})
