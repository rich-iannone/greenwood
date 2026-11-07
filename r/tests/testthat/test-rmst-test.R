# Replays fixtures/r/rmst_twogroup_lung.json (restricted means from survfit() summaries).

TOL <- 1e-9

test_that("the two-group comparison on lung matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("rmst_twogroup_lung")
  result <- rmst_test(Surv(time, status) ~ sex, data = survival::lung, tau = expected$tau)
  expect_identical(c(result$group1, result$group2), c("1", "2"))
  expect_equal(result$rmst1, expected$rmst1, tolerance = TOL)
  expect_equal(result$se1, expected$se1, tolerance = TOL)
  expect_equal(result$rmst2, expected$rmst2, tolerance = TOL)
  expect_equal(result$se2, expected$se2, tolerance = TOL)
  expect_equal(result$estimate, expected$difference, tolerance = TOL)
  expect_equal(result$std_error, expected$se_difference, tolerance = TOL)
  expect_equal(result$p_value, 2 * pnorm(-abs(expected$difference / expected$se_difference)))
  expect_false(result$stratified)
})

test_that("the stratified comparison matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("rmst_twogroup_lung")
  result <- rmst_test(
    Surv(time, status) ~ sex + strata(ph.ecog),
    data = survival::lung,
    tau = expected$tau
  )
  stratified <- expected$stratified
  expect_equal(result$estimate, stratified$difference, tolerance = TOL)
  expect_equal(result$std_error, stratified$se, tolerance = TOL)
  expect_equal(result$statistic, stratified$statistic, tolerance = TOL)
  expect_equal(result$p_value, stratified$p_value, tolerance = 1e-12)
  expect_true(result$stratified)
  expect_identical(result$method, "Stratified RMST difference (tau = 365)")
})

test_that("the group RMSTs are those of rmst()", {
  fit <- kaplan_meier(Surv(time, status) ~ sex, data = small)
  each <- rmst(fit, tau = 500)
  result <- rmst_test(Surv(time, status) ~ sex, data = small, tau = 500)
  expect_equal(c(result$rmst1, result$rmst2), each$estimate)
  expect_equal(c(result$se1, result$se2), each$std_error)
})

test_that("ratios and percentage differences use the delta method", {
  difference <- rmst_test(Surv(time, status) ~ sex, data = small, tau = 500)
  ratio <- rmst_test(Surv(time, status) ~ sex, data = small, tau = 500, estimand = "ratio")
  r1 <- difference$rmst1
  r2 <- difference$rmst2
  se_log <- sqrt((difference$se1 / r1)^2 + (difference$se2 / r2)^2)
  expect_equal(ratio$estimate, r1 / r2)
  expect_equal(ratio$std_error, r1 / r2 * se_log)
  expect_equal(ratio$conf_low, exp(log(r1 / r2) - qnorm(0.975) * se_log))
  expect_equal(ratio$statistic, log(r1 / r2) / se_log)

  percent <- rmst_test(
    Surv(time, status) ~ sex,
    data = small,
    tau = 500,
    estimand = "percentage_difference",
    conf_level = 0.9
  )
  expect_equal(percent$estimate, 100 * (r1 - r2) / r2)
  expect_equal(percent$std_error, 100 * difference$std_error / r2)
  expect_equal(percent$conf_high, percent$estimate + qnorm(0.95) * percent$std_error)
  expect_equal(percent$p_value, difference$p_value)
})

test_that("rmst_diff() is the difference as a one-row data frame", {
  result <- rmst_test(Surv(time, status) ~ sex, data = small, tau = 500)
  table <- rmst_diff(Surv(time, status) ~ sex, data = small, tau = 500)
  expect_named(table, c(
    "group1", "group2", "rmst1", "rmst2", "difference", "std_error", "conf_low", "conf_high",
    "statistic", "p_value"
  ))
  expect_identical(nrow(table), 1L)
  expect_equal(table$difference, result$estimate)
  expect_equal(table$p_value, result$p_value)
})

test_that("pairwise comparisons cover every pair and adjust the p-values", {
  three <- transform(small, grade = rep(c(0, 1, 2), length.out = 10))
  pairs <- pairwise_rmst_test(Surv(time, status) ~ grade, data = three, tau = 500)
  expect_identical(pairs$group1, c("0", "0", "1"))
  expect_identical(pairs$group2, c("1", "2", "2"))
  expect_equal(pairs$p_adjusted, p.adjust(pairs$p_value, "holm"))
  direct <- rmst_test(Surv(time, status) ~ grade, data = three, tau = 500, subset = grade != 1)
  expect_equal(pairs$estimate[2], direct$estimate)
  expect_error(rmst_test(Surv(time, status) ~ grade, data = three, tau = 500), "pairwise_rmst_test")
})

test_that("comparisons check their input", {
  expect_error(rmst_test(Surv(time, status) ~ 1, data = small, tau = 500), "grouping variable")
  expect_error(rmst_test(Surv(time, status) ~ sex, data = small, tau = NA), "single number")
  expect_error(rmst_test(Surv(time, status) ~ sex, data = small, tau = 500, estimand = "odds"))
  expect_error(
    rmst_test(Surv(time, status) ~ sex + strata(ecog), data = small, tau = 500, subset = ecog == 0 & sex == 1),
    "at least 2 groups"
  )
})

test_that("a comparison prints", {
  expect_snapshot({
    rmst_test(Surv(time, status) ~ sex, data = small, tau = 500)
    rmst_test(Surv(time, status) ~ sex + strata(ecog), data = small, tau = 500, estimand = "ratio")
  })
})
