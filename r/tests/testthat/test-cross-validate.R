# Checks cross_validate() by rebuilding each fold's fit and score from the folds it records, and
# checks the folds themselves.

lung_complete <- function() {
  lung <- survival::lung
  lung[stats::complete.cases(lung[c("time", "status", "age", "sex", "ph.ecog")]), ]
}

test_that("each score is the metric of a fit to the other folds", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  times <- c(180, 365, 540)
  set.seed(1)
  result <- cross_validate(cox_ph, Surv(time, status) ~ age + sex + ph.ecog,
    data = lung, metrics = c("concordance", "brier", "auc"), times = times
  )
  expect_identical(result$k, 5L)
  expect_identical(result$scores$metric, rep(c("concordance", "brier", "auc"), times = 5))
  for (i in 1:5) {
    train <- lung[result$folds != i, ]
    test <- lung[result$folds == i, ]
    fit <- cox_ph(Surv(time, status) ~ age + sex + ph.ecog, data = train)
    death <- Surv(test$time, test$status)
    lp <- predict(fit, newdata = test)
    p <- predict(fit, newdata = test, type = "survival", times = times)
    probs <- matrix(p$estimate, ncol = 3, byrow = TRUE)
    fold <- result$scores[result$scores$fold == i, ]
    expect_equal(fold$score, c(
      concordance_index(death, risk = lp),
      integrated_brier_score(death, survival_prob = probs, times = times),
      integrated_auc(death, marker = lp, times = times)
    ))
  }
  by_metric <- split(result$scores$score, result$scores$metric)
  expect_equal(result$results$mean, vapply(by_metric[result$results$metric], mean, numeric(1)),
    ignore_attr = TRUE
  )
  expect_equal(result$results$sd, vapply(by_metric[result$results$metric], stats::sd, numeric(1)),
    ignore_attr = TRUE
  )
})

test_that("an AFT fit ranks by the negative linear predictor", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  set.seed(2)
  result <- cross_validate(aft, Surv(time, status) ~ age + ph.ecog,
    data = lung, dist = "lognormal", k = 3
  )
  for (i in 1:3) {
    fit <- aft(Surv(time, status) ~ age + ph.ecog, data = lung[result$folds != i, ], dist = "lognormal")
    test <- lung[result$folds == i, ]
    expect_equal(result$scores$score[i],
      concordance_index(Surv(test$time, test$status), risk = -predict(fit, newdata = test))
    )
  }
  expect_gt(result$results$mean, 0.5)
})

test_that("a Royston-Parmar fit is scored by its linear predictor and survival curves", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  set.seed(4)
  times <- c(150, 300, 450)
  result <- cross_validate(royston_parmar, Surv(time, status) ~ age + ph.ecog,
    data = lung, df = 2, scale = "odds", k = 3, metrics = c("concordance", "brier"), times = times
  )
  for (i in 1:3) {
    fit <- royston_parmar(Surv(time, status) ~ age + ph.ecog,
      data = lung[result$folds != i, ], df = 2, scale = "odds"
    )
    test <- lung[result$folds == i, ]
    response <- Surv(test$time, test$status)
    fold <- result$scores[result$scores$fold == i, ]
    expect_equal(fold$score[1],
      concordance_index(response, risk = predict(fit, newdata = test, type = "lp"))
    )
    survival <- predict(fit, newdata = test, type = "survival", times = times)
    probs <- matrix(survival$estimate, ncol = length(times), byrow = TRUE)
    expect_equal(fold$score[2], integrated_brier_score(response, survival_prob = probs, times = times))
  }
})

test_that("the folds are balanced and follow the random seed", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  set.seed(3)
  first <- cross_validate(cox_ph, Surv(time, status) ~ age, data = lung, k = 7)
  set.seed(3)
  second <- cross_validate(cox_ph, Surv(time, status) ~ age, data = lung, k = 7)
  expect_identical(first$folds, second$folds)
  expect_identical(first$scores, second$scores)

  sizes <- tabulate(first$folds, 7)
  expect_lte(max(sizes) - min(sizes), 1L)
  # Each event status is spread as evenly as the fold sizes allow.
  by_status <- table(first$folds, lung$status)
  expect_true(all(apply(by_status, 2L, function(x) max(x) - min(x)) <= 1L))

  set.seed(3)
  plain <- cross_validate(cox_ph, Surv(time, status) ~ age, data = lung, k = 7, stratified = FALSE)
  expect_lte(max(tabulate(plain$folds, 7)) - min(tabulate(plain$folds, 7)), 1L)

  # The folds rotate, so a small status doesn't leave the last folds empty.
  folds <- cv_folds(c(rep(0, 3), rep(1, 3)), k = 4, stratified = TRUE)
  expect_identical(sort(tabulate(folds, 4)), c(1L, 1L, 2L, 2L))
})

test_that("arguments for the model pass through", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  lung$w <- 1
  set.seed(4)
  weighted <- cross_validate(cox_ph, Surv(time, status) ~ age + sex, data = lung, weights = w)
  set.seed(4)
  plain <- cross_validate(cox_ph, Surv(time, status) ~ age + sex, data = lung)
  expect_equal(weighted$scores, plain$scores)

  set.seed(5)
  breslow <- cross_validate(cox_ph, Surv(time, status) ~ age + sex, data = lung, ties = "breslow")
  i <- 1L
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung[breslow$folds != i, ], ties = "breslow")
  test <- lung[breslow$folds == i, ]
  expect_equal(breslow$scores$score[1],
    concordance_index(Surv(test$time, test$status), risk = predict(fit, newdata = test))
  )
})

test_that("rows with missing values are dropped first", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  set.seed(6)
  result <- cross_validate(cox_ph, Surv(time, status) ~ age + ph.ecog, data = lung)
  expect_identical(result$n_dropped, 1L)
  expect_length(result$folds, nrow(lung) - 1L)
  expect_output(print(result), "1 row was dropped")
  expect_output(print(result), "5-fold cross-validation of cox_ph()", fixed = TRUE)
  expect_identical(as.data.frame(result), result$results)
  expect_named(result$results, c("metric", "mean", "sd"))
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- lung_complete()
  f <- Surv(time, status) ~ age
  expect_error(cross_validate("cox_ph", f, data = lung), "model function")
  expect_error(cross_validate(cox_ph, f, data = as.list(lung)), "data frame")
  expect_error(cross_validate(cox_ph, f, data = lung, k = 1), "`k`")
  expect_error(cross_validate(cox_ph, f, data = lung, k = 2.5), "whole number")
  expect_error(cross_validate(cox_ph, f, data = lung[1:4, ], k = 5), "at most the number of rows")
  expect_error(cross_validate(cox_ph, f, data = lung, metrics = "c_index"), "`metrics`")
  expect_error(cross_validate(cox_ph, f, data = lung, metrics = "brier"), "two `times`")
  expect_error(cross_validate(cox_ph, f, data = lung, metrics = "auc", times = 365), "two `times`")
  expect_error(cross_validate(kaplan_meier, Surv(time, status) ~ sex, data = lung), "not `kaplan_meier`")
  expect_error(cross_validate(cox_ph, time ~ age, data = lung), "`Surv()`", fixed = TRUE)
  rare <- lung[1:60, ]
  rare$status <- ifelse(seq_len(60) <= 9, rare$status, 1)
  set.seed(7)
  expect_warning(cross_validate(cox_ph, f, data = rare, k = 5), "fewer than two per fold")
})
