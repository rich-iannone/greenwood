# Replays the AFT fixtures in fixtures/r/ (aft_*, aft_residuals_*, aft_threshold_weibull3, and the
# AFT parts of predict_*), and checks what the fixtures don't cover (weights, factors, standard
# errors of predictions) against survreg() directly.

TOL <- 1e-9
DISTS <- c("weibull", "exponential", "lognormal", "loglogistic")

lung_aft <- function(dist, ...) {
  aft(Surv(time, status) ~ age + sex, data = survival::lung, dist = dist, ...)
}

new_subjects <- data.frame(age = c(50, 70, 60), sex = c(1, 2, 1))

test_that("the fits on lung match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  for (dist in DISTS) {
    expected <- read_fixture(paste0("aft_", dist))
    fit <- lung_aft(dist)
    # survreg() stops at a looser tolerance for the log-normal, so its estimates are a little off.
    tol <- if (dist == "lognormal") 1e-7 else TOL
    table <- as.data.frame(fit)
    expect_identical(table$term, r_strings(expected$terms))
    expect_identical(fit$n, as.integer(expected$n))
    expect_equal(table$estimate, r_numbers(expected$coef), tolerance = tol)
    expect_equal(table$std_error, r_numbers(expected$coef_se), tolerance = tol)
    expect_equal(fit$scale, expected$scale, tolerance = tol)
    expect_equal(fit$loglik[2], expected$loglik, tolerance = TOL)
    if (dist == "exponential") {
      expect_false("log(scale)" %in% rownames(vcov(fit)))
    } else {
      expect_equal(sqrt(vcov(fit)["log(scale)", "log(scale)"]), expected$log_scale_se,
        tolerance = tol
      )
    }

    p <- r_numbers(expected$pred_p)
    q <- predict(fit, newdata = new_subjects, type = "quantile", p = p)
    expect_identical(names(q), c("subject", "p", "estimate", "std_error", "conf_low", "conf_high"))
    by_subject <- unlist(lapply(expected$pred_quantile, r_numbers))
    expect_equal(q$estimate, by_subject, tolerance = tol)
    expect_identical(q$subject, rep(1:3, each = length(p)))
  }
})

test_that("the residuals match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  for (dist in DISTS) {
    expected <- read_fixture(paste0("aft_residuals_", dist))
    fit <- lung_aft(dist)
    tol <- if (dist == "lognormal") 1e-6 else 1e-8
    expect_equal(unname(residuals(fit, type = "response")), r_numbers(expected$response),
      tolerance = tol
    )
    expect_equal(unname(residuals(fit, type = "deviance")), r_numbers(expected$deviance),
      tolerance = tol
    )
    dfbeta <- residuals(fit, type = "dfbeta")
    dfbetas <- residuals(fit, type = "dfbetas")
    expect_identical(colnames(dfbeta)[1:3], c("(Intercept)", "age", "sex"))
    for (j in 1:3) {
      term <- c("intercept", "age", "sex")[j]
      expect_equal(unname(dfbeta[, j]), r_numbers(expected[[paste0("dfbeta_", term)]]),
        tolerance = tol
      )
      expect_equal(unname(dfbetas[, j]), r_numbers(expected[[paste0("dfbetas_", term)]]),
        tolerance = tol
      )
    }
  }
})

test_that("the threshold Weibull matches the fixture", {
  skip_if_no_shared_cases()
  expected <- read_fixture("aft_threshold_weibull3")
  data <- data.frame(time = r_numbers(expected$data), status = 1)
  fit <- aft(Surv(time, status) ~ 1, data = data, threshold = TRUE)
  expect_true(fit$has_threshold)
  expect_identical(rownames(vcov(fit)), c("(Intercept)", "log(scale)", "threshold"))
  # The fixture comes from a bounded general-purpose optimizer, so it agrees to about 1e-4.
  expect_equal(1 / fit$scale, expected$shape, tolerance = 1e-3)
  expect_equal(exp(unname(coef(fit))), expected$scale, tolerance = 1e-3)
  expect_equal(fit$threshold, expected$thres, tolerance = 1e-3)
  expect_gte(fit$loglik[2], expected$loglik - 1e-8)
  expect_equal(fit$loglik[2], expected$loglik, tolerance = 1e-6)
  expect_true(all(is.finite(sqrt(diag(vcov(fit))))))

  # The threshold shifts every quantile.
  q <- predict(fit, newdata = data.frame(row.names = 1), type = "quantile", p = 0.5)
  w <- log(-log(0.5))
  expect_equal(q$estimate, unname(fit$threshold + exp(coef(fit) + fit$scale * w)))
  s <- predict(fit, newdata = data.frame(row.names = 1), type = "survival", times = c(1, 10))
  expect_identical(s$estimate[1], 1)
})

test_that("predicted medians and quantiles match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  medians <- read_fixture("predict_median")
  for (dist in c("weibull", "lognormal", "loglogistic")) {
    expected <- medians[[paste0("aft_", dist)]]
    m <- predict(lung_aft(dist), newdata = new_subjects, type = "quantile")
    tol <- if (dist == "lognormal") 1e-7 else TOL
    expect_equal(m$estimate, r_numbers(expected$median), tolerance = tol)
    expect_equal(m$conf_low, r_numbers(expected$lower), tolerance = tol)
    expect_equal(m$conf_high, r_numbers(expected$upper), tolerance = tol)
  }

  quantiles <- read_fixture("predict_quantile")
  p <- r_numbers(quantiles$p)
  q <- predict(lung_aft("weibull"), newdata = new_subjects, type = "quantile", p = p)
  # The fixture's rows are the probabilities and its columns the subjects.
  by_p <- function(rows) {
    matrix(unlist(lapply(rows, r_numbers)), nrow = length(p), byrow = TRUE)
  }
  expect_equal(q$estimate, c(by_p(quantiles$aft_weibull$quantiles)), tolerance = TOL)
  expect_equal(q$conf_low, c(by_p(quantiles$aft_weibull$lower)), tolerance = TOL)
  expect_equal(q$conf_high, c(by_p(quantiles$aft_weibull$upper)), tolerance = TOL)
})

test_that("standard errors of quantiles match survreg()", {
  skip_if_not_installed("survival")
  p <- c(0.1, 0.5, 0.9)
  for (dist in DISTS) {
    fit <- lung_aft(dist)
    reference <- survival::survreg(
      survival::Surv(time, status) ~ age + sex, data = survival::lung, dist = dist
    )
    q <- predict(fit, newdata = new_subjects, type = "quantile", p = p)
    expected <- predict(reference, new_subjects, type = "quantile", p = p, se.fit = TRUE)
    expect_equal(q$estimate, c(t(expected$fit)), tolerance = 1e-7)
    expect_equal(q$std_error, c(t(expected$se.fit)), tolerance = 1e-6)
    z <- stats::qnorm(0.975)
    se_log <- q$std_error / q$estimate
    expect_equal(q$conf_low, q$estimate * exp(-z * se_log))
  }
})

test_that("restricted means match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("predict_expectation")
  fit <- lung_aft("weibull")
  for (tau in r_numbers(expected$tau)) {
    e <- predict(fit, newdata = new_subjects, type = "expectation", tau = tau)
    expect_identical(names(e), c("subject", "tau", "estimate", "std_error", "conf_low", "conf_high"))
    expect_equal(e$estimate, r_numbers(expected$aft_weibull[[paste0("rmst_", tau)]]),
      tolerance = TOL
    )
    # The Weibull area through the incomplete gamma function.
    shape <- 1 / fit$scale
    scale <- exp(predict(fit, newdata = new_subjects))
    area <- scale * gamma(1 + 1 / shape) * stats::pgamma((tau / scale)^shape, 1 / shape)
    expect_equal(e$estimate, area, tolerance = 1e-10)
  }
})

test_that("weighted fits match survreg()", {
  skip_if_not_installed("survival")
  data <- survival::lung
  data$wt <- rep(c(0.5, 1.25, 2, 0.8), length.out = nrow(data))
  for (dist in DISTS) {
    fit <- aft(Surv(time, status) ~ age + sex, data = data, weights = wt, dist = dist)
    reference <- survival::survreg(
      survival::Surv(time, status) ~ age + sex, data = data, weights = wt, dist = dist
    )
    tol <- if (dist == "lognormal") 1e-6 else 1e-8
    expect_true(fit$has_weights)
    expect_equal(coef(fit), coef(reference), tolerance = tol)
    expect_equal(unname(vcov(fit)), unname(reference$var), tolerance = tol)
    expect_equal(fit$loglik, reference$loglik, tolerance = TOL)
    expect_equal(unname(residuals(fit, type = "response")),
      unname(residuals(reference, type = "response")),
      tolerance = tol
    )
    expect_equal(unname(residuals(fit, type = "deviance")),
      unname(residuals(reference, type = "deviance")),
      tolerance = tol
    )
    # survreg() leaves the case weights out of dfbeta. Here they're included, as for cox_ph().
    expect_equal(unname(residuals(fit, type = "dfbeta")),
      unname(residuals(reference, type = "dfbeta")) * data$wt,
      tolerance = tol
    )
  }
})

test_that("factors and missing values are handled like survreg()", {
  skip_if_not_installed("survival")
  data <- survival::lung
  data$ecog <- factor(data$ph.ecog)
  fit <- aft(Surv(time, status) ~ age + ecog, data = data, dist = "loglogistic")
  reference <- survival::survreg(
    survival::Surv(time, status) ~ age + ecog, data = data, dist = "loglogistic"
  )
  expect_equal(coef(fit), coef(reference), tolerance = 1e-8)
  expect_identical(fit$n_dropped, 1L)
  expect_identical(nobs(fit), 227L)
  expect_equal(
    predict(fit, newdata = data.frame(age = 60, ecog = factor(2, levels = 0:3)), type = "lp"),
    unname(predict(reference, data.frame(age = 60, ecog = factor(2, levels = 0:3)), type = "lp")),
    tolerance = 1e-8
  )
  expect_equal(fit$lr_test$df, 4L)
  expect_equal(
    fit$lr_test$statistic, 2 * (reference$loglik[2] - reference$loglik[1]),
    tolerance = 1e-7
  )
})

test_that("survival curves, means, and residuals are consistent", {
  fit <- aft(Surv(time, status) ~ age + sex, data = small_cox)
  lp <- predict(fit, newdata = new_subjects, type = "lp")
  expect_equal(lp, drop(cbind(1, new_subjects$age, new_subjects$sex) %*% coef(fit)))
  expect_equal(predict(fit), fit$linear_predictors)

  s <- predict(fit, newdata = new_subjects, type = "survival", times = c(100, 300))
  expect_identical(names(s), c("subject", "time", "estimate", "std_error", "conf_low", "conf_high"))
  z <- (log(s$time) - rep(lp, each = 2)) / fit$scale
  expect_equal(s$estimate, exp(-exp(z)))
  expect_true(all(s$conf_low <= s$estimate & s$estimate <= s$conf_high))
  h <- predict(fit, newdata = new_subjects, type = "cumhaz", times = c(100, 300))
  expect_equal(h$estimate, -log(s$estimate))
  expect_equal(h$conf_low, -log(s$conf_high))

  m <- predict(fit, newdata = new_subjects, type = "mean")
  expect_false("tau" %in% names(m))
  expect_equal(m$estimate, exp(lp) * gamma(1 + fit$scale))
  e <- predict(fit, newdata = new_subjects, type = "expectation", tau = 1e6)
  expect_equal(e$estimate, m$estimate, tolerance = 1e-6)

  # Martingale and Cox-Snell residuals: delta - H(t) and H(t).
  cumhaz <- exp((log(small_cox$time) - fit$linear_predictors) / fit$scale)
  expect_equal(unname(residuals(fit, type = "cox_snell")), cumhaz)
  expect_equal(unname(residuals(fit)), (small_cox$status == 2) - cumhaz)
  score <- residuals(fit, type = "score")
  expect_identical(colnames(score), c("(Intercept)", "age", "sex", "log(scale)"))
  expect_equal(unname(colSums(score)), rep(0, 4), tolerance = 1e-6)
  expect_equal(residuals(fit, type = "dfbeta"), score %*% vcov(fit))
})

test_that("the methods agree with the object", {
  fit <- aft(Surv(time, status) ~ age + sex, data = small_cox, dist = "lognormal")
  expect_identical(names(coef(fit)), c("(Intercept)", "age", "sex"))
  expect_identical(dim(vcov(fit)), c(4L, 4L))
  ll <- logLik(fit)
  expect_identical(attr(ll, "df"), 4L)
  expect_equal(AIC(fit), -2 * fit$loglik[2] + 8)
  expect_equal(BIC(fit), -2 * fit$loglik[2] + 4 * log(30))
  ratios <- as.data.frame(fit, exponentiate = TRUE)
  table <- as.data.frame(fit)
  expect_equal(ratios$estimate, exp(table$estimate))
  expect_equal(ratios$conf_low, exp(table$conf_low))
  narrow <- as.data.frame(fit, conf_level = 0.5)
  expect_true(all(narrow$conf_high - narrow$conf_low < table$conf_high - table$conf_low))
  exponential <- aft(Surv(time, status) ~ age, data = small_cox, dist = "exponential")
  expect_identical(exponential$scale, 1)
  expect_identical(attr(logLik(exponential), "df"), 2L)
})

test_that("bad input is rejected", {
  expect_error(
    aft(Surv(time, status) ~ age + strata(sex), data = small_cox),
    "doesn't support `strata()`", fixed = TRUE
  )
  expect_error(
    aft(Surv(time, status) ~ age + offset(sex), data = small_cox),
    "doesn't support `offset()`", fixed = TRUE
  )
  expect_error(aft(Surv(time, status) ~ age, data = small_cox, dist = "gompertz"), "should be one of")
  expect_error(aft(Surv(time, status) ~ age, data = small_cox, threshold = NA), "`threshold`")
  no_events <- transform(small_cox, status = 0)
  expect_error(aft(Surv(time, status) ~ age, data = no_events), "at least one event")
  zero <- transform(small_cox, time = replace(time, 1, 0))
  expect_error(aft(Surv(time, status) ~ age, data = zero), "positive times")
  fit <- aft(Surv(time, status) ~ age, data = small_cox)
  expect_error(predict(fit, newdata = 1), "must be a data frame")
  expect_error(predict(fit, newdata = small_cox, type = "survival"), "`times`")
  expect_error(predict(fit, newdata = small_cox, type = "expectation"), "`tau`")
})

test_that("the printout is stable", {
  fit <- aft(Surv(time, status) ~ age + sex, data = small_cox)
  expect_snapshot(print(fit))
  expect_snapshot(print(aft(Surv(time, status) ~ age, data = small_cox, dist = "exponential")))
})
