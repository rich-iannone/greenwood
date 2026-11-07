# Replays the Royston-Parmar anchors in fixtures/r/ (rp_weibull_anchor, rp_loglogistic_anchor): with
# df = 1 the model is a Weibull on the hazard scale and a log-logistic on the odds scale. Larger df
# are checked against Greenwood's Python values. Python fits by quasi-Newton steps with a
# numerical Hessian, so its estimates and standard errors agree to about 1e-5 and 1e-4.

lung_rp <- function(...) {
  royston_parmar(Surv(time, status) ~ age + sex, data = survival::lung, ...)
}

test_that("df = 1 reproduces the Weibull and log-logistic anchors", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  for (case in list(c("weibull", "hazard"), c("loglogistic", "odds"))) {
    expected <- read_fixture(paste0("rp_", case[1L], "_anchor"))
    fit <- lung_rp(df = 1, scale = case[2L])
    expect_true(fit$converged)
    expect_equal(fit$loglik, expected$loglik, tolerance = 1e-10)
    newdata <- data.frame(age = r_numbers(expected$newdata_age), sex = r_numbers(expected$newdata_sex))
    times <- r_numbers(expected$times)
    survival <- predict(fit, newdata = newdata, type = "survival", times = times)
    expect_equal(survival$estimate,
      c(r_numbers(expected$surv$subj1), r_numbers(expected$surv$subj2)),
      tolerance = 1e-9
    )
  }
})

test_that("df = 1 is the matching AFT model, reparametrized", {
  skip_if_not_installed("survival")
  for (case in list(c("weibull", "hazard"), c("loglogistic", "odds"))) {
    fit <- lung_rp(df = 1, scale = case[2L])
    reference <- aft(Surv(time, status) ~ age + sex, data = survival::lung, dist = case[1L])
    # log T = a + b x + sigma e gives eta = (log t - a - b x) / sigma.
    beta <- coef(reference)
    sigma <- reference$scale
    expect_equal(unname(coef(fit)), unname(c(-beta[1] / sigma, 1 / sigma, -beta[-1] / sigma)),
      tolerance = 1e-8
    )
    expect_equal(fit$loglik, reference$loglik[2], tolerance = 1e-12)
    expect_equal(AIC(fit), AIC(reference), tolerance = 1e-12)
  }
})

test_that("flexible fits match Python on lung", {
  skip_if_not_installed("survival")
  hazard <- lung_rp(df = 3)
  expect_equal(hazard$knots,
    c(1.6094379124341003, 5.1159918043966846, 5.82303715943831, 6.78332520060396),
    tolerance = 1e-14
  )
  expect_identical(names(coef(hazard)), c("gamma0", "gamma1", "gamma2", "gamma3", "age", "sex"))
  expect_equal(unname(coef(hazard)), c(
    -7.228464386147577, 1.0275131867292118, -0.09641985303406066, 0.11723217325823009,
    0.01614692618772489, -0.5101268340730931
  ), tolerance = 1e-4)
  expect_equal(unname(sqrt(diag(vcov(hazard)))), c(
    1.3260952741421268, 0.29679746559479714, 0.12971372675432552, 0.1849327619503349,
    0.00919257919219907, 0.16716226184592
  ), tolerance = 5e-3)
  # Python's optimizer stops just short of the maximum.
  expect_gte(hazard$loglik, -1146.5705368849094)
  expect_equal(hazard$loglik, -1146.5705368849094, tolerance = 1e-10)
  expect_equal(AIC(hazard), 2305.141073769819, tolerance = 1e-10)
  expect_equal(BIC(hazard), 2323.7767466132223, tolerance = 1e-10)
  expect_identical(nobs(hazard), 165)

  odds <- lung_rp(df = 3, scale = "odds")
  expect_equal(unname(coef(odds)), c(
    -7.252341279126419, 1.012181414812129, 0.19795949293018644, -0.4249196281737476,
    0.02429401856402443, -0.8920216610951792
  ), tolerance = 1e-4)
  expect_equal(odds$loglik, -1145.9467183683473, tolerance = 1e-10)

  null <- royston_parmar(Surv(time, status) ~ 1, data = survival::lung, df = 2)
  expect_identical(names(coef(null)), c("gamma0", "gamma1", "gamma2"))
  expect_equal(unname(coef(null)), c(-7.229658664118931, 1.1379504035375179, -0.01558045593009864),
    tolerance = 1e-4
  )
  expect_equal(null$loglik, -1153.5968288881304, tolerance = 1e-10)
})

test_that("predictions match Python on lung", {
  skip_if_not_installed("survival")
  newdata <- survival::lung[1:2, ]
  hazard <- lung_rp(df = 3)
  survival <- predict(hazard, newdata = newdata, type = "survival", times = c(180, 365))
  expect_named(survival, c("subject", "time", "estimate", "std_error", "conf_low", "conf_high"))
  expect_identical(survival$subject, c(1L, 1L, 2L, 2L))
  expect_equal(survival$estimate,
    c(0.6320986649234539, 0.29205094279378135, 0.6594467001576359, 0.32720225657347163),
    tolerance = 1e-4
  )
  rate <- predict(hazard, newdata = newdata, type = "hazard", times = c(180, 365))
  expect_equal(rate$estimate,
    c(0.0035285066441653121, 0.0046069099890032883, 0.0032026967117313844, 0.0041815240726331597),
    tolerance = 1e-4
  )
  cumhaz <- predict(hazard, newdata = newdata, type = "cumhaz", times = c(180, 365))
  expect_equal(cumhaz$estimate, -log(survival$estimate))
  quantiles <- predict(hazard, newdata = newdata, type = "quantile", p = c(0.25, 0.5))
  expect_named(quantiles, c("subject", "p", "estimate", "std_error", "conf_low", "conf_high"))
  expect_equal(quantiles$estimate,
    c(127.4930421180852, 241.73299560623593, 137.14751779658047, 258.95877797300238),
    tolerance = 1e-4
  )
  expect_equal(quantiles$conf_low,
    c(98.07344424198773, 197.46307718663681, 110.15456900772031, 220.38444611545052),
    tolerance = 1e-4
  )
  rmst <- predict(hazard, newdata = newdata, type = "expectation", tau = 365)
  expect_equal(rmst$estimate, c(231.69041957505397, 240.41070893312366), tolerance = 1e-4)
  # Python's limits are the estimate plus or minus z standard errors.
  expect_equal(rmst$std_error,
    (c(258.05937709, 261.00946286) - c(205.32146206, 219.81195501)) / (2 * stats::qnorm(0.975)),
    tolerance = 1e-3
  )

  odds <- lung_rp(df = 3, scale = "odds")
  rate <- predict(odds, newdata = newdata, type = "hazard", times = c(180, 365))
  expect_equal(rate$estimate,
    c(0.0035660716172771847, 0.0043021692439922802, 0.0032601872877573643, 0.0041168511102386094),
    tolerance = 1e-4
  )
  quantiles <- predict(odds, newdata = newdata, type = "quantile", p = 0.5)
  expect_equal(quantiles$estimate, c(228.65090010552422, 248.35235026211154), tolerance = 1e-4)
})

test_that("the predictions hang together", {
  skip_if_not_installed("survival")
  for (scale in c("hazard", "odds")) {
    fit <- lung_rp(df = 3, scale = scale)
    newdata <- data.frame(age = c(55, 70), sex = c(1, 2))
    times <- c(0, 100, 400)
    survival <- predict(fit, newdata = newdata, type = "survival", times = times)
    expect_identical(survival$estimate[c(1, 4)], c(1, 1))
    expect_true(all(survival$conf_low <= survival$estimate & survival$estimate <= survival$conf_high))
    rate <- predict(fit, newdata = newdata, type = "hazard", times = times)
    expect_true(all(is.na(rate$estimate[c(1, 4)])))
    # The hazard is the derivative of the cumulative hazard.
    h <- 1e-3
    up <- predict(fit, newdata = newdata, type = "cumhaz", times = 400 + h)$estimate
    down <- predict(fit, newdata = newdata, type = "cumhaz", times = 400 - h)$estimate
    expect_equal(rate$estimate[c(3, 6)], (up - down) / (2 * h), tolerance = 1e-6)
    # The median is where survival is one half, and the restricted mean is the area under it.
    median <- predict(fit, newdata = newdata, type = "quantile")
    at_median <- predict(fit, newdata = newdata[1, ], type = "survival", times = median$estimate[1])
    expect_equal(at_median$estimate, 0.5, tolerance = 1e-10)
    rmst <- predict(fit, newdata = newdata[1, ], type = "expectation", tau = 400)
    grid <- seq(0, 400, length.out = 4001)
    curve <- predict(fit, newdata = newdata[1, ], type = "survival", times = grid)$estimate
    expect_equal(rmst$estimate, sum((curve[-1] + curve[-4001]) / 2) * 0.1, tolerance = 1e-6)
    expect_identical(predict(fit, newdata = newdata, type = "expectation", tau = 0)$estimate, c(0, 0))
    # The linear predictor leaves out the spline.
    expect_equal(predict(fit, newdata = newdata, type = "lp"),
      drop(as.matrix(newdata) %*% coef(fit)[c("age", "sex")])
    )
    # Without newdata, the predictions are for the rows the model was fit to.
    expect_length(predict(fit, type = "lp"), fit$n)
  }

  # The quantile is NA when the curve doesn't reach it within the spline's range.
  fit <- lung_rp(df = 1)
  far <- predict(fit, newdata = data.frame(age = 50, sex = 1), type = "quantile", p = 1e-10)
  expect_true(is.na(far$estimate))
})

test_that("the delta method matches numerical derivatives", {
  skip_if_not_installed("survival")
  fit <- lung_rp(df = 2)
  newdata <- data.frame(age = 60, sex = 2)
  theta <- coef(fit)
  numeric_se <- function(f) {
    gradient <- vapply(seq_along(theta), function(j) {
      h <- 1e-6 * max(abs(theta[j]), 1)
      up <- down <- fit
      up$coefficients[j] <- theta[j] + h
      down$coefficients[j] <- theta[j] - h
      (f(up) - f(down)) / (2 * h)
    }, numeric(1))
    sqrt(drop(gradient %*% vcov(fit) %*% gradient))
  }
  for (type in c("survival", "hazard")) {
    reading <- function(object) predict(object, newdata = newdata, type = type, times = 300)$estimate
    expect_equal(predict(fit, newdata = newdata, type = type, times = 300)$std_error,
      numeric_se(reading),
      tolerance = 1e-5
    )
  }
  quantile_of <- function(object) predict(object, newdata = newdata, type = "quantile")$estimate
  expect_equal(predict(fit, newdata = newdata, type = "quantile")$std_error, numeric_se(quantile_of),
    tolerance = 1e-5
  )
  rmst_of <- function(object) predict(object, newdata = newdata, type = "expectation", tau = 500)$estimate
  expect_equal(predict(fit, newdata = newdata, type = "expectation", tau = 500)$std_error,
    numeric_se(rmst_of),
    tolerance = 1e-5
  )
})

test_that("weights, factors, and the table work", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- lung_rp(df = 2)
  lung$w <- 2
  doubled <- royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2, weights = w)
  expect_equal(coef(doubled), coef(fit), tolerance = 1e-8)
  expect_equal(doubled$loglik, 2 * fit$loglik)
  expect_equal(vcov(doubled), vcov(fit) / 2, tolerance = 1e-8)

  # Integer weights are copies of rows.
  lung$k <- rep(1:2, length.out = nrow(lung))
  weighted <- royston_parmar(Surv(time, status) ~ age + sex, data = lung, df = 2, weights = k)
  copies <- lung[rep(seq_len(nrow(lung)), lung$k), ]
  # The knots come from the rows, not the weights, so the copies are read with the same knots.
  u <- log(copies$time)
  at_copies <- rp_terms(coef(weighted), rp_basis(u, weighted$knots), u,
    as.matrix(copies[c("age", "sex")]), copies$status - 1, rep(1, nrow(copies)), rp_link("hazard"),
    derivatives = FALSE
  )
  expect_equal(weighted$loglik, at_copies$loglik, tolerance = 1e-12)

  lung$group <- factor(ifelse(lung$sex == 1, "m", "f"), levels = c("m", "f"))
  by_factor <- royston_parmar(Surv(time, status) ~ age + group, data = lung, df = 2)
  expect_identical(names(coef(by_factor))[4:5], c("age", "groupf"))
  expect_equal(unname(coef(by_factor)[5]), unname(coef(fit)["sex"]), tolerance = 1e-8)
  expect_equal(
    predict(by_factor, newdata = data.frame(age = 60, group = "f"), type = "survival", times = 300),
    predict(fit, newdata = data.frame(age = 60, sex = 2), type = "survival", times = 300),
    tolerance = 1e-8
  )

  table <- as.data.frame(fit)
  expect_named(table, c("term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"))
  expect_equal(table$std_error, unname(sqrt(diag(vcov(fit)))))
  ratios <- as.data.frame(fit, exponentiate = TRUE, conf_level = 0.9)
  expect_equal(ratios$estimate, exp(table$estimate))
  expect_equal(log(ratios$conf_high), table$estimate + stats::qnorm(0.95) * table$std_error)

  expect_output(print(fit), "Hazard scale (exp(coef) is a hazard ratio), df = 2", fixed = TRUE)
  expect_output(print(fit), "Knots (log time)", fixed = TRUE)
  expect_output(print(lung_rp(df = 1, scale = "odds")), "an odds ratio")
  expect_output(print(royston_parmar(Surv(time, status) ~ ph.ecog, data = lung, df = 1)), "1 row was dropped")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  expect_error(lung_rp(df = 0), "`df`")
  expect_error(lung_rp(df = 1.5), "whole number")
  expect_error(lung_rp(scale = "log"), "should be one of")
  expect_error(royston_parmar(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  lung$start <- 0
  expect_error(royston_parmar(Surv(start, time, status) ~ age, data = lung), "right")
  lung$zero <- lung$time
  lung$zero[1] <- 0
  expect_error(royston_parmar(Surv(zero, status) ~ age, data = lung), "positive times")
  expect_error(royston_parmar(Surv(time, status * 0) ~ age, data = lung), "at least one event")
  expect_error(royston_parmar(Surv(time, rep(1:2, c(227, 1))) ~ age, data = lung), "two or more")
  tied <- data.frame(time = c(rep(5, 20), 1, 10), status = 2, age = 1:22)
  expect_error(royston_parmar(Surv(time, status) ~ age, data = tied, df = 4), "Some knots coincide")
  lung$age2 <- 2 * lung$age
  expect_error(royston_parmar(Surv(time, status) ~ age + age2, data = lung), "collinear")
  fit <- lung_rp(df = 1)
  expect_error(predict(fit, type = "survival"), "needs `times`")
  expect_error(predict(fit, type = "survival", times = -1), "negative")
  expect_error(predict(fit, type = "quantile", p = 1), "`p`")
  expect_error(predict(fit, type = "expectation"), "needs `tau`")
  expect_error(predict(fit, newdata = list(age = 1), type = "lp"), "data frame")
  expect_error(predict(fit, newdata = data.frame(age = 1), type = "lp"), "sex")
})
