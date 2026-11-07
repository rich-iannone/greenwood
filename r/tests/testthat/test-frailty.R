# Checks shared frailty in cox_ph() against Greenwood's Python values (CoxPH with frailty_cluster
# and ties="breslow"), on the lung data by institution and on simulated clustered data.

clustered_data <- function() {
  set.seed(11)
  k <- 30
  m <- 12
  centre <- rep(seq_len(k), each = m)
  z <- stats::rgamma(k, shape = 2, rate = 2)[centre]
  x1 <- round(stats::rnorm(k * m), 3)
  x2 <- stats::rbinom(k * m, 1, 0.4)
  t <- round(stats::rexp(k * m, rate = 0.1 * z * exp(0.5 * x1 - 0.4 * x2)), 2)
  censor <- round(stats::rexp(k * m, rate = 0.05), 2)
  data.frame(
    time = pmin(t, censor), status = as.integer(t <= censor), x1 = x1, x2 = x2, centre = centre
  )
}

test_that("gamma frailty matches Python on the lung data", {
  skip_if_not_installed("survival")
  fit <- cox_ph(Surv(time, status) ~ age + sex + frailty(inst),
    data = survival::lung, ties = "breslow"
  )
  expect_identical(fit$n_dropped, 1L)
  expect_equal(unname(coef(fit)), c(0.01700004536148252, -0.5109965882474825), tolerance = 1e-9)
  expect_equal(unname(sqrt(diag(vcov(fit)))), c(0.00923134973524536, 0.1676827397987142),
    tolerance = 1e-9
  )
  # The variance is at the second point of the grid, so there's essentially no frailty.
  expect_equal(fit$frailty$theta, 1.3779723598335543e-08, tolerance = 1e-12)
  expect_identical(names(fit$frailty$effects)[1:5], c("3", "5", "1", "12", "7"))
  expect_equal(unname(fit$frailty$effects[1:5]), c(
    0.9999999752607969, 1.0000000022326831, 1.0000000553260842, 1.0000000165961136,
    0.9999999979148323
  ), tolerance = 1e-12)
  # The statistic is the difference of two log likelihoods of about 1e9 each, so only its first
  # digits are meaningful.
  expect_lt(abs(fit$frailty$test$statistic - 0.00043770764023065567), 1e-5)
  expect_equal(fit$loglik, c(-745.0130598090508, -738.0436406673442), tolerance = 1e-10)
  expect_equal(fit$lr_test$statistic, 13.938838283413133, tolerance = 1e-9)
  expect_equal(fit$wald_test$statistic, 13.312096634449233, tolerance = 1e-9)
  expect_equal(fit$score_test$statistic, 13.55473381433919, tolerance = 1e-9)
  expect_equal(fit$concordance, 0.6036517704025017, tolerance = 1e-12)
  expect_equal(unname(residuals(fit)[1:4]), c(
    -2.7556625888227604e-03, -4.9474823378522204e-01, -3.1199569306227053e+00,
    5.3012617117851946e-01
  ), tolerance = 1e-9)
  expect_equal(baseline_hazard(fit)$estimate[1:3],
    c(0.002972555990685601, 0.011919119914256373, 0.014958981698364491),
    tolerance = 1e-9
  )
})

test_that("lognormal frailty matches Python on the lung data", {
  skip_if_not_installed("survival")
  fit <- cox_ph(Surv(time, status) ~ age + sex + frailty(inst, distribution = "gaussian"),
    data = survival::lung, ties = "breslow"
  )
  expect_identical(fit$frailty$distribution, "lognormal")
  expect_equal(unname(coef(fit)), c(0.01726437266596513, -0.5117793345334088), tolerance = 1e-9)
  expect_equal(unname(sqrt(diag(vcov(fit)))), c(0.00927977978494242, 0.1681644651927698),
    tolerance = 1e-9
  )
  expect_equal(fit$frailty$theta, 0.014255476590025323, tolerance = 1e-9)
  expect_equal(fit$frailty$test$statistic, 0.05125652908554912, tolerance = 1e-8)
  expect_equal(fit$frailty$test$p_value, 0.4104455340191376, tolerance = 1e-9)
  expect_equal(unname(fit$frailty$effects[1:5]), c(
    -0.02234998341692958, 0.00175323667441445, 0.04268760625876464, 0.01304477041209319,
    -0.00243062553110609
  ), tolerance = 1e-8)
  expect_equal(fit$loglik, c(-745.0130605751312, -737.2836282111766), tolerance = 1e-10)
  expect_equal(fit$lr_test$statistic, 15.458864727909258, tolerance = 1e-9)
  expect_equal(fit$wald_test$statistic, 13.393183587279703, tolerance = 1e-9)
  expect_equal(fit$score_test$statistic, 13.554733637556907, tolerance = 1e-9)
  expect_equal(fit$concordance, 0.6033995763139312, tolerance = 1e-12)
  expect_equal(unname(residuals(fit)[1:4]), c(
    0.01363117856830676, -0.47186262041799365, -3.0719334288079887, 0.5296347547638391
  ), tolerance = 1e-9)
  expect_equal(baseline_hazard(fit)$estimate[1:3],
    c(0.002919249844401498, 0.011705356469470466, 0.014691042265573826),
    tolerance = 1e-9
  )
})

test_that("both frailties match Python on clustered data", {
  data <- clustered_data()
  gamma <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty(centre), data = data, ties = "breslow")
  expect_equal(unname(coef(gamma)), c(0.3658570201559487, -0.2054084812393365), tolerance = 1e-9)
  expect_equal(unname(sqrt(diag(vcov(gamma)))), c(0.07721422256436664, 0.15389579502887285),
    tolerance = 1e-9
  )
  expect_equal(gamma$frailty$theta, 0.4552268275507298, tolerance = 1e-12)
  expect_equal(gamma$frailty$test$statistic, 53.10837545382856, tolerance = 1e-9)
  expect_equal(gamma$frailty$test$p_value, 1.5782185756513694e-13, tolerance = 1e-8)
  expect_equal(unname(gamma$frailty$effects[1:3]),
    c(0.43766009668388334, 0.10231347680688893, 0.3907115010029258),
    tolerance = 1e-9
  )
  expect_equal(gamma$loglik, c(-876.323905171092, -864.1501869205606), tolerance = 1e-10)
  expect_equal(gamma$score_test$statistic, 24.16614155500352, tolerance = 1e-9)
  expect_equal(gamma$concordance, 0.5903864171942875, tolerance = 1e-12)
  expect_equal(unname(residuals(gamma)[1:3]),
    c(0.7779056817359236, 0.8611982976072083, -0.1471124588792808),
    tolerance = 1e-9
  )

  lognormal <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty.gaussian(centre),
    data = data, ties = "breslow"
  )
  expect_equal(unname(coef(lognormal)), c(0.3642285668290561, -0.1989911307916981),
    tolerance = 1e-9
  )
  expect_equal(unname(sqrt(diag(vcov(lognormal)))), c(0.08202556555845752, 0.1671707614090019),
    tolerance = 1e-9
  )
  expect_equal(lognormal$frailty$theta, 0.46010583956558304, tolerance = 1e-9)
  expect_equal(lognormal$frailty$test$statistic, 33.83502239670838, tolerance = 1e-9)
  expect_equal(unname(lognormal$frailty$effects[1:3]),
    c(-0.6420654907515725, -1.5677516926964983, -0.7274939938911195),
    tolerance = 1e-9
  )
  expect_equal(lognormal$loglik, c(-911.1076249417116, -866.8067195941426), tolerance = 1e-10)
  expect_equal(lognormal$score_test$statistic, 16.612437053831776, tolerance = 1e-9)
  expect_equal(unname(residuals(lognormal)[1:3]),
    c(0.7689917730411853, 0.8556207925782036, -0.15208043967117293),
    tolerance = 1e-9
  )
})

test_that("the frailty multiplies the risk scores in the diagnostics", {
  data <- clustered_data()
  for (term in c("frailty(centre)", "frailty.gaussian(centre)")) {
    f <- stats::as.formula(paste("Surv(time, status) ~ x1 + x2 +", term))
    fit <- cox_ph(f, data = data, ties = "breslow")
    # With Breslow's estimator, the martingale residuals add up to 0 only when the frailty is in
    # the expected events.
    expect_lt(abs(sum(residuals(fit))), 1e-9)
    # The score residuals add up to the score at the estimate, which is 0 when the coefficients
    # and the frailties are fit together. A gamma fit's frailties are updated after its last fit
    # of the coefficients.
    if (fit$frailty$distribution == "lognormal") {
      expect_lt(max(abs(colSums(residuals(fit, type = "score")))), 1e-6)
    }
    # Predictions are for a subject with a frailty of 1 (a log frailty of 0).
    newdata <- data.frame(x1 = 0.2, x2 = 1)
    survival <- predict(fit, newdata = newdata, type = "survival", times = c(5, 10))
    base <- baseline_hazard(fit)
    h0 <- step_at(base$time, base$estimate, c(5, 10), 0)
    expect_equal(survival$estimate, exp(-h0 * exp(sum(coef(fit) * c(0.2, 1)))))
  }
})

test_that("the clusters can be given as an argument", {
  data <- clustered_data()
  term <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty(centre), data = data, ties = "breslow")
  argument <- cox_ph(Surv(time, status) ~ x1 + x2,
    data = data, ties = "breslow", frailty = "gamma", frailty_cluster = centre
  )
  expect_equal(coef(argument), coef(term))
  expect_equal(argument$frailty$effects, term$frailty$effects)
  expect_identical(argument$frailty$cluster, "centre")

  lognormal <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty(centre, distribution = "lognormal"),
    data = data, ties = "breslow", frailty = "lognormal"
  )
  gaussian <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty.gaussian(centre),
    data = data, ties = "breslow"
  )
  expect_equal(coef(lognormal), coef(gaussian))

  # Unit weights change nothing.
  data$w <- 1
  weighted <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty(centre),
    data = data, ties = "breslow", weights = w
  )
  expect_equal(coef(weighted), coef(term))
  expect_equal(weighted$frailty$theta, term$frailty$theta)
})

test_that("the frailty is tested and printed", {
  data <- clustered_data()
  fit <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty(centre), data = data, ties = "breslow")
  test <- frailty_test(fit)
  expect_named(test, c("distribution", "theta", "statistic", "df", "p_value"))
  expect_identical(test$distribution, "gamma")
  expect_identical(test$df, 1L)
  expect_equal(test$p_value, 0.5 * stats::pchisq(test$statistic, 1, lower.tail = FALSE))
  expect_output(print(fit), "Shared gamma frailty by centre: theta = 0.455")
  expect_output(print(fit), "Frailty variance test (theta = 0): LR = 53.1", fixed = TRUE)

  lognormal <- cox_ph(Surv(time, status) ~ x1 + x2 + frailty.gaussian(centre),
    data = data, ties = "breslow"
  )
  expect_output(print(lognormal), "Shared lognormal frailty by centre: sigma2 = 0.46")
  expect_identical(frailty_test(lognormal)$distribution, "lognormal")

  plain <- cox_ph(Surv(time, status) ~ x1 + x2, data = data)
  expect_null(plain$frailty)
  expect_error(frailty_test(plain), "no frailty")
  expect_error(frailty_test(1), "`cox_ph()`", fixed = TRUE)
})

test_that("bad frailty input is rejected", {
  data <- clustered_data()
  data$start <- 0
  f <- Surv(time, status) ~ x1 + frailty(centre)
  expect_error(cox_ph(f, data = data), "ties = \"breslow\"", fixed = TRUE)
  expect_error(cox_ph(Surv(time, status) ~ x1 + strata(x2) + frailty(centre),
    data = data, ties = "breslow"
  ), "`strata()`", fixed = TRUE)
  expect_error(cox_ph(f, data = data, ties = "breslow", robust = TRUE), "robust")
  expect_error(cox_ph(f, data = data, ties = "breslow", cluster = x2), "robust")
  expect_error(cox_ph(Surv(start, time, status) ~ x1 + frailty(centre),
    data = data, ties = "breslow"
  ), "right-censored")
  expect_error(cox_ph(Surv(time, status) ~ x1, data = data, ties = "breslow", frailty = "gamma"),
    "needs the clusters"
  )
  expect_error(cox_ph(Surv(time, status) ~ x1, data = data, frailty_cluster = centre),
    "`frailty_cluster` needs"
  )
  expect_error(cox_ph(f, data = data, ties = "breslow", frailty = "lognormal"), "Use one")
  expect_error(cox_ph(f, data = data, ties = "breslow", frailty = "normal"), "`frailty` must")
  expect_error(cox_ph(f, data = data, ties = "breslow", frailty_cluster = centre), "not both")
  expect_error(cox_ph(f, data = data, ties = "breslow", frailty_theta = 0), "`frailty_theta`")
  expect_error(cox_ph(f, data = data, ties = "breslow", frailty_max_iter = 0), "`frailty_max_iter`")
  expect_error(cox_ph(Surv(time, status) ~ x1 + frailty(centre, theta = 1),
    data = data, ties = "breslow"
  ), "isn't supported")
  expect_error(cox_ph(Surv(time, status) ~ x1 + frailty.gamma(centre, distribution = "gaussian"),
    data = data, ties = "breslow"
  ), "two different")
  expect_error(cox_ph(Surv(time, status) ~ x1 + frailty(centre, x2),
    data = data, ties = "breslow"
  ), "exactly one")
  expect_error(cox_ph(Surv(time, status) ~ x1 + frailty(centre) + frailty(x2),
    data = data, ties = "breslow"
  ), "at most one")
  expect_error(kaplan_meier(Surv(time, status) ~ frailty(centre), data = data),
    "`kaplan_meier()` doesn't support `frailty()` terms", fixed = TRUE
  )
})
