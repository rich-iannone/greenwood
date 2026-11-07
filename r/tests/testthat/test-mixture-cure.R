# Replays the cure_ph_e1684 fixture in fixtures/r/, written from smcure::smcure() on the E1684
# melanoma trial (the fixture carries the data). The EM steps are checked against glm() and
# survival::coxph(), and larger models against Greenwood's Python values.

e1684 <- function() {
  expected <- read_fixture("cure_ph_e1684")
  as.data.frame(lapply(expected$data, r_numbers))
}

test_that("the fit reproduces the fixture", {
  skip_if_no_shared_cases()
  expected <- read_fixture("cure_ph_e1684")
  data <- e1684()
  fit <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = data, n_boot = NULL)
  expect_identical(c(fit$n, fit$n_event), c(284L, 196L))
  expect_equal(unname(fit$cure_coefficients), r_numbers(expected$cure_coef), tolerance = 1e-6)
  expect_equal(unname(fit$latency_coefficients), r_numbers(expected$latency_coef),
    tolerance = 1e-6
  )
  expect_named(fit$cure_coefficients, unlist(expected$cure_coef_names))
  expect_named(fit$latency_coefficients, unlist(expected$latency_coef_names))
  expect_equal(fit$baseline$time, r_numbers(expected$baseline_times), tolerance = 1e-12)
  expect_equal(fit$baseline$survival, r_numbers(expected$baseline_survival), tolerance = 1e-6)
  expect_named(coef(fit), c("cure.(Intercept)", "cure.TRT", "latency.TRT"))
})

test_that("the fits match Python", {
  skip_if_no_shared_cases()
  data <- e1684()
  fit <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = data, n_boot = NULL)
  expect_equal(unname(coef(fit)), c(1.2957164402987094, -0.5747481476999713, -0.1318354745285785),
    tolerance = 1e-8
  )
  expect_identical(fit$iterations, 31L)
  expect_true(fit$converged)
  survival <- predict(fit, newdata = data[1:2, ], times = c(0.5, 1, 3, 8, 10))
  expect_equal(survival$estimate, rep(c(
    0.7132979112546005, 0.604203481541711, 0.4329361235766218, 0.3670408630799252,
    0.327179793948314
  ), 2), tolerance = 1e-8)

  more <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + AGE + SEX,
    cure = ~ TRT + AGE, data = data, n_boot = NULL
  )
  expect_equal(unname(coef(more)), c(
    1.3316450344800919, -0.5897625373306786, 0.02056210458313994, -0.15006383009419397,
    -0.00762567849105541, 0.08244391080840563
  ), tolerance = 1e-7)
  expect_identical(more$iterations, 35L)
  expect_equal(predict(more, newdata = data[1:3, ], type = "uncured"),
    c(0.6259685556109846, 0.6539434242335616, 0.8591680369139609),
    tolerance = 1e-8
  )
  expect_equal(predict(more, newdata = data[1:2, ], times = c(0.5, 1, 3, 8, 10))$estimate, c(
    0.7250778264026982, 0.6220652924291977, 0.4650284609677655, 0.40596693270523315,
    0.37403144438901537, 0.722253863909574, 0.6159587660838899, 0.44954776159792326,
    0.3840912109017744, 0.34605657576643845
  ), tolerance = 1e-8)
})

test_that("each EM step is a quasi-binomial glm() and an offset coxph() fit", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  data <- e1684()
  set.seed(4)
  uncured <- ifelse(data$FAILCENS == 1, 1, stats::runif(nrow(data)))
  uncured[data$FAILCENS == 0][1:5] <- 0
  k <- rep(1:3, length.out = nrow(data))
  z <- cbind(1, data$TRT, data$AGE)
  x <- cbind(data$TRT, data$SEX)

  gamma <- cure_logistic(z, uncured, k, numeric(3))
  reference <- suppressWarnings(stats::glm(uncured ~ TRT + AGE,
    family = stats::quasibinomial(), data = data, weights = k,
    control = stats::glm.control(epsilon = 1e-14)
  ))
  expect_equal(gamma, unname(coef(reference)), tolerance = 1e-9)

  beta <- cure_cox(x, data$FAILTIME, data$FAILCENS, uncured, k, numeric(2))
  cox <- survival::coxph(
    survival::Surv(FAILTIME, FAILCENS) ~ TRT + SEX + offset(log(uncured)),
    data = cbind(data, uncured = uncured, k = k), weights = k, subset = uncured > 0,
    ties = "breslow", control = survival::coxph.control(eps = 1e-11, iter.max = 50)
  )
  expect_equal(beta, unname(coef(cox)), tolerance = 1e-9)

  # The baseline survival is the Breslow estimate with the weights in the risk sets.
  baseline <- cure_breslow(data$FAILTIME, data$FAILCENS, x, beta, uncured, k)
  events <- data$FAILCENS == 1
  t <- baseline$time[10]
  hazard <- vapply(baseline$time[1:10], function(s) {
    sum(k[events & data$FAILTIME == s]) /
      sum((k * uncured * exp(drop(x %*% beta)))[data$FAILTIME >= s])
  }, numeric(1))
  expect_equal(baseline$survival[10], exp(-sum(hazard)), tolerance = 1e-12)
  expect_identical(cure_baseline_at(baseline, c(0, t, max(baseline$time) + 1))[c(1, 3)], c(1, 0))
})

test_that("weights act as copies of rows", {
  skip_if_no_shared_cases()
  data <- e1684()
  data$k <- rep(1:3, length.out = nrow(data))
  weighted <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + SEX,
    cure = ~ TRT + AGE, data = data, weights = k, n_boot = NULL
  )
  copies <- data[rep(seq_len(nrow(data)), data$k), ]
  copied <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + SEX,
    cure = ~ TRT + AGE, data = copies, n_boot = NULL
  )
  expect_equal(coef(weighted), coef(copied), tolerance = 1e-10)
  expect_identical(weighted$iterations, copied$iterations)
  expect_equal(weighted$baseline, copied$baseline, tolerance = 1e-10)
  expect_true(weighted$has_weights)

  # Rows with weight 0 are left out.
  data$k[1:20] <- 0
  zeroed <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT,
    cure = ~TRT, data = data, weights = k, n_boot = NULL
  )
  dropped <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT,
    cure = ~TRT, data = data[-(1:20), ], weights = k, n_boot = NULL
  )
  expect_equal(coef(zeroed), coef(dropped), tolerance = 1e-12)
})

test_that("bootstrap standard errors come from stratified resamples", {
  skip_if_no_shared_cases()
  data <- e1684()
  f <- Surv(FAILTIME, FAILCENS) ~ TRT
  fit <- mixture_cure(f, cure = ~TRT, data = data, n_boot = NULL)
  table <- as.data.frame(fit)
  expect_named(table, c(
    "submodel", "term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"
  ))
  expect_identical(table$submodel, c("cure", "cure", "latency"))
  expect_identical(table$term, c("(Intercept)", "TRT", "TRT"))
  expect_true(all(is.na(table$std_error)))
  expect_error(vcov(fit), "`n_boot`")
  expect_null(fit$boot)

  set.seed(20)
  booted <- mixture_cure(f, cure = ~TRT, data = data, n_boot = 30)
  expect_identical(coef(booted), coef(fit))
  expect_identical(dim(booted$boot), c(30L, 3L))
  expect_equal(booted$std_error, apply(booted$boot, 2L, stats::sd))
  expect_equal(sqrt(diag(vcov(booted))), booted$std_error)
  table <- as.data.frame(booted)
  expect_equal(table$statistic, unname(coef(booted) / booted$std_error))
  expect_equal(table$conf_low, unname(coef(booted) - stats::qnorm(0.975) * booted$std_error))
  narrow <- as.data.frame(booted, conf_level = 0.8)
  expect_true(all(narrow$conf_high < table$conf_high))
  set.seed(20)
  again <- mixture_cure(f, cure = ~TRT, data = data, n_boot = 30)
  expect_identical(again$boot, booted$boot)
  expect_output(print(booted), "bootstrap resamples = 30", fixed = TRUE)
  # Python's standard errors (100 resamples) are 0.298, 0.345, and 0.166. These agree to within
  # the Monte Carlo error.
  expect_lt(max(abs(booted$std_error / c(0.29766383, 0.34542846, 0.1659128) - 1)), 0.4)
})

test_that("predictions level off at the cured fraction", {
  skip_if_no_shared_cases()
  data <- e1684()
  fit <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + SEX,
    cure = ~ TRT + AGE, data = data, n_boot = NULL
  )
  newdata <- data.frame(TRT = c(0, 1), SEX = c(1, 0), AGE = c(30, 60))
  lp <- predict(fit, newdata = newdata, type = "lp")
  expect_equal(lp, drop(as.matrix(newdata[c("TRT", "SEX")]) %*% fit$latency_coefficients))
  uncured <- predict(fit, newdata = newdata, type = "uncured")
  expect_equal(uncured, stats::plogis(drop(cbind(1, newdata$TRT, newdata$AGE) %*%
    fit$cure_coefficients)))
  last <- max(fit$baseline$time)
  survival <- predict(fit, newdata = newdata, times = c(0, 1, last, last + 1))
  expect_named(survival, c("subject", "time", "estimate"))
  expect_identical(survival$subject, rep(1:2, each = 4))
  expect_identical(survival$estimate[survival$time == 0], c(1, 1))
  expect_equal(survival$estimate[survival$time == last + 1], 1 - uncured)
  s0 <- fit$baseline$survival[findInterval(1, fit$baseline$time)]
  expect_equal(survival$estimate[survival$time == 1], 1 - uncured + uncured * s0^exp(lp))
  expect_identical(length(predict(fit, type = "uncured")), 284L)

  data$group <- factor(ifelse(data$TRT == 1, "treated", "control"))
  by_factor <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ group,
    cure = ~group, data = data, n_boot = NULL
  )
  plain <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = data, n_boot = NULL)
  expect_named(by_factor$cure_coefficients, c("(Intercept)", "grouptreated"))
  expect_equal(unname(coef(by_factor)), unname(coef(plain)), tolerance = 1e-10)
  expect_equal(
    predict(by_factor, newdata = data.frame(group = "treated"), times = c(1, 5)),
    predict(plain, newdata = data.frame(TRT = 1), times = c(1, 5)),
    tolerance = 1e-10
  )

  # One cured fraction for everyone, and no latency covariates.
  constant <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ 1, cure = ~1, data = data, n_boot = NULL)
  expect_named(coef(constant), "cure.(Intercept)")
  expect_identical(predict(constant, newdata = data[1:2, ], type = "lp"), c(0, 0))
  expect_output(print(constant), "No covariates.", fixed = TRUE)
})

test_that("print, missing values, and nobs work", {
  skip_if_no_shared_cases()
  data <- e1684()
  data$AGE[1:2] <- NA
  fit <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~ TRT + AGE, data = data,
    n_boot = NULL
  )
  expect_identical(c(fit$n, fit$n_dropped), c(282L, 2L))
  expect_identical(nobs(fit), 282L)
  expect_output(print(fit), "<mixture_cure> Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT + AGE",
    fixed = TRUE
  )
  expect_output(print(fit), "n = 282, events = ", fixed = TRUE)
  expect_output(print(fit), "Incidence (log odds of being uncured)", fixed = TRUE)
  expect_output(print(fit), "Latency (log hazard ratios among the uncured)", fixed = TRUE)
  expect_output(print(fit), "2 rows were dropped", fixed = TRUE)
  expect_warning(
    short <- mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = data,
      max_iter = 1, n_boot = NULL
    ),
    "didn't converge in 1 iterations"
  )
  expect_false(short$converged)
  expect_output(print(short), "converged = no", fixed = TRUE)
})

test_that("bad input is rejected", {
  skip_if_no_shared_cases()
  data <- e1684()
  f <- Surv(FAILTIME, FAILCENS) ~ TRT
  expect_error(mixture_cure(f, data = data), "`cure`")
  expect_error(mixture_cure(f, cure = FAILCENS ~ TRT, data = data), "one-sided")
  expect_error(mixture_cure(f, cure = ~TRT, data = data, max_iter = 0), "`max_iter`")
  expect_error(mixture_cure(f, cure = ~TRT, data = data, tol = 0), "`tol`")
  expect_error(mixture_cure(f, cure = ~TRT, data = data, n_boot = 0), "`n_boot`")
  expect_error(mixture_cure(f, cure = ~TRT, data = data, conf_level = 2), "conf_level")
  expect_error(
    mixture_cure(Surv(FAILTIME, FAILCENS * 0) ~ TRT, cure = ~TRT, data = data), "one event"
  )
  expect_error(
    mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + strata(SEX), cure = ~TRT, data = data),
    "strata"
  )
  expect_error(
    mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT + offset(AGE), cure = ~TRT, data = data),
    "offset"
  )
  expect_error(mixture_cure(f, cure = ~ TRT + offset(AGE), data = data), "offset")
  data$start <- 0
  expect_error(
    mixture_cure(Surv(start, FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = data),
    "right-censored"
  )
  fit <- mixture_cure(f, cure = ~ TRT + AGE, data = data, n_boot = NULL)
  expect_error(predict(fit, newdata = data[1:2, ]), "needs `times`")
  expect_error(predict(fit, newdata = list(TRT = 1), type = "lp"), "data frame")
  expect_error(predict(fit, newdata = data.frame(TRT = 1), type = "uncured"), "AGE")
  expect_error(predict(fit, times = "a"), "`times`")
  expect_error(predict(fit, type = "hazard"), "should be one of")
})
