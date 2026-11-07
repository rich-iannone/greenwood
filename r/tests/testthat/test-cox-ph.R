# Replays the Cox fixtures in fixtures/r/ (cox_*), and checks the parts the fixtures don't cover
# (weights, factors, counting-process residuals, and so on) against coxph() directly.

TOL <- 1e-9

lung_fit <- function(ties = "efron", ...) {
  cox_ph(Surv(time, status) ~ age + sex, data = survival::lung, ties = ties, ...)
}

expect_cox_matches <- function(fit, expected) {
  table <- as.data.frame(fit)
  hr <- as.data.frame(fit, exponentiate = TRUE)
  expect_identical(table$term, r_strings(expected$terms))
  expect_identical(fit$n, as.integer(expected$n))
  expect_equal(fit$n_event, expected$nevent)
  expect_equal(table$estimate, r_numbers(expected$coef), tolerance = TOL)
  expect_equal(table$std_error, r_numbers(expected$se), tolerance = TOL)
  expect_equal(table$statistic, r_numbers(expected$z), tolerance = 1e-9)
  expect_equal(table$p_value, r_numbers(expected$p), tolerance = 1e-9)
  expect_equal(hr$estimate, r_numbers(expected$exp_coef), tolerance = TOL)
  expect_equal(hr$conf_low, r_numbers(expected$conf_low), tolerance = TOL)
  expect_equal(hr$conf_high, r_numbers(expected$conf_high), tolerance = TOL)
  expect_equal(fit$loglik, c(expected$loglik_null, expected$loglik), tolerance = TOL)
  expect_equal(fit$lr_test$statistic, expected$lr$stat, tolerance = TOL)
  expect_equal(fit$lr_test$p_value, expected$lr$p, tolerance = 1e-9)
  expect_equal(fit$score_test$statistic, expected$score$stat, tolerance = TOL)
  expect_equal(fit$score_test$p_value, expected$score$p, tolerance = 1e-9)
  # The fixture holds the Wald statistic rounded to 2 decimals, as summary.coxph() prints it.
  expect_lt(abs(fit$wald_test$statistic - expected$wald$stat), 0.005)
  expect_identical(fit$wald_test$df, as.integer(expected$wald$df))
}

test_that("the fits on lung match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expect_cox_matches(lung_fit("efron"), read_fixture("cox_lung_age_sex_efron"))
  expect_cox_matches(lung_fit("breslow"), read_fixture("cox_lung_age_sex_breslow"))
  three <- cox_ph(Surv(time, status) ~ age + sex + ph.ecog, data = survival::lung)
  expect_cox_matches(three, read_fixture("cox_lung_three_efron"))
  expect_identical(three$n_dropped, 1L)
})

test_that("the stratified fit matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_strata")
  fit <- cox_ph(Surv(time, status) ~ age + ph.ecog + strata(sex), data = survival::lung)
  expect_identical(names(coef(fit)), r_strings(expected$terms))
  expect_equal(unname(coef(fit)), r_numbers(expected$coef), tolerance = TOL)
  expect_equal(unname(sqrt(diag(vcov(fit)))), r_numbers(expected$se), tolerance = TOL)
  expect_equal(fit$loglik, c(expected$loglik_null, expected$loglik), tolerance = TOL)
  expect_identical(fit$n, as.integer(expected$n))
  expect_equal(fit$n_event, expected$nevent)
  expect_equal(fit$lr_test$statistic, expected$lr, tolerance = TOL)
  expect_equal(fit$score_test$statistic, expected$score, tolerance = TOL)
  expect_lt(abs(fit$wald_test$statistic - expected$wald), 0.005)
  expect_identical(fit$strata$strata, c("sex=1", "sex=2"))
})

test_that("robust and clustered variances match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_robust")
  fit <- lung_fit("breslow", robust = TRUE)
  table <- as.data.frame(fit)
  expect_equal(table$estimate, r_numbers(expected$coef), tolerance = TOL)
  expect_equal(unname(sqrt(diag(fit$naive_var))), r_numbers(expected$naive_se), tolerance = TOL)
  expect_equal(table$std_error, r_numbers(expected$robust_se), tolerance = TOL)
  expect_equal(table$statistic, r_numbers(expected$z), tolerance = TOL)
  expect_equal(table$p_value, r_numbers(expected$p), tolerance = TOL)

  expected <- read_fixture("cox_cluster")
  by_term <- cox_ph(
    Surv(time, status) ~ age + sex + cluster(inst),
    data = survival::lung,
    ties = "breslow"
  )
  by_argument <- cox_ph(
    Surv(time, status) ~ age + sex,
    data = survival::lung,
    ties = "breslow",
    cluster = inst
  )
  for (fit in list(by_term, by_argument)) {
    expect_identical(fit$n, as.integer(expected$n))
    expect_true(fit$robust)
    expect_equal(unname(coef(fit)), r_numbers(expected$coef), tolerance = TOL)
    expect_equal(unname(sqrt(diag(vcov(fit)))), r_numbers(expected$robust_se), tolerance = TOL)
  }
})

test_that("weighted, stratified, and clustered fits and residuals match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_weighted_residuals")
  lung <- survival::lung
  lung$w <- r_numbers(expected$weights)
  lung <- lung[stats::complete.cases(lung[c("time", "status", "age", "sex", "ph.ecog")]), ]
  heart <- as.data.frame(lapply(expected$heart, r_numbers))
  # jsonlite writes a matrix as a list of rows.
  by_row <- function(rows) unlist(lapply(rows, r_numbers))
  for (ties in c("efron", "breslow")) {
    fits <- list(
      plain = cox_ph(Surv(time, status) ~ age + sex, data = lung, ties = ties),
      weighted = cox_ph(Surv(time, status) ~ age + sex, data = lung, weights = w, ties = ties,
        robust = TRUE
      ),
      strata = cox_ph(Surv(time, status) ~ age + ph.ecog + strata(sex), data = lung, weights = w,
        ties = ties, robust = TRUE
      ),
      counting = cox_ph(Surv(start, stop, event) ~ age + surgery + transplant, data = heart,
        ties = ties, cluster = id
      )
    )
    for (case in names(fits)) {
      fit <- fits[[case]]
      want <- expected[[ties]][[case]]
      naive <- if (is.null(fit$naive_var)) vcov(fit) else fit$naive_var
      expect_equal(unname(coef(fit)), r_numbers(want$coef), tolerance = TOL)
      expect_equal(unname(sqrt(diag(vcov(fit)))), r_numbers(want$se), tolerance = TOL)
      expect_equal(unname(sqrt(diag(naive))), r_numbers(want$naive_se), tolerance = TOL)
      expect_equal(fit$loglik, r_numbers(want$loglik), tolerance = TOL)
      expect_equal(fit$score_test$statistic, want$score_test, tolerance = TOL)
      expect_equal(unname(residuals(fit, type = "martingale")), r_numbers(want$martingale),
        tolerance = TOL
      )
      expect_equal(c(t(unname(residuals(fit, type = "score")))), by_row(want$score),
        tolerance = TOL
      )
      expect_equal(c(t(unname(residuals(fit, type = "dfbeta")))), by_row(want$dfbeta),
        tolerance = TOL
      )
    }
  }
})

test_that("the counting-process fit matches the fixture", {
  skip_if_no_shared_cases()
  expected <- read_fixture("cox_timevarying")
  heart <- data.frame(
    start = r_numbers(expected$start),
    stop = r_numbers(expected$stop),
    event = r_numbers(expected$event),
    age = r_numbers(expected$age),
    surgery = r_numbers(expected$surgery),
    transplant = r_numbers(expected$transplant)
  )
  fit <- cox_ph(
    Surv(start, stop, event) ~ age + surgery + transplant,
    data = heart,
    ties = "breslow"
  )
  expect_identical(fit$n, as.integer(expected$n))
  expect_equal(fit$n_event, expected$nevent)
  expect_equal(unname(coef(fit)), r_numbers(expected$coef), tolerance = TOL)
  expect_equal(unname(sqrt(diag(vcov(fit)))), r_numbers(expected$se), tolerance = TOL)
  expect_equal(fit$loglik[2], expected$loglik, tolerance = TOL)
})

test_that("baselines, predictions, and diagnostics match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  for (ties in c("breslow", "efron")) {
    expected <- read_fixture(paste0("cox_diag_", ties))
    fit <- lung_fit(ties)
    baseline <- baseline_hazard(fit)
    expect_equal(baseline$time, r_numbers(expected$basehaz_time))
    expect_equal(baseline$estimate, r_numbers(expected$basehaz_cumhaz), tolerance = TOL)
    expect_equal(predict(fit), r_numbers(expected$lp), tolerance = TOL)
    expect_equal(unname(residuals(fit)), r_numbers(expected$martingale), tolerance = TOL)
    schoenfeld <- residuals(fit, type = "schoenfeld")
    expect_equal(
      unname(sort(schoenfeld[, "age"])),
      sort(r_numbers(expected$schoenfeld$age)),
      tolerance = TOL
    )
    expect_equal(
      unname(sort(schoenfeld[, "sex"])),
      sort(r_numbers(expected$schoenfeld$sex)),
      tolerance = TOL
    )
    expect_equal(fit$concordance, expected$concordance, tolerance = TOL)
    expect_equal(fit$concordance_se, expected$concordance_se, tolerance = TOL)

    for (transform in c("identity", "log", "km", "rank")) {
      zph <- cox_zph(fit, transform = transform)$table
      reference <- expected[[paste0("zph_", transform)]]
      expect_identical(zph$term, c("age", "sex", "GLOBAL"))
      expect_equal(
        zph$statistic,
        c(reference$age$chisq, reference$sex$chisq, reference$global$chisq),
        tolerance = TOL
      )
      expect_equal(
        zph$p_value,
        c(reference$age$p, reference$sex$p, reference$global$p),
        tolerance = TOL
      )
      expect_identical(zph$df, c(1L, 1L, 2L))
    }

    newdata <- data.frame(
      age = r_numbers(expected$surv_newdata_age),
      sex = r_numbers(expected$surv_newdata_sex)
    )
    curves <- predict(
      fit,
      newdata = newdata,
      type = "survival",
      times = r_numbers(expected$surv_times)
    )
    expect_equal(
      curves$estimate,
      c(r_numbers(expected$surv$subj1), r_numbers(expected$surv$subj2)),
      tolerance = TOL
    )
  }
})

test_that("survival confidence limits match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  fit <- lung_fit("breslow")
  for (name in c("cox_survci_breslow", "cox_conditional_ci")) {
    expected <- read_fixture(name)
    newdata <- data.frame(
      age = r_numbers(expected$newdata_age),
      sex = r_numbers(expected$newdata_sex)
    )
    curves <- predict(
      fit,
      newdata = newdata,
      type = "survival",
      times = r_numbers(expected$times),
      conditional_after = expected$start_time
    )
    both <- function(field) c(r_numbers(field$subj1), r_numbers(field$subj2))
    expect_equal(curves$estimate, both(expected$surv), tolerance = TOL)
    expect_equal(curves$conf_low, both(expected$lower), tolerance = TOL)
    expect_equal(curves$conf_high, both(expected$upper), tolerance = TOL)
    expect_equal(curves$std_error / curves$estimate, both(expected$se_chaz), tolerance = TOL)
    expect_identical(curves$subject, rep(1:2, each = length(expected$times)))
  }
})

test_that("baseline hazard limits match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_basehaz_ci")
  fit <- lung_fit("breslow")
  baseline <- baseline_hazard(fit)
  at <- findInterval(r_numbers(expected$times), baseline$time)
  expect_equal(baseline$estimate[at], r_numbers(expected$cumhaz), tolerance = TOL)
  expect_equal(exp(-baseline$estimate[at]), r_numbers(expected$survival), tolerance = TOL)
  expect_equal(baseline$std_error[at], r_numbers(expected$se_cumhaz), tolerance = TOL)
  # Log-scale limits for the cumulative hazard are log-log limits for survival.
  expect_equal(exp(-baseline$conf_high[at]), r_numbers(expected$lower_loglog), tolerance = TOL)
  expect_equal(exp(-baseline$conf_low[at]), r_numbers(expected$upper_loglog), tolerance = TOL)

  plain <- baseline_hazard(fit, conf_type = "plain")
  expect_equal(plain$conf_low, pmax(plain$estimate - qnorm(0.975) * plain$std_error, 0))
  expect_equal(
    baseline_hazard(fit, centered = TRUE)$estimate,
    baseline$estimate * exp(sum(coef(fit) * fit$means)),
    tolerance = 1e-12
  )
})

test_that("residuals match the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_residuals_breslow")
  fit <- lung_fit("breslow")
  expect_equal(unname(residuals(fit, type = "deviance")), r_numbers(expected$deviance), tolerance = TOL)
  score <- residuals(fit, type = "score")
  expect_equal(unname(score[, "age"]), r_numbers(expected$score_age), tolerance = TOL)
  expect_equal(unname(score[, "sex"]), r_numbers(expected$score_sex), tolerance = TOL)
  dfbeta <- residuals(fit, type = "dfbeta")
  expect_equal(unname(dfbeta[, "age"]), r_numbers(expected$dfbeta_age), tolerance = TOL)
  expect_equal(unname(dfbeta[, "sex"]), r_numbers(expected$dfbeta_sex), tolerance = TOL)
  dfbetas <- residuals(fit, type = "dfbetas")
  expect_equal(unname(dfbetas[, "age"]), r_numbers(expected$dfbetas_age), tolerance = TOL)
  expect_equal(unname(dfbetas[, "sex"]), r_numbers(expected$dfbetas_sex), tolerance = TOL)
  expect_equal(unname(residuals(fit, type = "leverage")), r_numbers(expected$leverage), tolerance = TOL)
  scaled <- residuals(fit, type = "scaledsch")
  expect_equal(as.numeric(rownames(scaled)), r_numbers(expected$scaledsch_times))
  expect_equal(
    unname(sort(scaled[, "age"])),
    sort(r_numbers(expected$scaledsch_age)),
    tolerance = TOL
  )
  expect_equal(
    unname(sort(scaled[, "sex"])),
    sort(r_numbers(expected$scaledsch_sex)),
    tolerance = TOL
  )
})

test_that("the windowed proportional-hazards test matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("cox_zph_windowed")
  zph <- cox_zph(
    lung_fit(),
    transform = expected$transform,
    breaks = r_numbers(expected$breaks)
  )
  windows <- zph$windows
  expect_identical(unique(windows$window), c("(0, 180]", "(180, 365]", "(365, Inf]"))
  for (k in seq_along(expected$windows)) {
    reference <- expected$windows[[k]]
    rows <- windows[windows$window == unique(windows$window)[k], ]
    expect_equal(rows$start[1], reference$interval[[1]])
    expect_equal(rows$end[1], r_numbers(reference$interval[2]))
    expect_equal(rows$n_event[1], reference$n_events)
    expect_equal(
      rows$statistic,
      c(
        reference$per_term$age$chisq,
        reference$per_term$sex$chisq,
        reference$global_test$chisq
      ),
      tolerance = TOL
    )
    expect_equal(
      rows$p_value,
      c(
        reference$per_term$age$p_value,
        reference$per_term$sex$p_value,
        reference$global_test$p_value
      ),
      tolerance = TOL
    )
  }
})

test_that("quantiles and restricted means of the curves match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  fit <- lung_fit("breslow")
  median_fixture <- read_fixture("predict_median")
  newdata <- data.frame(
    age = r_numbers(median_fixture$newdata$age),
    sex = r_numbers(median_fixture$newdata$sex)
  )
  medians <- predict(fit, newdata = newdata, type = "quantile")
  expect_equal(medians$estimate, r_numbers(median_fixture$cox_breslow$median))
  expect_equal(medians$conf_low, r_numbers(median_fixture$cox_breslow$lower))
  expect_equal(medians$conf_high, r_numbers(median_fixture$cox_breslow$upper))

  quantile_fixture <- read_fixture("predict_quantile")
  p <- r_numbers(quantile_fixture$p)
  quantiles <- predict(fit, newdata = newdata, type = "quantile", p = p)
  expect_identical(quantiles$p, rep(p, 3))
  # The fixture's quantiles are one list per proportion, with one value per subject.
  expected <- do.call(rbind, lapply(quantile_fixture$cox_breslow$quantiles, r_numbers))
  expect_equal(quantiles$estimate, c(expected))

  expectation_fixture <- read_fixture("predict_expectation")
  for (tau in r_numbers(expectation_fixture$tau)) {
    means <- predict(fit, newdata = newdata, type = "expectation", tau = tau)
    expect_equal(
      means$estimate,
      r_numbers(expectation_fixture$cox_breslow[[paste0("rmst_", tau)]]),
      tolerance = TOL
    )
    expect_true(all(means$conf_low < means$estimate & means$estimate < means$conf_high))
  }
})

# -- against coxph() directly --------------------------------------------------------------------

# Fit the same model with cox_ph() and coxph(), with the formula's specials resolved in survival.
both_fits <- function(formula, data, ...) {
  # coxph() refits the model frame for residuals and curves, so `data` has to be visible from the
  # formula's environment.
  env <- list2env(list(data = data), parent = asNamespace("survival"))
  environment(formula) <- env
  arguments <- c(list(formula = formula, data = quote(data)), list(...))
  fit <- eval(as.call(c(quote(cox_ph), arguments)), env)
  reference <- eval(as.call(c(quote(coxph), arguments)), env)
  list(fit = fit, reference = reference)
}

expect_same_fit <- function(pair) {
  fit <- pair$fit
  reference <- pair$reference
  expect_equal(coef(fit), coef(reference), tolerance = TOL)
  expect_equal(unname(vcov(fit)), unname(vcov(reference)), tolerance = TOL)
  expect_equal(fit$loglik, reference$loglik, tolerance = TOL)
  expect_equal(fit$score_test$statistic, unname(reference$score), tolerance = TOL)
  expect_equal(fit$wald_test$statistic, unname(reference$wald.test), tolerance = TOL)
  # Subjects with the same covariates tie in the concordance only if their linear predictors are
  # bit-identical. coxph() computes them with `x %*% coef`, and an optimized BLAS (as on Linux) can
  # leave tied subjects a last bit apart, which moves the concordance. So the reference
  # concordance is survival's, computed from cox_ph()'s linear predictors, whose ties are exact.
  # The coefficients agree above, so these are coxph()'s linear predictors up to rounding.
  concordance <- survival::concordancefit(
    reference$y,
    fit$linear_predictors,
    fit$stratum,
    reference$weights,
    cluster = fit$cluster,
    reverse = TRUE,
    timefix = FALSE
  )
  expect_equal(fit$concordance, concordance$concordance, tolerance = TOL)
  expect_equal(fit$concordance_se, sqrt(concordance$var[[1L]]), tolerance = TOL)
  expect_equal(unname(residuals(fit)), unname(residuals(reference)), tolerance = TOL)
  for (type in c("score", "dfbeta", "dfbetas")) {
    expect_equal(
      unname(residuals(fit, type = type)),
      unname(residuals(reference, type = type)),
      tolerance = TOL
    )
  }
  expect_equal(
    sort(residuals(fit, type = "schoenfeld")),
    sort(residuals(reference, type = "schoenfeld")),
    tolerance = TOL
  )
  for (transform in c("km", "rank", "identity", "log")) {
    expect_equal(
      cox_zph(fit, transform = transform)$table$statistic,
      unname(survival::cox.zph(reference, transform = transform)$table[, "chisq"]),
      tolerance = TOL
    )
  }
}

test_that("weighted fits match coxph() for both tie methods", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$wt <- rep(c(0.5, 1.25, 2, 0.8), length.out = nrow(lung))
  for (ties in c("efron", "breslow")) {
    pair <- both_fits(
      Surv(time, status) ~ age + sex,
      lung,
      weights = quote(wt),
      ties = ties,
      robust = TRUE
    )
    expect_same_fit(pair)
  }
})

test_that("factors, strata, and counting-process data match coxph()", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$ecog <- factor(lung$ph.ecog)
  for (ties in c("efron", "breslow")) {
    factor_pair <- both_fits(Surv(time, status) ~ age + ecog + sex, lung, ties = ties)
    expect_same_fit(factor_pair)
    expect_identical(names(coef(factor_pair$fit)), c("age", "ecog1", "ecog2", "ecog3", "sex"))
    expect_identical(cox_zph(factor_pair$fit)$table$df, c(1L, 3L, 1L, 5L))

    expect_same_fit(both_fits(Surv(time, status) ~ age + ph.ecog + strata(sex), lung, ties = ties))
    expect_same_fit(both_fits(
      Surv(start, stop, event) ~ age + surgery + transplant,
      survival::heart,
      ties = ties
    ))
  }
})

test_that("survival curves match survfit() for strata, weights, and counting data", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$wt <- rep(c(0.5, 1.25, 2, 0.8), length.out = nrow(lung))
  times <- c(50, 100, 300, 500, 700)
  check <- function(pair, newdata) {
    curves <- predict(pair$fit, newdata = newdata, type = "survival", times = times)
    reference <- summary(survival::survfit(pair$reference, newdata = newdata), times = times)
    expect_equal(curves$estimate, c(reference$surv), tolerance = TOL)
    expect_equal(curves$conf_low, c(reference$lower), tolerance = TOL)
    expect_equal(curves$conf_high, c(reference$upper), tolerance = TOL)
  }
  for (ties in c("efron", "breslow")) {
    check(
      # coxph() turns on the robust variance for non-integer weights, and survfit() then gives
      # curve limits from it. The model-based limits need `robust = FALSE`.
      both_fits(
        Surv(time, status) ~ age + sex,
        lung,
        weights = quote(wt),
        ties = ties,
        robust = FALSE
      ),
      data.frame(age = c(50, 70), sex = c(1, 2))
    )
    check(
      both_fits(Surv(time, status) ~ age + ph.ecog + strata(sex), lung, ties = ties),
      data.frame(age = 60, ph.ecog = 1, sex = 2)
    )
    check(
      both_fits(Surv(start, stop, event) ~ age + surgery + transplant, survival::heart, ties = ties),
      data.frame(age = -10, surgery = 1, transplant = "1")
    )
  }
})

# -- behavior --------------------------------------------------------------------------------------

test_that("the methods give the usual pieces", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = small_cox)
  expect_s3_class(fit, "cox_ph")
  expect_named(coef(fit), c("age", "sex"))
  expect_identical(dim(vcov(fit)), c(2L, 2L))
  expect_s3_class(logLik(fit), "logLik")
  expect_identical(attr(logLik(fit), "df"), 2L)
  expect_equal(AIC(fit), -2 * fit$loglik[2] + 4)
  expect_equal(nobs(fit), sum(small_cox$status == 2))
  expect_true(fit$converged)
  expect_named(
    as.data.frame(fit),
    c("term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high")
  )
  expect_equal(predict(fit, type = "risk"), exp(predict(fit)))
  expect_equal(
    predict(fit, newdata = small_cox[1:3, ]),
    predict(fit)[1:3]
  )
})

test_that("survival predictions have one block per subject", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = small_cox)
  newdata <- data.frame(age = c(55, 70, 62), sex = c(1, 2, 1))
  curves <- predict(fit, newdata = newdata, type = "survival", times = c(0, 200, 400))
  expect_named(
    curves,
    c("subject", "strata", "time", "estimate", "std_error", "conf_low", "conf_high")
  )
  expect_identical(curves$subject, rep(1:3, each = 3))
  expect_identical(unique(curves$strata), "all")
  expect_equal(curves$estimate[curves$time == 0], rep(1, 3))
  cumhaz <- predict(fit, newdata = newdata, type = "cumhaz", times = c(0, 200, 400))
  expect_equal(exp(-cumhaz$estimate), curves$estimate)

  conditional <- predict(
    fit,
    newdata = newdata,
    type = "survival",
    times = c(100, 400),
    conditional_after = 200
  )
  expect_equal(conditional$estimate[conditional$time == 100], rep(1, 3))
  full <- predict(fit, newdata = newdata, type = "survival", times = c(199.999, 400))
  expect_equal(
    conditional$estimate[conditional$time == 400],
    full$estimate[full$time == 400] / full$estimate[full$time == 199.999]
  )
})

test_that("stratified predictions use each subject's stratum", {
  fit <- cox_ph(Surv(time, status) ~ age + strata(sex), data = small_cox)
  newdata <- data.frame(age = c(60, 60), sex = c(2, 1))
  curves <- predict(fit, newdata = newdata, type = "survival", times = 300)
  expect_identical(curves$strata, c("sex=2", "sex=1"))
  baseline <- baseline_hazard(fit)
  expect_identical(names(baseline)[1:2], c("strata", "sex"))
  expect_error(
    predict(fit, newdata = data.frame(age = 60, sex = 3), type = "survival", times = 300),
    "strata that the model wasn't fit to: sex=3"
  )
})

test_that("subjects with the same covariates get identical linear predictors", {
  skip_if_not_installed("survival")
  # The concordance tests these for exact equality, so a last-bit difference would break a tie.
  lung <- survival::lung
  lung$ecog <- factor(lung$ph.ecog)
  fits <- list(
    list(cox_ph(Surv(time, status) ~ age + sex, data = small_cox), small_cox[c("age", "sex")]),
    list(
      cox_ph(Surv(time, status) ~ age + ecog + sex, data = lung),
      stats::na.omit(lung[c("time", "status", "age", "ecog", "sex")])[c("age", "ecog", "sex")]
    )
  )
  for (pair in fits) {
    lp <- pair[[1L]]$linear_predictors
    covariates <- do.call(paste, pair[[2L]])
    expect_identical(length(unique(lp)), length(unique(covariates)))
    expect_identical(unname(lp), unname(predict(pair[[1L]], newdata = pair[[2L]])))
  }
})

test_that("Efron and Breslow agree without ties", {
  untied <- small_cox[!duplicated(small_cox$time), ]
  efron <- cox_ph(Surv(time, status) ~ age + sex, data = untied)
  breslow <- cox_ph(Surv(time, status) ~ age + sex, data = untied, ties = "breslow")
  expect_equal(coef(efron), coef(breslow), tolerance = 1e-12)
  expect_equal(efron$loglik, breslow$loglik, tolerance = 1e-12)
})

test_that("case weights act like repeated rows", {
  weighted <- cox_ph(
    Surv(time, status) ~ age + sex,
    data = small_cox,
    weights = rep(c(1, 2), length.out = nrow(small_cox)),
    ties = "breslow"
  )
  repeated <- small_cox[rep(seq_len(nrow(small_cox)), rep(c(1, 2), length.out = nrow(small_cox))), ]
  expanded <- cox_ph(Surv(time, status) ~ age + sex, data = repeated, ties = "breslow")
  expect_equal(coef(weighted), coef(expanded), tolerance = 1e-10)
  expect_equal(vcov(weighted), vcov(expanded), tolerance = 1e-10)
  expect_equal(weighted$loglik, expanded$loglik, tolerance = 1e-10)
})

test_that("bad input gives clear errors", {
  expect_error(
    cox_ph(Surv(time, status) ~ 1, data = small_cox),
    "needs at least one covariate"
  )
  expect_error(
    cox_ph(Surv(time, status) ~ age + I(2 * age), data = small_cox),
    "collinear"
  )
  expect_error(
    cox_ph(Surv(time, status) ~ age, data = transform(small_cox, status = 0)),
    "at least one event"
  )
  expect_error(
    cox_ph(Surv(time, status) ~ age + offset(sex), data = small_cox),
    "offset"
  )
  expect_error(cox_ph(Surv(time, status) ~ age, data = small_cox, ties = "exact"))
  fit <- cox_ph(Surv(time, status) ~ age, data = small_cox)
  expect_error(cox_zph(lm(time ~ age, data = small_cox)), "`cox_ph\\(\\)` fit")
  expect_error(predict(fit, newdata = data.frame(sex = 1)), "age")
})

test_that("the printouts are stable", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = small_cox)
  expect_snapshot({
    fit
    cox_zph(fit, transform = "identity", breaks = 300)
  })
})
