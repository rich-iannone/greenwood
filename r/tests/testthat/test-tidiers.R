# tidy(), glance(), and augment(). The tables are checked against each model's own methods and
# fields, and the glance() values on lung against the Python package's (the numbers below were
# printed by greenwood.glance() on the same data).

lung_full <- function() {
  skip_if_not_installed("survival")
  survival::lung
}

test_that("tidy() gives the model's table and passes its arguments on", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = small_cox)
  expect_identical(tidy(fit), as.data.frame(fit))
  expect_identical(
    tidy(fit, exponentiate = TRUE, conf_level = 0.9),
    as.data.frame(fit, exponentiate = TRUE, conf_level = 0.9)
  )
  km <- kaplan_meier(Surv(time, status) ~ sex, data = small)
  expect_identical(tidy(km), as.data.frame(km))
  test <- logrank_test(Surv(time, status) ~ sex, data = small)
  expect_identical(tidy(test), as.data.frame(test))

  # The generics are those of the generics package, so broom-style code finds them.
  expect_identical(tidy, generics::tidy)
  expect_identical(glance, generics::glance)
  expect_identical(augment, generics::augment)
})

test_that("tidy() works for every model", {
  lung <- lung_full()
  lung <- lung[stats::complete.cases(lung[c("time", "status", "age", "sex", "ph.ecog")]), ]
  f <- Surv(time, status) ~ age + sex
  coefficient_columns <- c(
    "term", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"
  )
  for (fit in list(
    cox_ph(f, data = lung), aft(f, data = lung), royston_parmar(f, data = lung),
    piecewise_exponential(f, data = lung), buckley_james(f, data = lung)
  )) {
    expect_named(tidy(fit), coefficient_columns)
  }
  expect_named(
    tidy(cox_net(f, data = lung, penalizer = 0.05)), c("term", "estimate", "hazard_ratio")
  )
  expect_named(tidy(ipc_ridge(f, data = lung)), c("term", "estimate"))
  expect_identical(tidy(survival_tree(f, data = lung))$node[1], 1L)

  set.seed(1)
  forest <- random_survival_forest(f, data = lung, n_estimators = 5)
  set.seed(2)
  importance <- variable_importance(forest, n_repeats = 2)
  set.seed(2)
  expect_identical(tidy(forest, n_repeats = 2), importance)
  boost <- gradient_boosting_survival_analysis(f, data = lung, n_estimators = 5)
  expect_identical(tidy(boost), variable_importance(boost))

  mg <- small_cr
  sb <- survival_boost(Surv(time, event) ~ age, data = mg, n_estimators = 2)
  expect_identical(tidy(sb), data.frame(term = "age"))
})

test_that("glance() of a Cox model matches the fields and the Python values", {
  lung <- lung_full()
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
  out <- glance(fit)
  expect_identical(nrow(out), 1L)
  expect_named(out, c(
    "n", "nevent", "loglik", "aic", "lr_statistic", "df", "lr_p_value", "frailty_theta",
    "frailty_lrt_statistic", "frailty_lrt_p_value"
  ))
  expect_identical(out$n, 228L)
  expect_identical(out$nevent, 165)
  expect_equal(out$loglik, -742.8482457837704, tolerance = 1e-9)
  expect_equal(out$aic, 1489.6964915675408, tolerance = 1e-9)
  expect_equal(out$aic, stats::AIC(fit))
  expect_equal(out$lr_statistic, 14.123111213248876, tolerance = 1e-8)
  expect_identical(out$df, 2L)
  expect_equal(out$lr_p_value, 0.0008574432116275308, tolerance = 1e-7)
  expect_true(all(is.na(out[c("frailty_theta", "frailty_lrt_statistic", "frailty_lrt_p_value")])))

  frail <- cox_ph(Surv(time, status) ~ age + frailty(inst), data = lung, ties = "breslow")
  out <- glance(frail)
  expect_identical(out$frailty_theta, frail$frailty$theta)
  expect_identical(out$frailty_lrt_statistic, frail$frailty$test$statistic)
  expect_identical(out$frailty_lrt_p_value, frail$frailty$test$p_value)

  # A cause-specific Cox model is a Cox model.
  csc <- cause_specific_cox(Surv(time, event) ~ age, data = small_cr, cause = "pcm")
  expect_named(glance(csc), names(out))
  expect_identical(tidy(csc), as.data.frame(csc))
})

test_that("glance() of the parametric and flexible models matches the Python values", {
  lung <- lung_full()
  f <- Surv(time, status) ~ age + sex
  out <- glance(aft(f, data = lung))
  expect_identical(out$dist, "weibull")
  expect_equal(out$scale, 0.7540512562105107, tolerance = 1e-6)
  expect_identical(out$threshold, 0)
  expect_identical(out$Q, NA_real_)
  expect_equal(out$loglik, -1147.0544314334734, tolerance = 1e-9)
  expect_equal(out$aic, 2302.1088628669468, tolerance = 1e-9)
  expect_equal(glance(aft(f, data = lung, dist = "exponential"))$aic, 2318.1980742780493,
    tolerance = 1e-9
  )
  out <- glance(aft(f, data = lung, dist = "gengamma"))
  expect_equal(out$Q, 0.985829818969026, tolerance = 1e-7)
  expect_equal(out$aic, 2 * 1147.05223250424 + 2 * 5, tolerance = 1e-9)

  out <- glance(parametric(Surv(time, status) ~ 1, data = lung, dist = "weibull"))
  expect_named(out, c("dist", "n", "nevent", "loglik", "aic", "bic"))
  expect_equal(out$aic, 2311.702376178812, tolerance = 1e-9)
  expect_equal(out$bic, 2318.561067436721, tolerance = 1e-9)

  out <- glance(piecewise_exponential(f, data = lung))
  expect_identical(out$n_intervals, 4L)
  expect_equal(out$aic, 2303.7340805416216, tolerance = 1e-8)
  expect_equal(out$bic, 2324.310154315348, tolerance = 1e-8)
  expect_identical(out$df, 6L)
  expect_equal(out$lr_statistic, 13.782762310371709, tolerance = 1e-6)

  out <- glance(royston_parmar(f, data = lung))
  expect_identical(out$scale, "hazard")
  expect_identical(out$n_knots, 3L)
  expect_identical(out$df, 6L)
  # The knots differ from Python's in the fifth digit, so the likelihood does slightly.
  expect_equal(out$aic, 2305.141073769819, tolerance = 1e-5)
  expect_equal(out$bic, -2 * out$loglik + log(165) * 6)

  out <- glance(cox_net(f, data = lung, penalizer = 0.05))
  expect_equal(out$aic, 1490.5675144097077, tolerance = 1e-6)
  expect_equal(out$bic, 1496.551953768268, tolerance = 1e-6)
  expect_equal(out$effective_df, 1.92676896901372, tolerance = 1e-6)
  expect_identical(out$n_nonzero, 2L)

  out <- glance(ipc_ridge(f, data = lung))
  expect_identical(out$alpha, 1)
  expect_identical(out$n_features, 2L)
  expect_identical(glance(aalen_additive(f, data = lung))$test, "aalen")
})

test_that("glance() of curves has one row per curve, with the same layout for one curve", {
  lung <- lung_full()
  out <- glance(kaplan_meier(Surv(time, status) ~ sex, data = lung))
  expect_named(
    out, c("strata", "sex", "n_start", "events", "median", "median_lower", "median_upper")
  )
  expect_identical(out$strata, c("sex=1", "sex=2"))
  expect_identical(out$n_start, c(138, 90))
  expect_identical(out$events, c(112, 53))
  expect_identical(out$median, c(270, 426))
  expect_identical(out$median_lower, c(212, 348))
  expect_identical(out$median_upper, c(310, 550))
  one <- glance(kaplan_meier(Surv(time, status) ~ 1, data = lung))
  expect_named(one, c("strata", "n_start", "events", "median", "median_lower", "median_upper"))
  expect_identical(one$strata, "all")

  out <- glance(nelson_aalen(Surv(time, status) ~ sex, data = lung))
  expect_equal(out$max_cumhaz, c(3.162838772299265, 2.3219147753804834), tolerance = 1e-12)

  fit <- turnbull(Surv(time, status) ~ 1, data = survival::veteran)
  out <- glance(fit)
  expect_named(out, c("strata", "n", "n_atoms", "n_ambiguous", "n_iter", "converged"))
  # The Python values.
  expect_identical(out$n_atoms, 179L)
  expect_identical(out$n_ambiguous, 82L)
  expect_identical(out$n, 137)
  expect_true(out$converged)

  out <- glance(aalen_johansen(Surv(time, event) ~ sex, data = small_cr))
  expect_named(out, c("strata", "sex", "n_causes", "causes"))
  expect_identical(out$causes, c("pcm, death", "pcm, death"))
  out <- glance(multi_state(Surv(time, event) ~ 1, data = small_cr))
  expect_named(out, c("strata", "n_states", "states", "n_times"))
  expect_identical(out$n_states, 3L)
})

test_that("glance() of the competing-risks and machine-learning models reads their fields", {
  fg <- fine_gray(Surv(time, event) ~ age, data = small_cr, cause = "pcm")
  expect_identical(glance(fg)$loglik, fg$loglik[2])
  pfg <- penalized_fine_gray(Surv(time, event) ~ age, data = small_cr, penalizer = 0.01)
  expect_named(glance(pfg), c("n", "nevent", "penalizer", "l1_ratio", "n_nonzero"))

  f <- Surv(time, status) ~ age + sex
  tree <- survival_tree(f, data = small_cox)
  expect_identical(glance(tree)$n_leaves, tree$n_leaves)
  set.seed(1)
  forest <- random_survival_forest(f, data = small_cox, n_estimators = 3, oob_score = TRUE)
  out <- glance(forest)
  expect_identical(out$oob_concordance, forest$oob_score)
  expect_identical(out$n_features, 2L)
  boost <- gradient_boosting_survival_analysis(f, data = small_cox, n_estimators = 3)
  expect_identical(glance(boost)$learning_rate, boost$learning_rate)
  set.seed(1)
  crf <- competing_risk_forest(Surv(time, event) ~ age, data = small_cr, n_estimators = 2)
  expect_identical(glance(crf)$causes, "pcm, death")
  expect_identical(glance(crf)$oob_concordance, NA_real_)
  set.seed(1)
  sb <- survival_boost(Surv(time, event) ~ age, data = small_cr, n_estimators = 2)
  expect_identical(glance(sb)$n_causes, 2L)

  bj <- buckley_james(f, data = small_cox)
  expect_identical(glance(bj)$n_boot, 0L)
  set.seed(1)
  bj <- buckley_james(f, data = small_cox, n_boot = 4)
  expect_identical(glance(bj)$n_boot, 4L)
})

test_that("augment() adds the linear predictor and the residuals", {
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = small_cox)
  out <- augment(fit, data = small_cox)
  expect_named(out, c(names(small_cox), ".fitted", ".resid"))
  expect_equal(out$.fitted, predict(fit, type = "lp"))
  expect_equal(out$.resid, unname(residuals(fit)))
  out <- augment(fit, data = small_cox, type_predict = "risk", type_residuals = "deviance")
  expect_equal(out$.fitted, predict(fit, type = "risk"))
  expect_equal(out$.resid, unname(residuals(fit, type = "deviance")))

  # Rows dropped for missing values get NA.
  holes <- small_cox
  holes$age[c(2, 5)] <- NA
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = holes)
  out <- augment(fit, data = holes)
  expect_identical(which(is.na(out$.fitted)), c(2L, 5L))
  expect_identical(which(is.na(out$.resid)), c(2L, 5L))
  expect_equal(out$.resid[-c(2, 5)], unname(residuals(fit)))

  new <- data.frame(age = c(55, 70), sex = 1)
  out <- augment(fit, newdata = new)
  expect_identical(out$.fitted, predict(fit, newdata = new))
  expect_false(".resid" %in% names(out))

  expect_error(augment(fit), "needs the data frame the model was fit to")
  expect_error(augment(fit, data = small_cox[1:5, ]), "`data` has 5 rows")
  expect_error(augment(fit, data = as.list(holes)), "must be a data frame")
  expect_error(augment(fit, newdata = as.list(new)), "must be a data frame")
})
