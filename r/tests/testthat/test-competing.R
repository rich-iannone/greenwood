# Replays the competing-risks fixtures in fixtures/r/ (cif_mgus2, finegray_mgus2_pcm, and
# grays_test_mgus2_*), and checks what they don't cover (groups, predictions, the special cases)
# directly.

TOL <- 1e-9

test_that("the cumulative incidence on mgus2 matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  expected <- read_fixture("cif_mgus2")
  for (conf_type in c("plain", "log", "log-log")) {
    fit <- aalen_johansen(Surv(etime, event_f) ~ 1, data = mg, conf_type = conf_type)
    table <- as.data.frame(fit)
    expect_identical(names(table), c(
      "strata", "cause", "time", "n_risk", "estimate", "std_error", "conf_low", "conf_high"
    ))
    pcm <- table[table$cause == "pcm", ]
    death <- table[table$cause == "death", ]
    expect_equal(pcm$time, r_numbers(expected$time), tolerance = TOL)
    expect_equal(death$time, r_numbers(expected$time), tolerance = TOL)
    expect_equal(pcm$n_risk, r_numbers(expected$n_risk), tolerance = TOL)
    expect_equal(pcm$estimate, r_numbers(expected$cif_pcm), tolerance = TOL)
    expect_equal(death$estimate, r_numbers(expected$cif_death), tolerance = TOL)
    expect_equal(pcm$std_error, r_numbers(expected$se_pcm), tolerance = 1e-8)
    expect_equal(death$std_error, r_numbers(expected$se_death), tolerance = 1e-8)
    key <- gsub("-", "", conf_type)
    for (cause in c("pcm", "death")) {
      rows <- table[table$cause == cause, ]
      expect_equal(rows$conf_low, r_numbers(expected[[paste0("lower_", cause, "_", key)]]),
        tolerance = 1e-8
      )
      expect_equal(rows$conf_high, r_numbers(expected[[paste0("upper_", cause, "_", key)]]),
        tolerance = 1e-8
      )
    }
  }
})

test_that("the cumulative incidences and the event-free curve add up to 1", {
  fit <- aalen_johansen(Surv(time, event) ~ 1, data = small_cr)
  table <- as.data.frame(fit)
  km <- as.data.frame(kaplan_meier(Surv(time, status > 0) ~ 1, data = small_cr))
  total <- table$estimate[table$cause == "pcm"] + table$estimate[table$cause == "death"]
  expect_equal(total, 1 - km$estimate)
  expect_identical(fit$n_event, c(pcm = 6, death = 7))

  # With one cause, the cumulative incidence is one minus Kaplan-Meier.
  one_cause <- transform(small_cr, event = factor(status > 0, c(FALSE, TRUE), c("censor", "any")))
  single <- as.data.frame(aalen_johansen(Surv(time, event) ~ 1, data = one_cause))
  expect_equal(single$estimate, 1 - km$estimate)
})

test_that("groups are fit separately", {
  fit <- aalen_johansen(Surv(time, event) ~ sex, data = small_cr, conf_type = "log")
  table <- as.data.frame(fit)
  expect_identical(names(table)[1:3], c("strata", "sex", "cause"))
  expect_identical(unique(table$strata), c("sex=F", "sex=M"))
  for (s in c("F", "M")) {
    one <- as.data.frame(aalen_johansen(Surv(time, event) ~ 1, data = small_cr[small_cr$sex == s, ],
      conf_type = "log"
    ))
    rows <- table[table$sex == s, ]
    expect_equal(rows$estimate, one$estimate)
    expect_equal(rows$conf_high, one$conf_high)
  }
})

test_that("predict() reads the curves as steps", {
  fit <- aalen_johansen(Surv(time, event) ~ 1, data = small_cr)
  table <- as.data.frame(fit)
  p <- predict(fit, times = c(10, 30, 100, 1000))
  expect_identical(names(p), c(
    "strata", "cause", "time", "estimate", "std_error", "conf_low", "conf_high"
  ))
  pcm <- table[table$cause == "pcm", ]
  expect_equal(p$estimate[p$cause == "pcm"], c(0, pcm$estimate[c(1, 6)], pcm$estimate[nrow(pcm)]))
  expect_identical(p$conf_low[1], 0)

  # The log limits are missing where the estimate is 0.
  logged <- aalen_johansen(Surv(time, event) ~ 1, data = small_cr, conf_type = "log-log")
  expect_true(is.nan(predict(logged, times = 10)$conf_low[1]))
  expect_true(is.nan(as.data.frame(logged)$conf_low[1 + nrow(pcm)]))
})

test_that("Gray's test matches the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  expected <- read_fixture("grays_test_mgus2_sex")
  for (cause in c("pcm", "death")) {
    test <- grays_test(Surv(etime, event_f) ~ sex, data = mg, cause = cause)
    expect_equal(test$statistic, expected[[cause]]$statistic, tolerance = 1e-6)
    expect_equal(test$p_value, expected[[cause]]$p_value, tolerance = 1e-6)
    expect_identical(test$df, as.integer(expected[[cause]]$df))
  }
  mg$age_group <- ifelse(mg$age < 60, "young", ifelse(mg$age < 70, "mid", "old"))
  expected <- read_fixture("grays_test_mgus2_age")
  test <- grays_test(Surv(etime, event_f) ~ age_group, data = mg, cause = "pcm")
  expect_equal(test$statistic, expected$pcm$statistic, tolerance = 1e-6)
  expect_equal(test$p_value, expected$pcm$p_value, tolerance = 1e-6)
  expect_identical(test$df, 2L)
})

test_that("Gray's test counts and labels are consistent", {
  test <- grays_test(Surv(time, event) ~ sex, data = small_cr)
  expect_s3_class(test, "greenwood_test")
  expect_identical(test$method, "Gray's test (cause = \"pcm\")")
  expect_identical(names(test$groups), c("group", "sex", "n", "observed", "expected"))
  expect_equal(sum(test$groups$observed), sum(small_cr$status == 1))
  expect_equal(sum(test$groups$expected), sum(test$groups$observed))
  expect_equal(grays_test(Surv(time, event) ~ sex, data = small_cr, cause = "pcm"), test)

  # Which group comes last doesn't matter.
  flipped <- transform(small_cr, sex = ifelse(sex == "F", "Z", sex))
  expect_equal(
    grays_test(Surv(time, event) ~ sex, data = flipped, cause = 2)$statistic,
    grays_test(Surv(time, event) ~ sex, data = small_cr, cause = 2)$statistic
  )
})

test_that("the Fine-Gray fit on mgus2 matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  expected <- read_fixture("finegray_mgus2_pcm")
  fit <- fine_gray(Surv(etime, event_f) ~ age + sex, data = mg, cause = "pcm")
  table <- as.data.frame(fit)
  expect_identical(table$term, r_strings(expected$terms))
  expect_equal(table$estimate, r_numbers(expected$coef), tolerance = TOL)
  expect_equal(unname(sqrt(diag(fit$naive_var))), r_numbers(expected$naive_se), tolerance = TOL)
  expect_equal(table$std_error, r_numbers(expected$robust_se), tolerance = TOL)
  expect_equal(fit$n_event, 115)
  expect_identical(nobs(fit), 1384L)
})

test_that("Fine-Gray cumulative incidence matches the weighted Cox fit", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  fit <- fine_gray(Surv(etime, event_f) ~ age + sex, data = mg, cause = "pcm")
  newdata <- data.frame(age = c(60, 75), sex = c("F", "M"))
  times <- c(30, 60, 120, 240, 400)
  cif <- predict(fit, newdata = newdata, type = "cif", times = times)
  expect_identical(names(cif), c("subject", "time", "estimate"))

  expanded <- survival::finegray(
    survival::Surv(etime, event_f) ~ age + sex + id, data = mg, etype = "pcm"
  )
  reference <- survival::coxph(
    survival::Surv(fgstart, fgstop, fgstatus) ~ age + sex, weights = fgwt, data = expanded,
    ties = "breslow"
  )
  curves <- summary(survival::survfit(reference, newdata = newdata), times = times)
  expect_equal(cif$estimate, c(1 - curves$surv), tolerance = 1e-9)
})

test_that("with no competing events, Fine-Gray is the Breslow Cox model", {
  data <- transform(small_cr, event = factor(ifelse(status == 2, 0, status), 0:2,
    c("censor", "pcm", "death")
  ))
  fit <- fine_gray(Surv(time, event) ~ age + sex, data = data)
  cox <- cox_ph(Surv(time, status == 1) ~ age + sex, data = small_cr, ties = "breslow",
    robust = TRUE
  )
  expect_equal(coef(fit), coef(cox), tolerance = 1e-8)
  expect_equal(fit$naive_var, cox$naive_var, tolerance = 1e-8)
  expect_equal(vcov(fit), vcov(cox), tolerance = 1e-8)
  expect_equal(fit$loglik, cox$loglik, tolerance = 1e-8)
})

test_that("Fine-Gray predictions and tables are consistent", {
  fit <- fine_gray(Surv(time, event) ~ age + sex, data = small_cr, cause = "death")
  expect_identical(fit$cause, "death")
  expect_true(fit$converged)
  expect_identical(names(coef(fit)), c("age", "sexM"))
  lp <- predict(fit)
  expect_equal(lp, fit$linear_predictors)
  expect_equal(predict(fit, newdata = small_cr), lp)
  expect_equal(predict(fit, newdata = small_cr[1:3, ], type = "risk"), exp(lp[1:3]))

  cif <- predict(fit, newdata = small_cr[1:2, ], type = "cif")
  expect_identical(cif$time, rep(fit$baseline$time, 2))
  expect_equal(cif$estimate, 1 - exp(-rep(fit$baseline$cumhaz, 2) * rep(exp(lp[1:2]), each = nrow(fit$baseline))))
  early <- predict(fit, newdata = small_cr[1, ], type = "cif", times = 1)
  expect_identical(early$estimate, 0)

  ratios <- as.data.frame(fit, exponentiate = TRUE)
  table <- as.data.frame(fit)
  expect_equal(ratios$estimate, exp(table$estimate))
  expect_equal(table$std_error, unname(sqrt(diag(vcov(fit)))))
  expect_equal(table$statistic, table$estimate / table$std_error)
})

test_that("bad input is rejected", {
  expect_error(
    aalen_johansen(Surv(time, status > 0) ~ 1, data = small_cr),
    "needs a competing-risks response"
  )
  counting <- transform(small_cr, start = 0)
  expect_error(
    aalen_johansen(Surv(start, time, event) ~ 1, data = counting),
    "doesn't support delayed entry"
  )
  expect_error(aalen_johansen(Surv(time, event) ~ strata(sex), data = small_cr), "`strata()`",
    fixed = TRUE
  )
  expect_error(aalen_johansen(Surv(time, event) ~ 1, data = small_cr, conf_type = "logit"))
  expect_error(grays_test(Surv(time, event) ~ 1, data = small_cr), "grouping variables")
  expect_error(grays_test(Surv(time, event) ~ sex, data = small_cr, cause = "relapse"), "`cause`")
  expect_error(grays_test(Surv(time, event) ~ sex, data = small_cr, cause = 3), "`cause`")
  expect_error(
    grays_test(Surv(time, event) ~ sex, data = small_cr[small_cr$sex == "F", ]),
    "at least two groups"
  )
  expect_error(fine_gray(Surv(time, event) ~ 1, data = small_cr), "at least one covariate")
  no_pcm <- transform(small_cr, event = factor(ifelse(status == 1, 0, status), 0:2,
    c("censor", "pcm", "death")
  ))
  expect_error(fine_gray(Surv(time, event) ~ age, data = no_pcm), "no events of cause \"pcm\"")
  fit <- fine_gray(Surv(time, event) ~ age, data = small_cr)
  expect_error(predict(fit, newdata = 1), "must be a data frame")
  expect_error(predict(fit, type = "cif", times = "a"), "`times`")
})

test_that("the printouts are stable", {
  expect_snapshot(print(aalen_johansen(Surv(time, event) ~ 1, data = small_cr)))
  expect_snapshot(print(aalen_johansen(Surv(time, event) ~ sex, data = small_cr)))
  expect_snapshot(print(grays_test(Surv(time, event) ~ sex, data = small_cr, cause = "death")))
  expect_snapshot(print(fine_gray(Surv(time, event) ~ age + sex, data = small_cr)))
})
