# The cause-specific Cox model is cox_ph() on the recoded response, so it's checked against
# cox_ph() and survival::coxph() on that response, and against Greenwood's Python values.

mgus2_formula <- Surv(etime, event_f) ~ age + sex

test_that("the fits match Python", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  pcm <- cause_specific_cox(mgus2_formula, data = mg, cause = "pcm")
  expect_equal(unname(coef(pcm)), c(0.013038569824972193, -0.025137789294278696),
    tolerance = 1e-9
  )
  expect_equal(unname(sqrt(diag(vcov(pcm)))), c(0.008258685911711918, 0.18845584819278138),
    tolerance = 1e-9
  )
  expect_identical(c(pcm$n, pcm$n_event), c(1384L, 115))
  expect_equal(pcm$concordance, 0.5635771434096347, tolerance = 1e-9)

  death <- cause_specific_cox(mgus2_formula, data = mg, cause = "death")
  expect_equal(unname(coef(death)), c(0.06482366356286133, 0.3932258637315012), tolerance = 1e-9)
  expect_equal(unname(sqrt(diag(vcov(death)))), c(0.0036202755680845885, 0.06969810457968525),
    tolerance = 1e-9
  )
  expect_identical(death$n_event, 860)

  breslow <- cause_specific_cox(Surv(etime, event_f) ~ age + sex + hgb,
    data = mg, cause = "pcm", ties = "breslow"
  )
  expect_equal(unname(coef(breslow)), c(
    0.009954483277179328, 0.12250251445954849, -0.12843138198265094
  ), tolerance = 1e-9)
  expect_identical(breslow$n_dropped, 13L)
})

test_that("the fit is cox_ph() with the other causes as censoring", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  mg$is_pcm <- as.numeric(mg$event == 1)
  fit <- cause_specific_cox(mgus2_formula, data = mg, cause = "pcm")
  manual <- cox_ph(Surv(etime, is_pcm) ~ age + sex, data = mg)
  expect_s3_class(fit, c("cause_specific_cox", "cox_ph"), exact = TRUE)
  expect_equal(coef(fit), coef(manual))
  expect_equal(vcov(fit), vcov(manual))
  expect_equal(fit$loglik, manual$loglik)
  expect_equal(baseline_hazard(fit), baseline_hazard(manual))
  expect_equal(residuals(fit, type = "martingale"), residuals(manual, type = "martingale"))
  expect_equal(as.data.frame(fit, exponentiate = TRUE), as.data.frame(manual, exponentiate = TRUE))
  newdata <- data.frame(age = c(60, 75), sex = c("F", "M"))
  expect_equal(
    predict(fit, newdata = newdata, type = "survival", times = c(100, 200)),
    predict(manual, newdata = newdata, type = "survival", times = c(100, 200))
  )
  expect_equal(cox_zph(fit)$table, cox_zph(manual)$table)

  reference <- survival::coxph(survival::Surv(etime, event == 1) ~ age + sex, data = mg)
  expect_equal(unname(coef(fit)), unname(coef(reference)), tolerance = 1e-9)
  expect_equal(unname(vcov(fit)), unname(vcov(reference)), tolerance = 1e-9)

  # The cause can be given by number.
  expect_identical(coef(cause_specific_cox(mgus2_formula, data = mg, cause = 1)), coef(fit))
  expect_identical(coef(cause_specific_cox(mgus2_formula, data = mg)), coef(fit))
  expect_identical(fit$cause, "pcm")
  expect_identical(fit$states, c("pcm", "death"))
})

test_that("the arguments of cox_ph() pass through", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  mg$is_death <- as.numeric(mg$event == 2)
  mg$k <- rep(1:3, length.out = nrow(mg))
  fit <- cause_specific_cox(Surv(etime, event_f) ~ age + strata(sex) + cluster(id),
    data = mg, cause = "death", weights = k, subset = age > 50, ties = "breslow"
  )
  manual <- cox_ph(Surv(etime, is_death) ~ age + strata(sex) + cluster(id),
    data = mg, weights = k, subset = age > 50, ties = "breslow"
  )
  expect_equal(coef(fit), coef(manual))
  expect_equal(vcov(fit), vcov(manual))
  expect_true(fit$robust && fit$has_weights)
  expect_identical(fit$n, manual$n)
  expect_equal(
    predict(fit, newdata = data.frame(age = 70, sex = "M"), times = 100),
    predict(manual, newdata = data.frame(age = 70, sex = "M"), times = 100)
  )

  # Counting-process data, with delayed entry.
  mg$entry <- ifelse(seq_len(nrow(mg)) %% 4 == 0, pmin(mg$etime / 2, 10), 0)
  counting <- cause_specific_cox(Surv(entry, etime, event_f) ~ age + sex,
    data = mg, cause = "pcm", conf_level = 0.9
  )
  mg$is_pcm <- as.numeric(mg$event == 1)
  manual_counting <- cox_ph(Surv(entry, etime, is_pcm) ~ age + sex, data = mg, conf_level = 0.9)
  expect_equal(coef(counting), coef(manual_counting))
  expect_equal(as.data.frame(counting), as.data.frame(manual_counting))
})

test_that("print names the cause", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  fit <- cause_specific_cox(mgus2_formula, data = mg, cause = "death")
  expect_output(print(fit), "<cause_specific_cox> Surv(etime, event_f) ~ age + sex", fixed = TRUE)
  expect_output(print(fit), "Cause \"death\", with the other causes taken as censoring.",
    fixed = TRUE
  )
  expect_output(print(fit), "n = 1384, events = 860", fixed = TRUE)
  expect_output(print(cox_ph(Surv(time, status) ~ age, data = survival::lung)), "<cox_ph>")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  expect_error(
    cause_specific_cox(Surv(etime, death) ~ age, data = mg),
    "competing-risks response"
  )
  expect_error(cause_specific_cox(mgus2_formula, data = mg, cause = "relapse"), "`cause`")
  expect_error(cause_specific_cox(mgus2_formula, data = mg, cause = 3), "`cause`")
  mg$event_f[mg$event_f == "pcm"] <- "censor"
  expect_error(cause_specific_cox(mgus2_formula, data = mg, cause = "pcm"), "no events of cause")
  expect_error(
    cause_specific_cox(mgus2_formula, data = mg, cause = 2, ties = "exact"),
    "should be one of"
  )
  fit <- cause_specific_cox(mgus2_formula, data = mgus2_competing(), cause = "death")
  expect_error(
    cross_validate(cause_specific_cox, mgus2_formula, data = mgus2_competing(), cause = "death"),
    "not `cause_specific_cox` objects"
  )
  expect_s3_class(fit, "cause_specific_cox")
})
