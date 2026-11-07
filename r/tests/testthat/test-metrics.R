# Replays the metric fixtures in fixtures/r/ (td_auc_lung, concordance_ipcw_lung, brier_lung, and
# the concordance of cox_diag_efron) and checks the other metrics against values from the Python
# package and against their special cases.

TOL <- 1e-9

lung_death <- function() {
  lung <- survival::lung
  list(data = lung, death = Surv(lung$time, lung$status))
}

test_that("the time-dependent AUC and Uno's C match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  lung <- lung_death()
  auc <- read_fixture("td_auc_lung")
  marker <- r_numbers(auc$marker)
  times <- r_numbers(auc$times)
  expect_equal(time_dependent_auc(lung$death, marker = marker, times = times), r_numbers(auc$auc),
    tolerance = TOL
  )
  expected <- read_fixture("concordance_ipcw_lung")
  expect_equal(concordance_index_ipcw(lung$death, risk = marker), expected$c_ipcw_default,
    tolerance = TOL
  )
  expect_equal(concordance_index_ipcw(lung$death, risk = marker, tau = 365),
    expected$c_ipcw_tau365,
    tolerance = TOL
  )
})

test_that("the Brier score and Harrell's C of a Cox model match the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  lung <- lung_death()
  expected <- read_fixture("brier_lung")
  times <- r_numbers(expected$times)
  fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung$data)
  p <- predict(fit, newdata = lung$data, type = "survival", times = times)
  probs <- matrix(p$estimate, ncol = length(times), byrow = TRUE)
  expect_equal(brier_score(lung$death, survival_prob = probs, times = times),
    r_numbers(expected$brier),
    tolerance = TOL
  )

  efron <- read_fixture("cox_diag_efron")
  expect_equal(concordance_index(lung$death, risk = predict(fit)), efron$concordance,
    tolerance = TOL
  )
  expect_equal(concordance_index(lung$death, risk = predict(fit)), fit$concordance, tolerance = TOL)

  # Values from the Python package on the same data.
  expect_equal(integrated_brier_score(lung$death, survival_prob = probs, times = times),
    0.21262046811957802,
    tolerance = TOL
  )
  expect_equal(integrated_auc(lung$death, marker = predict(fit), times = times),
    0.6084157733720682,
    tolerance = TOL
  )
  cal <- calibration(lung$death, predicted = probs[, 2], time = 365, n_bins = 5)
  expect_identical(names(cal), c(
    "bin", "n", "predicted", "observed", "observed_lower", "observed_upper"
  ))
  expect_identical(cal$n, c(45L, 44L, 45L, 45L, 49L))
  expect_equal(cal$observed, c(0.226515, 0.363082, 0.408116, 0.540004, 0.514285), tolerance = 1e-5)
  expect_equal(cal$observed_upper, c(0.408414, 0.554357, 0.585537, 0.737310, 0.702909),
    tolerance = 1e-5
  )
})

test_that("the competing-risks metrics match the Python package", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  outcome <- Surv(mg$etime, mg$event_f)
  times <- c(60, 120, 240)
  inc <- outer(mg$age / 200, times / 240)
  expect_equal(brier_score_incidence(outcome, incidence_prob = inc, times = times, cause = 1),
    c(0.035845191604019076, 0.07306379048271719, 0.15809499940920163),
    tolerance = TOL
  )
  expect_equal(
    integrated_brier_score_incidence(outcome, incidence_prob = inc, times = times, cause = "pcm"),
    0.09520442697842899,
    tolerance = TOL
  )
  expect_equal(concordance_index_incidence(outcome, incidence_prob = mg$age / 100, cause = 2),
    0.6522759106455989,
    tolerance = TOL
  )
  expect_equal(
    concordance_index_incidence(outcome, incidence_prob = mg$age / 100, cause = 1, tau = 200),
    0.4176230765028965,
    tolerance = TOL
  )
  probs <- array(c(inc * 0.3, inc), dim = c(nrow(mg), length(times), 2L))
  probs <- aperm(probs, c(1L, 3L, 2L))
  expect_equal(accuracy_in_time(outcome, incidence_probs = probs, times = times),
    c(0.6388478581979321, 0.3507896924355777, 0.14313919052319843),
    tolerance = TOL
  )
  expect_equal(calibration_incidence(outcome, incidence_prob = inc, times = times, cause = 1),
    c(0.053925550031431325, 0.11233635799844777, 0.2523033360876522),
    tolerance = TOL
  )
})

test_that("concordance behaves at the extremes", {
  death <- Surv(c(1, 2, 3, 4, 5), c(1, 1, 0, 1, 1))
  expect_identical(concordance_index(death, risk = 5:1), 1)
  expect_identical(concordance_index(death, risk = 1:5), 0)
  expect_identical(concordance_index(death, risk = rep(0, 5)), 0.5)

  # A censored subject tied with an event is still at risk, and tied event times aren't compared.
  tied <- Surv(c(2, 2, 2, 3), c(1, 1, 0, 1))
  # The pairs are (1, 3), (1, 4), (2, 3), and (2, 4), and the last has tied risks.
  expect_equal(concordance_index(tied, risk = c(2, 0, 1, 0)), 2.5 / 4)

  # Without censoring, Uno's C is Harrell's C.
  exact <- Surv(small$time, rep(1, 10))
  risk <- small$ecog + small$sex / 3
  expect_equal(concordance_index_ipcw(exact, risk = risk), concordance_index(exact, risk = risk))
})

test_that("the Brier score and AUC reduce to their uncensored forms", {
  exact <- Surv(c(1, 2, 3, 4, 5, 6), rep(1, 6))
  probs <- cbind(c(0.1, 0.4, 0.5, 0.7, 0.8, 0.9), c(0.05, 0.2, 0.3, 0.5, 0.6, 0.85))
  times <- c(2.5, 4.5)
  status <- cbind(as.numeric(1:6 > 2.5), as.numeric(1:6 > 4.5))
  expect_equal(brier_score(exact, survival_prob = probs, times = times), colMeans((status - probs)^2))
  constant <- matrix(0.5, 6, 3)
  expect_equal(integrated_brier_score(exact, survival_prob = constant, times = c(1, 2, 4)), 0.25)
  expect_equal(brier_score(exact, survival_prob = probs[, 1], times = 2.5),
    brier_score(exact, survival_prob = probs, times = times)[1]
  )

  marker <- c(6, 4, 5, 2, 3, 1)
  # At 2.5 the cases are 1 and 2, the controls 3 to 6, and 7 of the 8 pairs are concordant.
  auc <- time_dependent_auc(exact, marker = marker, times = c(0.5, 2.5, 4.5, 6))
  expect_true(is.na(auc[1]))
  expect_equal(auc[2], 7 / 8)
  expect_true(is.na(auc[4]))
  expect_equal(auc[3], 7 / 8)
  # The NA at 0.5 is dropped, and the AUC at 3.5 is 1.
  expect_equal(integrated_auc(exact, marker = marker, times = c(0.5, 2.5, 3.5, 4.5)), 15 / 16)
})

test_that("calibration bins the predictions", {
  death <- Surv(small_cox$time, small_cox$status)
  predicted <- seq(0.2, 0.8, length.out = 30)
  cal <- calibration(death, predicted = predicted, time = 300, n_bins = 3)
  expect_identical(cal$bin, 1:3)
  expect_identical(sum(cal$n), 30L)
  expect_equal(cal$predicted, c(mean(predicted[1:10]), mean(predicted[11:20]), mean(predicted[21:30])))
  first <- small_cox[1:10, ]
  km <- predict(kaplan_meier(Surv(time, status) ~ 1, data = first), times = 300)
  expect_equal(cal$observed[1], km$estimate)

  # Tied predictions leave empty bins out.
  expect_lt(nrow(calibration(death, predicted = rep(0.5, 30), time = 300, n_bins = 4)), 4L)
  # Before the first time, the observed survival is 1.
  expect_identical(calibration(death, predicted = predicted, time = 1, n_bins = 2)$observed, c(1, 1))
})

test_that("with one cause, the incidence metrics are the survival ones", {
  data <- transform(small_cox, event = factor(status - 1, 0:1, c("censor", "death")))
  outcome <- Surv(data$time, data$event)
  death <- Surv(small_cox$time, small_cox$status)
  surv <- matrix(seq(0.9, 0.3, length.out = 60), 30, 2)
  times <- c(200, 400)
  expect_equal(
    brier_score_incidence(outcome, incidence_prob = 1 - surv, times = times, cause = "death"),
    brier_score(death, survival_prob = surv, times = times)
  )
  expect_equal(
    concordance_index_incidence(outcome, incidence_prob = small_cox$age, cause = 1, tau = 500),
    concordance_index_ipcw(death, risk = small_cox$age, tau = 500)
  )
  km <- predict(kaplan_meier(Surv(time, status) ~ 1, data = small_cox), times = times)
  expect_equal(
    calibration_incidence(outcome, incidence_prob = 1 - surv, times = times, cause = 1),
    abs(colMeans(1 - surv) - (1 - km$estimate))
  )
})

test_that("calibration_incidence() reads the reference curve as steps", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  expected <- read_fixture("calibration_incidence_mgus2")
  times <- r_numbers(expected$times)
  # With every prediction 0, the error is the reference itself.
  zeros <- matrix(0, nrow(mg), length(times))
  outcome <- Surv(mg$etime, mg$event_f)
  expect_equal(calibration_incidence(outcome, incidence_prob = zeros, times = times, cause = 1),
    r_numbers(expected$cif_pcm),
    tolerance = TOL
  )
})

test_that("accuracy in time compares the most likely outcome", {
  outcome <- Surv(small_cr$time[1:6], small_cr$event[1:6])
  # Subjects: pcm at 30, death at 45, censored at 60, death at 72, pcm at 80, death at 95.
  probs <- array(0, dim = c(6, 2, 2))
  probs[, , 1] <- cbind(c(0.6, 0, 0, 0, 0, 0), c(0, 0.6, 0, 0, 0, 0))
  probs[, , 2] <- cbind(c(0.8, 0.1, 0.1, 0.1, 0.1, 0.1), c(0.1, 0.8, 0.1, 0.1, 0.1, 0.8))
  acc <- accuracy_in_time(outcome, incidence_probs = probs, times = c(50, 100))
  # At 50, everyone is right. At 100, the censored subject is out and subjects 4 and 5 are wrong.
  expect_equal(acc, c(1, 3 / 5))
})

test_that("bad input is rejected", {
  death <- Surv(small$time, small$status)
  expect_error(concordance_index(small$time, risk = small$ecog), "`Surv()` object", fixed = TRUE)
  expect_error(concordance_index(death, risk = 1:3), "one per subject")
  expect_error(concordance_index(death, risk = c(NA, 2:10)), "missing")
  expect_error(concordance_index(Surv(small$time, rep(0, 10)), risk = 1:10), "comparable pairs")
  expect_error(concordance_index_ipcw(Surv(small$time, rep(0, 10)), risk = 1:10), "no events")
  expect_error(concordance_index_ipcw(death, risk = 1:10, tau = 10), "`tau`")
  expect_error(
    concordance_index(Surv(c(1, 2), c(3, 4), c(1, 0)), risk = 1:2),
    "right-censored data"
  )
  expect_error(brier_score(death, survival_prob = matrix(0.5, 10, 3), times = 1:2), "10 by 2")
  expect_error(integrated_brier_score(death, survival_prob = matrix(0.5, 10, 1), times = 100),
    "at least 2"
  )
  expect_error(integrated_auc(death, marker = 1:10, times = c(1, 2)), "fewer than two")
  expect_error(calibration(death, predicted = 1:10 / 10, time = 100, n_bins = 1), "`n_bins`")
  outcome <- Surv(small_cr$time, small_cr$event)
  expect_error(brier_score_incidence(death, incidence_prob = matrix(0, 10, 1), times = 1, cause = 1),
    "competing-risks data"
  )
  expect_error(
    brier_score_incidence(outcome, incidence_prob = matrix(0, 18, 1), times = 1, cause = "relapse"),
    "`cause`"
  )
  expect_error(accuracy_in_time(outcome, incidence_probs = matrix(0, 18, 2), times = 1:2),
    "dimensions 18"
  )
  no_pcm <- Surv(small_cr$time, factor(ifelse(small_cr$status == 1, 0, small_cr$status), 0:2,
    c("censor", "pcm", "death")
  ))
  expect_error(concordance_index_incidence(no_pcm, incidence_prob = small_cr$age, cause = "pcm"),
    "no events of cause"
  )
})
