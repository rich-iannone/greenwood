# Survival along a path of covariates, from predict.cox_ph(trajectory =). The estimates are checked
# against the fixture and against survival's survfit() with `id`. The standard errors are checked
# against the delta method with the derivative taken numerically. survfit() takes the derivative
# of each interval on its own, so its standard errors only agree up to the first change of the
# covariates.

pbcseq_episodes <- function() {
  skip_if_not_installed("survival")
  pbcseq <- survival::pbcseq
  baseline <- pbcseq[!duplicated(pbcseq$id), c("id", "futime", "status", "sex")]
  names(baseline)[2] <- "time"
  long <- split_episodes(baseline,
    visits = pbcseq[c("id", "day", "bili", "albumin", "protime")], id = "id", time = "time",
    event = "status", visit_time = "day"
  )
  long <- long[stats::complete.cases(long), ]
  long$dead <- long$status == 2
  long
}

subject_1 <- function() {
  data.frame(
    tstart = c(0, 192),
    tstop = c(192, 400),
    bili = c(14.5, 21.3),
    albumin = c(2.6, 2.94),
    protime = c(12.2, 11.2)
  )
}

tvc_formula <- Surv(tstart, tstop, dead) ~ bili + albumin + protime

# The cumulative hazard along `trajectory` at `times`, with the coefficients set to `beta`, and
# the baseline variance term, summed over the event times one at a time.
trajectory_terms <- function(fit, trajectory, times, beta = fit$coefficients) {
  detail <- cox_terms(beta, fit$setup, detail = TRUE)$detail
  x <- sweep(cox_new_design(fit, trajectory), 2L, fit$means)
  stratum <- cox_new_strata(fit, trajectory)
  k <- nrow(trajectory)
  ends <- c(trajectory$tstop[-k], Inf)
  cumhaz <- numeric(length(times))
  var0 <- numeric(length(times))
  for (j in seq_along(times)) {
    for (i in seq_len(k)) {
      piece <- detail[[stratum[i]]]
      risk <- exp(sum(x[i, ] * beta))
      at <- piece$times > trajectory$tstart[i] & piece$times <= min(times[j], ends[i])
      cumhaz[j] <- cumhaz[j] + risk * sum(piece$hazard[at])
      var0[j] <- var0[j] + risk^2 * sum(piece$hazard2[at])
    }
  }
  list(cumhaz = cumhaz, var0 = var0)
}

test_that("the survival of pbcseq's subject 1 matches the fixture", {
  skip_if_no_shared_cases()
  fixture <- read_fixture("tvc_trajectory_subj1")
  fit <- cox_ph(tvc_formula, data = pbcseq_episodes())
  out <- suppressWarnings(
    predict(fit, trajectory = subject_1(), times = unlist(fixture$query))
  )
  expect_named(
    out, c("subject", "strata", "time", "estimate", "std_error", "conf_low", "conf_high")
  )
  expect_identical(out$subject, rep(1L, 6))
  expect_identical(out$strata, rep("all", 6))
  expect_identical(out$time, as.numeric(unlist(fixture$query)))
  expect_equal(out$estimate, unlist(fixture$surv), tolerance = 1e-9)
})

test_that("the estimates and the standard errors follow their definitions", {
  long <- pbcseq_episodes()
  fit <- cox_ph(tvc_formula, data = long)
  times <- c(50, 120, 192, 250, 400, 520)
  out <- suppressWarnings(predict(fit, trajectory = subject_1(), type = "cumhaz", times = times))
  terms <- trajectory_terms(fit, subject_1(), times)
  expect_equal(out$estimate, terms$cumhaz, tolerance = 1e-12)

  # The derivative of the cumulative hazard with respect to the coefficients, numerically.
  h <- 1e-6
  gradient <- vapply(seq_along(fit$coefficients), function(j) {
    step <- replace(numeric(length(fit$coefficients)), j, h)
    up <- trajectory_terms(fit, subject_1(), times, fit$coefficients + step)$cumhaz
    down <- trajectory_terms(fit, subject_1(), times, fit$coefficients - step)$cumhaz
    (up - down) / (2 * h)
  }, numeric(length(times)))
  variance <- terms$var0 + rowSums((gradient %*% fit$var) * gradient)
  expect_equal(out$std_error, sqrt(variance), tolerance = 1e-7)
  z <- stats::qnorm(0.975)
  expect_equal(out$conf_low, out$estimate * exp(-z * out$std_error / out$estimate))
  expect_equal(out$conf_high, out$estimate * exp(z * out$std_error / out$estimate))

  # The survival curve, with the standard error of survival and log-scale limits.
  surv <- suppressWarnings(predict(fit, trajectory = subject_1(), times = times))
  expect_equal(surv$estimate, exp(-out$estimate))
  expect_equal(surv$std_error, surv$estimate * out$std_error)
  expect_equal(surv$conf_low, surv$estimate * exp(-z * out$std_error))
  expect_equal(surv$conf_high, pmin(surv$estimate * exp(z * out$std_error), 1))
})

test_that("it agrees with survfit() for a subject with an id", {
  skip_if_not_installed("survival")
  long <- pbcseq_episodes()
  fit <- cox_ph(tvc_formula, data = long)
  reference <- survival::coxph(
    survival::Surv(tstart, tstop, dead) ~ bili + albumin + protime, data = long
  )
  path <- subject_1()
  path$id <- 1
  path$dead <- FALSE
  curve <- survival::survfit(reference, newdata = path, id = id)
  times <- c(25, 100, 192, 300, 400)
  expected <- summary(curve, times = times)
  out <- predict(fit, trajectory = subject_1(), times = times)
  expect_equal(out$estimate, expected$surv, tolerance = 1e-10)
  first <- times <= 192
  expect_equal(out$std_error[first], expected$std.err[first], tolerance = 1e-8)
  expect_equal(out$conf_low[first], expected$lower[first], tolerance = 1e-8)

  # With strata, and a subject who moves between them.
  strata_fit <- cox_ph(Surv(tstart, tstop, dead) ~ bili + albumin + strata(sex), data = long)
  # coxph() finds the stratifying variables by the name `strata`.
  strata_formula <- local({
    strata <- survival::strata
    survival::Surv(tstart, tstop, dead) ~ bili + albumin + strata(sex)
  })
  strata_reference <- survival::coxph(strata_formula, data = long)
  expect_equal(unname(coef(strata_fit)), unname(coef(strata_reference)), tolerance = 1e-8)
  path <- data.frame(
    tstart = c(0, 300, 700), tstop = c(300, 700, 1500), bili = c(1, 3, 6),
    albumin = c(3.5, 3.2, 2.9), sex = factor(c("f", "m", "m"), levels = levels(long$sex))
  )
  out <- predict(strata_fit, trajectory = path, times = c(150, 300, 500, 700, 1200))
  expect_identical(out$strata, c("sex=f", "sex=f", "sex=m", "sex=m", "sex=m"))
  path$id <- 1
  path$dead <- FALSE
  curve <- survival::survfit(strata_reference, newdata = path, id = id)
  expect_equal(
    out$estimate, summary(curve, times = c(150, 300, 500, 700, 1200))$surv,
    tolerance = 1e-10
  )
})

test_that("one interval gives the curve of fixed covariates", {
  fit <- cox_ph(tvc_formula, data = pbcseq_episodes())
  path <- data.frame(tstart = 0, tstop = 9000, bili = 3, albumin = 3.1, protime = 11)
  times <- c(100, 1000, 3000)
  fixed <- predict(fit, newdata = path, type = "survival", times = times)
  expect_equal(predict(fit, trajectory = path, times = times), fixed, tolerance = 1e-12)
  expect_equal(
    predict(fit, trajectory = path, type = "cumhaz", times = times),
    predict(fit, newdata = path, type = "cumhaz", times = times),
    tolerance = 1e-12
  )

  # The default times are those of the fit, and the default type is survival.
  default <- predict(fit, trajectory = path)
  expect_identical(default$time, sort(unique(fit$setup$stop)))
  expect_equal(default$estimate, predict(fit, newdata = path, type = "survival")$estimate)

  # Splitting the interval in two changes nothing.
  split <- data.frame(
    tstart = c(0, 700), tstop = c(700, 9000), bili = 3, albumin = 3.1, protime = 11
  )
  expect_equal(predict(fit, trajectory = split, times = times), fixed, tolerance = 1e-12)

  # No hazard builds up before the first interval or in a gap between two.
  late <- data.frame(
    tstart = c(500, 1500), tstop = c(1000, 9000), bili = 3, albumin = 3.1, protime = 11
  )
  out <- predict(fit, trajectory = late, type = "cumhaz", times = c(400, 1000, 1500))
  expect_identical(out$estimate[1], 0)
  expect_identical(out$estimate[2], out$estimate[3])
  expect_identical(out$strata, rep("all", 3))
})

test_that("times past the end of the trajectory give a warning", {
  fit <- cox_ph(tvc_formula, data = pbcseq_episodes())
  expect_warning(
    predict(fit, trajectory = subject_1(), times = c(300, 500, 600)),
    "`trajectory` ends at 400, so 2 of `times` are carried past it",
    fixed = TRUE
  )
  expect_no_warning(predict(fit, trajectory = subject_1(), times = c(300, 400)))
})

test_that("bad input is caught", {
  long <- pbcseq_episodes()
  fit <- cox_ph(tvc_formula, data = long)
  path <- subject_1()
  expect_error(
    predict(fit, newdata = path, trajectory = path),
    "one of `newdata` and `trajectory`"
  )
  expect_error(predict(fit, trajectory = path, type = "risk"), "\"survival\" or \"cumhaz\"")
  expect_error(
    predict(fit, trajectory = path, conditional_after = 10),
    "`conditional_after` can't be used"
  )
  expect_error(predict(fit, trajectory = as.list(path)), "must be a data frame")
  expect_error(predict(fit, trajectory = path[-1]), "doesn't have `tstart`")
  expect_error(predict(fit, trajectory = path[0, ]), "must be numbers")
  expect_error(
    predict(fit, trajectory = transform(path, tstop = c(192, 192))),
    "0 <= tstart < tstop"
  )
  expect_error(
    predict(fit, trajectory = transform(path, tstart = c(-1, 192))),
    "0 <= tstart < tstop"
  )
  expect_error(
    predict(fit, trajectory = transform(path, tstart = c(0, 100))),
    "must not overlap"
  )
  expect_error(predict(fit, trajectory = transform(path, bili = c(1, NA))), "missing covariate")
  expect_error(predict(fit, trajectory = path[c("tstart", "tstop", "bili")]))
  strata_fit <- cox_ph(Surv(tstart, tstop, dead) ~ bili + strata(sex), data = long)
  expect_error(
    predict(strata_fit, trajectory = transform(path, sex = "x")),
    "strata that the model wasn't fit to"
  )
})
