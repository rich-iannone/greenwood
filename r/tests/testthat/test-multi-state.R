# Replays the multi-state fixture in fixtures/r/ (multistate_mgus2) and checks the special cases
# and the methods of multi_state().

TOL <- 1e-9

# The mgus -> pcm -> death intervals of mgus2, as in the fixture generator.
mgus2_illness_death <- function() {
  mg <- survival::mgus2
  progressed <- mg$pstat == 1 & mg$ptime < mg$futime
  first <- data.frame(
    id = mg$id,
    t0 = 0,
    t1 = ifelse(progressed, mg$ptime, mg$futime),
    from = "mgus",
    ev = ifelse(progressed, "pcm",
      ifelse(mg$death == 1, "death", ifelse(mg$pstat == 1, "pcm", "censor"))
    )
  )
  second <- data.frame(
    id = mg$id[progressed],
    t0 = mg$ptime[progressed],
    t1 = mg$futime[progressed],
    from = "pcm",
    ev = ifelse(mg$death[progressed] == 1, "death", "censor")
  )
  out <- rbind(first, second)
  out <- out[out$t1 > out$t0, ]
  out$ev <- factor(out$ev, levels = c("censor", "pcm", "death"))
  out$from <- factor(out$from, levels = c("mgus", "pcm", "death"))
  out
}

test_that("the occupancy on mgus2 matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("multistate_mgus2")
  fit <- multi_state(Surv(t0, t1, ev) ~ 1, data = mgus2_illness_death(), istate = from)
  expect_identical(fit$states, r_strings(expected$states))
  table <- as.data.frame(fit)
  expect_identical(names(table), c("strata", "state", "time", "estimate"))
  for (state in fit$states) {
    rows <- table[table$state == state, ]
    expect_equal(rows$time, r_numbers(expected$time), tolerance = TOL)
    expect_equal(rows$estimate, r_numbers(expected[[state]]), tolerance = TOL)
  }
})

test_that("competing risks are the special case with one starting state", {
  fit <- multi_state(Surv(time, event) ~ 1, data = small_cr)
  expect_identical(fit$states, c("(s0)", "pcm", "death"))
  table <- as.data.frame(fit)
  cif <- as.data.frame(aalen_johansen(Surv(time, event) ~ 1, data = small_cr))
  km <- as.data.frame(kaplan_meier(Surv(time, status > 0) ~ 1, data = small_cr))
  expect_equal(table$estimate[table$state == "(s0)"], km$estimate)
  for (cause in c("pcm", "death")) {
    expect_equal(table$estimate[table$state == cause], cif$estimate[cif$cause == cause])
  }

  # The occupancies add up to 1 at every time, and groups are fit separately.
  grouped <- as.data.frame(multi_state(Surv(time, event) ~ sex, data = small_cr))
  expect_identical(names(grouped)[1:3], c("strata", "sex", "state"))
  totals <- tapply(grouped$estimate, list(grouped$strata, grouped$time), sum)
  expect_equal(unname(totals[!is.na(totals)]), rep(1, sum(!is.na(totals))))
  women <- as.data.frame(multi_state(Surv(time, event) ~ 1, data = small_cr[small_cr$sex == "F", ]))
  expect_equal(grouped$estimate[grouped$sex == "F"], women$estimate)
})

test_that("predict() starts from the initial distribution", {
  intervals <- data.frame(
    start = c(0, 20, 0, 0, 35, 0, 0, 0),
    stop = c(20, 60, 45, 35, 90, 70, 30, 40),
    from = c("healthy", "ill", "healthy", "healthy", "ill", "healthy", "healthy", "ill"),
    to = c("ill", "dead", "dead", "ill", "censor", "censor", "dead", "dead")
  )
  intervals$to <- factor(intervals$to, levels = c("censor", "ill", "dead"))
  fit <- multi_state(Surv(start, stop, to) ~ 1, data = intervals, istate = from)
  expect_identical(fit$states, c("healthy", "ill", "dead"))
  p <- predict(fit, times = c(10, 20, 1000))
  expect_identical(names(p), c("strata", "state", "time", "estimate"))
  # Five of the six rows starting at time 0 are healthy and one is ill. At time 20, one of the
  # five healthy subjects falls ill.
  expect_equal(p$estimate[p$time == 10], c(5, 1, 0) / 6)
  expect_equal(p$estimate[p$time == 20], c(4, 2, 0) / 6)
  table <- as.data.frame(fit)
  last <- table[table$time == max(table$time), ]
  expect_equal(p$estimate[p$time == 1000], last$estimate)
})

test_that("bad input is rejected", {
  expect_error(multi_state(Surv(time, status > 0) ~ 1, data = small_cr), "multi-state response")
  expect_error(multi_state(Surv(time, event) ~ strata(sex), data = small_cr), "`strata()`",
    fixed = TRUE
  )
  fit <- multi_state(Surv(time, event) ~ 1, data = small_cr)
  expect_error(predict(fit, times = "a"), "`times`")
})

test_that("the printout is stable", {
  expect_snapshot(print(multi_state(Surv(time, event) ~ 1, data = small_cr)))
})
