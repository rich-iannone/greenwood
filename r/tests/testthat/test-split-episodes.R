# split_episodes() is checked against the rules for each interval, and through a Cox fit to the
# pbcseq visits that matches the fixture made with survival's tmerge() and coxph().

se_baseline <- function() {
  data.frame(id = 1:4, time = c(10, 8, 12, 5), event = c(1, 0, 1, 1), sex = c("f", "m", "f", "m"))
}

se_visits <- function() {
  data.frame(
    id = c(1L, 1L, 2L, 3L, 3L, 3L),
    day = c(0, 5, 0, 2, 4, 12),
    bili = c(1.2, 2.4, 0.8, 3.1, 2.9, 4)
  )
}

split_se <- function(baseline = se_baseline(), visits = se_visits(), ...) {
  split_episodes(baseline,
    visits = visits, id = "id", time = "time", event = "event", visit_time = "day", ...
  )
}

test_that("each visit starts an interval that holds its values", {
  out <- split_se()
  expect_identical(names(out), c("id", "tstart", "tstop", "event", "sex", "bili"))
  expect_identical(out$id, c(1L, 1L, 2L, 3L, 3L, 3L, 4L))
  expect_identical(out$tstart, c(0, 5, 0, 0, 2, 4, 0))
  expect_identical(out$tstop, c(5, 10, 8, 2, 4, 12, 5))
  # The event stays on the interval that ends with follow-up.
  expect_identical(out$event, c(0, 1, 0, 0, 0, 1, 1))
  expect_identical(out$sex, c("f", "f", "m", "f", "f", "f", "m"))
  # Before the first visit, and for a subject with no visits, the values are missing. A visit at
  # the end of follow-up starts no interval.
  expect_identical(out$bili, c(1.2, 2.4, 0.8, NA, 3.1, 2.9, NA))
})

test_that("carry_forward = FALSE stops follow-up at the last visit", {
  out <- split_se(carry_forward = FALSE)
  expect_identical(out$id, c(1L, 2L, 3L, 3L))
  expect_identical(out$tstart, c(0, 0, 2, 4))
  expect_identical(out$tstop, c(5, 8, 4, 12))
  # Subject 1 loses its event with the interval after its last visit. Subject 2 keeps its one
  # interval, and subject 3's last visit is at the end of follow-up, so it keeps its event.
  expect_identical(out$event, c(0, 0, 0, 1))
  expect_identical(out$bili, c(1.2, 0.8, 3.1, 2.9))
})

test_that("repeated visit times keep the last values", {
  visits <- data.frame(id = c(1, 1, 1), day = c(0, 3, 3), bili = c(1, 2, 3))
  out <- split_se(se_baseline()[1, ], visits = visits)
  expect_identical(out$tstart, c(0, 3))
  expect_identical(out$tstop, c(3, 10))
  expect_identical(out$bili, c(1, 3))

  # The visits can come in any order.
  shuffled <- se_visits()[c(6, 3, 1, 5, 2, 4), ]
  expect_identical(split_se(visits = shuffled), split_se())
})

test_that("the covariates can be chosen", {
  visits <- se_visits()
  visits$albumin <- 3:8
  out <- split_se(visits = visits, covariates = "albumin", baseline_covariates = character(0))
  expect_identical(names(out), c("id", "tstart", "tstop", "event", "albumin"))
  expect_identical(out$albumin, c(3L, 4L, 5L, NA, 6L, 7L, NA))

  # The default leaves out the follow-up columns of `visits`.
  visits$time <- 99
  visits$event <- 1
  expect_identical(names(split_se(visits = visits)), c(names(split_se()), "albumin"))

  # The same subjects and order as `baseline`, whatever the id type.
  baseline <- se_baseline()[c(3, 1, 2, 4), ]
  baseline$id <- c("c", "a", "b", "d")
  visits <- se_visits()
  visits$id <- c("a", "a", "b", "c", "c", "c")
  out <- split_se(baseline, visits = visits)
  expect_identical(out$id, c("c", "c", "c", "a", "a", "b", "d"))
  expect_identical(out$bili, c(NA, 3.1, 2.9, 1.2, 2.4, 0.8, NA))
})

test_that("factor and logical events are kept on the last interval", {
  baseline <- se_baseline()
  baseline$event <- factor(c("death", "censor", "death", "relapse"),
    levels = c("censor", "relapse", "death")
  )
  out <- split_se(baseline)
  expect_identical(levels(out$event), levels(baseline$event))
  expect_identical(
    as.character(out$event),
    c("censor", "death", "censor", "censor", "censor", "death", "relapse")
  )

  baseline$event <- c(TRUE, FALSE, TRUE, TRUE)
  expect_identical(split_se(baseline)$event, c(FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE))
})

test_that("a long table can be its own baseline", {
  long <- merge(se_visits(), se_baseline(), by = "id")
  out <- split_episodes(long,
    visits = long, id = "id", time = "time", event = "event", visit_time = "day",
    covariates = "bili"
  )
  expect_identical(out, split_se(baseline = se_baseline()[1:3, -4]))

  # The follow-up columns must stay the same within a subject.
  long$time[2] <- 11
  expect_error(
    split_episodes(long,
      visits = long, id = "id", time = "time", event = "event", visit_time = "day"
    ),
    "Subject 1 has more than one value of `time` in `baseline` (10 and 11)",
    fixed = TRUE
  )
})

test_that("the episodes give the time-varying Cox fit of pbcseq", {
  skip_if_not_installed("survival")
  skip_if_no_shared_cases()
  fixture <- read_fixture("tvc_pbcseq")
  pbcseq <- survival::pbcseq
  check_fit <- function(long) {
    long <- long[stats::complete.cases(long[c("bili", "albumin", "protime")]), ]
    expect_identical(nrow(long), as.integer(fixture$n))
    expect_identical(sum(long$status == 2), as.integer(fixture$nevent))
    fit <- cox_ph(Surv(tstart, tstop, status == 2) ~ bili + albumin + protime, data = long)
    expect_equal(unname(coef(fit)), unlist(fixture$coef), tolerance = 1e-6)
    expect_equal(fit$loglik[2], fixture$loglik, tolerance = 1e-6)
  }

  baseline <- pbcseq[!duplicated(pbcseq$id), c("id", "futime", "status")]
  names(baseline)[2] <- "time"
  two_tables <- split_episodes(baseline,
    visits = pbcseq[c("id", "day", "bili", "albumin", "protime")], id = "id", time = "time",
    event = "status", visit_time = "day"
  )
  check_fit(two_tables)

  # The long table on its own, with every column other than the follow-up as a covariate.
  one_table <- split_episodes(pbcseq,
    visits = pbcseq, id = "id", time = "futime", event = "status", visit_time = "day"
  )
  expect_false("futime" %in% names(one_table))
  expect_identical(sum(one_table$status > 0), sum(!duplicated(pbcseq$id) & pbcseq$status > 0))
  check_fit(one_table)
})

test_that("bad input is caught", {
  expect_error(split_se(baseline = list(id = 1)), "`baseline` must be a data frame")
  expect_error(split_se(visits = 1), "`visits` must be a data frame")
  expect_error(
    split_episodes(se_baseline(),
      visits = se_visits(), id = 1, time = "time", event = "event", visit_time = "day"
    ),
    "`id` must be a column name"
  )
  expect_error(split_se(covariates = c("bili", "alb")), "`alb` is not in `visits`")
  expect_error(
    split_se(baseline_covariates = c("age", "stage")),
    "`age`, `stage` are not in `baseline`"
  )
  expect_error(split_se(carry_forward = NA), "carry_forward")

  visits <- se_visits()
  visits$sex <- "f"
  expect_error(
    split_se(visits = visits, covariates = c("bili", "sex")),
    "`sex` is both fixed (from `baseline`) and time-varying",
    fixed = TRUE
  )
  visits$day[2] <- 11
  expect_error(
    split_se(visits = visits),
    "Subject 1 has a visit at 11, after its follow-up ends at 10"
  )
  visits$day[2] <- NA
  expect_error(split_se(visits = visits), "`day` must be numbers with no missing values")
  baseline <- se_baseline()
  baseline$time[1] <- NA
  expect_error(split_se(baseline), "`time` must be numbers with no missing values")
  baseline <- se_baseline()
  baseline$event <- c("a", "b", "a", "b")
  expect_error(split_se(baseline), "must be logical, numeric, or a factor")
  baseline <- se_baseline()
  baseline$id[1] <- NA
  expect_error(split_se(baseline), "`id` has missing values")

  visits <- se_visits()
  visits$id <- visits$id + 10L
  expect_error(split_se(visits = visits, carry_forward = FALSE), "produced no rows")
})
