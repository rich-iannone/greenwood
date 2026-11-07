# duration(): elapsed times from dates, in each unit, and in formulas.

test_that("the elapsed time is in the chosen unit", {
  start <- as.Date(c("2021-03-01", "2021-03-15", "2021-04-01"))
  end <- as.Date(c("2022-01-10", "2021-09-22", "2023-01-01"))
  days <- as.numeric(end - start)
  expect_identical(duration(start, end), days)
  expect_equal(duration(start, end, unit = "weeks"), days / 7)
  expect_equal(duration(start, end, unit = "months"), days / 30.4375)
  expect_equal(duration(start, end, unit = "years"), days / 365.25)
  expect_equal(duration(start, end, unit = "hours"), days * 24)
  expect_equal(duration(start, end, unit = "minutes"), days * 1440)
  expect_equal(duration(start, end, unit = "seconds"), days * 86400)
  # One start for every end.
  expect_identical(duration(start[1], end), as.numeric(end - start[1]))
})

test_that("date-times, character dates, and missing values are read", {
  t0 <- as.POSIXct("2021-03-01 06:00:00", tz = "UTC")
  t1 <- as.POSIXct("2021-03-02 18:00:00", tz = "UTC")
  expect_identical(duration(t0, t1, unit = "hours"), 36)
  expect_identical(duration(as.POSIXlt(t0), t1, unit = "hours"), 36)
  # A date is midnight UTC.
  expect_identical(duration(as.Date("2021-03-01"), t1, unit = "hours"), 42)
  expect_identical(duration("2021-03-01", "2021-03-11"), 10)
  expect_identical(duration("2021-03-01T06:00:00", "2021-03-01 18:30", unit = "hours"), 12.5)
  expect_identical(duration(c("2021-03-01", NA), c("2021-03-03", "2021-03-04")), c(2, NA))
  expect_identical(duration(as.Date(c("2021-03-01", NA)), as.Date("2021-03-05")), c(4, NA))
})

test_that("bad input is caught", {
  start <- as.Date(c("2021-03-01", "2021-03-15"))
  end <- as.Date(c("2021-02-01", "2021-04-15"))
  expect_error(
    duration(start, end),
    "1 row has `end` earlier than `start`, which gives a negative duration. The first is row 1."
  )
  expect_error(duration(1:2, end), "`1:2` must hold dates or date-times for `duration()`",
    fixed = TRUE
  )
  expect_error(duration(start, end, unit = "fortnights"), "`unit` must be one of")
  expect_error(duration("yesterday", "2021-03-01"), "isn't a date: \"yesterday\"")
  expect_error(duration(start, as.Date(rep("2022-01-01", 3))), "must have the same length")
  expect_error(duration(list(), start), "not a list")
})

test_that("it works inside a formula", {
  study <- data.frame(
    enroll = as.Date(c("2021-03-01", "2021-03-15", "2021-04-01", "2021-05-01")),
    exit = as.Date(c("2022-01-10", "2021-09-22", "2023-01-01", "2021-12-24")),
    died = c(TRUE, FALSE, TRUE, TRUE)
  )
  fit <- kaplan_meier(Surv(duration(enroll, exit, unit = "years"), died) ~ 1, data = study)
  expected <- kaplan_meier(
    Surv(as.numeric(exit - enroll) / 365.25, died) ~ 1,
    data = study
  )
  expect_equal(fit$curves, expected$curves)
})
