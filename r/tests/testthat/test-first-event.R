# first_event() is checked on small cases from its rules, against the competing-risks response
# built by hand from mgus2, and through the Fine-Gray fit of the fixture made with survival's
# finegray() and coxph().

test_that("the earliest event wins, and a censored row ends at its latest time", {
  y <- first_event(list(a = list(c(5, 9, 4), c(1, 0, 0)), b = list(c(3, 8, 6), c(1, 1, 0))))
  expect_s3_class(y, "Surv")
  expect_identical(attr(y, "type"), "mright")
  expect_identical(attr(y, "states"), c("a", "b"))
  expect_identical(unname(unclass(y)[, "time"]), c(3, 8, 6))
  expect_identical(unname(unclass(y)[, "status"]), c(2, 2, 0))

  # Logical events work too.
  logical <- first_event(list(
    a = list(c(5, 9, 4), c(TRUE, FALSE, FALSE)), b = list(c(3, 8, 6), c(TRUE, TRUE, FALSE))
  ))
  expect_identical(unclass(logical), unclass(y))
})

test_that("ties follow the order of the endpoints", {
  a_first <- first_event(list(a = list(5, 1), b = list(5, 1)))
  b_first <- first_event(list(b = list(5, 1), a = list(5, 1)))
  expect_identical(attr(a_first, "states")[unclass(a_first)[, "status"]], "a")
  expect_identical(attr(b_first, "states")[unclass(b_first)[, "status"]], "b")
})

test_that("censor_at and start set the censoring and entry times", {
  y <- first_event(
    list(a = list(c(5, 9), c(0, 0)), b = list(c(3, 8), c(0, 0))),
    censor_at = c(10, 11), start = c(1, 2)
  )
  expect_identical(attr(y, "type"), "mcounting")
  expect_identical(unname(unclass(y)[, "start"]), c(1, 2))
  expect_identical(unname(unclass(y)[, "stop"]), c(10, 11))
  expect_identical(unname(unclass(y)[, "status"]), c(0, 0))

  # One censoring time for every row, and none for a row with an event.
  y <- first_event(list(a = list(c(5, 9), c(1, 0))), censor_at = 20)
  expect_identical(unname(unclass(y)[, "time"]), c(5, 20))
})

test_that("a row with a missing endpoint time is missing", {
  y <- first_event(list(a = list(c(5, NA, 4), c(1, 0, 0)), b = list(c(6, 8, NA), c(0, 1, 0))))
  expect_identical(unname(unclass(y)[, "time"]), c(5, NA, NA))
  frame <- data.frame(a_time = c(5, NA, 4, 8), b_time = c(6, 8, NA, 7), x = c(1, 2, 3, 5))
  frame$a <- c(1, 0, 0, 0)
  frame$b <- c(0, 1, 0, 1)
  fit <- aalen_johansen(
    first_event(list(a = list(a_time, a), b = list(b_time, b))) ~ 1,
    data = frame
  )
  expect_identical(fit$n, 2L)
})

test_that("an event after another endpoint's follow-up ended gives a warning", {
  expect_warning(
    first_event(list(pcm = list(80, 1), death = list(60, 0))),
    "row 1 has \"pcm\" at 80, but its \"death\" follow-up stops at 60",
    fixed = TRUE
  )
  expect_warning(
    first_event(list(
      pcm = list(c(80, 90, 10), c(1, 1, 1)), death = list(c(60, 70, 50), c(0, 0, 0))
    )),
    "2 rows have an event recorded after"
  )
  expect_no_warning(first_event(list(pcm = list(60, 1), death = list(80, 0))))
})

test_that("it builds the competing-risks response of mgus2", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  y <- with(mg, first_event(list(pcm = list(ptime, pstat), death = list(futime, death))))
  expect_no_warning(
    with(mg, first_event(list(pcm = list(ptime, pstat), death = list(futime, death))))
  )
  expected <- Surv(mg$etime, mg$event_f)
  expect_identical(attr(y, "states"), attr(expected, "states"))
  expect_identical(unclass(y)[, "time"], unclass(expected)[, "time"])
  expect_identical(unclass(y)[, "status"], unclass(expected)[, "status"])

  # In a formula, with the columns found in `data`.
  skip_if_no_shared_cases()
  fixture <- read_fixture("finegray_mgus2_pcm")
  fit <- fine_gray(
    first_event(list(pcm = list(ptime, pstat), death = list(futime, death))) ~ age + sex,
    data = mg, cause = "pcm"
  )
  expect_equal(unname(coef(fit)), unlist(fixture$coef), tolerance = 1e-6)
})

test_that("bad input is caught", {
  expect_error(first_event(list()), "at least one endpoint")
  expect_error(first_event(list(list(1, 1))), "its own name")
  expect_error(first_event(list(a = list(1, 1), a = list(2, 1))), "its own name")
  expect_error(first_event(list(a = list(1, 1, 1))), "must be a list of its time and event")
  expect_error(first_event(list(a = list("1", 1))), "isn't numeric")
  expect_error(first_event(list(a = list(1, 3))), "must be logical or 0/1")
  expect_error(first_event(list(a = list(1, NA))), "must be logical or 0/1")
  expect_error(
    first_event(list(a = list(c(1, 2), c(1, 0)), b = list(1, 1))),
    "same length"
  )
  expect_error(first_event(list(a = list(1, 0)), censor_at = c(1, 2)), "`censor_at`")
  expect_error(first_event(list(a = list(1, 0)), start = c(0, 0)), "`start`")
})
