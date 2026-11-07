# event_time() beyond the shared conformance cases: the condition fields, the vector methods the
# fixtures don't record, and event_time() responses in model formulas.

et <- function() {
  event_time(
    time = c(7, 5, 3, 2, NA),
    status = c("e", "r", "l", "i", NA),
    time_max = c(NA, NA, NA, 4, NA)
  )
}

test_that("errors are classed and carry the check and the locations", {
  e <- tryCatch(
    event_time(time = c(1, 2, 3), status = c("e", "x", "y")),
    greenwood_event_time_error = function(e) e
  )
  expect_identical(e$check, "status_invalid_value")
  expect_identical(e$locations, c(2L, 3L))
  expect_s3_class(e, "error")

  # A long list of locations shows its first and last ones, as cli does.
  expect_error(
    event_time(time = -(1:30), status = rep("e", 30)),
    paste(
      "Problem at locations 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ...,",
      "29, and 30."
    ),
    fixed = TRUE
  )
  expect_error(
    event_time(time = 1:3, status = c("e", "e", NA)),
    "Problem at location 3.",
    fixed = TRUE
  )
  expect_error(
    event_time(time = factor(1), status = "e"),
    "Can't convert `time` <factor> to <double>.",
    fixed = TRUE
  )
  expect_error(event_time(time = 1, status = 1L), "not an integer.", fixed = TRUE)
  expect_error(event_time(time = 1, status = c(1, 2)), "not a double vector.", fixed = TRUE)
  expect_error(new_event_time(time = NULL), "not `NULL`.", fixed = TRUE)
  expect_error(as_surv(data.frame()), "Can't convert a <data.frame> object", fixed = TRUE)
})

test_that("empty input gives an empty vector", {
  x <- event_time(time = double(), status = character())
  expect_length(x, 0L)
  expect_identical(format(x), character())
  expect_identical(extract_time(x), double())
  expect_identical(nrow(as_surv(x)), 0L)
  expect_identical(attr(as_surv(x), "type"), "right")
  expect_identical(event_time(time = NULL, status = character()), x)
})

test_that("a missing index gives a missing element", {
  x <- et()
  out <- x[c(1, NA)]
  expect_identical(is.na(out), c(FALSE, TRUE))
  expect_identical(format(out), c("7 ", NA))
  expect_identical(extract_status(out), c("e", NA))
  expect_identical(unname(unclass(as_surv(out))[, "status"]), c(1, NA))
  expect_identical(x[c(TRUE, NA, FALSE, FALSE, FALSE)], out)
})

test_that("subsetting follows vctrs' rules", {
  x <- et()
  expect_identical(x[], x)
  expect_identical(x[0], x[integer()])
  expect_identical(x[TRUE], x)
  expect_identical(x[c(-1, -2)], x[3:5])
  expect_identical(x[2.7], x[2])
  expect_error(x[c(9, 10)], "Locations 9 and 10 don't exist. There are only 5 elements.")
  expect_error(x[1][2], "Location 2 doesn't exist. There is only 1 element.")
  expect_error(x[c(TRUE, FALSE)], "Logical subscript must be size 1 or 5, not 2.")
  expect_error(x["a"], "Can't use character names")
  expect_error(x[c(-1, 2)], "Can't mix negative and positive locations.")
  expect_error(x[[1:2]], "takes one location")
  expect_identical(x[[4]], x[4])
})

test_that("assignment replaces, recycles, and grows", {
  x <- et()
  y <- event_time(time = c(1, 9), status = c("r", "i"), time_max = c(NA, 12))
  z <- x
  z[2:3] <- y[1]
  expect_identical(format(z[1:3]), c("7 ", "1+", "1+"))
  z[[1]] <- y[2]
  expect_identical(format(z[1]), "[9, 12]")
  z <- y
  z[4] <- x[1]
  expect_identical(format(z), c("1+", "[9, 12]", NA, "7 "))
  expect_identical(is.na(z), c(FALSE, FALSE, TRUE, FALSE))
  z <- x
  z[c(TRUE, FALSE, FALSE, FALSE, FALSE)] <- y[2]
  expect_identical(format(z[1]), "[9, 12]")
  expect_error({
    z[1:3] <- y
  }, "Can't recycle input of size 2 to size 3.")
  expect_error({
    z[NA] <- y[1]
  }, "missing location")
})

test_that("comparison, duplicates, and conversions", {
  x <- et()
  expect_identical(x == x, c(TRUE, TRUE, TRUE, TRUE, NA))
  expect_identical(x != x[1], c(FALSE, TRUE, TRUE, TRUE, TRUE))
  # Two missing rows are duplicates, but a missing time with a status differs from one without.
  missing_with_status <- event_time(time = NA, status = "e")
  expect_identical(duplicated(c(x, x[5], missing_with_status)), c(rep(FALSE, 5), TRUE, FALSE))
  expect_length(unique(c(x, rev(x))), 5L)
  expect_identical(rep(x[1:2], times = 2), x[c(1, 2, 1, 2)])
  expect_identical(rep(x[1:2], each = 2), x[c(1, 1, 2, 2)])
  expect_true(anyNA(x))
  expect_false(anyNA(x[1:4]))
  expect_identical(lapply(x[1:2], format), list("7 ", "5+"))

  expect_error(x < x, "`<` isn't defined")
  expect_error(x == 1, "can only be compared with another")
  expect_error(sort(x), "no order")
  expect_error(order(x), "no order")
  expect_error(c(x, 1), "Can't combine `..1` <event_time> and `..2` <double>.", fixed = TRUE)
  expect_error(as.numeric(x), "to <double>", fixed = TRUE)
  expect_error(extract_time(x, 1), "`...` must be empty.", fixed = TRUE)
})

test_that("a data frame holds one in a column", {
  frame <- data.frame(id = 1:5, times = et())
  expect_named(frame, c("id", "times"))
  expect_s3_class(frame$times, "event_time")
  expect_identical(frame[2:3, "times"], et()[2:3])
  expect_identical(nrow(frame[frame$id > 2, ]), 3L)
  expect_named(as.data.frame(et()), "et()")
})

test_that("as_tibble() gives the three component columns", {
  skip_if_not_installed("tibble")
  frame <- tibble::as_tibble(et())
  expect_s3_class(frame, "tbl_df")
  expect_identical(frame$time, c(7, 5, 3, 2, NA))
  expect_identical(frame$time_max, c(NA, NA, NA, 4, NA))
})

test_that("as_surv() returns a Surv() response unchanged", {
  s <- Surv(c(1, 2), c(1, 0))
  expect_identical(as_surv(s), s)
  expect_s3_class(as_surv(et()), "greenwood_surv")
})

test_that("an event_time() response works in model formulas", {
  data <- small
  data$code <- ifelse(data$status == 2, "e", "r")
  expect_equal(
    kaplan_meier(event_time(time, code) ~ sex, data = data)$curves,
    kaplan_meier(Surv(time, status) ~ sex, data = data)$curves
  )
  fit <- cox_ph(greenwood::event_time(time, code) ~ ecog, data = data)
  expect_equal(coef(fit), coef(cox_ph(Surv(time, status) ~ ecog, data = data)))

  visits <- data.frame(
    lower = c(2, 4, 1, 6, 3, 8, 5, 0),
    upper = c(4, NA, 3, 9, NA, 11, NA, 2),
    code = c("i", "r", "i", "i", "r", "i", "e", "i")
  )
  visits$upper[visits$code == "e"] <- NA
  interval <- turnbull(event_time(lower, code, time_max = upper) ~ 1, data = visits)
  # Surv(type = "interval2") reads an exact time as equal bounds and right-censoring as a missing
  # upper bound.
  visits$right <- ifelse(visits$code == "e", visits$lower, visits$upper)
  reference <- turnbull(Surv(lower, right, type = "interval2") ~ 1, data = visits)
  expect_equal(interval$curves[[1]]$table, reference$curves[[1]]$table)

  out <- cross_validate(
    cox_ph,
    formula = event_time(time, code) ~ age + sex,
    data = transform(small_cox, code = ifelse(status == 2, "e", "r")),
    k = 3
  )
  expect_s3_class(out, "greenwood_cv")
})
