test_that("Surv() returns a response matrix", {
  y <- Surv(c(5, 6, 4), c(1, 0, 1))
  expect_s3_class(y, c("greenwood_surv", "Surv"), exact = TRUE)
  expect_identical(colnames(unclass(y)), c("time", "status"))
  expect_identical(attr(y, "type"), "right")
  expect_identical(length(y), 3L)
})

test_that("Surv() agrees with the reference implementation", {
  skip_if_not_installed("survival")
  cases <- list(
    list(c(5, 6, 4), c(2, 1, 2)),
    list(c(0, 2, 1), c(5, 6, 4), c(1, 0, 1)),
    list(c(1, 2, NA, 3), c(1, NA, 4, 5), type = "interval2"),
    list(c(5, 6), factor(c("pcm", "censor"), levels = c("censor", "pcm")))
  )
  for (args in cases) {
    ours <- do.call(Surv, args)
    theirs <- do.call(survival::Surv, args)
    expect_identical(attr(ours, "type"), attr(theirs, "type"))
    expect_identical(attr(ours, "states"), attr(theirs, "states"))
    expect_equal(unclass(ours)[, ], unclass(theirs)[, ], ignore_attr = TRUE)
  }
})

test_that("Surv objects work with the survival package", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- survival::survfit(Surv(time, status) ~ 1, data = lung)
  reference <- survival::survfit(survival::Surv(time, status) ~ 1, data = lung)
  expect_equal(fit$surv, reference$surv)
})

test_that("subsetting rows keeps the type and states", {
  y <- Surv(c(5, 6, 7), factor(c("censor", "pcm", "death"), levels = c("censor", "pcm", "death")))
  sub <- y[2:3]
  expect_s3_class(sub, "greenwood_surv")
  expect_identical(attr(sub, "type"), "mright")
  expect_identical(attr(sub, "states"), c("pcm", "death"))
  expect_identical(length(sub), 2L)
  expect_identical(y[, "time"], c(5, 6, 7))
})

test_that("is.na() flags rows with any missing column", {
  y <- suppressWarnings(Surv(c(0, 6, NA), c(5, 6, 4), c(1, 0, 1)))
  expect_identical(is.na(y), c(FALSE, TRUE, TRUE))
})

test_that("Surv objects format and print compactly", {
  expect_snapshot({
    Surv(c(5, 6, NA), c(1, 0, 1))
    Surv(c(5, 6), c(1, 0), type = "left")
    Surv(c(0, 2), c(5, 6), c(1, 0))
    Surv(c(5, 6, 4, 2), c(NA, NA, NA, 4), c(0, 1, 2, 3), type = "interval")
    Surv(c(5, 6, 7), factor(c("censor", "pcm", "death"), levels = c("censor", "pcm", "death")))
    Surv(numeric(), numeric())
  })
})
