test_that("Surv() is the reference implementation", {
  expect_identical(Surv, survival::Surv)
})
