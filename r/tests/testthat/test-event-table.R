test_that("the event table on lung matches the fixture", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  expected <- read_fixture("lung_km_overall")$overall
  table <- event_table(Surv(time, status) ~ 1, data = survival::lung)
  expect_named(table, c("strata", "time", "n_risk", "n_event", "n_censor"))
  expect_equal(table$time, r_numbers(expected$time))
  expect_equal(table$n_risk, r_numbers(expected$n_risk))
  expect_equal(table$n_event, r_numbers(expected$n_event))
  expect_equal(table$n_censor, r_numbers(expected$n_censor))
})

test_that("groups follow strata as their own columns", {
  table <- event_table(Surv(time, status) ~ sex + ecog, data = small)
  expect_named(table, c("strata", "sex", "ecog", "time", "n_risk", "n_event", "n_censor"))
  expect_identical(unique(table$strata), c("sex=1, ecog=0", "sex=1, ecog=1", "sex=2, ecog=0", "sex=2, ecog=1"))
  expect_identical(sum(table$n_event + table$n_censor), nrow(small) + 0)
})

test_that("weights are summed and start times delay entry", {
  table <- event_table(Surv(time, status) ~ 1, data = small, weights = rep(0.5, 10))
  expect_identical(table$n_risk[1], 5)
  entry <- data.frame(start = c(0, 2, 0), stop = c(3, 4, 1), event = c(1, 1, 0))
  table <- event_table(Surv(start, stop, event) ~ 1, data = entry)
  expect_identical(table$time, c(1, 3, 4))
  expect_identical(table$n_risk, c(2, 2, 1))
})
