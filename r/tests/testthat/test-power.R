# Checks the log-rank sample size and power against values from the Python package and against
# Schoenfeld's closed form.

test_that("the events, power, and sample size match the Python package", {
  expect_identical(logrank_n_events(hazard_ratio = 0.5), 66L)
  expect_identical(logrank_n_events(hazard_ratio = 0.5, power = 0.9, allocation = 1 / 3), 99L)
  expect_identical(logrank_n_events(hazard_ratio = 1.5, alpha = 0.025, sides = 1), 191L)
  expect_equal(logrank_power(hazard_ratio = 0.5, n_events = 60), 0.7656462084964221, tolerance = 1e-12)
  expect_equal(logrank_power(hazard_ratio = 2, n_events = 60, allocation = 0.3, sides = 1),
    0.7926283123845915,
    tolerance = 1e-12
  )
  expect_equal(logrank_power(hazard_ratio = 0.8, n_events = 123.5, alpha = 0.01),
    0.09078673394599995,
    tolerance = 1e-12
  )
  expect_identical(logrank_sample_size(hazard_ratio = 0.5, prob_event = 0.4, power = 0.9), 219L)
  expect_identical(logrank_sample_size(hazard_ratio = 0.7, prob_event = 0.25, allocation = 0.6), 1029L)
})

test_that("the power at the needed number of events is the target", {
  for (hr in c(0.5, 0.75, 1.6)) {
    exact <- schoenfeld_events(hr, power = 0.85, alpha = 0.05, allocation = 0.4, sides = 2)
    expect_equal(logrank_power(hazard_ratio = hr, n_events = exact, allocation = 0.4), 0.85)
    expect_gte(logrank_power(
      hazard_ratio = hr, n_events = logrank_n_events(hazard_ratio = hr, power = 0.85, allocation = 0.4),
      allocation = 0.4
    ), 0.85)
  }
  # A hazard ratio and its inverse need the same number of events, as do the two allocations.
  expect_identical(logrank_n_events(hazard_ratio = 0.6), logrank_n_events(hazard_ratio = 1 / 0.6))
  expect_identical(
    logrank_n_events(hazard_ratio = 0.6, allocation = 0.3),
    logrank_n_events(hazard_ratio = 0.6, allocation = 0.7)
  )
  # When everyone has the event, the sample size is the number of events.
  expect_identical(logrank_sample_size(hazard_ratio = 0.6, prob_event = 1), logrank_n_events(hazard_ratio = 0.6))
})

test_that("the design is checked", {
  expect_error(logrank_n_events(hazard_ratio = 1), "not 1")
  expect_error(logrank_n_events(hazard_ratio = -0.5), "positive")
  expect_error(logrank_n_events(hazard_ratio = 0.5, power = 1), "`power`")
  expect_error(logrank_n_events(hazard_ratio = 0.5, alpha = 0), "`alpha`")
  expect_error(logrank_n_events(hazard_ratio = 0.5, allocation = 1.2), "`allocation`")
  expect_error(logrank_n_events(hazard_ratio = 0.5, sides = 3), "`sides`")
  expect_error(logrank_power(hazard_ratio = 0.5, n_events = 0), "`n_events`")
  expect_error(logrank_sample_size(hazard_ratio = 0.5, prob_event = 0), "`prob_event`")
  expect_error(logrank_sample_size(hazard_ratio = 0.5, prob_event = 1.1), "`prob_event`")
})
