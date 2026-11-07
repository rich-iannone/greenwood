# Replays the Aalen additive model fixtures in fixtures/r/ (aalen_lung_age_sex, aalen_veteran),
# written from survival::aareg(). Weighted and counting-process fits, which the fixtures don't
# cover, are checked against aareg() directly, and the predictions against Greenwood's Python
# values.

fixture_matrix <- function(columns) {
  do.call(cbind, lapply(columns, r_numbers))
}

test_that("the fits reproduce the fixtures", {
  skip_if_no_shared_cases()
  skip_if_not_installed("survival")
  lung <- survival::lung
  veteran <- survival::veteran
  cases <- list(
    list(
      name = "aalen_lung_age_sex",
      fit = aalen_additive(Surv(time, status) ~ age + sex, data = lung)
    ),
    list(
      name = "aalen_veteran",
      fit = aalen_additive(Surv(time, status) ~ trt + karno + diagtime + age, data = veteran)
    )
  )
  for (case in cases) {
    expected <- read_fixture(case$name)
    fit <- case$fit
    n <- r_numbers(expected$n)
    expect_identical(c(fit$n, fit$n_times), as.integer(n[1:2]))
    expect_identical(colnames(fit$increments), unlist(expected$term_names))
    expect_equal(fit$times, r_numbers(expected$event_times), tolerance = 1e-12)
    expect_equal(fit$n_risk, r_numbers(expected$nrisk), tolerance = 1e-12)
    expect_equal(unname(fit$increments), unname(fixture_matrix(expected$coef_increments)),
      tolerance = 1e-9
    )
    # The cumulative coefficients after the last event at each time.
    cumulative <- fixture_matrix(expected$cumulative_coefs)
    last <- !duplicated(fit$times, fromLast = TRUE)
    coefs <- cumulative_coefficients(fit)
    expect_equal(coefs$estimate, as.vector(cumulative[last, ]), tolerance = 1e-9)
    table <- as.data.frame(fit)
    expect_equal(table$slope, r_numbers(expected$summary_slope), tolerance = 1e-9)
    expect_equal(table$estimate, r_numbers(expected$summary_coef), tolerance = 1e-9)
    expect_equal(table$std_error, r_numbers(expected$summary_se), tolerance = 1e-9)
    expect_equal(table$statistic, r_numbers(expected$summary_z), tolerance = 1e-9)
    expect_equal(table$p_value, r_numbers(expected$summary_p), tolerance = 1e-9)
    expect_equal(unname(fit$test_statistic), r_numbers(expected$test_statistic), tolerance = 1e-9)
    expect_equal(unname(fit$test_var), unname(fixture_matrix(expected$test_var)), tolerance = 1e-9)
  }
})

test_that("number-at-risk weights, weighted fits, and counting-process data match aareg()", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  table_of <- function(reference) {
    unname(summary(reference)$table)
  }
  ours <- function(fit) {
    columns <- c("slope", "estimate", "std_error", "statistic", "p_value")
    unname(as.matrix(as.data.frame(fit)[columns]))
  }

  nrisk <- aalen_additive(Surv(time, status) ~ age + sex, data = lung, test = "nrisk")
  reference <- survival::aareg(survival::Surv(time, status) ~ age + sex,
    data = lung, test = "nrisk"
  )
  expect_equal(ours(nrisk), table_of(reference), tolerance = 1e-10)
  expect_equal(unname(nrisk$increments), unname(reference$coefficient), tolerance = 1e-10)
  expect_output(print(nrisk), "Number-at-risk test weights")

  set.seed(1)
  lung$w <- stats::runif(nrow(lung), 0.5, 2)
  lung$ecog <- factor(ifelse(lung$ph.ecog >= 2, "2+", lung$ph.ecog))
  weighted <- aalen_additive(Surv(time, status) ~ age + ecog, data = lung, weights = w)
  reference <- survival::aareg(survival::Surv(time, status) ~ age + ecog, data = lung, weights = w)
  expect_identical(colnames(weighted$increments), c("Intercept", "age", "ecog1", "ecog2+"))
  expect_equal(unname(weighted$increments), unname(reference$coefficient), tolerance = 1e-10)
  expect_equal(weighted$n_risk, reference$nrisk, tolerance = 1e-12)
  expect_equal(ours(weighted), table_of(reference), tolerance = 1e-10)
  expect_identical(weighted$n_dropped, 1L)

  heart <- survival::heart
  counting <- aalen_additive(Surv(start, stop, event) ~ age + surgery, data = heart)
  reference <- survival::aareg(survival::Surv(start, stop, event) ~ age + surgery, data = heart)
  expect_equal(unname(counting$increments), unname(reference$coefficient), tolerance = 1e-10)
  expect_equal(ours(counting), table_of(reference), tolerance = 1e-10)
  expect_equal(counting$wald_test$statistic, drop(summary(reference)$chisq), tolerance = 1e-10)
})

test_that("the covariate test is the chi-square of the statistics", {
  skip_if_not_installed("survival")
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = survival::lung)
  reference <- summary(
    survival::aareg(survival::Surv(time, status) ~ age + sex, data = survival::lung)
  )
  expect_equal(fit$wald_test$statistic, drop(reference$chisq), tolerance = 1e-10)
  expect_identical(fit$wald_test$df, 2L)
  expect_equal(fit$wald_test$p_value, stats::pchisq(fit$wald_test$statistic, 2, lower.tail = FALSE))
  expect_equal(sqrt(diag(vcov(fit))), as.data.frame(fit)$std_error, ignore_attr = TRUE)
  expect_identical(coef(fit), fit$coefficients)
  expect_identical(nobs(fit), 228L)
})

test_that("predictions match Python", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
  times <- c(0, 180, 365, 2000)
  survival <- predict(fit, newdata = lung[1:3, ], type = "survival", times = times)
  expect_named(survival, c("subject", "time", "estimate", "std_error", "conf_low", "conf_high"))
  expect_identical(survival$subject, rep(1:3, each = 4))
  expect_equal(survival$estimate, c(
    1, 0.6020046287160741, 0.3001381411608335, 0.03412813323883351,
    1, 0.6260210617548203, 0.32089595582324404, 0.04542746481999475,
    1, 0.6769664928933177, 0.36681775996443816, 0.08048787007188488
  ), tolerance = 1e-12)
  cumhaz <- predict(fit, newdata = lung[1:3, ], type = "cumhaz", times = times)
  expect_equal(cumhaz$estimate, c(
    0, 0.5074901448057424, 1.203512439773848, 3.3776332131692985,
    0, 0.46837126347226665, 1.1366383335532682, 3.091638404862258,
    0, 0.3901335008053153, 1.0028901211121086, 2.519648788248177
  ), tolerance = 1e-12)
  expect_equal(survival$conf_low, exp(-cumhaz$conf_high))
  expect_equal(survival$std_error, cumhaz$std_error)

  # Without newdata or times, the rows the model was fit to at the event times used.
  all <- predict(fit)
  expect_identical(nrow(all), fit$n * fit$n_times)
  expect_identical(unique(all$time), unique(fit$times))
})

test_that("the predictions and cumulative coefficients hang together", {
  skip_if_not_installed("survival")
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = survival::lung)
  coefs <- cumulative_coefficients(fit)
  expect_named(coefs, c(
    "term", "time", "n_risk", "n_event", "estimate", "std_error", "conf_low", "conf_high"
  ))
  expect_identical(unique(coefs$term), c("Intercept", "age", "sex"))
  expect_identical(nrow(coefs), 3L * fit$n_times)
  sex <- coefs[coefs$term == "sex", ]
  last <- !duplicated(fit$times, fromLast = TRUE)
  expect_equal(sex$std_error, sqrt(cumsum(fit$increments[, "sex"]^2))[last])
  expect_equal(sex$conf_high - sex$estimate, stats::qnorm(0.975) * sex$std_error)
  expect_identical(sex$n_event[sex$time == 11], 3)
  narrow <- cumulative_coefficients(fit, conf_level = 0.5)
  expect_lt(max(narrow$conf_high - narrow$conf_low), max(coefs$conf_high - coefs$conf_low))

  # The cumulative hazard is the intercept plus the covariates times their coefficients, and its
  # variance sums the outer products of the increments.
  newdata <- data.frame(age = 60, sex = 2)
  t <- 400
  cumhaz <- predict(fit, newdata = newdata, type = "cumhaz", times = t)
  b <- vapply(c("Intercept", "age", "sex"), function(term) {
    rows <- coefs[coefs$term == term & coefs$time <= t, ]
    rows$estimate[nrow(rows)]
  }, numeric(1))
  expect_equal(cumhaz$estimate, sum(b * c(1, 60, 2)))
  used <- fit$times <= t
  increments <- fit$increments[used, ] %*% c(1, 60, 2)
  expect_equal(cumhaz$std_error, sqrt(sum(increments^2)))

  # Before the first event, nothing has accumulated. A negative cumulative hazard is floored at 0.
  early <- predict(fit, newdata = newdata, type = "cumhaz", times = 1)
  expect_identical(c(early$estimate, early$std_error), c(0, 0))
  young <- predict(fit, newdata = data.frame(age = -500, sex = 2), type = "cumhaz", times = 400)
  expect_identical(young$estimate, 0)
  expect_gte(young$conf_low, 0)
})

test_that("weights act as copies of rows", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung <- lung[stats::complete.cases(lung[c("time", "status", "age", "sex")]), ]
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
  lung$two <- 2
  doubled <- aalen_additive(Surv(time, status) ~ age + sex, data = lung, weights = two, nmin = 12)
  expect_equal(doubled$increments, fit$increments)
  expect_equal(as.data.frame(doubled)$statistic, as.data.frame(fit)$statistic)
  expect_equal(coef(doubled), coef(fit))

  lung$k <- rep(1:2, length.out = nrow(lung))
  weighted <- aalen_additive(Surv(time, status) ~ age + sex, data = lung, weights = k)
  copies <- lung[rep(seq_len(nrow(lung)), lung$k), ]
  copied <- aalen_additive(Surv(time, status) ~ age + sex, data = copies)
  expect_equal(cumulative_coefficients(weighted)[c("time", "n_risk", "n_event", "estimate")],
    cumulative_coefficients(copied)[c("time", "n_risk", "n_event", "estimate")],
    ignore_attr = TRUE
  )
  newdata <- data.frame(age = 60, sex = 1)
  expect_equal(
    predict(weighted, newdata = newdata, times = c(100, 300))$estimate,
    predict(copied, newdata = newdata, times = c(100, 300))$estimate
  )
})

test_that("nmin and collinear risk sets leave event times out", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
  expect_identical(fit$nmin, 6)
  expect_true(all(fit$n_risk >= 6))
  large <- aalen_additive(Surv(time, status) ~ age + sex, data = lung, nmin = 100)
  expect_true(all(large$n_risk >= 100))
  expect_lt(large$n_times, fit$n_times)
  expect_identical(large$times, fit$times[fit$n_risk >= 100])

  # The one subject with ph.ecog = 3 dies at day 118, after which that column is all zeros.
  lung$ecog <- factor(lung$ph.ecog)
  sparse <- aalen_additive(Surv(time, status) ~ age + ecog, data = lung)
  expect_identical(max(sparse$times), 118)
  expect_true(all(is.finite(sparse$increments)))
})

test_that("factors, the table, and print work", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  fit <- aalen_additive(Surv(time, status) ~ age + sex, data = lung)
  lung$group <- factor(ifelse(lung$sex == 1, "m", "f"), levels = c("m", "f"))
  by_factor <- aalen_additive(Surv(time, status) ~ age + group, data = lung)
  expect_identical(colnames(by_factor$increments), c("Intercept", "age", "groupf"))
  expect_equal(
    predict(by_factor, newdata = data.frame(age = 60, group = "f"), times = 300),
    predict(fit, newdata = data.frame(age = 60, sex = 2), times = 300)
  )

  table <- as.data.frame(fit)
  expect_named(table, c(
    "term", "slope", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"
  ))
  expect_equal(table$statistic, unname(fit$test_statistic / sqrt(diag(fit$test_var))))
  narrow <- as.data.frame(fit, conf_level = 0.9)
  expect_equal(narrow$conf_high, table$estimate + stats::qnorm(0.95) * table$std_error)

  expect_output(print(fit), "Aalen test weights. n = 228, events = 165, event times used = 138",
    fixed = TRUE
  )
  expect_output(print(fit), "Chi-square test of the covariates: 13.8 on 2 df", fixed = TRUE)
  dropped <- aalen_additive(Surv(time, status) ~ ph.ecog, data = lung)
  expect_output(print(dropped), "1 row was dropped")
})

test_that("bad input is rejected", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  f <- Surv(time, status) ~ age + sex
  expect_error(aalen_additive(f, data = lung, test = "variance"), "should be one of")
  expect_error(aalen_additive(f, data = lung, nmin = -1), "`nmin`")
  expect_error(aalen_additive(f, data = lung, qrtol = -1), "`qrtol`")
  expect_error(aalen_additive(Surv(time, status) ~ 1, data = lung), "at least one covariate")
  expect_error(aalen_additive(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(aalen_additive(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(aalen_additive(Surv(time, status * 0) ~ age, data = lung), "at least one event")
  expect_error(aalen_additive(f, data = lung, nmin = 1000), "No event time")
  lung$age2 <- 2 * lung$age
  expect_error(aalen_additive(Surv(time, status) ~ age + age2, data = lung), "collinear")
  expect_error(
    aalen_additive(Surv(time, time2 = time, event = status, type = "interval") ~ age, data = lung),
    "right-censored or counting-process"
  )
  fit <- aalen_additive(f, data = lung)
  expect_error(predict(fit, newdata = list(age = 1)), "data frame")
  expect_error(predict(fit, newdata = data.frame(age = 1)), "sex")
  expect_error(predict(fit, times = "a"), "`times`")
  expect_error(predict(fit, type = "hazard"), "should be one of")
})
