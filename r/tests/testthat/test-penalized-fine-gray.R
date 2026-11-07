# The penalized Fine-Gray model has no R reference here, so it's checked against Greenwood's Python
# values, against fine_gray() for an unpenalized fit, and against the optimality (KKT) conditions
# of the penalized fit, with the gradient from fine_gray()'s own pseudo partial likelihood.

pfg_formula <- Surv(etime, event_f) ~ age + sex + hgb

pfg_data <- function() {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  mg <- mg[stats::complete.cases(mg[c("etime", "event_f", "age", "sex", "hgb")]), ]
  rownames(mg) <- NULL
  mg
}

# The largest violation of the optimality conditions of a penalized_fine_gray() fit to `pfg_data()`,
# on the standardized scale: the gradient of the smooth part plus the lasso subgradient should
# be 0.
pfg_kkt_violation <- function(fit, mg) {
  x <- cbind(mg$age, as.numeric(mg$sex == "M"), mg$hgb)
  setup <- fg_setup(mg$etime, as.integer(mg$event_f) - 1L, match(fit$cause, fit$states))
  coefs <- unname(coef(fit))
  score <- fg_terms(coefs, sweep(x, 2L, fit$center), setup, detail = TRUE)$gradient / fit$scale
  b <- coefs * fit$scale
  lasso <- fit$penalizer * fit$l1_ratio
  smooth <- -score / nrow(x) + fit$penalizer * (1 - fit$l1_ratio) * b
  max(ifelse(b != 0, abs(smooth + lasso * sign(b)), pmax(abs(smooth) - lasso, 0)))
}

# FISTA stops when a step is small rather than when the gradient is, and the rounding of each
# step decides where, so the fits agree with Python's to about 1e-5 and are optimal to about 1e-6.
test_that("the fits match Python", {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  lasso <- penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = 0.01,
    l1_ratio = 1
  )
  expect_identical(names(coef(lasso)), c("age", "sexM", "hgb"))
  expect_equal(unname(coef(lasso)), c(-0.00804888606246963, 0, 0), tolerance = 1e-4)
  expect_identical(lasso$n_nonzero, 1L)
  expect_identical(c(lasso$n, lasso$n_event, lasso$n_dropped), c(1371L, 114L, 13L))
  expect_equal(
    predict(lasso, newdata = mg[c(1, 2, 6), ]),
    c(-0.1410286951076918, -0.06053983448299549, -0.15712646723263107),
    tolerance = 1e-4
  )
  expect_equal(
    predict(lasso, newdata = mg[2, ], type = "cif", times = c(24, 120))$estimate,
    c(0.016407734533762874, 0.05950071965220172),
    tolerance = 1e-5
  )

  small <- penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = 0.002,
    l1_ratio = 1
  )
  expect_equal(unname(coef(small)), c(-0.01555323506830837, -0.19957789370125636, 0),
    tolerance = 1e-4
  )
  expect_equal(
    predict(small, newdata = mg[2, ], type = "cif", times = c(24, 120))$estimate,
    c(0.016984128359537443, 0.06150812249656912),
    tolerance = 1e-5
  )

  mixed <- penalized_fine_gray(pfg_formula, data = mg, cause = "death", penalizer = 0.02,
    l1_ratio = 0.5
  )
  expect_equal(unname(coef(mixed)), c(
    0.04914437251060211, 0.42629221201909034, -0.1113440010101622
  ), tolerance = 1e-5)
  expect_identical(mixed$n_event, 855L)
  expect_equal(
    predict(mixed, newdata = mg[c(1, 2, 6), ]),
    c(0.6515145750209784, 0.33822125153121685, 1.1983643322633055),
    tolerance = 1e-5
  )

  ridge <- penalized_fine_gray(pfg_formula, data = mg, cause = "death", penalizer = 0.05,
    l1_ratio = 0
  )
  expect_equal(unname(coef(ridge)), c(
    0.04657399850544474, 0.4296523675867115, -0.1155250228052474
  ), tolerance = 1e-5)
  expect_equal(
    predict(ridge, newdata = mg[2, ], type = "cif", times = c(24, 120))$estimate,
    c(0.19624468969105424, 0.640366508945901),
    tolerance = 1e-5
  )
})

test_that("the fits are optimal", {
  mg <- pfg_data()
  settings <- list(
    list("pcm", 0.01, 1), list("pcm", 0.002, 0.5), list("death", 0.02, 0.5),
    list("death", 0.05, 0)
  )
  for (s in settings) {
    fit <- penalized_fine_gray(pfg_formula,
      data = mg, cause = s[[1]], penalizer = s[[2]], l1_ratio = s[[3]]
    )
    expect_true(fit$converged)
    expect_lt(pfg_kkt_violation(fit, mg), 1e-5)
  }
})

test_that("an unpenalized fit is the Fine-Gray fit", {
  mg <- pfg_data()
  fit <- penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = 0)
  fg <- fine_gray(pfg_formula, data = mg, cause = "pcm")
  expect_true(fit$converged)
  expect_equal(coef(fit), coef(fg), tolerance = 1e-3)
  expect_equal(fit$loglik, fg$loglik[2L], tolerance = 1e-9)
  expect_equal(unname(fit$center), unname(fg$means))
  expect_equal(fit$baseline$time, fg$baseline$time)
  expect_equal(fit$baseline$cumhaz, fg$baseline$cumhaz, tolerance = 1e-4)
  expect_equal(predict(fit), predict(fg), tolerance = 1e-3)
  newdata <- data.frame(age = c(55, 80), sex = c("F", "M"), hgb = c(14, 10))
  expect_equal(
    predict(fit, newdata = newdata, type = "cif", times = c(50, 150, 300)),
    predict(fg, newdata = newdata, type = "cif", times = c(50, 150, 300)),
    tolerance = 1e-4
  )

  # The other cause, and the log pseudo partial likelihood at the penalized estimate.
  death <- penalized_fine_gray(pfg_formula, data = mg, cause = "death", penalizer = 0)
  expect_equal(coef(death), coef(fine_gray(pfg_formula, data = mg, cause = "death")),
    tolerance = 1e-4
  )
  ridge <- penalized_fine_gray(pfg_formula, data = mg, cause = 2, penalizer = 0.05, l1_ratio = 0)
  x <- cbind(mg$age, as.numeric(mg$sex == "M"), mg$hgb)
  setup <- fg_setup(mg$etime, as.integer(mg$event_f) - 1L, 2L)
  expect_equal(
    ridge$loglik,
    fg_terms(unname(coef(ridge)), sweep(x, 2L, colMeans(x)), setup)$loglik
  )
})

test_that("the penalty shrinks and selects", {
  mg <- pfg_data()
  nonzero <- vapply(c(0.001, 0.005, 0.01, 0.05), function(lambda) {
    penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = lambda,
      l1_ratio = 1
    )$n_nonzero
  }, integer(1))
  expect_true(all(diff(nonzero) <= 0))
  expect_identical(nonzero[c(1, 4)], c(3L, 0L))

  # With every coefficient at 0, every subject gets the baseline curve.
  empty <- penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = 0.05,
    l1_ratio = 1
  )
  expect_equal(predict(empty), rep(0, nrow(mg)))
  curve <- predict(empty, newdata = mg[1:2, ], type = "cif", times = c(100, 300))
  expect_identical(curve$estimate[1:2], curve$estimate[3:4])
  expect_equal(curve$estimate[1:2], 1 - exp(-step_at(
    empty$baseline$time, empty$baseline$cumhaz, c(100, 300), 0
  )))

  # Without standardizing, the penalty falls on the coefficients in their own units, and the
  # badly scaled covariates slow FISTA down.
  expect_warning(
    unscaled <- penalized_fine_gray(pfg_formula,
      data = mg, cause = "death", penalizer = 0.02, l1_ratio = 0, standardize = FALSE
    ),
    "didn't converge in 1000 iterations"
  )
  expect_false(unscaled$converged)
  expect_identical(unname(unscaled$scale), c(1, 1, 1))

  # Collinear covariates are allowed, and a ridge penalty splits the effect between them.
  mg$age2 <- mg$age
  twins <- penalized_fine_gray(Surv(etime, event_f) ~ age + age2 + sex,
    data = mg, cause = "death", penalizer = 0.05, l1_ratio = 0
  )
  expect_equal(coef(twins)[["age"]], coef(twins)[["age2"]], tolerance = 1e-6)
})

test_that("predictions and coefficients have the documented forms", {
  mg <- pfg_data()
  fit <- penalized_fine_gray(pfg_formula, data = mg, cause = "pcm", penalizer = 0.002)
  lp <- predict(fit)
  expect_equal(mean(lp), 0, tolerance = 1e-12)
  expect_identical(predict(fit, newdata = mg), lp)
  expect_equal(predict(fit, type = "risk"), exp(lp))
  newdata <- data.frame(age = c(50, 70), sex = c("M", "F"), hgb = c(12, 14))
  expect_equal(
    predict(fit, newdata = newdata),
    drop(cbind(newdata$age, newdata$sex == "M", newdata$hgb) %*% coef(fit)) -
      sum(fit$center * coef(fit))
  )
  cif <- predict(fit, newdata = newdata, type = "cif", times = c(300, 100, 0))
  expect_named(cif, c("subject", "time", "estimate"))
  expect_identical(cif$subject, rep(1:2, each = 3))
  expect_identical(cif$time, rep(c(300, 100, 0), 2))
  expect_identical(cif$estimate[c(3, 6)], c(0, 0))
  expect_gt(cif$estimate[1], cif$estimate[2])
  default <- predict(fit, newdata = newdata[1, ], type = "cif")
  expect_identical(default$time, sort(unique(mg$etime[mg$event_f == "pcm"])))
  expect_true(all(diff(default$estimate) >= 0))

  table <- as.data.frame(fit)
  expect_named(table, c("term", "estimate", "hazard_ratio"))
  expect_identical(table$term, c("age", "sexM", "hgb"))
  expect_equal(table$hazard_ratio, exp(table$estimate))
  expect_identical(nobs(fit), nrow(mg))

  # The cause can be given by number, and the first cause is the default.
  expect_identical(
    coef(penalized_fine_gray(pfg_formula, data = mg, cause = 1, penalizer = 0.002)),
    coef(fit)
  )
  expect_identical(coef(penalized_fine_gray(pfg_formula, data = mg, penalizer = 0.002)), coef(fit))
  expect_identical(fit$cause, "pcm")
  expect_identical(fit$states, c("pcm", "death"))

  # Subsets work as in fine_gray().
  older <- penalized_fine_gray(pfg_formula, data = mg, subset = age > 60, penalizer = 0.002)
  expect_identical(
    coef(older),
    coef(penalized_fine_gray(pfg_formula, data = mg[mg$age > 60, ], penalizer = 0.002))
  )
})

test_that("print shows the fit", {
  skip_if_not_installed("survival")
  fit <- penalized_fine_gray(pfg_formula, data = mgus2_competing(), cause = "pcm",
    penalizer = 0.01, l1_ratio = 1
  )
  expect_output(print(fit), "<penalized_fine_gray> Surv(etime, event_f) ~ age + sex + hgb",
    fixed = TRUE
  )
  expect_output(print(fit), paste(
    "Elastic net, penalizer = 0.01, l1_ratio = 1. Subdistribution hazards for cause \"pcm\".",
    "n = 1371, events = 114",
    sep = "\n"
  ), fixed = TRUE)
  expect_output(print(fit), "Nonzero coefficients: 1 of 3. Log pseudo partial likelihood",
    fixed = TRUE
  )
  expect_output(print(fit), "13 rows were dropped", fixed = TRUE)
})

test_that("bad input is rejected", {
  mg <- pfg_data()
  expect_error(penalized_fine_gray(pfg_formula, data = mg, penalizer = -1), "`penalizer`")
  expect_error(penalized_fine_gray(pfg_formula, data = mg, l1_ratio = 2), "`l1_ratio`")
  expect_error(penalized_fine_gray(pfg_formula, data = mg, standardize = "yes"), "`standardize`")
  expect_error(penalized_fine_gray(pfg_formula, data = mg, max_iter = 0), "`max_iter`")
  expect_error(penalized_fine_gray(pfg_formula, data = mg, cause = "relapse"), "`cause`")
  expect_error(penalized_fine_gray(pfg_formula, data = mg, cause = 3), "`cause`")
  expect_error(
    penalized_fine_gray(Surv(etime, death) ~ age, data = mg),
    "competing-risks response"
  )
  mg$entry <- 0
  expect_error(
    penalized_fine_gray(Surv(entry, etime, event_f) ~ age, data = mg),
    "delayed entry"
  )
  expect_error(penalized_fine_gray(Surv(etime, event_f) ~ 1, data = mg), "at least one covariate")
  expect_error(
    penalized_fine_gray(Surv(etime, event_f) ~ age + strata(sex), data = mg),
    "strata"
  )
  expect_error(
    penalized_fine_gray(Surv(etime, event_f) ~ age + offset(hgb), data = mg),
    "offset"
  )
  none <- mg
  none$event_f[none$event_f == "pcm"] <- "censor"
  expect_error(penalized_fine_gray(pfg_formula, data = none, cause = "pcm"), "no events of cause")
  fit <- penalized_fine_gray(pfg_formula, data = mg)
  expect_error(predict(fit, newdata = as.matrix(mg)), "data frame")
  expect_error(predict(fit, newdata = mg[c("age", "sex")]), "hgb")
  expect_error(predict(fit, type = "survival"), "should be one of")
})
