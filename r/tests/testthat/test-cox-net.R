# The elastic-net Cox model has no R reference here, so it's checked against Greenwood's Python
# values, against cox_ph() for an unpenalized fit, and against the optimality (KKT) conditions of
# the penalized fit, with the gradient from cox_ph()'s own partial likelihood.

lung_net_formula <- Surv(time, status) ~ age + sex + ph.ecog + ph.karno + wt.loss

# The largest violation of the optimality conditions of a cox_net() fit, on the standardized
# scale: the gradient of the smooth part plus the lasso subgradient should be 0.
kkt_violation <- function(fit, x, start, stop, event, w = rep(1, nrow(x))) {
  setup <- cox_setup(x, start, stop, event, w, rep(1L, nrow(x)), efron = FALSE)
  score <- cox_terms(unname(coef(fit)), setup)$score / fit$scale
  b <- unname(coef(fit)) * fit$scale
  lasso <- fit$penalizer * fit$l1_ratio
  smooth <- -score / sum(w) + fit$penalizer * (1 - fit$l1_ratio) * b
  max(ifelse(b != 0, abs(smooth + lasso * sign(b)), pmax(abs(smooth) - lasso, 0)))
}

lung_complete <- function() {
  skip_if_not_installed("survival")
  vars <- c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")
  lung <- stats::na.omit(survival::lung[vars])
  rownames(lung) <- NULL
  lung
}

test_that("the fits match Python", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lasso <- cox_net(lung_net_formula, data = lung, penalizer = 0.05, l1_ratio = 1)
  expect_equal(unname(coef(lasso)), c(
    0.00617421931022444, -0.4092623117988817, 0.36732247880324065, 0, -0.00069759902893235
  ), tolerance = 1e-9)
  expect_equal(lasso$loglik, -661.6791289944867, tolerance = 1e-12)
  expect_identical(lasso$effective_df, 4L)
  expect_equal(AIC(lasso), 1331.3582579889735, tolerance = 1e-12)
  expect_equal(BIC(lasso), 1343.4273773362331, tolerance = 1e-12)
  expect_identical(c(lasso$n, lasso$n_event, lasso$n_dropped), c(213L, 151, 15L))
  expect_equal(
    predict(lasso, newdata = lung[1:3, ], type = "lp"),
    c(NA, -0.14767139607680407, -0.22176202779949733),
    tolerance = 1e-9
  )
  expect_equal(
    predict(lasso, newdata = lung[2, ], type = "survival", times = c(180, 365))$estimate,
    c(0.7850848335120472, 0.48253832735015123),
    tolerance = 1e-9
  )

  strong <- cox_net(lung_net_formula, data = lung, penalizer = 0.2, l1_ratio = 1)
  expect_equal(unname(coef(strong)), c(0, 0, 0.03951371597370969, 0, 0), tolerance = 1e-4)

  expect_warning(
    unscaled <- cox_net(lung_net_formula,
      data = lung, penalizer = 0.05, l1_ratio = 1, standardize = FALSE
    ),
    "didn't converge in 1000 iterations"
  )
  expect_equal(unname(coef(unscaled)), c(
    0.01491239703023745, -0.2516039280114083, 0.3058017096769829, -0.00100717052454584,
    -0.00466773720332675
  ), tolerance = 1e-9)

  # The stopping rule ends FISTA a little short of the optimum, at a point that depends on the
  # rounding of each step, so the strong lasso, ridge, and elastic-net fits agree to about 1e-5.
  ridge <- cox_net(lung_net_formula, data = lung, penalizer = 0.05, l1_ratio = 0)
  expect_equal(unname(coef(ridge)), c(
    0.01396522586893512, -0.562592626121532, 0.5907951287822489, 0.00895622998978396,
    -0.00754605274959325
  ), tolerance = 1e-4)
  expect_equal(ridge$effective_df, 4.502103711181385, tolerance = 1e-5)
  mixed <- cox_net(lung_net_formula, data = lung, penalizer = 0.1, l1_ratio = 0.5)
  expect_equal(unname(coef(mixed)), c(
    0.006160954116284212, -0.377939758213229, 0.33735734631707026, 0, -0.00014972030846913269
  ), tolerance = 1e-4)
  expect_equal(mixed$effective_df, 3.70796145564968, tolerance = 1e-5)
})

# FISTA stops when a step is small rather than when the gradient is, so the fits below are
# optimal to about 1e-5 (in the gradient) and 1e-4 (in the coefficients), not to the last digit.
test_that("the fits are optimal", {
  lung <- lung_complete()
  x <- as.matrix(lung[3:7])
  event <- as.numeric(lung$status == 2)
  for (setting in list(c(0.05, 1), c(0.05, 0), c(0.1, 0.5), c(0.01, 0.2))) {
    fit <- cox_net(lung_net_formula,
      data = lung, penalizer = setting[1], l1_ratio = setting[2], tol = 1e-10, max_iter = 1e4
    )
    expect_lt(kkt_violation(fit, x, rep(-Inf, nrow(x)), lung$time, event), 1e-5)
  }

  # The log partial likelihood is the Breslow one at the estimate.
  fit <- cox_net(lung_net_formula, data = lung, penalizer = 0.05, l1_ratio = 0)
  reference <- survival::coxph(survival::Surv(time, status) ~ age + sex + ph.ecog + ph.karno +
    wt.loss, data = lung, ties = "breslow", init = unname(coef(fit)),
  control = survival::coxph.control(iter.max = 0))
  expect_equal(fit$loglik, reference$loglik[2L], tolerance = 1e-10)

  # The ridge effective degrees of freedom, from the information matrix of the standardized fit.
  xs <- sweep(sweep(x, 2L, fit$center), 2L, fit$scale, "/")
  setup <- cox_setup(xs, rep(-Inf, nrow(x)), lung$time, event, rep(1, nrow(x)),
    rep(1L, nrow(x)),
    efron = FALSE
  )
  info <- cox_terms(unname(coef(fit)) * fit$scale, setup)$info
  expect_equal(fit$effective_df, sum(diag(info %*% solve(info + diag(0.05 * nrow(x), 5)))))
})

test_that("an unpenalized fit is the Breslow Cox fit", {
  lung <- lung_complete()
  fit <- cox_net(lung_net_formula, data = lung, penalizer = 0, tol = 1e-10, max_iter = 1e4)
  cox <- cox_ph(lung_net_formula, data = lung, ties = "breslow")
  expect_true(fit$converged)
  expect_equal(coef(fit), coef(cox), tolerance = 1e-3)
  expect_equal(fit$loglik, cox$loglik[2L], tolerance = 1e-9)
  expect_equal(fit$effective_df, 5L)
  expect_equal(predict(fit, type = "lp"), unname(predict(cox, type = "lp")), tolerance = 1e-3)
  newdata <- lung[c(5, 50), ]
  expect_equal(
    predict(fit, newdata = newdata, type = "survival", times = c(100, 300, 500))$estimate,
    predict(cox, newdata = newdata, type = "survival", times = c(100, 300, 500))$estimate,
    tolerance = 1e-4
  )
})

test_that("counting-process data and weights work", {
  lung <- lung_complete()
  n <- nrow(lung)
  # Splitting each subject's time in two leaves the risk sets alone and doubles the total weight,
  # so the split fit at penalizer lambda is the fit to the original rows at 2 lambda.
  split <- rbind(
    transform(lung, start = 0, stop = time / 2, event = 0),
    transform(lung, start = time / 2, stop = time, event = as.numeric(status == 2))
  )
  counting <- cox_net(Surv(start, stop, event) ~ age + sex + ph.ecog + ph.karno + wt.loss,
    data = split, penalizer = 0.05, l1_ratio = 1, tol = 1e-10, max_iter = 1e4
  )
  right <- cox_net(lung_net_formula,
    data = lung, penalizer = 0.1, l1_ratio = 1, tol = 1e-10, max_iter = 1e4
  )
  expect_equal(coef(counting), coef(right), tolerance = 1e-3)
  expect_equal(counting$loglik, right$loglik, tolerance = 1e-5)
  x <- as.matrix(split[3:7])
  expect_lt(kkt_violation(counting, x, split$start, split$stop, split$event), 1e-5)

  # Delayed entry, checked against cox_ph() without a penalty.
  lung$entry <- ifelse(seq_len(n) %% 3 == 0, pmin(lung$time / 2, 30), 0)
  delayed <- Surv(entry, time, status) ~ age + sex + ph.ecog + ph.karno + wt.loss
  expect_equal(
    coef(cox_net(delayed, data = lung, penalizer = 0, tol = 1e-10, max_iter = 1e4)),
    coef(cox_ph(delayed, data = lung, ties = "breslow")),
    tolerance = 1e-3
  )

  # Whole-number weights act as copies of the rows.
  lung$k <- rep(1:3, length.out = n)
  weighted <- cox_net(lung_net_formula,
    data = lung, weights = k, penalizer = 0.05, l1_ratio = 0.5, tol = 1e-10, max_iter = 1e4
  )
  copies <- cox_net(lung_net_formula,
    data = lung[rep(seq_len(n), lung$k), ], penalizer = 0.05, l1_ratio = 0.5, tol = 1e-10,
    max_iter = 1e4
  )
  expect_equal(coef(weighted), coef(copies), tolerance = 1e-4)
  expect_equal(weighted$loglik, copies$loglik, tolerance = 1e-6)
  expect_equal(weighted$effective_df, copies$effective_df, tolerance = 1e-6)
  expect_equal(weighted$center, copies$center)
  expect_equal(weighted$scale, copies$scale)
  expect_equal(
    predict(weighted, newdata = lung[1:4, ], type = "survival", times = c(200, 400)),
    predict(copies, newdata = lung[1:4, ], type = "survival", times = c(200, 400)),
    tolerance = 1e-4
  )
  expect_true(weighted$has_weights)
  x <- as.matrix(lung[3:7])
  expect_lt(
    kkt_violation(weighted, x, rep(-Inf, n), lung$time, as.numeric(lung$status == 2), lung$k),
    1e-5
  )

  # Rows with a weight of 0 are left out.
  lung$k0 <- as.numeric(seq_len(n) > 10)
  expect_equal(
    coef(cox_net(lung_net_formula, data = lung, weights = k0, penalizer = 0.05)),
    coef(cox_net(lung_net_formula, data = lung[-(1:10), ], penalizer = 0.05))
  )
})

test_that("the penalty shrinks and selects", {
  lung <- lung_complete()
  nonzero <- vapply(c(0.01, 0.05, 0.1, 0.2, 0.5), function(lambda) {
    cox_net(lung_net_formula, data = lung, penalizer = lambda, l1_ratio = 1)$n_nonzero
  }, integer(1))
  expect_true(all(diff(nonzero) <= 0))
  expect_identical(nonzero[5], 0L)

  ridge <- cox_net(lung_net_formula, data = lung, penalizer = 0.5, l1_ratio = 0)
  expect_identical(ridge$n_nonzero, 5L)
  cox <- cox_ph(lung_net_formula, data = lung, ties = "breslow")
  expect_lt(sqrt(sum((coef(ridge) * ridge$scale)^2)), sqrt(sum((coef(cox) * ridge$scale)^2)))

  # With every coefficient at 0, every subject gets the baseline curve.
  empty <- cox_net(lung_net_formula, data = lung, penalizer = 0.5, l1_ratio = 1)
  expect_identical(empty$effective_df, 0L)
  expect_equal(predict(empty, type = "lp"), rep(0, nrow(lung)))
  curve <- predict(empty, newdata = lung[1:2, ], type = "survival", times = c(100, 300))
  na <- nelson_aalen(Surv(time, status) ~ 1, data = lung)
  expect_equal(curve$estimate[1:2], exp(-predict(na, times = c(100, 300))$estimate))

  # Collinear covariates are allowed, and a ridge penalty splits the effect between them.
  lung$age2 <- lung$age
  twins <- cox_net(Surv(time, status) ~ age + age2 + sex, data = lung, penalizer = 0.1,
    l1_ratio = 0
  )
  expect_equal(coef(twins)[["age"]], coef(twins)[["age2"]], tolerance = 1e-6)
})

test_that("predictions have the documented forms", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  lung$sex_f <- factor(lung$sex, 1:2, c("male", "female"))
  fit <- cox_net(Surv(time, status) ~ age + sex_f + ph.ecog, data = lung, penalizer = 0.02)
  expect_identical(names(coef(fit)), c("age", "sex_ffemale", "ph.ecog"))
  lp <- predict(fit, type = "lp")
  expect_equal(mean(lp), 0, tolerance = 1e-12)
  expect_equal(predict(fit, type = "risk"), exp(lp))
  newdata <- data.frame(age = c(50, 70), sex_f = c("female", "male"), ph.ecog = c(0, 2))
  expect_equal(
    predict(fit, newdata = newdata),
    unname(drop(cbind(newdata$age, newdata$sex_f == "female", newdata$ph.ecog) %*% coef(fit) -
      sum(fit$center * coef(fit))))
  )
  surv <- predict(fit, newdata = newdata, type = "survival", times = c(400, 100, 0))
  expect_named(surv, c("subject", "time", "estimate"))
  expect_identical(surv$subject, rep(1:2, each = 3))
  expect_identical(surv$time, rep(c(400, 100, 0), 2))
  expect_identical(surv$estimate[c(3, 6)], c(1, 1))
  expect_lt(surv$estimate[1], surv$estimate[2])
  default <- predict(fit, newdata = newdata[1, ], type = "survival")
  expect_identical(default$time, fit$times)
  expect_true(all(diff(default$estimate) <= 0))

  table <- as.data.frame(fit)
  expect_named(table, c("term", "estimate", "hazard_ratio"))
  expect_equal(table$hazard_ratio, exp(table$estimate))
  expect_identical(nobs(fit), fit$n_event)
  ll <- logLik(fit)
  expect_identical(attr(ll, "df"), fit$effective_df)
})

test_that("print shows the fit", {
  skip_if_not_installed("survival")
  fit <- cox_net(lung_net_formula, data = survival::lung, penalizer = 0.05, l1_ratio = 1)
  expect_output(print(fit), "<cox_net> Surv(time, status) ~ age + sex", fixed = TRUE)
  expect_output(print(fit), "Elastic net, penalizer = 0.05, l1_ratio = 1. n = 213, events = 151",
    fixed = TRUE
  )
  expect_output(print(fit), "Nonzero coefficients: 4 of 5. Effective df = 4", fixed = TRUE)
  expect_output(print(fit), "15 rows were dropped", fixed = TRUE)
})

test_that("bad input is rejected", {
  lung <- lung_complete()
  expect_error(cox_net(lung_net_formula, data = lung, penalizer = -1), "`penalizer`")
  expect_error(cox_net(lung_net_formula, data = lung, l1_ratio = 1.5), "`l1_ratio`")
  expect_error(cox_net(lung_net_formula, data = lung, standardize = NA), "`standardize`")
  expect_error(cox_net(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(cox_net(Surv(time, status) ~ age + cluster(sex), data = lung), "cluster")
  expect_error(cox_net(Surv(time, status) ~ 1, data = lung), "at least one covariate")
  expect_error(cox_net(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(
    cox_net(Surv(time, time + 1, type = "interval2") ~ age, data = lung),
    "right-censored or counting-process"
  )
  lung$none <- 0
  expect_error(cox_net(Surv(time, none) ~ age, data = lung), "at least one event")
  fit <- cox_net(lung_net_formula, data = lung)
  expect_error(predict(fit, newdata = as.matrix(lung)), "data frame")
  expect_error(predict(fit, newdata = lung[c("age", "sex")]), "ph.ecog")
})

test_that("cv_coxnet() scores each fold", {
  skip_if_not_installed("survival")
  lung <- survival::lung
  set.seed(5)
  cv <- cv_coxnet(lung_net_formula, data = lung, n_penalizers = 6, k = 4)
  expect_s3_class(cv, "cv_coxnet")
  expect_identical(cv$n_dropped, 15L)
  expect_identical(nrow(cv$path), 6L)
  expect_true(all(diff(cv$path$penalizer) < 0))
  expect_equal(cv$path$penalizer[1], 0.2178819590522021, tolerance = 1e-9)
  expect_equal(cv$path$penalizer[6], 0.00021788195905220195, tolerance = 1e-9)
  expect_identical(as.data.frame(cv), cv$path)
  expect_named(cv$scores, c("penalizer", "fold", "score", "n_nonzero"))

  # Rebuild one penalizer's scores from the recorded folds.
  used <- lung[stats::complete.cases(lung[all.vars(lung_net_formula)]), ]
  lambda <- cv$path$penalizer[3]
  rebuilt <- vapply(1:4, function(i) {
    fit <- cox_net(lung_net_formula, data = used[cv$folds != i, ], penalizer = lambda, l1_ratio = 1)
    test <- used[cv$folds == i, ]
    concordance_index(Surv(test$time, test$status), risk = predict(fit, newdata = test))
  }, numeric(1))
  expect_equal(cv$scores$score[cv$scores$penalizer == lambda], rebuilt)
  expect_equal(cv$path$mean_score[3], mean(rebuilt))
  expect_equal(cv$path$std_score[3], stats::sd(rebuilt))

  # The best penalizer and the one-standard-error rule.
  best <- which.max(cv$path$mean_score)
  expect_identical(cv$best_penalizer, cv$path$penalizer[best])
  within <- cv$path$mean_score >= cv$path$mean_score[best] - cv$path$std_score[best]
  expect_identical(cv$penalizer_1se, max(cv$path$penalizer[within]))
  expect_gte(cv$penalizer_1se, cv$best_penalizer)
  expect_true(all(cv$path$mean_score > 0 & cv$path$mean_score < 1))

  # The same seed gives the same folds.
  set.seed(5)
  expect_identical(cv_coxnet(lung_net_formula, data = lung, n_penalizers = 6, k = 4)$path, cv$path)

  # Given penalizers are sorted from the largest, and ridge gets a larger lambda_max.
  set.seed(5)
  given <- cv_coxnet(lung_net_formula, data = lung, penalizers = c(0.01, 0.1, 0.05), k = 4)
  expect_identical(given$path$penalizer, c(0.1, 0.05, 0.01))
  set.seed(5)
  ridge <- cv_coxnet(lung_net_formula, data = lung, l1_ratio = 0, n_penalizers = 3, k = 4)
  expect_equal(ridge$path$penalizer[1], 100 * cv$path$penalizer[1])
  expect_true(all(ridge$path$n_nonzero == 5))

  expect_output(print(cv), "<cv_coxnet> Surv(time, status) ~ age", fixed = TRUE)
  expect_output(print(cv), "4-fold cross-validation, l1_ratio = 1, concordance (higher is better)",
    fixed = TRUE
  )
  expect_output(print(cv), "1-SE penalizer:", fixed = TRUE)
})

test_that("cv_coxnet() scores the Brier score", {
  lung <- lung_complete()
  times <- c(100, 200, 300)
  set.seed(9)
  cv <- cv_coxnet(lung_net_formula,
    data = lung, metric = "brier", times = times, penalizers = c(0.2, 0.02), k = 3
  )
  rebuilt <- vapply(1:3, function(i) {
    fit <- cox_net(lung_net_formula, data = lung[cv$folds != i, ], penalizer = 0.02, l1_ratio = 1)
    test <- lung[cv$folds == i, ]
    p <- predict(fit, newdata = test, type = "survival", times = times)
    probs <- matrix(p$estimate, ncol = length(times), byrow = TRUE)
    integrated_brier_score(Surv(test$time, test$status), survival_prob = probs, times = times)
  }, numeric(1))
  expect_equal(cv$scores$score[cv$scores$penalizer == 0.02], rebuilt)
  expect_identical(cv$best_penalizer, cv$path$penalizer[which.min(cv$path$mean_score)])
  expect_output(print(cv), "brier (lower is better)", fixed = TRUE)
})

test_that("cv_coxnet() rejects bad input", {
  lung <- lung_complete()
  expect_error(cv_coxnet(lung_net_formula, data = lung, k = 1), "`k`")
  expect_error(cv_coxnet(lung_net_formula, data = lung, k = 2.5), "whole number")
  expect_error(cv_coxnet(lung_net_formula, data = lung, k = 1000), "at most the number of rows")
  expect_error(cv_coxnet(lung_net_formula, data = lung, metric = "brier"), "two `times`")
  expect_error(cv_coxnet(lung_net_formula, data = lung, metric = "auc"), "should be one of")
  expect_error(cv_coxnet(lung_net_formula, data = lung, penalizers = -1), "`penalizers`")
  expect_error(cv_coxnet(lung_net_formula, data = lung, eps = 2), "`eps`")
  expect_error(cv_coxnet(lung_net_formula, data = lung, l1_ratio = -0.1), "`l1_ratio`")
  lung$start <- 0
  expect_error(
    cv_coxnet(Surv(start, time, status) ~ age, data = lung),
    "needs right-censored data"
  )
  set.seed(1)
  expect_warning(
    cv_coxnet(Surv(time, status) ~ age + sex, data = lung[1:12, ], penalizers = 0.1, k = 6),
    "fewer than two per fold"
  )
})

test_that("cross_validate() scores cox_net() fits", {
  lung <- lung_complete()
  set.seed(3)
  cv <- cross_validate(cox_net, lung_net_formula,
    data = lung, penalizer = 0.05, l1_ratio = 1, k = 3, metrics = c("concordance", "brier"),
    times = c(100, 300)
  )
  rebuilt <- vapply(1:3, function(i) {
    fit <- cox_net(lung_net_formula, data = lung[cv$folds != i, ], penalizer = 0.05, l1_ratio = 1)
    test <- lung[cv$folds == i, ]
    concordance_index(Surv(test$time, test$status), risk = predict(fit, newdata = test))
  }, numeric(1))
  expect_equal(cv$scores$score[cv$scores$metric == "concordance"], rebuilt)
})
