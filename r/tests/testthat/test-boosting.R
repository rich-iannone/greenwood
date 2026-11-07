# The boosting models have no R reference here. The Cox gradient boosting fits that use no random
# numbers are checked against Greenwood's Python values, its gradients against
# survival::coxph()'s martingale residuals, and its regression trees against a brute-force search
# for the best split. survival_boost() is checked by rebuilding its sampled targets, inverse
# probability of censoring weights, and trees by hand.

boost_formula <- Surv(time, status) ~ age + sex + ph.ecog + ph.karno + wt.loss

boost_lung <- function() {
  skip_if_not_installed("survival")
  vars <- c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")
  lung <- stats::na.omit(survival::lung[vars])
  rownames(lung) <- NULL
  lung
}

boost_x <- function(lung) {
  as.matrix(lung[c("age", "sex", "ph.ecog", "ph.karno", "wt.loss")])
}

# The rows of x in each leaf of a regression tree.
boost_leaves <- function(tree, x) {
  leaf <- tree_route(tree, x)
  split(seq_len(nrow(x)), leaf)
}

test_that("the Cox gradient boosting matches Python", {
  lung <- boost_lung()
  rows <- c(1, 2, 21, 51, 101, 151, 201)

  fit <- gradient_boosting_survival_analysis(boost_formula, data = lung)
  expect_equal(
    predict(fit, type = "lp")[rows],
    c(
      -0.11036316363292484, -1.0630700636617283, -0.5334959172452711, 0.39892855441530484,
      0.08556761324037404, -0.8515615812020737, -0.16649161696546
    ),
    tolerance = 1e-9
  )
  expect_equal(
    predict(fit, newdata = lung[c(1, 51), ], type = "survival", times = c(5, 200, 500))$estimate,
    c(
      0.99729923015667632, 0.76692170153977524, 0.26697128579300872, 0.99550964799359742,
      0.64300268707850228, 0.11106384727680381
    ),
    tolerance = 1e-9
  )
  # ph.ecog and ph.karno tie on some splits, and Python credits the tie to either column. Their
  # total matches.
  expect_equal(
    unname(fit$importance[c("age", "sex", "wt.loss")]),
    c(0.32250456598643246, 0.10031720083059013, 0.33467634115763323),
    tolerance = 1e-9
  )
  expect_equal(
    sum(fit$importance[c("ph.ecog", "ph.karno")]),
    0.18578558594992675 + 0.05671630607541738,
    tolerance = 1e-9
  )

  stumps <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 40, learning_rate = 0.2, max_depth = 1, min_samples_leaf = 10
  )
  expect_equal(
    predict(stumps)[rows],
    c(
      rep(0.8986702078373769, 3), 0.5710867465005234, 1.10795435601889, 0.8127400058694841,
      1.1649152616818987
    ),
    tolerance = 1e-10
  )
  expect_equal(
    unname(stumps$importance),
    c(
      0.11975250422457716, 0.28370086503129, 0.43266889724207397, 0.02700766603868529,
      0.13687006746337368
    ),
    tolerance = 1e-10
  )
  expect_equal(
    stumps$cumhaz[c(1, 11, 51, length(stumps$cumhaz))],
    c(0.00437693565650994, 0.06410839780137784, 0.3651623033814289, 2.979485721884812),
    tolerance = 1e-10
  )
})

test_that("the gradient is the martingale residual and the baseline is Breslow's", {
  lung <- boost_lung()
  set.seed(5)
  s <- stats::rnorm(nrow(lung), sd = 0.5)
  event <- lung$status == 2
  gradient <- boost_cox_gradient(s, lung$time, event)

  cox <- survival::coxph(survival::Surv(time, status) ~ offset(s),
    data = cbind(lung, s = s), ties = "breslow"
  )
  expect_equal(gradient$residuals, unname(stats::residuals(cox, type = "martingale")),
    tolerance = 1e-10
  )
  times <- sort(unique(lung$time[event]))
  breslow <- cumsum(vapply(times, function(t) {
    sum(event & lung$time == t) / sum(exp(s)[lung$time >= t])
  }, numeric(1)))
  expect_equal(gradient$times, times)
  expect_equal(gradient$cumhaz, breslow, tolerance = 1e-12)

  # The fit keeps the baseline at its final scores, which sum the trees' steps.
  fit <- gradient_boosting_survival_analysis(boost_formula, data = lung, n_estimators = 15)
  x <- boost_x(lung)
  scores <- Reduce(`+`, lapply(fit$trees, function(tree) 0.1 * regression_tree_predict(tree, x)))
  expect_equal(fit$scores, scores, tolerance = 1e-12)
  expect_equal(predict(fit, newdata = lung, type = "lp"), fit$scores, tolerance = 1e-12)
  final <- boost_cox_gradient(fit$scores, lung$time, event)
  expect_identical(fit$times, final$times)
  expect_identical(fit$cumhaz, final$cumhaz)
})

test_that("each regression tree fits its gradient by least squares", {
  lung <- boost_lung()
  x <- boost_x(lung)
  residuals <- boost_cox_gradient(numeric(nrow(lung)), lung$time, lung$status == 2)$residuals

  # A brute-force search over every feature and cut-point for the first split.
  sse <- function(y) sum((y - mean(y))^2)
  candidates <- do.call(rbind, lapply(seq_len(ncol(x)), function(f) {
    values <- sort(unique(x[, f]))
    cuts <- (values[-1L] + values[-length(values)]) / 2
    do.call(rbind, lapply(cuts, function(cut) {
      left <- x[, f] <= cut
      if (sum(left) < 5 || sum(!left) < 5) {
        return(NULL)
      }
      gain <- sse(residuals) - sse(residuals[left]) - sse(residuals[!left])
      data.frame(feature = f, threshold = cut, gain = gain)
    }))
  }))
  best <- candidates[which.max(candidates$gain), ]

  stump <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 1, learning_rate = 1, max_depth = 1, min_samples_leaf = 5
  )
  tree <- stump$trees[[1]]
  expect_identical(tree$feature[1], best$feature)
  expect_equal(tree$threshold[1], best$threshold)
  expect_equal(sum(tree$importance), best$gain, tolerance = 1e-10)
  left <- x[, best$feature] <= best$threshold
  expected <- ifelse(left, mean(residuals[left]), mean(residuals[!left]))
  expect_equal(predict(stump, type = "lp"), expected, tolerance = 1e-12)
  expect_equal(unname(stump$importance), as.numeric(seq_len(ncol(x)) == best$feature))

  # In a deeper tree, every leaf holds min_samples_leaf rows and predicts the mean of its rows.
  deep <- grow_regression_tree(x, residuals, max_depth = 4, min_leaf = 7, mtry = ncol(x))
  leaves <- boost_leaves(deep, x)
  expect_true(all(lengths(leaves) >= 7))
  for (leaf in names(leaves)) {
    expect_equal(deep$value[as.integer(leaf)], mean(residuals[leaves[[leaf]]]), tolerance = 1e-12)
  }
  depth <- integer(length(deep$feature))
  for (id in which(deep$feature > 0L)) {
    depth[c(deep$left[id], deep$right[id])] <- depth[id] + 1L
  }
  expect_lte(max(depth), 4L)
  # The importance is the total reduction in squared error, the root's to its leaves'.
  leaf_sse <- sum(vapply(leaves, function(rows) sse(residuals[rows]), numeric(1)))
  expect_equal(sum(deep$importance), sse(residuals) - leaf_sse, tolerance = 1e-10)

  # sse_best_split() keeps the first feature on ties.
  twins <- cbind(x[, "ph.ecog"], x[, "ph.ecog"])
  expect_identical(sse_best_split(twins, residuals, c(2L, 1L), 3)$feature, 2L)
})

test_that("nodes stop splitting by depth, size, and a constant gradient", {
  lung <- boost_lung()
  x <- boost_x(lung)
  # No node holds 2 * min_samples_leaf rows, and the martingale residuals sum to 0.
  flat <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 3, min_samples_leaf = 110
  )
  expect_true(all(vapply(flat$trees, function(tree) length(tree$feature), integer(1)) == 1L))
  expect_equal(predict(flat, type = "lp"), numeric(nrow(lung)), tolerance = 1e-12)
  expect_identical(unname(flat$importance), numeric(5))
  expect_identical(variable_importance(flat)$importance, numeric(5))

  stumps <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 5, max_depth = 1
  )
  expect_true(all(vapply(stumps$trees, function(tree) length(tree$feature), integer(1)) == 3L))

  constant <- grow_regression_tree(x, rep(0.3, nrow(x)), max_depth = 3, min_leaf = 1, mtry = 5L)
  expect_identical(constant$feature, 0L)
  expect_identical(constant$value, 0.3)
})

test_that("subsampling and feature sampling draw from the generator", {
  lung <- boost_lung()
  x <- boost_x(lung)
  n <- nrow(lung)
  set.seed(11)
  fit <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 1, subsample = 0.5, max_features = 2
  )
  expect_identical(fit$n_features_split, 2L)
  residuals <- boost_cox_gradient(numeric(n), lung$time, lung$status == 2)$residuals
  set.seed(11)
  rows <- sample.int(n, round(0.5 * n))
  tree <- grow_regression_tree(x[rows, ], residuals[rows], max_depth = 3, min_leaf = 3, mtry = 2L)
  expect_identical(fit$trees[[1]], tree)

  set.seed(12)
  a <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 10, subsample = 0.7, max_features = "sqrt"
  )
  set.seed(12)
  b <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 10, subsample = 0.7, max_features = "sqrt"
  )
  expect_identical(a$scores, b$scores)
  expect_identical(a$n_features_split, 2L)
  set.seed(13)
  c <- gradient_boosting_survival_analysis(boost_formula,
    data = lung, n_estimators = 10, subsample = 0.7, max_features = "sqrt"
  )
  expect_false(identical(a$scores, c$scores))

  # The full fit draws nothing.
  set.seed(14)
  gradient_boosting_survival_analysis(boost_formula, data = lung, n_estimators = 3)
  after <- stats::runif(1)
  set.seed(14)
  expect_identical(stats::runif(1), after)
})

test_that("predict() gives scores and curves in every form", {
  lung <- boost_lung()
  fit <- gradient_boosting_survival_analysis(boost_formula, data = lung, n_estimators = 20)
  lp <- predict(fit, type = "lp")
  expect_identical(predict(fit), exp(lp))
  expect_equal(predict(fit, newdata = lung[1:4, ], type = "lp"), lp[1:4], tolerance = 1e-12)

  times <- c(1, 100, 365, 2000)
  cumhaz <- predict(fit, newdata = lung[1:3, ], type = "cumhaz", times = times)
  expect_named(cumhaz, c("subject", "time", "estimate"))
  expect_identical(cumhaz$subject, rep(1:3, each = 4))
  expect_identical(cumhaz$time, rep(times, 3))
  baseline <- c(0, fit$cumhaz)[findInterval(times, fit$times) + 1L]
  expect_equal(cumhaz$estimate, as.vector(outer(baseline, exp(lp[1:3]))), tolerance = 1e-12)
  survival <- predict(fit, newdata = lung[1:3, ], type = "survival", times = times)
  expect_equal(survival$estimate, exp(-cumhaz$estimate), tolerance = 1e-12)
  # Before the first event the survival is 1, and past the last it stays at the last step.
  expect_identical(survival$estimate[survival$time == 1], rep(1, 3))
  expect_equal(cumhaz$estimate[cumhaz$time == 2000], exp(lp[1:3]) * max(fit$cumhaz),
    tolerance = 1e-12
  )

  default <- predict(fit, type = "survival")
  expect_identical(nrow(default), nrow(lung) * length(fit$times))
  expect_identical(unique(default$time), fit$times)

  # Rows with a missing covariate get missing predictions.
  newdata <- lung[1:3, ]
  newdata$age[2] <- NA
  expect_identical(is.na(predict(fit, newdata = newdata)), c(FALSE, TRUE, FALSE))
  curves <- predict(fit, newdata = newdata, type = "survival", times = times)
  expect_identical(is.na(curves$estimate), rep(c(FALSE, TRUE, FALSE), each = 4))

  # Factor covariates are coded as in the fit.
  lung$sex <- factor(lung$sex, levels = 1:2, labels = c("male", "female"))
  factored <- gradient_boosting_survival_analysis(boost_formula, data = lung, n_estimators = 20)
  expect_named(factored$importance, c("age", "sexfemale", "ph.ecog", "ph.karno", "wt.loss"))
  expect_equal(predict(factored), predict(fit), tolerance = 1e-12)
  expect_equal(predict(factored, newdata = lung[5:6, ]), predict(fit)[5:6], tolerance = 1e-12)
})

test_that("variable_importance(), nobs(), and print() describe the fit", {
  lung <- boost_lung()
  fit <- gradient_boosting_survival_analysis(boost_formula, data = lung, n_estimators = 20)
  importance <- variable_importance(fit)
  expect_named(importance, c("term", "importance"))
  expect_equal(sum(importance$importance), 1)
  expect_false(is.unsorted(rev(importance$importance)))
  expect_identical(importance$importance, unname(sort(fit$importance, decreasing = TRUE)))
  expect_identical(nobs(fit), nrow(lung))

  full <- gradient_boosting_survival_analysis(boost_formula, data = survival::lung,
    n_estimators = 20, learning_rate = 0.05
  )
  expect_output(print(full), "<gradient_boosting_survival_analysis> Surv(time, status) ~ age",
    fixed = TRUE
  )
  expect_output(print(full), paste(
    "20 regression trees of depth at most 3, learning rate 0.05, Cox partial likelihood.",
    "n = 213, events = 151, features = 5",
    sep = "\n"
  ), fixed = TRUE)
  expect_output(print(full), "15 rows were dropped", fixed = TRUE)
})

test_that("cross_validate() scores the Cox gradient boosting", {
  lung <- boost_lung()
  set.seed(21)
  cv <- cross_validate(gradient_boosting_survival_analysis, boost_formula,
    data = lung, n_estimators = 15, k = 3
  )
  set.seed(21)
  folds <- cv_folds(lung$status, 3L, TRUE)
  rebuilt <- vapply(1:3, function(i) {
    fit <- gradient_boosting_survival_analysis(boost_formula,
      data = lung[folds != i, ], n_estimators = 15
    )
    test <- lung[folds == i, ]
    concordance_index(Surv(test$time, test$status), risk = predict(fit, newdata = test))
  }, numeric(1))
  expect_equal(cv$scores$score, rebuilt)
})

test_that("the Cox gradient boosting rejects bad input", {
  lung <- boost_lung()
  gbm <- function(...) gradient_boosting_survival_analysis(boost_formula, data = lung, ...)
  expect_error(gbm(n_estimators = 0), "`n_estimators`")
  expect_error(gbm(learning_rate = 0), "`learning_rate` must be greater than 0")
  expect_error(gbm(learning_rate = -1), "`learning_rate`")
  expect_error(gbm(max_depth = 0), "`max_depth`")
  expect_error(gbm(min_samples_leaf = 0), "`min_samples_leaf`")
  expect_error(gbm(subsample = 0), "`subsample` must be a number greater than 0 and at most 1")
  expect_error(gbm(subsample = 1.5), "`subsample`")
  expect_error(gbm(max_features = 0), "`max_features`")
  expect_error(
    gradient_boosting_survival_analysis(Surv(time, status) ~ age + strata(sex), data = lung),
    "strata"
  )
  expect_error(gradient_boosting_survival_analysis(Surv(time, status) ~ 1, data = lung),
    "at least one covariate"
  )
  fit <- gbm(n_estimators = 2)
  expect_error(predict(fit, newdata = as.matrix(lung)), "data frame")
  expect_error(predict(fit, type = "cumulative_hazard"), "should be one of")
  expect_error(predict(fit, type = "survival", times = "a"), "`times`")
})

# -- survival_boost() ------------------------------------------------------------------------

boost_mgus <- function() {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  mg <- mg[stats::complete.cases(mg[c("etime", "event_f", "age", "sex", "hgb")]), ]
  mg <- mg[seq(1, nrow(mg), by = 5), ]
  rownames(mg) <- NULL
  mg
}

# The Kaplan-Meier curve of the censoring times, with the events at a time before the censorings.
censoring_curve <- function(time, censored) {
  at <- sort(unique(time[censored]))
  g <- cumprod(vapply(at, function(c) {
    1 - sum(censored & time == c) / (sum(time > c) + sum(censored & time == c))
  }, numeric(1)))
  function(t) {
    i <- findInterval(t, at)
    out <- c(1, g)[i + 1L]
    out[t <= 0] <- 1
    out
  }
}

# The boosting rounds of survival_boost(), rebuilt from the same draws.
rebuild_survival_boost <- function(mg, seed, n_rounds, learning_rate, max_depth, min_leaf,
                                   hard_zero_fraction = 0.1, horizons = 3, eps = 0.05) {
  set.seed(seed)
  x <- cbind(age = mg$age, sexM = as.numeric(mg$sex == "M"), hgb = mg$hgb)
  n <- nrow(mg)
  time <- mg$etime
  status <- mg$event
  g <- censoring_curve(time, status == 0)
  g_floor <- function(t) pmax(g(t), eps)
  rounds <- list()
  for (b in seq_len(n_rounds)) {
    tau <- numeric(0)
    for (h in seq_len(horizons)) {
      draw <- stats::runif(n, 0, max(time))
      draw[sample.int(n, floor(hard_zero_fraction * n))] <- 0
      tau <- c(tau, draw)
    }
    subject <- rep(seq_len(n), horizons)
    t_i <- time[subject]
    target <- ifelse(t_i > tau, 0, status[subject])
    weight <- numeric(length(tau))
    weight[t_i > tau] <- 1 / g_floor(tau[t_i > tau])
    event <- t_i <= tau & status[subject] > 0
    weight[event] <- 1 / g_floor(t_i[event])
    keep <- weight > 0
    xa <- cbind(tau[keep], x[subject[keep], ])
    scores <- matrix(0, nrow(xa), 3)
    for (trees in rounds) {
      for (k in 1:3) {
        scores[, k] <- scores[, k] + learning_rate * trees[[k]]$value[tree_route(trees[[k]], xa)]
      }
    }
    probs <- exp(scores) / rowSums(exp(scores))
    rounds[[b]] <- lapply(1:3, function(k) {
      y <- weight[keep] * ((target[keep] == k - 1) - probs[, k])
      grow_regression_tree(xa, y, max_depth, min_leaf, mtry = 4L)
    })
  }
  rounds
}

test_that("survival_boost() boosts the weighted class targets at the sampled horizons", {
  mg <- boost_mgus()
  set.seed(7)
  fit <- survival_boost(Surv(etime, event_f) ~ age + sex + hgb,
    data = mg, n_estimators = 3, learning_rate = 0.3, max_depth = 2
  )
  rounds <- rebuild_survival_boost(mg,
    seed = 7, n_rounds = 3, learning_rate = 0.3, max_depth = 2, min_leaf = 10
  )
  expect_equal(fit$rounds, rounds, tolerance = 1e-12)

  set.seed(8)
  none <- survival_boost(Surv(etime, event_f) ~ age + sex + hgb,
    data = mg, n_estimators = 2, hard_zero_fraction = 0, epsilon_censoring = 0.3
  )
  rounds <- rebuild_survival_boost(mg,
    seed = 8, n_rounds = 2, learning_rate = 0.05, max_depth = 3, min_leaf = 10,
    hard_zero_fraction = 0, eps = 0.3
  )
  expect_equal(none$rounds, rounds, tolerance = 1e-12)

  # censoring_steps() is the censoring curve.
  steps <- censoring_steps(mg$etime, mg$event == 0)
  expect_equal(steps$g, censoring_curve(mg$etime, mg$event == 0)(steps$time), tolerance = 1e-12)
})

test_that("survival_boost() predictions are distributions over the classes", {
  mg <- boost_mgus()
  set.seed(9)
  fit <- survival_boost(Surv(etime, event_f) ~ age + sex + hgb,
    data = mg, n_estimators = 5, learning_rate = 0.3
  )
  times <- c(0, 24, 120, 500)
  cif <- predict(fit, newdata = mg[1:3, ], times = times)
  expect_named(cif, c("subject", "cause", "time", "estimate"))
  expect_identical(cif$subject, rep(1:3, each = 8))
  expect_identical(cif$cause, rep(rep(c("pcm", "death"), each = 4), 3))
  expect_identical(cif$time, rep(times, 6))
  survival <- predict(fit, newdata = mg[1:3, ], type = "survival", times = times)
  expect_named(survival, c("subject", "time", "estimate"))
  expect_identical(survival$subject, rep(1:3, each = 4))
  total <- survival$estimate + cif$estimate[cif$cause == "pcm"] +
    cif$estimate[cif$cause == "death"]
  expect_equal(total, rep(1, 12), tolerance = 1e-12)
  expect_true(all(cif$estimate > 0 & cif$estimate < 1))

  # The class probabilities are the softmax of the summed tree scores at (time, covariates).
  xa <- cbind(times, matrix(c(mg$age[2], mg$sex[2] == "M", mg$hgb[2]), 4, 3, byrow = TRUE))
  scores <- Reduce(`+`, lapply(fit$rounds, function(trees) {
    0.3 * vapply(trees, function(tree) tree$value[tree_route(tree, xa)], numeric(4))
  }))
  probs <- exp(scores) / rowSums(exp(scores))
  expect_equal(survival$estimate[5:8], probs[, 1], tolerance = 1e-12)
  expect_equal(cif$estimate[9:16], as.vector(probs[, 2:3]), tolerance = 1e-12)

  default <- predict(fit)
  expect_identical(nrow(default), nrow(mg) * 2L * length(fit$times))
  expect_equal(
    predict(fit, newdata = mg[1:3, ], type = "survival")$estimate,
    predict(fit, type = "survival")$estimate[seq_len(3 * length(fit$times))],
    tolerance = 1e-12
  )

  newdata <- mg[1:3, ]
  newdata$hgb[3] <- NA
  missing <- predict(fit, newdata = newdata, times = times)
  expect_identical(is.na(missing$estimate), rep(c(FALSE, TRUE), c(16, 8)))
})

test_that("survival_boost() picks up a covariate that moves one cause", {
  set.seed(41)
  n <- 400
  x <- stats::rnorm(n)
  first <- stats::rexp(n, 0.02 * exp(1.5 * x))
  second <- stats::rexp(n, 0.02)
  censor <- stats::rexp(n, 0.005)
  time <- pmin(first, second, censor)
  status <- ifelse(censor == time, 0, ifelse(first == time, 1, 2))
  data <- data.frame(time = time, x = x, noise = stats::rnorm(n),
    event = factor(status, 0:2, c("censor", "a", "b"))
  )
  fit <- survival_boost(Surv(time, event) ~ x + noise,
    data = data, n_estimators = 30, learning_rate = 0.3
  )
  cif <- predict(fit, newdata = data.frame(x = c(-1, 1), noise = 0), times = 40)
  a <- cif$estimate[cif$cause == "a"]
  b <- cif$estimate[cif$cause == "b"]
  expect_gt(a[2], a[1] + 0.2)
  expect_lt(b[2], b[1])
})

test_that("survival_boost() builds its time grid from the event times", {
  mg <- boost_mgus()
  event_times <- sort(mg$etime[mg$event > 0])
  set.seed(1)
  fine <- survival_boost(Surv(etime, event_f) ~ age, data = mg, n_estimators = 1,
    n_time_grid_steps = 1000
  )
  expect_identical(fine$times, unique(event_times))
  set.seed(1)
  coarse <- survival_boost(Surv(etime, event_f) ~ age, data = mg, n_estimators = 1,
    n_time_grid_steps = 10
  )
  expect_identical(coarse$times,
    unique(unname(stats::quantile(event_times, seq(0, 1, length.out = 10))))
  )
  expect_identical(range(coarse$times), range(event_times))
})

test_that("survival_boost() is reproducible with set.seed()", {
  mg <- boost_mgus()
  boost <- function(seed) {
    set.seed(seed)
    survival_boost(Surv(etime, event_f) ~ age + sex + hgb,
      data = mg, n_estimators = 3, subsample = 0.6, max_features = 2
    )
  }
  a <- boost(3)
  expect_identical(a$n_features_split, 2L)
  expect_identical(a$rounds, boost(3)$rounds)
  expect_false(identical(a$rounds, boost(4)$rounds))
})

test_that("survival_boost() prints and rejects bad input", {
  mg <- boost_mgus()
  set.seed(2)
  fit <- survival_boost(Surv(etime, event_f) ~ age + sex + hgb, data = mg, n_estimators = 2)
  expect_identical(nobs(fit), nrow(mg))
  expect_output(print(fit), "<survival_boost> Surv(etime, event_f) ~ age + sex + hgb",
    fixed = TRUE
  )
  expect_output(print(fit), paste0(
    "2 boosting rounds of 3 regression trees (one per class) of depth at most 3, ",
    "learning rate 0.05.\nCauses: pcm, death. n = ", nrow(mg), ", events = ", sum(mg$event > 0),
    ", features = 3"
  ), fixed = TRUE)

  sb <- function(...) survival_boost(Surv(etime, event_f) ~ age, data = mg, ...)
  expect_error(sb(n_time_grid_steps = 0), "`n_time_grid_steps`")
  expect_error(sb(hard_zero_fraction = 1), "`hard_zero_fraction` must be a number at least 0")
  expect_error(sb(hard_zero_fraction = -0.1), "`hard_zero_fraction`")
  expect_error(sb(n_horizons_per_observation = 0), "`n_horizons_per_observation`")
  expect_error(sb(epsilon_censoring = 0), "`epsilon_censoring` must be greater than 0")
  expect_error(sb(learning_rate = 0), "`learning_rate`")
  expect_error(sb(subsample = 2), "`subsample`")
  expect_error(survival_boost(Surv(etime, event > 0) ~ age, data = mg),
    "competing-risks response"
  )
  expect_error(survival_boost(Surv(etime, event_f) ~ 1, data = mg), "at least one covariate")
  expect_error(survival_boost(Surv(etime, event_f) ~ age + offset(hgb), data = mg), "offset")
  mg$event_f[] <- "censor"
  expect_error(survival_boost(Surv(etime, event_f) ~ age, data = mg), "at least one event")
  expect_error(predict(fit, type = "cumhaz"), "should be one of")
  expect_error(predict(fit, newdata = as.matrix(mg)), "data frame")
})
