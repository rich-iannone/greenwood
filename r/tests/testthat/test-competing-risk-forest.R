# The competing-risks forest has no R reference here. The trees that use no random numbers are
# checked against Greenwood's Python values, the splits against the composite log-rank statistic
# computed directly from its definition, the leaves against aalen_johansen() of their subjects,
# and the forests by rebuilding their bagging, out-of-bag scores, and permutation importance.

crf_formula <- Surv(etime, event_f) ~ age + male + hgb + creat

# Every `by`-th complete row of mgus2, with numeric covariates so the design matches the data.
crf_mgus <- function(by = 4) {
  skip_if_not_installed("survival")
  mg <- mgus2_competing()
  mg <- mg[stats::complete.cases(mg[c("etime", "event_f", "age", "sex", "hgb", "creat")]), ]
  mg <- mg[seq(1, nrow(mg), by = by), ]
  rownames(mg) <- NULL
  mg$male <- as.numeric(mg$sex == "M")
  mg[c("etime", "event", "event_f", "age", "male", "hgb", "creat")]
}

# A forest with only its `b`-th tree.
one_tree <- function(fit, b) {
  fit$trees <- fit$trees[b]
  fit
}

# The composite log-rank statistic for splitting off the rows in `left`, from its definition.
cr_statistic <- function(time, event, left) {
  num <- 0
  var <- 0
  for (cause in sort(unique(event[event > 0]))) {
    for (t in sort(unique(time[event > 0]))) {
      d <- sum(time == t & event == cause)
      at_risk <- time >= t | (time < t & event > 0 & event != cause)
      y <- sum(time >= t)
      y_all <- sum(at_risk)
      frac <- sum(at_risk & left) / y_all
      num <- num - frac * d
      if (y >= 2) {
        var <- var + d * frac * (1 - frac) * (y_all - d) / (y_all - 1)
      }
    }
    num <- num + sum(left & event == cause)
  }
  if (var > 0) num^2 / var else 0
}

test_that("the trees match Python", {
  mg <- crf_mgus()
  rows <- c(1, 2, 11, 51, 101, 201)
  times <- c(12, 60, 120, 240)
  grow <- function(...) {
    competing_risk_forest(crf_formula,
      data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL, ...
    )
  }

  shallow <- grow(max_depth = 3)
  expect_equal(
    predict(shallow, type = "risk", cause = "pcm")[rows],
    c(rep(7.383215962441318, 3), 7.533333333333333, 6.790040908358913, 6.790040908358913),
    tolerance = 1e-12
  )
  expect_equal(
    predict(shallow, newdata = mg[c(1, 51), ], cause = "pcm", times = times)$estimate,
    c(0.01408450704225352, 0.05633802816901411, 0.07194835680751177, 0.08755868544600945, 0,
      rep(0.06666666666666667, 3)),
    tolerance = 1e-12
  )
  expect_equal(
    predict(shallow, type = "risk", cause = "death")[rows],
    c(rep(83.74330985915492, 3), 65.16666666666669, 28.351904885844743, 28.351904885844743),
    tolerance = 1e-12
  )
  expect_equal(
    predict(shallow, newdata = mg[c(1, 51), ], cause = "death", times = times)$estimate,
    c(0.32394366197183094, 0.6338028169014083, 0.8656103286384974, 0.9124413145539904, 0.2,
      0.26666666666666666, 0.8333333333333337, 0.9333333333333338),
    tolerance = 1e-12
  )

  sized <- grow(min_samples_leaf = 10)
  expect_equal(
    predict(sized, type = "risk", cause = "pcm")[rows],
    c(0, 7.833333333333336, 0, 7.533333333333333, 13.059999999999999, 9.7),
    tolerance = 1e-12
  )
  expect_equal(
    predict(sized, newdata = mg[c(1, 51), ], cause = "pcm", times = times)$estimate,
    c(rep(0, 5), rep(0.06666666666666667, 3)),
    tolerance = 1e-12
  )
  expect_equal(
    predict(sized, type = "risk", cause = "death")[rows],
    c(91.4, 89.49999999999997, 98.60000000000001, 65.16666666666669, 27.659999999999997,
      31.319999999999993),
    tolerance = 1e-12
  )
  expect_equal(
    predict(sized, newdata = mg[c(1, 51), ], cause = "death", times = times)$estimate,
    c(0.1, 0.8999999999999999, 0.9999999999999999, 0.9999999999999999, 0.2, 0.26666666666666666,
      0.8333333333333337, 0.9333333333333338),
    tolerance = 1e-12
  )
})

test_that("each split maximizes the composite log-rank statistic", {
  mg <- crf_mgus(by = 12)
  x <- as.matrix(mg[c("age", "male", "hgb", "creat")])
  min_leaf <- 5
  split <- logrank_cr_best_split(x, mg$etime, mg$event, seq_len(ncol(x)), min_leaf)

  # Every cut-point between two distinct values with `min_leaf` rows on each side.
  best <- -Inf
  for (f in seq_len(ncol(x))) {
    values <- sort(unique(x[, f]))
    for (cut in (values[-1] + values[-length(values)]) / 2) {
      left <- x[, f] <= cut
      if (sum(left) >= min_leaf && sum(!left) >= min_leaf) {
        score <- cr_statistic(mg$etime, mg$event, left)
        if (score > best + 1e-9) {
          best <- score
          found <- list(feature = f, threshold = cut)
        }
      }
    }
  }
  expect_equal(split$score, best, tolerance = 1e-10)
  expect_identical(split$feature, found$feature)
  expect_equal(split$threshold, found$threshold)
  expect_equal(
    cr_statistic(mg$etime, mg$event, x[, split$feature] <= split$threshold), split$score,
    tolerance = 1e-10
  )

  # The statistic of the tree's root is the same.
  fit <- competing_risk_forest(crf_formula,
    data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL, max_depth = 1,
    min_samples_leaf = min_leaf
  )
  expect_identical(fit$trees[[1]]$feature[1], found$feature)
  expect_equal(fit$trees[[1]]$threshold[1], found$threshold)

  # No split when no cut-point leaves enough rows on each side.
  expect_null(logrank_cr_best_split(x, mg$etime, mg$event, 2L, nrow(x)))
})

test_that("the leaves hold the Aalen-Johansen curves of their subjects", {
  mg <- crf_mgus(by = 6)
  fit <- competing_risk_forest(crf_formula,
    data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL, max_depth = 2
  )
  tree <- fit$trees[[1]]
  members <- split(seq_len(fit$n), tree$node_of)
  expect_gt(length(members), 1)
  times <- c(10, 50, 100, 200, 300)
  for (rows in members) {
    aj <- predict(aalen_johansen(Surv(etime, event_f) ~ 1, data = mg[rows, ]), times = times)
    for (cause in fit$states) {
      expect_equal(
        predict(fit, newdata = mg[rows[1], ], cause = cause, times = times)$estimate,
        aj$estimate[aj$cause == cause],
        tolerance = 1e-12
      )
    }
  }

  # A leaf's risk score is its cumulative incidence summed over the event times of the fit.
  curves <- predict(fit, cause = "death")
  expect_equal(
    predict(fit, type = "risk", cause = "death"),
    as.vector(tapply(curves$estimate, curves$subject, sum)),
    tolerance = 1e-12
  )
})

test_that("the stopping rules hold", {
  mg <- crf_mgus(by = 6)
  grow <- function(...) {
    competing_risk_forest(crf_formula,
      data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL, ...
    )$trees[[1]]
  }

  stump <- grow(max_depth = 1)
  expect_identical(sum(stump$feature == 0L), 2L)

  sized <- grow(min_samples_leaf = 15)
  expect_gte(min(tabulate(sized$node_of, length(sized$feature))[sized$feature == 0L]), 15)

  # A node with fewer than `min_samples_split` subjects is a leaf.
  root <- grow(min_samples_split = nrow(mg) + 1)
  expect_identical(length(root$feature), 1L)

  # With one leaf, the forest gives the curves of the whole sample.
  fit <- competing_risk_forest(crf_formula,
    data = mg, n_estimators = 1, bootstrap = FALSE, min_samples_leaf = nrow(mg)
  )
  aj <- predict(aalen_johansen(Surv(etime, event_f) ~ 1, data = mg), times = c(50, 150))
  expect_equal(
    predict(fit, newdata = mg[1, ], cause = 2, times = c(50, 150))$estimate,
    aj$estimate[aj$cause == "death"],
    tolerance = 1e-12
  )

  # A node with no events is not split, and its curves stay at 0.
  x <- as.matrix(mg[c("age", "male", "hgb", "creat")])
  settings <- tree_settings(NULL, 6, 3, NULL, ncol(x), "competing")
  quiet <- grow_logrank_tree(x, mg$etime, integer(nrow(mg)), settings)
  expect_identical(quiet$feature, 0L)
  expect_identical(sum(quiet$node_of == 1L), nrow(mg))
  curves <- cr_leaf_cif(quiet, mg$etime, integer(nrow(mg)), c(10, 100), 2L)
  expect_identical(curves$cif, rep(list(matrix(0, 1, 2)), 2))
})

test_that("the forest bags its trees and averages them", {
  mg <- crf_mgus(by = 6)
  n <- nrow(mg)

  # Trying every covariate draws no random numbers, so a one-tree fit without bagging repeats.
  set.seed(3)
  plain <- competing_risk_forest(crf_formula,
    data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL
  )
  after <- stats::runif(1)
  set.seed(3)
  expect_identical(stats::runif(1), after)
  again <- competing_risk_forest(crf_formula,
    data = mg, n_estimators = 1, bootstrap = FALSE, max_features = NULL
  )
  expect_identical(predict(again, type = "risk"), predict(plain, type = "risk"))

  # A bagged tree is the tree grown on its bootstrap sample.
  set.seed(11)
  bagged <- competing_risk_forest(crf_formula, data = mg, n_estimators = 1, max_features = NULL)
  set.seed(11)
  rows <- sample.int(n, n, replace = TRUE)
  single <- competing_risk_forest(crf_formula,
    data = mg[rows, ], n_estimators = 1, bootstrap = FALSE, max_features = NULL
  )
  expect_identical(bagged$trees[[1]]$feature, single$trees[[1]]$feature)
  expect_equal(bagged$trees[[1]]$threshold, single$trees[[1]]$threshold)
  expect_identical(bagged$trees[[1]]$rows, rows)
  expect_identical(bagged$oob[, 1], !seq_len(n) %in% rows)
  for (cause in bagged$states) {
    expect_equal(
      predict(bagged, newdata = mg, cause = cause, times = single$times),
      predict(single, newdata = mg, cause = cause),
      tolerance = 1e-12
    )
  }

  # The forest's predictions are the averages of its trees'.
  set.seed(5)
  fit <- competing_risk_forest(crf_formula, data = mg, n_estimators = 4, max_features = 2)
  expect_length(fit$trees, 4)
  times <- c(20, 100, 250)
  for (cause in fit$states) {
    risk <- lapply(seq_len(4), function(b) predict(one_tree(fit, b), type = "risk", cause = cause))
    expect_equal(predict(fit, type = "risk", cause = cause), Reduce(`+`, risk) / 4)
    cif <- lapply(seq_len(4), function(b) {
      predict(one_tree(fit, b), cause = cause, times = times)$estimate
    })
    expect_equal(predict(fit, cause = cause, times = times)$estimate, Reduce(`+`, cif) / 4)
  }

  # The cumulative incidences of the causes add to at most 1.
  total <- predict(fit, cause = 1)$estimate + predict(fit, cause = 2)$estimate
  expect_true(all(total <= 1 + 1e-12))

  # The same seed gives the same forest.
  set.seed(5)
  same <- competing_risk_forest(crf_formula, data = mg, n_estimators = 4, max_features = 2)
  expect_identical(predict(same, type = "risk"), predict(fit, type = "risk"))
})

test_that("the out-of-bag concordance and importance are rebuilt from the trees", {
  mg <- crf_mgus(by = 8)
  n <- nrow(mg)
  set.seed(21)
  fit <- competing_risk_forest(crf_formula, data = mg, n_estimators = 6, oob_score = TRUE)

  # The out-of-bag risk of one cause, and one minus its concordance. The trees are added one at a
  # time, as in the forest: rowSums() adds in extended precision on some platforms, and the
  # different rounding can break a near-tie between two subjects' risks.
  oob_error <- function(data, cause) {
    total <- numeric(n)
    count <- numeric(n)
    for (b in seq_len(6)) {
      risk <- predict(one_tree(fit, b), newdata = data, type = "risk", cause = cause)
      total <- total + ifelse(fit$oob[, b], risk, 0)
      count <- count + fit$oob[, b]
    }
    scored <- count > 0
    oob_risk <- total[scored] / count[scored]
    response <- Surv(mg$etime[scored], mg$event[scored] == cause)
    1 - concordance_index(response, risk = oob_risk)
  }
  expect_equal(fit$oob_score, 1 - oob_error(mg, 1), tolerance = 1e-12)
  expect_output(print(fit), "Out-of-bag concordance for pcm")

  # The importance of a covariate: the mean rise in the out-of-bag error over shuffles of it.
  hand_importance <- function(cause, n_repeats) {
    base <- oob_error(mg, cause)
    vapply(c("age", "male", "hgb", "creat"), function(v) {
      mean(vapply(seq_len(n_repeats), function(r) {
        shuffled <- mg
        shuffled[[v]] <- mg[[v]][sample.int(n)]
        oob_error(shuffled, cause) - base
      }, numeric(1)))
    }, numeric(1))
  }
  for (cause in 1:2) {
    set.seed(8)
    importance <- variable_importance(fit, cause = cause, n_repeats = 2)
    set.seed(8)
    expected <- hand_importance(cause, 2)
    o <- order(expected, decreasing = TRUE)
    expect_identical(importance$term, names(expected)[o])
    expect_equal(importance$importance, unname(expected[o]), tolerance = 1e-12)
  }
  set.seed(8)
  by_name <- variable_importance(fit, cause = "death", n_repeats = 2)
  set.seed(8)
  expect_identical(by_name, variable_importance(fit, cause = 2, n_repeats = 2))

  plain <- competing_risk_forest(crf_formula, data = mg, n_estimators = 2)
  expect_true(is.na(plain$oob_score))
  expect_no_match(capture.output(print(plain)), "Out-of-bag")
})

test_that("predict() handles its forms", {
  mg <- crf_mgus(by = 8)
  set.seed(2)
  fit <- competing_risk_forest(crf_formula, data = mg, n_estimators = 3)

  default <- predict(fit)
  expect_named(default, c("subject", "time", "estimate"))
  expect_identical(nrow(default), nrow(mg) * length(fit$times))
  expect_identical(unique(default$time), fit$times)
  expect_identical(default, predict(fit, newdata = mg, cause = "pcm", times = fit$times))

  # Before the first event time the incidence is 0, and after the last it stays put.
  early <- predict(fit, newdata = mg[1:3, ], cause = "death", times = c(0, max(fit$times) + 10))
  expect_identical(early$estimate[early$time == 0], c(0, 0, 0))
  last <- predict(fit, newdata = mg[1:3, ], cause = "death", times = max(fit$times))
  expect_identical(early$estimate[early$time > max(fit$times)], last$estimate)

  # Rows with a missing covariate get NA.
  gappy <- mg[1:4, ]
  gappy$hgb[2] <- NA
  risk <- predict(fit, newdata = gappy, type = "risk", cause = "death")
  expect_identical(is.na(risk), c(FALSE, TRUE, FALSE, FALSE))
  expect_equal(risk[-2], predict(fit, newdata = mg[c(1, 3, 4), ], type = "risk", cause = "death"))
  curves <- predict(fit, newdata = gappy, times = c(50, 100))
  expect_identical(is.na(curves$estimate), rep(c(FALSE, TRUE, FALSE, FALSE), each = 2))

  expect_error(predict(fit, cause = "relapse"), "`cause` must be one of the states")
  expect_error(predict(fit, cause = 3), "`cause` must be one of the states")
  expect_error(predict(fit, type = "survival"))
  expect_error(predict(fit, newdata = mg["age"]))
})

test_that("the fit records its inputs", {
  mg <- crf_mgus(by = 8)
  mg$age[1] <- NA
  set.seed(4)
  fit <- competing_risk_forest(Surv(etime, event_f) ~ age + factor(male) + hgb,
    data = mg, n_estimators = 2
  )
  expect_s3_class(fit, "competing_risk_forest")
  expect_identical(nobs(fit), nrow(mg) - 1L)
  expect_identical(fit$states, c("pcm", "death"))
  expect_identical(fit$n_event, sum(mg$event[-1] > 0))
  expect_identical(fit$times, sort(unique(mg$etime[-1][mg$event[-1] > 0])))
  expect_identical(fit$n_features_split, 1L)

  out <- capture.output(print(fit))
  expect_match(out[1], "<competing_risk_forest> Surv(etime, event_f) ~ age + factor(male) + hgb",
    fixed = TRUE
  )
  expect_match(out[2], "2 competing-risks log-rank trees, 1 of 3 features tried per split, boot",
    fixed = TRUE
  )
  expect_match(out[3], "Causes: pcm, death.", fixed = TRUE)
  expect_true(any(grepl("1 row was dropped for missing values.", out, fixed = TRUE)))
})

test_that("bad input is caught", {
  mg <- crf_mgus(by = 8)
  expect_error(
    competing_risk_forest(Surv(etime, event > 0) ~ age, data = mg),
    "competing-risks"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age + strata(male), data = mg),
    "strata"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age + offset(hgb), data = mg),
    "offset"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ 1, data = mg),
    "needs at least one covariate"
  )
  censored <- mg
  censored$event_f[] <- "censor"
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age, data = censored),
    "needs at least one event"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age, data = mg, bootstrap = FALSE,
      oob_score = TRUE
    ),
    "bootstrap"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age, data = mg, n_estimators = 0),
    "n_estimators"
  )
  expect_error(
    competing_risk_forest(Surv(etime, event_f) ~ age, data = mg, max_features = 0),
    "max_features"
  )
  plain <- competing_risk_forest(Surv(etime, event_f) ~ age,
    data = mg, n_estimators = 1, bootstrap = FALSE
  )
  expect_error(variable_importance(plain), "bootstrap = TRUE")
  set.seed(1)
  bagged <- competing_risk_forest(Surv(etime, event_f) ~ age, data = mg, n_estimators = 1)
  expect_error(variable_importance(bagged, cause = "relapse"), "`cause` must be one of")
  expect_error(variable_importance(bagged, n_repeats = 0), "n_repeats")
})
