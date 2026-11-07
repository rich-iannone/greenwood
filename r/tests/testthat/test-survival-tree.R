# The survival trees and forests have no R reference here. The trees that use no random numbers
# are checked against Greenwood's Python values, the splits against survival::survdiff()'s
# log-rank statistic, the leaves against kaplan_meier() and nelson_aalen() of their subjects, and
# the forests by rebuilding their bagging, out-of-bag scores, and permutation importance.

tree_formula <- Surv(time, status) ~ age + sex + ph.ecog + ph.karno + wt.loss

tree_lung <- function() {
  skip_if_not_installed("survival")
  vars <- c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")
  lung <- stats::na.omit(survival::lung[vars])
  rownames(lung) <- NULL
  lung
}

# The subjects (rows of the training data) in each leaf of a fitted survival_tree().
leaf_members <- function(fit) {
  split(seq_len(fit$n), fit$tree$node_of)
}

test_that("the trees match Python", {
  lung <- tree_lung()
  rows <- c(1, 2, 21, 51, 101, 151, 201)

  shallow <- survival_tree(tree_formula, data = lung, max_depth = 3)
  expect_equal(
    predict(shallow)[1:8],
    c(rep(95.15345287886001, 5), 168.543253968254, 168.543253968254, 95.15345287886001),
    tolerance = 1e-12
  )
  expect_equal(
    predict(shallow, newdata = lung[1:3, ], type = "survival", times = c(100, 365, 700))$estimate,
    rep(c(0.91489361702127669, 0.42573855204850464, 0.10169069414644284), 3),
    tolerance = 1e-12
  )

  full <- survival_tree(tree_formula, data = lung)
  expect_equal(
    predict(full)[rows],
    c(106.49999999999999, 20.5, 20.5, 43.33333333333333, 58, 54, 10),
    tolerance = 1e-12
  )

  sized <- survival_tree(tree_formula, data = lung, min_samples_leaf = 10, min_samples_split = 20)
  expect_equal(
    predict(sized)[rows[-2]],
    c(134.94605949605946, 45.57525968183863, 30.76071428571428, 84.7059523809524,
      59.253968253968246, 176.2705294705295),
    tolerance = 1e-12
  )
  expect_equal(
    predict(sized, newdata = lung[c(1, 51), ], type = "cumhaz", times = c(5, 200, 500))$estimate,
    c(0, 0.42259407259407256, 2.0015623265623264, 0.0625, 0.12916666666666665,
      0.32559523809523805),
    tolerance = 1e-12
  )
})

test_that("each split maximizes the log-rank statistic", {
  lung <- tree_lung()
  fit <- survival_tree(tree_formula, data = lung, max_depth = 2)
  nodes <- as.data.frame(fit)
  root <- nodes[1, ]
  chisq <- function(data, term, threshold) {
    survival::survdiff(survival::Surv(time, status) ~ I(data[[term]] <= threshold),
      data = data
    )$chisq
  }
  x <- as.matrix(lung[3:7])
  event <- lung$status == 2
  best <- logrank_best_split(x, lung$time, event, 1:5, 3)
  expect_identical(colnames(x)[best$feature], root$term)
  expect_identical(best$threshold, root$threshold)
  expect_equal(best$score, chisq(lung, root$term, root$threshold), tolerance = 1e-10)

  # Every cut-point of each covariate that leaves 3 subjects on each side scores lower.
  for (term in colnames(x)) {
    values <- sort(unique(lung[[term]]))
    cuts <- (values[-1] + values[-length(values)]) / 2
    sizes <- vapply(cuts, function(cut) sum(lung[[term]] <= cut), numeric(1))
    cuts <- cuts[sizes >= 3 & nrow(lung) - sizes >= 3]
    scores <- vapply(cuts, function(cut) chisq(lung, term, cut), numeric(1))
    one <- logrank_best_split(x, lung$time, event, match(term, colnames(x)), 3)
    expect_equal(one$score, max(scores), tolerance = 1e-10)
    expect_identical(one$threshold, cuts[which.max(scores)])
    expect_lte(max(scores), best$score + 1e-10)
  }

  # The second split on the left is the best one among the subjects there.
  left <- lung[lung[[root$term]] <= root$threshold, ]
  child <- nodes[root$left, ]
  expect_equal(
    logrank_best_split(as.matrix(left[3:7]), left$time, left$status == 2, 1:5, 3)$threshold,
    child$threshold
  )
  expect_identical(child$n, nrow(left))
  expect_identical(child$n_event, sum(left$status == 2))
})

test_that("the leaves hold the curves of their subjects", {
  lung <- tree_lung()
  fit <- survival_tree(tree_formula, data = lung, min_samples_leaf = 15)
  times <- fit$times
  expect_identical(times, sort(unique(lung$time[lung$status == 2])))
  members <- leaf_members(fit)
  expect_identical(length(members), fit$n_leaves)
  for (rows in members) {
    leaf <- lung[rows, ]
    km <- kaplan_meier(Surv(time, status) ~ 1, data = leaf)
    na <- nelson_aalen(Surv(time, status) ~ 1, data = leaf)
    query <- c(0, times, max(lung$time) + 1)
    subject <- lung[rows[1], ]
    expect_equal(
      predict(fit, newdata = subject, type = "survival", times = query)$estimate,
      predict(km, times = query)$estimate,
      tolerance = 1e-12
    )
    expect_equal(
      predict(fit, newdata = subject, type = "cumhaz", times = query)$estimate,
      predict(na, times = query)$estimate,
      tolerance = 1e-12
    )
    # The risk score is the cumulative hazard summed over the event times of the whole data.
    expect_equal(predict(fit)[rows], rep(sum(predict(na, times = times)$estimate), length(rows)))
  }
})

test_that("the stopping rules hold", {
  lung <- tree_lung()
  for (settings in list(list(NULL, 6, 3), list(4, 6, 3), list(NULL, 40, 12), list(2, 2, 1))) {
    fit <- survival_tree(tree_formula,
      data = lung, max_depth = settings[[1]], min_samples_split = settings[[2]],
      min_samples_leaf = settings[[3]]
    )
    nodes <- as.data.frame(fit)
    inner <- !is.na(nodes$term)
    leaves <- nodes[!inner, ]
    expect_identical(fit$n_leaves, nrow(leaves))
    expect_identical(fit$n_nodes, 2L * fit$n_leaves - 1L)
    expect_true(all(nodes$n >= settings[[3]]))
    expect_true(all(nodes$n[inner] >= settings[[2]]))
    expect_true(all(nodes$n_event[inner] > 0))
    if (!is.null(settings[[1]])) {
      expect_lte(max(nodes$depth), settings[[1]])
    }
    expect_identical(nodes$n[nodes$left[inner]] + nodes$n[nodes$right[inner]], nodes$n[inner])
    expect_identical(nodes$depth[nodes$left[inner]], nodes$depth[inner] + 1L)
    expect_identical(sum(leaves$n), nrow(lung))
    expect_identical(sum(leaves$n_event), sum(lung$status == 2))
  }

  # A node with no events isn't split, and without a covariate that separates its subjects
  # there's no split at all.
  lung$status[lung$ph.ecog == 0] <- 1
  fit <- survival_tree(Surv(time, status) ~ ph.ecog, data = lung)
  nodes <- as.data.frame(fit)
  expect_true(all(nodes$n_event[!is.na(nodes$term)] > 0))
  lung$same <- 1
  flat <- survival_tree(Surv(time, status) ~ same, data = lung)
  expect_identical(flat$n_nodes, 1L)
  km <- kaplan_meier(Surv(time, status) ~ 1, data = lung)
  expect_equal(
    predict(flat, newdata = lung[1, ], type = "survival")$estimate,
    predict(km, times = flat$times)$estimate
  )
})

test_that("random splits and feature subsets follow the seed", {
  lung <- tree_lung()
  set.seed(4)
  random <- survival_tree(tree_formula, data = lung, splitter = "random", max_depth = 4)
  set.seed(4)
  again <- survival_tree(tree_formula, data = lung, splitter = "random", max_depth = 4)
  expect_identical(as.data.frame(random), as.data.frame(again))
  set.seed(5)
  other <- survival_tree(tree_formula, data = lung, splitter = "random", max_depth = 4)
  expect_false(identical(as.data.frame(random), as.data.frame(other)))

  # Each random cut-point lies within the range of its covariate among the node's subjects.
  nodes <- as.data.frame(random)
  inner <- which(!is.na(nodes$term))
  members <- list(seq_len(nrow(lung)))
  for (id in seq_len(nrow(nodes))) {
    if (!is.na(nodes$term[id])) {
      rows <- members[[id]]
      values <- lung[[nodes$term[id]]][rows]
      expect_gt(nodes$threshold[id], min(values))
      expect_lt(nodes$threshold[id], max(values))
      members[[nodes$left[id]]] <- rows[values <= nodes$threshold[id]]
      members[[nodes$right[id]]] <- rows[values > nodes$threshold[id]]
    }
  }
  expect_true(length(inner) > 0)

  # With all the covariates tried, the best-split tree doesn't use the generator.
  set.seed(1)
  survival_tree(tree_formula, data = lung, max_depth = 3)
  expect_identical(stats::runif(1), {
    set.seed(1)
    stats::runif(1)
  })

  # A covariate subset changes the tree, and is drawn from the seed.
  set.seed(2)
  subset_tree <- survival_tree(tree_formula, data = lung, max_features = 2)
  expect_identical(subset_tree$n_features_split, 2L)
  set.seed(2)
  expect_identical(
    as.data.frame(survival_tree(tree_formula, data = lung, max_features = 2)),
    as.data.frame(subset_tree)
  )
})

test_that("max_features is read as documented", {
  expect_identical(resolve_max_features(NULL, 10), 10L)
  expect_identical(resolve_max_features("sqrt", 10), 3L)
  expect_identical(resolve_max_features("log2", 10), 3L)
  expect_identical(resolve_max_features("log2", 1), 1L)
  expect_identical(resolve_max_features(4, 10), 4L)
  expect_identical(resolve_max_features(40, 10), 10L)
  expect_identical(resolve_max_features(0.25, 10), 2L)
  expect_identical(resolve_max_features(0.01, 10), 1L)
  expect_error(resolve_max_features("all", 10), "`max_features` must be")
  expect_error(resolve_max_features(2.5, 10), "`max_features` must be")
  expect_error(resolve_max_features(0, 10), "`max_features` must be")
})

test_that("predictions have the documented forms", {
  lung <- tree_lung()
  lung$sex_f <- factor(lung$sex, 1:2, c("male", "female"))
  fit <- survival_tree(Surv(time, status) ~ age + sex_f + ph.ecog, data = lung, max_depth = 3)
  expect_identical(colnames(fit$x), c("age", "sex_ffemale", "ph.ecog"))
  risk <- predict(fit)
  expect_length(risk, nrow(lung))
  expect_identical(predict(fit, newdata = lung), risk)
  expect_identical(nobs(fit), nrow(lung))

  newdata <- data.frame(age = c(50, NA, 70), sex_f = c("female", "male", "male"),
    ph.ecog = c(0, 1, 2)
  )
  expect_identical(is.na(predict(fit, newdata = newdata)), c(FALSE, TRUE, FALSE))
  surv <- predict(fit, newdata = newdata, type = "survival", times = c(400, 100, 0))
  expect_named(surv, c("subject", "time", "estimate"))
  expect_identical(surv$subject, rep(1:3, each = 3))
  expect_identical(surv$time, rep(c(400, 100, 0), 3))
  expect_identical(surv$estimate[c(3, 9)], c(1, 1))
  expect_true(all(is.na(surv$estimate[4:6])))
  cumhaz <- predict(fit, newdata = newdata, type = "cumhaz", times = c(400, 100, 0))
  expect_identical(cumhaz$estimate[c(3, 9)], c(0, 0))
  default <- predict(fit, newdata = newdata[1, ], type = "survival")
  expect_identical(default$time, fit$times)
  expect_true(all(diff(default$estimate) <= 0))

  nodes <- as.data.frame(fit)
  expect_named(nodes, c("node", "depth", "term", "threshold", "left", "right", "n", "n_event"))
  expect_true(all(nodes$term[!is.na(nodes$term)] %in% colnames(fit$x)))
  expect_identical(nodes$node, seq_len(fit$n_nodes))

  # Subsets work as for the other models.
  expect_identical(
    predict(survival_tree(tree_formula, data = lung, subset = age > 60, max_depth = 2)),
    predict(survival_tree(tree_formula, data = lung[lung$age > 60, ], max_depth = 2))
  )
})

test_that("a forest of one tree is that tree", {
  lung <- tree_lung()
  tree <- survival_tree(tree_formula, data = lung, min_samples_leaf = 5)
  one <- random_survival_forest(tree_formula,
    data = lung, n_estimators = 1, min_samples_leaf = 5, max_features = NULL, bootstrap = FALSE
  )
  expect_equal(predict(one), predict(tree))
  newdata <- lung[c(3, 30, 90), ]
  expect_equal(
    predict(one, newdata = newdata, type = "survival", times = c(50, 300, 600)),
    predict(tree, newdata = newdata, type = "survival", times = c(50, 300, 600))
  )

  set.seed(8)
  extra <- extra_survival_trees(tree_formula, data = lung, n_estimators = 1, max_features = NULL)
  set.seed(8)
  random <- survival_tree(tree_formula, data = lung, splitter = "random", max_features = NULL)
  expect_equal(predict(extra), predict(random))

  # A bagged tree is the tree of its bootstrap sample, with its curves on the event times of the
  # whole data.
  set.seed(3)
  bagged <- random_survival_forest(tree_formula,
    data = lung, n_estimators = 1, max_features = NULL
  )
  set.seed(3)
  rows <- sample.int(nrow(lung), nrow(lung), replace = TRUE)
  expect_identical(bagged$trees[[1]]$rows, rows)
  expect_identical(bagged$oob[, 1], !seq_len(nrow(lung)) %in% rows)
  sample_tree <- survival_tree(tree_formula, data = lung[rows, ])
  expect_identical(bagged$trees[[1]]$feature, sample_tree$tree$feature)
  expect_identical(bagged$trees[[1]]$threshold, sample_tree$tree$threshold)
  members <- split(rows, bagged$trees[[1]]$node_of)
  leaf <- lung[members[[1]], ]
  subject <- lung[members[[1]][1], ]
  km <- kaplan_meier(Surv(time, status) ~ 1, data = leaf)
  expect_equal(
    predict(bagged, newdata = subject, type = "survival")$estimate,
    predict(km, times = bagged$times)$estimate
  )
})

test_that("forests average their trees and follow the seed", {
  lung <- tree_lung()
  set.seed(11)
  forest <- random_survival_forest(tree_formula, data = lung, n_estimators = 20)
  expect_s3_class(forest, c("random_survival_forest", "survival_forest"))
  expect_identical(forest$n_features_split, 2L)
  expect_identical(length(forest$trees), 20L)
  set.seed(11)
  expect_identical(
    predict(random_survival_forest(tree_formula, data = lung, n_estimators = 20)),
    predict(forest)
  )

  # The risk score is the average of the trees' summed cumulative hazards, and the curves are the
  # averages of the trees' curves.
  newdata <- lung[c(5, 50, 150), ]
  x <- as.matrix(newdata[3:7])
  by_tree <- vapply(forest$trees, function(tree) {
    tree$mortality[match(tree_route(tree, x), tree$leaves)]
  }, numeric(3))
  expect_equal(predict(forest, newdata = newdata), rowMeans(by_tree))
  curves <- vapply(forest$trees, function(tree) {
    rows <- tree$rows
    leaf <- tree_leaf_curves(tree, lung$time[rows], lung$status[rows] == 2, forest$times)
    leaf$survival[match(tree_route(tree, x), leaf$leaves), findInterval(c(100, 300), forest$times)]
  }, matrix(0, 3, 2))
  expect_equal(
    predict(forest, newdata = newdata, type = "survival", times = c(100, 300))$estimate,
    as.vector(t(apply(curves, c(1, 2), mean)))
  )

  set.seed(12)
  extra <- extra_survival_trees(tree_formula, data = lung, n_estimators = 10)
  expect_s3_class(extra, c("extra_survival_trees", "survival_forest"))
  expect_false(extra$bootstrap)
  expect_null(extra$trees[[1]]$rows)
  expect_true(is.na(extra$oob_score))
  expect_identical(nobs(extra), nrow(lung))
})

test_that("the out-of-bag concordance and variable importance can be rebuilt", {
  lung <- tree_lung()
  set.seed(21)
  forest <- random_survival_forest(tree_formula,
    data = lung, n_estimators = 30, oob_score = TRUE
  )
  x <- as.matrix(lung[3:7])
  # The trees are added one at a time, as in the forest: rowMeans() adds in extended precision on
  # some platforms, and the different rounding can break a near-tie between two subjects' risks.
  oob_risk <- function(design) {
    total <- numeric(nrow(lung))
    count <- numeric(nrow(lung))
    for (b in seq_along(forest$trees)) {
      tree <- forest$trees[[b]]
      out <- tree$mortality[match(tree_route(tree, design), tree$leaves)]
      total <- total + ifelse(forest$oob[, b], out, 0)
      count <- count + forest$oob[, b]
    }
    ifelse(count > 0, total / count, NA)
  }
  risk <- oob_risk(x)
  scored <- !is.na(risk)
  expect_identical(scored, rowSums(forest$oob) > 0)
  expect_equal(
    forest$oob_score,
    concordance_index(Surv(lung$time[scored], lung$status[scored]), risk = risk[scored])
  )
  expect_gt(forest$oob_score, 0.5)

  set.seed(22)
  importance <- variable_importance(forest, n_repeats = 2)
  expect_named(importance, c("term", "importance"))
  expect_setequal(importance$term, colnames(x))
  expect_true(all(diff(importance$importance) <= 0))
  base <- 1 - forest$oob_score
  set.seed(22)
  rebuilt <- vapply(seq_len(ncol(x)), function(j) {
    mean(vapply(1:2, function(r) {
      shuffled <- x
      shuffled[, j] <- x[sample.int(nrow(x)), j]
      risk <- oob_risk(shuffled)
      1 - concordance_index(Surv(lung$time[scored], lung$status[scored]), risk = risk[scored])
    }, numeric(1))) - base
  }, numeric(1))
  expect_equal(importance$importance, rebuilt[match(importance$term, colnames(x))])

  # Extra trees on bootstrap samples are scored the same way.
  set.seed(23)
  extra <- extra_survival_trees(tree_formula,
    data = lung, n_estimators = 30, bootstrap = TRUE, oob_score = TRUE
  )
  expect_true(extra$oob_score > 0 && extra$oob_score < 1)
  expect_error(
    variable_importance(extra_survival_trees(tree_formula, data = lung, n_estimators = 2)),
    "bootstrap = TRUE"
  )
})

test_that("print shows the fits", {
  skip_if_not_installed("survival")
  fit <- survival_tree(tree_formula, data = survival::lung, max_depth = 3)
  expect_output(print(fit), "<survival_tree> Surv(time, status) ~ age + sex", fixed = TRUE)
  expect_output(print(fit),
    "Log-rank splits, 15 nodes, 8 leaves. n = 213, events = 151, features = 5",
    fixed = TRUE
  )
  expect_output(print(fit), "15 rows were dropped", fixed = TRUE)
  set.seed(1)
  forest <- random_survival_forest(tree_formula,
    data = survival::lung, n_estimators = 5, oob_score = TRUE
  )
  expect_output(print(forest), "<random_survival_forest> Surv(time, status)", fixed = TRUE)
  expect_output(print(forest), paste(
    "5 log-rank trees, 2 of 5 features tried per split, bootstrap samples.",
    "n = 213, events = 151"
  ), fixed = TRUE)
  expect_output(print(forest), "Out-of-bag concordance = ", fixed = TRUE)
  extra <- extra_survival_trees(tree_formula, data = survival::lung, n_estimators = 5)
  expect_output(print(extra), "5 log-rank trees, 2 of 5 features tried per split. n = 213",
    fixed = TRUE
  )
})

test_that("cross_validate() scores the trees and forests", {
  lung <- tree_lung()
  set.seed(31)
  cv <- cross_validate(random_survival_forest, tree_formula,
    data = lung, n_estimators = 10, k = 3, metrics = c("concordance", "brier"),
    times = c(100, 300)
  )
  set.seed(31)
  folds <- cv_folds(lung$status, 3L, TRUE)
  expect_identical(cv$folds, folds)
  rebuilt <- vapply(1:3, function(i) {
    fit <- random_survival_forest(tree_formula, data = lung[folds != i, ], n_estimators = 10)
    test <- lung[folds == i, ]
    concordance_index(Surv(test$time, test$status), risk = predict(fit, newdata = test))
  }, numeric(1))
  # The folds' fits draw from the generator after the folds, in order.
  expect_equal(cv$scores$score[cv$scores$metric == "concordance"], rebuilt)

  tree_cv <- cross_validate(survival_tree, tree_formula, data = lung, max_depth = 2, k = 3)
  expect_true(all(tree_cv$scores$score > 0 & tree_cv$scores$score < 1))
})

test_that("bad input is rejected", {
  lung <- tree_lung()
  expect_error(survival_tree(tree_formula, data = lung, max_depth = 0), "`max_depth`")
  expect_error(survival_tree(tree_formula, data = lung, min_samples_split = 1),
    "`min_samples_split`"
  )
  expect_error(survival_tree(tree_formula, data = lung, min_samples_leaf = 0.5),
    "`min_samples_leaf`"
  )
  expect_error(survival_tree(tree_formula, data = lung, max_features = "all"), "`max_features`")
  expect_error(survival_tree(tree_formula, data = lung, splitter = "worst"), "should be one of")
  expect_error(random_survival_forest(tree_formula, data = lung, n_estimators = 0),
    "`n_estimators`"
  )
  expect_error(random_survival_forest(tree_formula, data = lung, bootstrap = NA), "`bootstrap`")
  expect_error(
    random_survival_forest(tree_formula, data = lung, bootstrap = FALSE, oob_score = TRUE),
    "needs `bootstrap = TRUE`"
  )
  expect_error(survival_tree(Surv(time, status) ~ age + strata(sex), data = lung), "strata")
  expect_error(survival_tree(Surv(time, status) ~ age + offset(sex), data = lung), "offset")
  expect_error(survival_tree(Surv(time, status) ~ 1, data = lung), "at least one covariate")
  lung$start <- 0
  expect_error(
    random_survival_forest(Surv(start, time, status) ~ age, data = lung),
    "needs right-censored data"
  )
  lung$none <- 0
  expect_error(extra_survival_trees(Surv(time, none) ~ age, data = lung), "at least one event")
  fit <- survival_tree(tree_formula, data = lung, max_depth = 2)
  expect_error(predict(fit, newdata = as.matrix(lung)), "data frame")
  expect_error(predict(fit, newdata = lung[c("age", "sex")]), "ph.ecog")
  expect_error(predict(fit, type = "lp"), "should be one of")
  expect_error(predict(fit, type = "survival", times = "a"), "`times`")
})
