# Log-rank survival trees and forests: survival_tree(), random_survival_forest(),
# extra_survival_trees(), variable_importance(), and their methods. Tested by
# tests/testthat/test-survival-tree.R, against Greenwood's Python values for the trees that use no
# random numbers, against log-rank statistics and leaf curves built by hand, and for the forests'
# bagging, out-of-bag, and permutation logic.

#' A log-rank survival tree
#'
#' `survival_tree()` splits the subjects into groups with similar survival. Starting from all of
#' them, it looks for the covariate and the cut-point that best separate the survival of the two
#' groups they make, by the two-sample log-rank statistic, and splits there. It then splits each
#' group in the same way, until a group is too small to split, has no events, or reaches
#' `max_depth`. A cut-point is the midpoint between two neighboring values of the covariate, and
#' each side of a split keeps at least `min_samples_leaf` subjects.
#'
#' Each final group (a leaf) holds the Nelson-Aalen cumulative hazard and the Kaplan-Meier survival
#' curve of its subjects, at the event times of the whole data. A subject's prediction is the curve
#' of the leaf their covariates lead to. The risk score is the leaf's cumulative hazard summed over
#' those event times (the ensemble mortality of Ishwaran et al., 2008), larger for an earlier
#' event. No proportional hazards or other form for the effects of the covariates is assumed.
#'
#' With `splitter = "random"`, each covariate gets a single cut-point drawn at random between its
#' smallest and largest values in the group, and the best of those is used. With `max_features`
#' smaller than the number of covariates, each split looks at a random subset of them. Both use
#' R's random number generator, so call [set.seed()] first for results that can be repeated. A
#' single tree is mostly useful to read. For prediction, the forests average many of them (see
#' [random_survival_forest()]).
#'
#' @param formula A formula with a right-censored [Surv()] response on the left and the covariates
#'   on the right, such as `Surv(time, status) ~ age + sex`. Factors are coded with treatment
#'   contrasts, and each column is split on separately.
#' @param data A data frame holding the variables in `formula`.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param max_depth The largest number of splits from the first group to a leaf, or `NULL` (the
#'   default) for no limit.
#' @param min_samples_split The smallest group that can be split, at least 2. The default is `6`.
#' @param min_samples_leaf The fewest subjects on each side of a split, at least 1. The default is
#'   `3`.
#' @param max_features The number of covariates tried at each split: `NULL` (all of them, the
#'   default), `"sqrt"` or `"log2"` (the square root or base-2 logarithm of the number of columns,
#'   rounded down), a whole number, or a fraction between 0 and 1 of the columns. At least one is
#'   always tried.
#' @param splitter `"best"` (the default) to try every cut-point, or `"random"` to try one random
#'   cut-point per covariate.
#'
#' @return A `survival_tree` object. Use [predict()] for risk scores and survival or cumulative
#'   hazard curves, and [as.data.frame()] for the nodes of the tree. The object also holds `times`
#'   (the event times the curves are given at), `n`, `n_event`, `n_nodes`, and `n_leaves`.
#'
#' @references Ishwaran, H., Kogalur, U. B., Blackstone, E. H., and Lauer, M. S. (2008). Random
#'   survival forests. *The Annals of Applied Statistics*, 2(3), 841-860.
#'
#' @seealso [random_survival_forest()] and [extra_survival_trees()] for forests of these trees.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#'     2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57, 67,
#'     49, 50, 58, 72, 70, 60, 70, 53, 74),
#'   ph.ecog = c(1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 1, 1, 1, 1,
#'     2, 1, 1)
#' )
#'
#' tree <- survival_tree(Surv(time, status) ~ age + ph.ecog, data = lung, max_depth = 2)
#' tree
#'
#' # The splits and the leaves
#' as.data.frame(tree)
#'
#' # Survival curves for two new subjects
#' predict(tree, newdata = data.frame(age = c(55, 72), ph.ecog = c(0, 2)), type = "survival",
#'   times = c(100, 300, 500)
#' )
#' @export
survival_tree <- function(formula,
                          data,
                          subset,
                          na.action = stats::na.omit,
                          max_depth = NULL,
                          min_samples_split = 6,
                          min_samples_leaf = 3,
                          max_features = NULL,
                          splitter = c("best", "random")) {
  splitter <- match.arg(splitter)
  check_tree_args(max_depth, min_samples_split, min_samples_leaf)
  inputs <- tree_inputs(
    match.call(expand.dots = FALSE), parent.frame(), formula,
    if (missing(data)) NULL else data, "survival_tree()"
  )
  settings <- tree_settings(
    max_depth, min_samples_split, min_samples_leaf, max_features, ncol(inputs$x), splitter
  )
  grid <- sort(unique(inputs$time[inputs$event]))
  tree <- grow_logrank_tree(inputs$x, inputs$time, inputs$event, settings)
  tree <- tree_add_leaves(tree, inputs$time, inputs$event, grid)

  structure(
    c(
      list(
        tree = tree,
        times = grid,
        n_nodes = tree$n_nodes,
        n_leaves = length(tree$leaves),
        max_features = max_features,
        splitter = splitter
      ),
      tree_fit_fields(inputs, settings, formula, match.call())
    ),
    class = "survival_tree"
  )
}

#' Random survival forests and extra survival trees
#'
#' `random_survival_forest()` grows `n_estimators` log-rank survival trees (see [survival_tree()]),
#' each on a bootstrap sample of the subjects (drawn with replacement), with a random subset of
#' `max_features` covariates tried at each split. A prediction is the average of the trees' curves,
#' and the risk score is the average cumulative hazard summed over the event times (the ensemble
#' mortality), larger for an earlier event. Averaging many trees that differ by chance gives much
#' steadier predictions than any one of them, while still finding nonlinear effects and
#' interactions without assuming proportional hazards (Ishwaran et al., 2008).
#'
#' `extra_survival_trees()` grows extremely randomized trees: each covariate tried at a split gets
#' a single cut-point drawn at random between its smallest and largest values, and the best of
#' those is used. Following the extra-trees convention, each tree is grown on all of the subjects
#' (`bootstrap = FALSE`) by default.
#'
#' With bootstrap samples, each subject is left out of about a third of the trees. The out-of-bag
#' concordance (with `oob_score = TRUE`) scores each subject with only those trees, which estimates
#' how well the forest ranks new subjects without setting data aside, and [variable_importance()]
#' measures how much worse that score gets when a covariate's values are shuffled. The forests use
#' R's random number generator, so call [set.seed()] first for results that can be repeated.
#'
#' @inheritParams survival_tree
#' @param n_estimators The number of trees, at least 1. The default is `100`.
#' @param max_features The number of covariates tried at each split: `"sqrt"` (the square root of
#'   the number of columns, rounded down, the default), `"log2"`, `NULL` (all of them), a whole
#'   number, or a fraction between 0 and 1 of the columns.
#' @param bootstrap Whether to grow each tree on a bootstrap sample. The default is `TRUE` for
#'   `random_survival_forest()` and `FALSE` for `extra_survival_trees()`. The out-of-bag
#'   concordance and [variable_importance()] need it.
#' @param oob_score Whether to compute the out-of-bag concordance. The default is `FALSE`.
#'
#' @return A `random_survival_forest` or `extra_survival_trees` object (both are
#'   `survival_forest` objects). Use [predict()] for risk scores and survival or cumulative hazard
#'   curves, and [variable_importance()] for the permutation importance of each covariate. The
#'   object also holds `times` (the event times the curves are given at), `oob_score` (the
#'   out-of-bag concordance, or `NA`), `n`, and `n_event`.
#'
#' @references Ishwaran, H., Kogalur, U. B., Blackstone, E. H., and Lauer, M. S. (2008). Random
#'   survival forests. *The Annals of Applied Statistics*, 2(3), 841-860.
#'
#'   Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. *Machine
#'   Learning*, 63(1), 3-42.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#'     2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57, 67,
#'     49, 50, 58, 72, 70, 60, 70, 53, 74),
#'   ph.ecog = c(1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 1, 1, 1, 1,
#'     2, 1, 1),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1,
#'     1, 1)
#' )
#'
#' set.seed(1)
#' forest <- random_survival_forest(Surv(time, status) ~ age + ph.ecog + sex,
#'   data = lung, n_estimators = 50, oob_score = TRUE
#' )
#' forest
#'
#' # The risk scores and survival curves of two new subjects
#' newdata <- data.frame(age = c(55, 72), ph.ecog = c(0, 2), sex = c(2, 1))
#' predict(forest, newdata = newdata)
#' predict(forest, newdata = newdata, type = "survival", times = c(100, 300))
#'
#' # Extra trees, grown on bootstrap samples so they can be scored out of bag
#' extra <- extra_survival_trees(Surv(time, status) ~ age + ph.ecog + sex,
#'   data = lung, n_estimators = 50, bootstrap = TRUE, oob_score = TRUE
#' )
#' variable_importance(extra, n_repeats = 3)
#' @export
random_survival_forest <- function(formula,
                                   data,
                                   subset,
                                   na.action = stats::na.omit,
                                   n_estimators = 100,
                                   max_depth = NULL,
                                   min_samples_split = 6,
                                   min_samples_leaf = 3,
                                   max_features = "sqrt",
                                   bootstrap = TRUE,
                                   oob_score = FALSE) {
  inputs <- forest_inputs(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data,
    "random_survival_forest()", n_estimators, max_depth, min_samples_split, min_samples_leaf,
    bootstrap, oob_score
  )
  grow_forest(
    inputs, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
    bootstrap, oob_score, "best", "random_survival_forest", formula, match.call()
  )
}

#' @rdname random_survival_forest
#' @export
extra_survival_trees <- function(formula,
                                 data,
                                 subset,
                                 na.action = stats::na.omit,
                                 n_estimators = 100,
                                 max_depth = NULL,
                                 min_samples_split = 6,
                                 min_samples_leaf = 3,
                                 max_features = "sqrt",
                                 bootstrap = FALSE,
                                 oob_score = FALSE) {
  inputs <- forest_inputs(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data,
    "extra_survival_trees()", n_estimators, max_depth, min_samples_split, min_samples_leaf,
    bootstrap, oob_score
  )
  grow_forest(
    inputs, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
    bootstrap, oob_score, "random", "extra_survival_trees", formula, match.call()
  )
}

# -- fitting ---------------------------------------------------------------------------------

check_tree_args <- function(max_depth, min_samples_split, min_samples_leaf) {
  if (!is.null(max_depth)) {
    check_count(max_depth, "max_depth", min = 1)
  }
  check_count(min_samples_split, "min_samples_split", min = 2)
  check_count(min_samples_leaf, "min_samples_leaf", min = 1)
}

# The number of covariates tried at each split, from 1 to p.
resolve_max_features <- function(max_features, p) {
  value <- if (is.null(max_features)) {
    p
  } else if (identical(max_features, "sqrt")) {
    floor(sqrt(p))
  } else if (identical(max_features, "log2")) {
    floor(log2(p))
  } else if (is.numeric(max_features) && length(max_features) == 1L && !is.na(max_features) &&
    max_features > 0 && (max_features < 1 || max_features == round(max_features))) {
    if (max_features < 1) floor(max_features * p) else max_features
  } else {
    stop(
      paste(
        "`max_features` must be \"sqrt\", \"log2\", a whole number, a fraction between 0 and 1,",
        "or `NULL`."
      ),
      call. = FALSE
    )
  }
  as.integer(min(max(value, 1), p))
}

tree_settings <- function(max_depth, min_samples_split, min_samples_leaf, max_features, p,
                          splitter) {
  list(
    max_depth = if (is.null(max_depth)) Inf else max_depth,
    min_split = min_samples_split,
    min_leaf = min_samples_leaf,
    mtry = resolve_max_features(max_features, p),
    splitter = splitter
  )
}

# The covariates and the right-censored response of a tree model's formula.
tree_inputs <- function(call, env, formula, data, fn) {
  frame <- gw_model_frame(call, env, formula, data)
  if (length(frame$strata) > 0L || !is.null(frame$cluster)) {
    stop(sprintf("`%s` doesn't support `strata()` or `cluster()` terms.", fn), call. = FALSE)
  }
  if (!is.null(split_formula(formula, data)$offset)) {
    stop(sprintf("`%s` doesn't support `offset()` terms.", fn), call. = FALSE)
  }
  parts <- surv_parts(frame$response, fn, types = "right")
  design <- covariate_design(frame, check_rank = FALSE)
  if (ncol(design$x) == 0L) {
    stop(sprintf("`%s` needs at least one covariate.", fn), call. = FALSE)
  }
  event <- unname(parts$event) == 1
  if (!any(event)) {
    stop(sprintf("`%s` needs at least one event.", fn), call. = FALSE)
  }
  list(frame = frame, design = design, x = design$x, time = unname(parts$stop), event = event)
}

# The covariates and the competing-risks response of a tree or boosting model's formula, with
# `status` the cause of each row (0 for censored).
cr_tree_inputs <- function(call, env, formula, data, fn) {
  frame <- gw_model_frame(call, env, formula, data)
  cr_no_specials(frame, fn)
  if (!is.null(split_formula(formula, data)$offset)) {
    stop(sprintf("`%s` doesn't support `offset()` terms.", fn), call. = FALSE)
  }
  cr <- cr_response(frame, fn)
  design <- covariate_design(frame, check_rank = FALSE)
  if (ncol(design$x) == 0L) {
    stop(sprintf("`%s` needs at least one covariate.", fn), call. = FALSE)
  }
  if (!any(cr$status > 0L)) {
    stop(sprintf("`%s` needs at least one event.", fn), call. = FALSE)
  }
  list(
    frame = frame, design = design, x = design$x, time = cr$time, status = cr$status,
    states = cr$states
  )
}

tree_fit_fields <- function(inputs, settings, formula, call) {
  list(
    max_depth = if (is.finite(settings$max_depth)) settings$max_depth else NULL,
    min_samples_split = settings$min_split,
    min_samples_leaf = settings$min_leaf,
    n_features_split = settings$mtry,
    x = inputs$x,
    time = inputs$time,
    event = inputs$event,
    n = length(inputs$time),
    n_event = sum(inputs$event > 0),
    terms = inputs$design$terms,
    xlevels = inputs$design$xlevels,
    contrasts = inputs$design$contrasts,
    n_dropped = inputs$frame$n_dropped,
    formula = formula,
    call = call
  )
}

forest_inputs <- function(call, env, formula, data, fn, n_estimators, max_depth,
                          min_samples_split, min_samples_leaf, bootstrap, oob_score,
                          read_inputs = tree_inputs) {
  check_count(n_estimators, "n_estimators", min = 1)
  check_tree_args(max_depth, min_samples_split, min_samples_leaf)
  check_flag(bootstrap, "bootstrap")
  check_flag(oob_score, "oob_score")
  if (oob_score && !bootstrap) {
    stop("`oob_score = TRUE` needs `bootstrap = TRUE`.", call. = FALSE)
  }
  read_inputs(call, env, formula, data, fn)
}

grow_forest <- function(inputs, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                        max_features, bootstrap, oob_score, splitter, cls, formula, call) {
  x <- inputs$x
  time <- inputs$time
  event <- inputs$event
  n <- length(time)
  settings <- tree_settings(
    max_depth, min_samples_split, min_samples_leaf, max_features, ncol(x), splitter
  )
  grid <- sort(unique(time[event]))
  trees <- vector("list", n_estimators)
  oob <- matrix(FALSE, n, n_estimators)
  for (b in seq_len(n_estimators)) {
    rows <- if (bootstrap) sample.int(n, n, replace = TRUE) else seq_len(n)
    if (bootstrap) {
      oob[, b] <- tabulate(rows, n) == 0L
    }
    tree <- grow_logrank_tree(x[rows, , drop = FALSE], time[rows], event[rows], settings)
    tree <- tree_add_leaves(tree, time[rows], event[rows], grid)
    tree$rows <- if (bootstrap) rows else NULL
    trees[[b]] <- tree
  }

  fit <- structure(
    c(
      list(
        trees = trees,
        times = grid,
        n_estimators = as.integer(n_estimators),
        max_features = max_features,
        bootstrap = bootstrap,
        oob = oob,
        oob_score = NA_real_
      ),
      tree_fit_fields(inputs, settings, formula, call)
    ),
    class = c(cls, "survival_forest")
  )
  if (oob_score) {
    fit$oob_score <- forest_oob_concordance(fit, x)
  }
  fit
}

# Grow one log-rank tree on the rows given. The nodes are numbered in the order they're made
# (each node, then all of its left branch, then its right branch). Each node records its split
# covariate and cut-point (0 and NA for a leaf), its children, its depth, and its numbers of
# subjects and events. `node_of` gives the leaf of each row. `event` is logical, or for
# competing_risk_forest() the cause of each row (0 for censored).
grow_logrank_tree <- function(x, time, event, settings) {
  n <- length(time)
  p <- ncol(x)
  size <- 2L * n - 1L
  feature <- integer(size)
  threshold <- rep(NA_real_, size)
  left <- integer(size)
  right <- integer(size)
  depth <- integer(size)
  n_node <- integer(size)
  n_event <- integer(size)
  node_of <- integer(n)
  count <- 0L
  stack <- list(list(rows = seq_len(n), depth = 0L, parent = 0L, side = 0L))
  while (length(stack) > 0L) {
    item <- stack[[length(stack)]]
    stack[[length(stack)]] <- NULL
    count <- count + 1L
    id <- count
    if (item$side == 1L) {
      left[item$parent] <- id
    } else if (item$side == 2L) {
      right[item$parent] <- id
    }
    rows <- item$rows
    depth[id] <- item$depth
    n_node[id] <- length(rows)
    n_event[id] <- sum(event[rows] > 0)
    split <- NULL
    if (length(rows) >= settings$min_split && n_event[id] > 0L &&
      item$depth < settings$max_depth) {
      features <- if (settings$mtry < p) sample.int(p, settings$mtry) else seq_len(p)
      split_fn <- switch(settings$splitter,
        random = logrank_random_split,
        competing = logrank_cr_best_split,
        logrank_best_split
      )
      split <- split_fn(
        x[rows, , drop = FALSE], time[rows], event[rows], features, settings$min_leaf
      )
    }
    if (is.null(split)) {
      node_of[rows] <- id
      next
    }
    feature[id] <- split$feature
    threshold[id] <- split$threshold
    goes_left <- x[rows, split$feature] <= split$threshold
    child_depth <- item$depth + 1L
    stack[[length(stack) + 1L]] <- list(
      rows = rows[!goes_left], depth = child_depth, parent = id, side = 2L
    )
    stack[[length(stack) + 1L]] <- list(
      rows = rows[goes_left], depth = child_depth, parent = id, side = 1L
    )
  }
  keep <- seq_len(count)
  list(
    feature = feature[keep],
    threshold = threshold[keep],
    left = left[keep],
    right = right[keep],
    depth = depth[keep],
    n = n_node[keep],
    n_event = n_event[keep],
    node_of = node_of,
    n_nodes = count
  )
}

# What the log-rank statistics of a node's splits need: for each distinct event time of the node,
# the numbers at risk and with an event, and the hypergeometric variance factor
# d (n - d) / (n - 1), and for each row the number of those times at or before its own.
logrank_node <- function(time, event) {
  tau <- sort(unique(time[event]))
  m <- length(tau)
  k <- findInterval(time, tau)
  n_risk <- rev(cumsum(rev(tabulate(k, m))))
  d <- tabulate(k[event], m)
  list(
    m = m,
    k = k,
    n_risk = n_risk,
    d = d,
    var_factor = ifelse(n_risk > 1, d * (n_risk - d) / (n_risk - 1), 0)
  )
}

# The running sums down each column of a matrix.
col_cumsum <- function(x) {
  n <- nrow(x)
  sums <- matrix(cumsum(x), n)
  sums - rep(c(0, sums[n, -ncol(x)]), each = n)
}

# The split of a node with the largest log-rank statistic over `features`: every cut-point between
# two distinct values that leaves `min_leaf` rows on each side. NULL when there is no such split.
# A tie goes to the first covariate tried and the smallest cut-point.
logrank_best_split <- function(x, time, event, features, min_leaf) {
  node <- logrank_node(time, event)
  n <- length(time)
  left_size <- seq_len(n - 1L)
  best <- NULL
  for (f in features) {
    col <- unname(x[, f])
    o <- order(col)
    sorted <- col[o]
    valid <- sorted[-n] < sorted[-1L] & left_size >= min_leaf & n - left_size >= min_leaf
    if (!any(valid)) {
      next
    }
    # The numbers at risk on the left, with the first p rows (in covariate order) on the left.
    n_left <- col_cumsum(outer(node$k[o], seq_len(node$m), ">="))
    frac <- n_left / rep(node$n_risk, each = n)
    u <- cumsum(event[o]) - drop(frac %*% node$d)
    v <- drop((frac * (1 - frac)) %*% node$var_factor)
    chi <- ifelse(v > 0, u * u / v, 0)[-n]
    chi[!valid] <- -Inf
    p <- which.max(chi)
    if (is.null(best) || chi[p] > best$score) {
      best <- list(feature = f, threshold = 0.5 * (sorted[p] + sorted[p + 1L]), score = chi[p])
    }
  }
  best
}

# The extremely randomized split of a node: one cut-point for each covariate, drawn uniformly
# between its smallest and largest values, keeping the one with the largest log-rank statistic.
logrank_random_split <- function(x, time, event, features, min_leaf) {
  node <- logrank_node(time, event)
  n <- length(time)
  best <- NULL
  for (f in features) {
    col <- unname(x[, f])
    lo <- min(col)
    hi <- max(col)
    if (lo >= hi) {
      next
    }
    threshold <- stats::runif(1L, lo, hi)
    goes_left <- col <= threshold
    n_left <- sum(goes_left)
    if (n_left < min_leaf || n - n_left < min_leaf) {
      next
    }
    at_risk <- rev(cumsum(rev(tabulate(node$k[goes_left], node$m))))
    frac <- at_risk / node$n_risk
    u <- sum(tabulate(node$k[goes_left & event], node$m) - frac * node$d)
    v <- sum(node$var_factor * frac * (1 - frac))
    if (v <= 0) {
      next
    }
    if (is.null(best) || u * u / v > best$score) {
      best <- list(feature = f, threshold = threshold, score = u * u / v)
    }
  }
  best
}

# The leaves of a grown tree and the summed cumulative hazard (mortality) of each.
tree_add_leaves <- function(tree, time, event, grid) {
  curves <- tree_leaf_curves(tree, time, event, grid)
  tree$leaves <- curves$leaves
  tree$mortality <- rowSums(curves$cumhaz)
  tree
}

# The Nelson-Aalen cumulative hazard and Kaplan-Meier survival of each leaf at the times of
# `grid`, from the rows the tree was grown on. Where none of a leaf's rows are at risk, the curves
# stay level.
tree_leaf_curves <- function(tree, time, event, grid) {
  leaves <- which(tree$feature == 0L)
  n_leaves <- length(leaves)
  m <- length(grid)
  leaf <- match(tree$node_of, leaves)
  k <- findInterval(time, grid)
  exits <- matrix(tabulate(leaf + n_leaves * k, n_leaves * (m + 1L)), n_leaves, m + 1L)
  events <- matrix(
    tabulate(leaf[event] + n_leaves * (k[event] - 1L), n_leaves * m), n_leaves, m
  )
  at_risk <- matrix(0, n_leaves, m)
  running <- numeric(n_leaves)
  for (i in rev(seq_len(m))) {
    running <- running + exits[, i + 1L]
    at_risk[, i] <- running
  }
  cumhaz <- matrix(0, n_leaves, m)
  survival <- matrix(1, n_leaves, m)
  h <- numeric(n_leaves)
  s <- rep(1, n_leaves)
  for (i in seq_len(m)) {
    r <- at_risk[, i]
    e <- events[, i]
    h <- h + ifelse(r > 0, e / r, 0)
    s <- s * ifelse(r > 0, 1 - e / r, 1)
    cumhaz[, i] <- h
    survival[, i] <- s
  }
  list(leaves = leaves, cumhaz = cumhaz, survival = survival)
}

# The node each row of x ends up in, following the splits from the first node. A row with a
# missing value of a covariate it's split on gets NA. The predict() methods give NA for a row
# with any missing covariate.
tree_route <- function(tree, x) {
  node <- rep(1L, nrow(x))
  # The rows not yet at a leaf, and their nodes.
  rows <- seq_len(nrow(x))
  repeat {
    at <- node[rows]
    inner <- tree$feature[at] > 0L
    rows <- rows[inner]
    if (length(rows) == 0L) {
      break
    }
    at <- at[inner]
    goes_left <- x[cbind(rows, tree$feature[at])] <= tree$threshold[at]
    child <- tree$right[at]
    left <- which(goes_left)
    child[left] <- tree$left[at[left]]
    child[is.na(goes_left)] <- NA_integer_
    node[rows] <- child
    rows <- rows[!is.na(child)]
  }
  node
}

# -- prediction ------------------------------------------------------------------------------

#' Predictions from survival trees and forests
#'
#' `predict()` gives the risk score of each row of `newdata` (or of the subjects the model was fit
#' to), or its survival or cumulative hazard curve. For a forest, these are the averages over the
#' trees. The risk score is the cumulative hazard summed over the event times of the fit (the
#' ensemble mortality), larger for an earlier event. `as.data.frame()` gives the nodes of a
#' single tree.
#'
#' @param object,x A [survival_tree()], [random_survival_forest()], or [extra_survival_trees()]
#'   fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"risk"` (the default), `"survival"`, or `"cumhaz"`.
#' @param times For `"survival"` and `"cumhaz"`, the times to evaluate at. The default is the
#'   event times of the fit.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `"risk"`, a numeric vector with one value per row of `newdata`. For `"survival"`
#'   and `"cumhaz"`, a data frame with one row per subject and time: `subject` (the row of
#'   `newdata`), `time`, and `estimate`. A row with a missing covariate gets `NA`.
#'   `as.data.frame()` gives one row per node of the tree, in the order they were made: `node`,
#'   `depth`, `term` (the covariate split on, `NA` for a leaf), `threshold` (rows with a value at
#'   or below it go left), `left` and `right` (the child nodes, `NA` for a leaf), `n`, and
#'   `n_event`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#'     2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57, 67,
#'     49, 50, 58, 72, 70, 60, 70, 53, 74),
#'   ph.ecog = c(1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 1, 1, 1, 1,
#'     2, 1, 1)
#' )
#' tree <- survival_tree(Surv(time, status) ~ age + ph.ecog, data = lung, max_depth = 2)
#'
#' # The risk scores of the first five subjects
#' predict(tree)[1:5]
#'
#' # Cumulative hazard curves for a new subject
#' predict(tree, newdata = data.frame(age = 65, ph.ecog = 1), type = "cumhaz",
#'   times = c(100, 200, 400)
#' )
#' @name survival_tree_predict
#' @export
predict.survival_tree <- function(object,
                                  newdata,
                                  type = c("risk", "survival", "cumhaz"),
                                  times,
                                  ...) {
  type <- match.arg(type)
  x <- tree_new_x(object, if (missing(newdata)) NULL else newdata)
  tree_predict(object, list(object$tree), x, type, if (missing(times)) NULL else times)
}

#' @rdname survival_tree_predict
#' @export
predict.survival_forest <- function(object,
                                    newdata,
                                    type = c("risk", "survival", "cumhaz"),
                                    times,
                                    ...) {
  type <- match.arg(type)
  x <- tree_new_x(object, if (missing(newdata)) NULL else newdata)
  tree_predict(object, object$trees, x, type, if (missing(times)) NULL else times)
}

tree_new_x <- function(object, newdata) {
  if (is.null(newdata)) {
    return(object$x)
  }
  if (!is.data.frame(newdata)) {
    stop("`newdata` must be a data frame.", call. = FALSE)
  }
  cox_new_design(
    list(
      terms = object$terms, xlevels = object$xlevels, contrasts = object$contrasts,
      coefficients = stats::setNames(numeric(ncol(object$x)), colnames(object$x))
    ),
    newdata
  )
}

# The average over `trees` of the risk scores, or of the curves at `times`, of the rows of x.
tree_predict <- function(object, trees, x, type, times) {
  missing <- !stats::complete.cases(x)
  if (type == "risk") {
    risk <- numeric(nrow(x))
    for (tree in trees) {
      risk <- risk + tree$mortality[match(tree_route(tree, x), tree$leaves)]
    }
    risk[missing] <- NA
    return(risk / length(trees))
  }
  times <- if (is.null(times)) object$times else check_times(times)
  curves <- matrix(0, nrow(x), length(object$times))
  for (tree in trees) {
    rows <- if (is.null(tree$rows)) seq_len(object$n) else tree$rows
    leaf <- tree_leaf_curves(tree, object$time[rows], object$event[rows], object$times)
    at <- match(tree_route(tree, x), leaf$leaves)
    curves <- curves + leaf[[type]][at, , drop = FALSE]
  }
  curves[missing, ] <- NA
  curve_frame(curves / length(trees), object$times, times, if (type == "survival") 1 else 0)
}

# Curves (one row per subject, one column per time of `grid`) read as step functions at `times`,
# in the long layout, with `baseline` before the first time.
curve_frame <- function(curves, grid, times, baseline) {
  i <- findInterval(times, grid)
  values <- matrix(baseline, nrow(curves), length(times))
  values[, i > 0L] <- curves[, i[i > 0L], drop = FALSE]
  values[is.na(curves[, 1L]), ] <- NA
  long_curves(values, times)
}

# Curves (one row per subject, one column per time) in the long layout.
long_curves <- function(values, times) {
  data.frame(
    subject = rep(seq_len(nrow(values)), each = length(times)),
    time = rep(times, times = nrow(values)),
    estimate = as.vector(t(values))
  )
}

#' @rdname survival_tree_predict
#' @export
as.data.frame.survival_tree <- function(x, row.names = NULL, optional = FALSE, ...) {
  tree <- x$tree
  inner <- tree$feature > 0L
  data.frame(
    node = seq_len(tree$n_nodes),
    depth = tree$depth,
    term = ifelse(inner, colnames(x$x)[pmax(tree$feature, 1L)], NA_character_),
    threshold = tree$threshold,
    left = ifelse(inner, tree$left, NA_integer_),
    right = ifelse(inner, tree$right, NA_integer_),
    n = tree$n,
    n_event = tree$n_event,
    stringsAsFactors = FALSE
  )
}

#' @export
nobs.survival_tree <- function(object, ...) {
  object$n
}

#' @export
nobs.survival_forest <- function(object, ...) {
  object$n
}

#' @export
print.survival_tree <- function(x, digits = getOption("digits"), ...) {
  cat("<survival_tree> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "Log-rank splits, %d nodes, %d leaves. n = %d, events = %d, features = %d\n",
    x$n_nodes, x$n_leaves, x$n, x$n_event, ncol(x$x)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

#' @export
print.survival_forest <- function(x, digits = getOption("digits"), ...) {
  cat("<", class(x)[1L], "> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    "%d log-rank trees, %d of %d features tried per split%s. n = %d, events = %d\n",
    x$n_estimators, x$n_features_split, ncol(x$x),
    if (x$bootstrap) ", bootstrap samples" else "", x$n, x$n_event
  ))
  if (!is.na(x$oob_score)) {
    cat(sprintf(
      "Out-of-bag concordance = %s\n", format(x$oob_score, digits = max(3L, digits - 3L))
    ))
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- out of bag ------------------------------------------------------------------------------

# Each training subject's average risk score over the trees it was left out of, for the rows of
# x (the training covariates, possibly with a column shuffled). `score` gives a tree's risk score
# for each of its leaves. `scored` marks the subjects left out of at least one tree.
forest_oob_risk <- function(object, x, score = function(tree) tree$mortality) {
  risk <- numeric(object$n)
  count <- numeric(object$n)
  for (b in seq_along(object$trees)) {
    out <- object$oob[, b]
    if (!any(out)) {
      next
    }
    tree <- object$trees[[b]]
    node <- tree_route(tree, x[out, , drop = FALSE])
    risk[out] <- risk[out] + score(tree)[match(node, tree$leaves)]
    count[out] <- count[out] + 1
  }
  scored <- count > 0
  list(risk = risk[scored] / count[scored], scored = scored)
}

# The out-of-bag concordance of the risk scores with the events `event` (logical).
forest_oob_concordance <- function(object, x, event = object$event,
                                   score = function(tree) tree$mortality) {
  oob <- forest_oob_risk(object, x, score)
  if (sum(oob$scored) < 2L) {
    return(NA_real_)
  }
  response <- Surv(object$time[oob$scored], as.numeric(event[oob$scored]))
  concordance_index(response, risk = oob$risk)
}

# The permutation importance of each covariate: the mean increase in the out-of-bag error over
# `n_repeats` shuffles of its column.
forest_permutation_importance <- function(object, n_repeats, event = object$event,
                                          score = function(tree) tree$mortality) {
  check_count(n_repeats, "n_repeats", min = 1)
  if (!object$bootstrap) {
    stop("`variable_importance()` needs a forest grown with `bootstrap = TRUE`.", call. = FALSE)
  }
  x <- object$x
  n <- object$n
  oob_error <- function(design) 1 - forest_oob_concordance(object, design, event, score)
  base <- oob_error(x)
  importance <- vapply(seq_len(ncol(x)), function(j) {
    increase <- 0
    for (r in seq_len(n_repeats)) {
      shuffled <- x
      shuffled[, j] <- x[sample.int(n), j]
      increase <- increase + oob_error(shuffled) - base
    }
    increase / n_repeats
  }, numeric(1))
  importance_table(colnames(x), importance)
}

#' Variable importance
#'
#' `variable_importance()` measures how much each covariate contributes to a tree model's
#' predictions.
#'
#' For [random_survival_forest()], [extra_survival_trees()], and [competing_risk_forest()] fits,
#' it's the permutation importance: the values of one covariate are shuffled among the subjects,
#' the out-of-bag error (one minus the out-of-bag concordance) is computed again, and the
#' importance is how much it went up, averaged over `n_repeats` shuffles. Shuffling a covariate
#' the forest doesn't use changes nothing, so its importance is about 0, and it can come out a
#' little below. The forest must be grown on bootstrap samples. For a competing-risks forest, the
#' concordance is that of one cause's risk score, with the other causes as censoring. The shuffles
#' use R's random number generator, so call [set.seed()] first for results that can be repeated.
#'
#' For [gradient_boosting_survival_analysis()] fits, it's the impurity importance: the total
#' reduction in squared error of the splits on each covariate, over all of the trees, as a share
#' of the total over all covariates.
#'
#' @param object A [random_survival_forest()], [extra_survival_trees()],
#'   [competing_risk_forest()], or [gradient_boosting_survival_analysis()] fit.
#' @param n_repeats For forests, the number of shuffles of each covariate. The default is `5`.
#' @param ... Unused.
#'
#' @return A data frame with one row per covariate (column of the design), from the most to the
#'   least important: `term` and `importance`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#'     2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57, 67,
#'     49, 50, 58, 72, 70, 60, 70, 53, 74),
#'   ph.ecog = c(1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 1, 1, 1, 1,
#'     2, 1, 1)
#' )
#'
#' set.seed(1)
#' forest <- random_survival_forest(Surv(time, status) ~ age + ph.ecog,
#'   data = lung, n_estimators = 50
#' )
#' variable_importance(forest, n_repeats = 3)
#'
#' boosted <- gradient_boosting_survival_analysis(Surv(time, status) ~ age + ph.ecog,
#'   data = lung, n_estimators = 50, max_depth = 2
#' )
#' variable_importance(boosted)
#' @export
variable_importance <- function(object, ...) {
  UseMethod("variable_importance")
}

#' @rdname variable_importance
#' @export
variable_importance.survival_forest <- function(object, n_repeats = 5, ...) {
  forest_permutation_importance(object, n_repeats)
}

importance_table <- function(terms, importance) {
  o <- order(importance, decreasing = TRUE)
  data.frame(term = terms[o], importance = importance[o], stringsAsFactors = FALSE)
}
