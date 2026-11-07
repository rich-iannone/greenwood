# Gradient boosting with regression trees: gradient_boosting_survival_analysis() for the Cox
# partial likelihood and survival_boost() for competing risks, with their methods. Tested by
# tests/testthat/test-boosting.R, against Greenwood's Python values for the fits that use no
# random numbers, against the gradients and splits computed by hand, and for the sampling of
# the competing-risks targets.

#' Gradient boosting for the Cox model
#'
#' `gradient_boosting_survival_analysis()` builds a risk score \eqn{F(x)} as a sum of small
#' regression trees, in place of the linear predictor of [cox_ph()]. It starts from
#' \eqn{F = 0}, and at each of `n_estimators` rounds fits a squared-error regression tree of depth
#' at most `max_depth` to the martingale residuals of the current score,
#' \eqn{\delta_i - e^{F(x_i)} \hat H_0(t_i)}, which are the negative gradient of the negative log
#' partial likelihood, with \eqn{\hat H_0} the Breslow estimate of the baseline cumulative hazard.
#' The tree's predictions, shrunk by `learning_rate`, are added to the score. The trees can find
#' nonlinear effects and, with `max_depth` above 1, interactions, so no linear form is assumed
#' for the covariates, while the score still acts on the hazard proportionally.
#'
#' A smaller `learning_rate` needs more trees but usually predicts better. With `subsample` below
#' 1, each tree is grown on a random share of the subjects (drawn without replacement, and at
#' least `2 * min_samples_leaf` of them), which is stochastic gradient boosting. With
#' `max_features` smaller than the number of covariates, each split looks at a random subset of
#' them. Both use R's random number generator, so call [set.seed()] first for results that can be
#' repeated.
#'
#' Survival curves are \eqn{\exp\{-\hat H_0(t) e^{F(x)}\}}, with \eqn{\hat H_0} the Breslow
#' estimate at the final scores.
#'
#' @inheritParams survival_tree
#' @param n_estimators The number of boosting rounds (trees), at least 1. The default is `100`.
#' @param learning_rate The shrinkage of each tree, a positive number. The default is `0.1`.
#' @param max_depth The largest depth of each tree, at least 1. The default is `3`.
#' @param min_samples_leaf The fewest subjects in each leaf of a tree, at least 1. The default is
#'   `3`.
#' @param subsample The share of the subjects each tree is grown on, greater than 0 and at most 1.
#'   The default is `1` (all of them).
#'
#' @return A `gradient_boosting_survival_analysis` object. Use [predict()] for risk scores and
#'   survival or cumulative hazard curves, and [variable_importance()] for the share of the
#'   reduction in squared error credited to each covariate. The object also holds `times` and
#'   `cumhaz` (the baseline cumulative hazard at the event times), `n`, and `n_event`.
#'
#' @references Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine.
#'   *The Annals of Statistics*, 29(5), 1189-1232.
#'
#'   Ridgeway, G. (1999). The state of boosting. *Computing Science and Statistics*, 31, 172-181.
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
#' fit <- gradient_boosting_survival_analysis(Surv(time, status) ~ age + ph.ecog,
#'   data = lung, n_estimators = 50, learning_rate = 0.05, max_depth = 2
#' )
#' fit
#'
#' # Risk scores and survival curves for two new subjects
#' newdata <- data.frame(age = c(55, 72), ph.ecog = c(0, 2))
#' predict(fit, newdata = newdata)
#' predict(fit, newdata = newdata, type = "survival", times = c(100, 300))
#'
#' variable_importance(fit)
#' @export
gradient_boosting_survival_analysis <- function(formula,
                                                data,
                                                subset,
                                                na.action = stats::na.omit,
                                                n_estimators = 100,
                                                learning_rate = 0.1,
                                                max_depth = 3,
                                                min_samples_leaf = 3,
                                                subsample = 1,
                                                max_features = NULL) {
  check_boost_args(n_estimators, learning_rate, max_depth, min_samples_leaf, subsample)
  inputs <- tree_inputs(
    match.call(expand.dots = FALSE), parent.frame(), formula,
    if (missing(data)) NULL else data, "gradient_boosting_survival_analysis()"
  )
  x <- inputs$x
  time <- inputs$time
  event <- inputs$event
  n <- length(time)
  mtry <- resolve_max_features(max_features, ncol(x))
  size <- min(max(2 * min_samples_leaf, round(subsample * n)), n)
  scores <- numeric(n)
  importance <- numeric(ncol(x))
  trees <- vector("list", n_estimators)
  for (b in seq_len(n_estimators)) {
    residuals <- boost_cox_gradient(scores, time, event)$residuals
    rows <- if (subsample < 1) sample.int(n, size) else seq_len(n)
    tree <- grow_regression_tree(
      x[rows, , drop = FALSE], residuals[rows], max_depth, min_samples_leaf, mtry
    )
    importance <- importance + tree$importance
    scores <- scores + learning_rate * regression_tree_predict(tree, x)
    trees[[b]] <- tree
  }
  baseline <- boost_cox_gradient(scores, time, event)
  total <- sum(importance)

  structure(
    list(
      trees = trees,
      n_estimators = as.integer(n_estimators),
      learning_rate = learning_rate,
      max_depth = max_depth,
      min_samples_leaf = min_samples_leaf,
      subsample = subsample,
      max_features = max_features,
      n_features_split = mtry,
      scores = scores,
      times = baseline$times,
      cumhaz = baseline$cumhaz,
      importance = stats::setNames(if (total > 0) importance / total else importance, colnames(x)),
      x = x,
      n = n,
      n_event = sum(event),
      terms = inputs$design$terms,
      xlevels = inputs$design$xlevels,
      contrasts = inputs$design$contrasts,
      n_dropped = inputs$frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "gradient_boosting_survival_analysis"
  )
}

check_boost_args <- function(n_estimators, learning_rate, max_depth, min_samples_leaf,
                             subsample) {
  check_count(n_estimators, "n_estimators", min = 1)
  check_number(learning_rate, "learning_rate", min = 0)
  if (learning_rate == 0) {
    stop("`learning_rate` must be greater than 0.", call. = FALSE)
  }
  check_count(max_depth, "max_depth", min = 1)
  check_count(min_samples_leaf, "min_samples_leaf", min = 1)
  if (!is.numeric(subsample) || length(subsample) != 1L || is.na(subsample) || subsample <= 0 ||
    subsample > 1) {
    stop("`subsample` must be a number greater than 0 and at most 1.", call. = FALSE)
  }
}

# The martingale residuals of the Cox model with risk scores `scores`, and the Breslow baseline
# cumulative hazard at the event times.
boost_cox_gradient <- function(scores, time, event) {
  relative <- exp(scores)
  times <- sort(unique(time))
  m <- length(times)
  k <- match(time, times)
  risk <- rev(cumsum(rev(tabulate_sum(k, relative, m))))
  d <- tabulate(k[event], m)
  cumhaz <- cumsum(ifelse(risk > 0, d / risk, 0))
  list(
    residuals = as.numeric(event) - relative * cumhaz[k],
    times = times[d > 0],
    cumhaz = cumhaz[d > 0]
  )
}

# Grow a squared-error regression tree of depth at most `max_depth`. A node is split when it has
# at least 2 * min_leaf rows and y isn't constant on it, at the cut-point that most reduces the
# squared error, if any does. `importance` is the reduction credited to each column. Nodes are
# numbered as in grow_logrank_tree(), and `value` is the mean of y on each node.
grow_regression_tree <- function(x, y, max_depth, min_leaf, mtry) {
  n <- length(y)
  p <- ncol(x)
  size <- max(2L * n - 1L, 1L)
  feature <- integer(size)
  threshold <- rep(NA_real_, size)
  left <- integer(size)
  right <- integer(size)
  value <- numeric(size)
  importance <- numeric(p)
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
    node_y <- y[rows]
    value[id] <- if (length(rows) > 0L) mean(node_y) else 0
    if (item$depth >= max_depth || length(rows) < 2L * min_leaf ||
      diff(range(node_y)) <= 0) {
      next
    }
    features <- if (mtry < p) sample.int(p, mtry) else seq_len(p)
    split <- sse_best_split(x[rows, , drop = FALSE], node_y, features, min_leaf)
    if (is.null(split) || split$score <= 0) {
      next
    }
    importance[split$feature] <- importance[split$feature] + split$score
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
    value = value[keep],
    importance = importance
  )
}

# The split of a node with the largest reduction in squared error over `features`, among the
# cut-points between two distinct values that leave `min_leaf` rows on each side. `score` is the
# reduction. A tie goes to the first covariate tried and the smallest cut-point.
sse_best_split <- function(x, y, features, min_leaf) {
  n <- length(y)
  total <- sum(y)
  parent <- total * total / n
  n_left <- seq_len(n - 1L)
  n_right <- n - n_left
  best <- NULL
  for (f in features) {
    col <- unname(x[, f])
    o <- order(col)
    sorted <- col[o]
    valid <- sorted[-n] < sorted[-1L] & n_left >= min_leaf & n_right >= min_leaf
    if (!any(valid)) {
      next
    }
    csum <- cumsum(y[o])[-n]
    objective <- csum * csum / n_left + (total - csum)^2 / n_right
    objective[!valid] <- -Inf
    p <- which.max(objective)
    gain <- objective[p] - parent
    if (is.null(best) || gain > best$score) {
      best <- list(feature = f, threshold = 0.5 * (sorted[p] + sorted[p + 1L]), score = gain)
    }
  }
  best
}

regression_tree_predict <- function(tree, x) {
  tree$value[tree_route(tree, x)]
}

#' Predictions from gradient boosting for the Cox model
#'
#' `predict()` gives the risk score of each row of `newdata` (or of the subjects the model was fit
#' to), or its survival or cumulative hazard curve.
#'
#' @param object A [gradient_boosting_survival_analysis()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"risk"` (the relative hazard \eqn{e^{F(x)}}, the default),
#'   `"lp"` (the score \eqn{F(x)}), `"survival"`, or `"cumhaz"`.
#' @param times For `"survival"` and `"cumhaz"`, the times to evaluate at. The default is the
#'   event times of the fit.
#' @param ... Unused.
#'
#' @return For `"risk"` and `"lp"`, a numeric vector with one value per row of `newdata`. For
#'   `"survival"` and `"cumhaz"`, a data frame with one row per subject and time: `subject` (the
#'   row of `newdata`), `time`, and `estimate`. A row with a missing covariate gets `NA`.
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
#' fit <- gradient_boosting_survival_analysis(Surv(time, status) ~ age + ph.ecog,
#'   data = lung, n_estimators = 50, max_depth = 2
#' )
#'
#' # The scores of the first five subjects
#' predict(fit, type = "lp")[1:5]
#'
#' predict(fit, newdata = data.frame(age = 65, ph.ecog = 1), type = "cumhaz",
#'   times = c(100, 200, 400)
#' )
#' @export
predict.gradient_boosting_survival_analysis <- function(object,
                                                        newdata,
                                                        type = c(
                                                          "risk", "lp", "survival", "cumhaz"
                                                        ),
                                                        times,
                                                        ...) {
  type <- match.arg(type)
  scores <- if (missing(newdata)) {
    object$scores
  } else {
    x <- tree_new_x(object, newdata)
    scores <- numeric(nrow(x))
    for (tree in object$trees) {
      scores <- scores + object$learning_rate * regression_tree_predict(tree, x)
    }
    scores[!stats::complete.cases(x)] <- NA
    scores
  }
  if (type == "lp") {
    return(scores)
  }
  if (type == "risk") {
    return(exp(scores))
  }
  times <- if (missing(times)) object$times else check_times(times)
  cumhaz <- outer(exp(scores), step_at(object$times, object$cumhaz, times, 0))
  long_curves(if (type == "survival") exp(-cumhaz) else cumhaz, times)
}

#' @rdname variable_importance
#' @export
variable_importance.gradient_boosting_survival_analysis <- function(object, ...) {
  importance_table(names(object$importance), unname(object$importance))
}

#' @export
nobs.gradient_boosting_survival_analysis <- function(object, ...) {
  object$n
}

#' @export
print.gradient_boosting_survival_analysis <- function(x, digits = getOption("digits"), ...) {
  cat("<gradient_boosting_survival_analysis> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    paste0(
      "%d regression trees of depth at most %d, learning rate %s, Cox partial likelihood.\n",
      "n = %d, events = %d, features = %d\n"
    ),
    x$n_estimators, as.integer(x$max_depth), format(x$learning_rate), x$n, x$n_event,
    ncol(x$x)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}

# -- SurvivalBoost ---------------------------------------------------------------------------

#' Gradient boosting for competing risks
#'
#' `survival_boost()` estimates the cumulative incidence of each cause, and the probability of
#' being free of any event, as functions of time and the covariates, by multi-class gradient
#' boosting (Alberge et al., 2025). At a time \eqn{t}, each subject is in one of \eqn{K + 1}
#' classes: still free of any event, or having had an event of cause \eqn{k} by \eqn{t}. The model
#' gives the probabilities of the classes as the softmax of \eqn{K + 1} scores, each a sum of
#' regression trees on \eqn{t} and the covariates.
#'
#' At each of `n_estimators` rounds, every subject gets `n_horizons_per_observation` random times
#' \eqn{\tau}, drawn uniformly up to the largest follow-up time, with a share `hard_zero_fraction`
#' of them set to 0. A subject still under follow-up at \eqn{\tau} counts as event-free, weighted
#' by \eqn{1 / G(\tau)}. A subject with an event by \eqn{\tau} counts in the class of its cause,
#' weighted by \eqn{1 / G(T)}, with \eqn{T} its event time. A subject censored by \eqn{\tau} is
#' left out. \eqn{G} is the Kaplan-Meier curve of the censoring times (with events tied with a
#' censoring taken to come first), kept at least `epsilon_censoring`. These inverse-probability-of-
#' censoring weights make up for the subjects left out. For each class, a regression tree on
#' \eqn{(\tau, x)} is then fit to the weighted gradient of the multi-class log loss,
#' \eqn{w (y_k - p_k)}, with \eqn{p_k} the current model's probability of the class at
#' \eqn{(\tau, x)}, and added to the class's score, shrunk by `learning_rate`.
#'
#' The model uses R's random number generator for the times, and for `subsample` and
#' `max_features` (which work as in [gradient_boosting_survival_analysis()], with the time counted
#' as one of the covariates), so call [set.seed()] first for results that can be repeated.
#'
#' @inheritParams gradient_boosting_survival_analysis
#' @param formula A formula with a competing-risks response on the left, `Surv(time, event)` with
#'   `event` a factor whose first level is censoring, and the covariates on the right, such as
#'   `Surv(time, event) ~ age + sex`.
#' @param learning_rate The shrinkage of each tree, a positive number. The default is `0.05`.
#' @param min_samples_leaf The fewest rows in each leaf of a tree, at least 1. The default is `10`.
#' @param subsample The share of the rows (the subjects at their times \eqn{\tau}) each round's
#'   trees are grown on, greater than 0 and at most 1. The default is `1`.
#' @param max_features The number of columns (the time and the covariates) tried at each split:
#'   `NULL` (all of them, the default), `"sqrt"`, `"log2"`, a whole number, or a fraction between
#'   0 and 1 of the columns.
#' @param n_time_grid_steps The number of times in the default grid of [predict()]: the event times
#'   if there are no more of them than this, and otherwise this many of their quantiles. The
#'   default is `100`.
#' @param hard_zero_fraction The share of each round's times set to 0, at least 0 and below 1. The
#'   default is `0.1`.
#' @param n_horizons_per_observation The number of random times per subject in each round, at
#'   least 1. The default is `3`.
#' @param epsilon_censoring The smallest value of the censoring curve used in the weights, a
#'   positive number. The default is `0.05`.
#'
#' @return A `survival_boost` object. Use [predict()] for the cumulative incidence of each cause
#'   and the probability of being free of any event. The object also holds `times` (the default
#'   times of the predictions), `states` (the causes), `n`, and `n_event` (the number of events of
#'   any cause).
#'
#' @references Alberge, J., Maladiere, V., Grisel, O., Abecassis, J., and Varoquaux, G. (2025).
#'   Survival models: proper scoring rule and stochastic optimization with competing risks.
#'   *Proceedings of the 28th International Conference on Artificial Intelligence and Statistics*.
#'
#' @seealso [aalen_johansen()] for the cumulative incidence without covariates, and [fine_gray()]
#'   for a regression model of it.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95,
#'     35, 140),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61, 81, 57),
#'   hgb = c(13, 12, 14, 11, 13, 12, 15, 12, 11, 10, 14, 13, 12, 11, 15, 12, 13, 12, 10, 14)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' set.seed(1)
#' fit <- survival_boost(Surv(time, event) ~ age + hgb,
#'   data = mgus, n_estimators = 30, max_depth = 2, min_samples_leaf = 5
#' )
#' fit
#'
#' # The cumulative incidence of each cause for a new subject
#' predict(fit, newdata = data.frame(age = 75, hgb = 11), times = c(50, 150))
#' @export
survival_boost <- function(formula,
                           data,
                           subset,
                           na.action = stats::na.omit,
                           n_estimators = 100,
                           learning_rate = 0.05,
                           max_depth = 3,
                           min_samples_leaf = 10,
                           subsample = 1,
                           max_features = NULL,
                           n_time_grid_steps = 100,
                           hard_zero_fraction = 0.1,
                           n_horizons_per_observation = 3,
                           epsilon_censoring = 0.05) {
  check_boost_args(n_estimators, learning_rate, max_depth, min_samples_leaf, subsample)
  check_count(n_time_grid_steps, "n_time_grid_steps", min = 1)
  if (!is.numeric(hard_zero_fraction) || length(hard_zero_fraction) != 1L ||
    is.na(hard_zero_fraction) || hard_zero_fraction < 0 || hard_zero_fraction >= 1) {
    stop("`hard_zero_fraction` must be a number at least 0 and below 1.", call. = FALSE)
  }
  check_count(n_horizons_per_observation, "n_horizons_per_observation", min = 1)
  check_number(epsilon_censoring, "epsilon_censoring", min = 0)
  if (epsilon_censoring == 0) {
    stop("`epsilon_censoring` must be greater than 0.", call. = FALSE)
  }
  fn <- "survival_boost()"
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  cr_no_specials(frame, fn)
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
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
  x <- design$x
  time <- cr$time
  status <- cr$status
  n <- length(time)
  n_classes <- length(cr$states) + 1L

  event_times <- sort(time[status > 0L])
  grid <- if (length(event_times) > n_time_grid_steps) {
    unique(unname(stats::quantile(event_times, seq(0, 1, length.out = n_time_grid_steps))))
  } else {
    unique(event_times)
  }
  censoring <- censoring_steps(time, status == 0L)
  g_at <- function(t) {
    i <- findInterval(t, censoring$time)
    g <- ifelse(i > 0L, censoring$g[pmax(i, 1L)], 1)
    ifelse(t <= 0, 1, pmax(g, epsilon_censoring))
  }
  t_max <- max(time)
  n_zero <- floor(hard_zero_fraction * n)
  mtry <- resolve_max_features(max_features, ncol(x) + 1L)

  rounds <- list()
  for (b in seq_len(n_estimators)) {
    horizon <- unlist(lapply(seq_len(n_horizons_per_observation), function(h) {
      tau <- stats::runif(n, 0, t_max)
      tau[sample.int(n, n_zero)] <- 0
      tau
    }))
    subject <- rep(seq_len(n), times = n_horizons_per_observation)
    t_i <- time[subject]
    later <- t_i > horizon
    had_event <- !later & status[subject] > 0L
    target <- ifelse(later, 0L, status[subject])
    weight <- ifelse(later, 1 / g_at(horizon), ifelse(had_event, 1 / g_at(t_i), 0))
    active <- which(weight > 0)
    if (length(active) < 2L) {
      next
    }
    xa <- cbind(horizon[active], x[subject[active], , drop = FALSE])
    probs <- softmax(boost_scores(rounds, xa, learning_rate, n_classes))
    rows <- if (subsample < 1) {
      sample.int(length(active), min(
        max(2 * min_samples_leaf, round(subsample * length(active))), length(active)
      ))
    } else {
      seq_along(active)
    }
    w <- weight[active]
    rounds[[length(rounds) + 1L]] <- lapply(seq_len(n_classes), function(k) {
      gradient <- w * ((target[active] == k - 1L) - probs[, k])
      grow_regression_tree(xa[rows, , drop = FALSE], gradient[rows], max_depth,
        min_samples_leaf, mtry
      )
    })
  }

  structure(
    list(
      rounds = rounds,
      n_estimators = as.integer(n_estimators),
      learning_rate = learning_rate,
      max_depth = max_depth,
      min_samples_leaf = min_samples_leaf,
      n_features_split = mtry,
      times = grid,
      states = cr$states,
      x = x,
      n = n,
      n_event = sum(status > 0L),
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "survival_boost"
  )
}

# The class scores of the rows of xa (the time, then the covariates) from the boosting rounds.
boost_scores <- function(rounds, xa, learning_rate, n_classes) {
  scores <- matrix(0, nrow(xa), n_classes)
  for (trees in rounds) {
    for (k in seq_len(n_classes)) {
      scores[, k] <- scores[, k] + learning_rate * regression_tree_predict(trees[[k]], xa)
    }
  }
  scores
}

softmax <- function(scores) {
  e <- exp(scores - apply(scores, 1L, max))
  e / rowSums(e)
}

#' Predictions from gradient boosting for competing risks
#'
#' `predict()` gives the cumulative incidence of each cause, or the probability of being free of
#' any event, for each row of `newdata` (or for the subjects the model was fit to), at `times`.
#' At each time, these are the softmax of the class scores, so the incidences of all the causes
#' and the probability of being free of any event add up to 1. The curves aren't constrained to
#' be monotone in time.
#'
#' @param object A [survival_boost()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type `"cif"` (the default) for the cumulative incidence of each cause, or `"survival"`
#'   for the probability of being free of any event.
#' @param times The times to evaluate at. The default is the time grid of the fit (see the
#'   `n_time_grid_steps` argument of [survival_boost()]).
#' @param ... Unused.
#'
#' @return A data frame. For `"cif"`, one row per subject, cause, and time: `subject` (the row of
#'   `newdata`), `cause`, `time`, and `estimate`. For `"survival"`, one row per subject and time:
#'   `subject`, `time`, and `estimate`. A row with a missing covariate gets `NA`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95,
#'     35, 140),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61, 81, 57),
#'   hgb = c(13, 12, 14, 11, 13, 12, 15, 12, 11, 10, 14, 13, 12, 11, 15, 12, 13, 12, 10, 14)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' set.seed(1)
#' fit <- survival_boost(Surv(time, event) ~ age + hgb,
#'   data = mgus, n_estimators = 30, max_depth = 2, min_samples_leaf = 5
#' )
#'
#' newdata <- data.frame(age = c(60, 80), hgb = c(14, 10))
#' predict(fit, newdata = newdata, times = 100)
#' predict(fit, newdata = newdata, type = "survival", times = c(50, 100, 200))
#' @export
predict.survival_boost <- function(object,
                                   newdata,
                                   type = c("cif", "survival"),
                                   times,
                                   ...) {
  type <- match.arg(type)
  x <- tree_new_x(object, if (missing(newdata)) NULL else newdata)
  times <- if (missing(times)) object$times else check_times(times)
  n <- nrow(x)
  n_times <- length(times)
  n_classes <- length(object$states) + 1L
  # Every subject at every time, as rows ordered by time within subject.
  xa <- cbind(rep(times, times = n), x[rep(seq_len(n), each = n_times), , drop = FALSE])
  probs <- softmax(boost_scores(object$rounds, xa, object$learning_rate, n_classes))
  probs[!stats::complete.cases(xa), ] <- NA
  if (type == "survival") {
    return(data.frame(
      subject = rep(seq_len(n), each = n_times),
      time = rep(times, times = n),
      estimate = probs[, 1L]
    ))
  }
  causes <- length(object$states)
  # Reorder from (subject, time) rows and cause columns to subject, then cause, then time.
  index <- array(seq_len(n * n_times), c(n_times, n))
  estimate <- vapply(seq_len(n), function(i) {
    as.vector(probs[index[, i], -1L, drop = FALSE])
  }, numeric(n_times * causes))
  data.frame(
    subject = rep(seq_len(n), each = n_times * causes),
    cause = rep(rep(object$states, each = n_times), times = n),
    time = rep(times, times = n * causes),
    estimate = as.vector(estimate),
    stringsAsFactors = FALSE
  )
}

#' @export
nobs.survival_boost <- function(object, ...) {
  object$n
}

#' @export
print.survival_boost <- function(x, digits = getOption("digits"), ...) {
  cat("<survival_boost> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    paste0(
      "%d boosting rounds of %d regression trees (one per class) of depth at most %d, ",
      "learning rate %s.\nCauses: %s. n = %d, events = %d, features = %d\n"
    ),
    length(x$rounds), length(x$states) + 1L, as.integer(x$max_depth), format(x$learning_rate),
    paste(x$states, collapse = ", "), x$n, x$n_event, ncol(x$x)
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}
