# The competing-risks forest: competing_risk_forest() and its methods. Tested by
# tests/testthat/test-competing-risk-forest.R, against Greenwood's Python values for the trees
# that use no random numbers, the split statistic computed by hand, and aalen_johansen() in each
# leaf.

#' Competing-risks random forests
#'
#' `competing_risk_forest()` grows `n_estimators` trees for a competing-risks response, each on a
#' bootstrap sample of the subjects, with a random subset of `max_features` covariates tried at
#' each split. Each split is the one with the largest composite log-rank statistic over the
#' causes, with the risk set of each cause keeping the subjects who had another cause earlier (as
#' for the subdistribution hazard), so the splits separate the cumulative incidence curves
#' directly. Each leaf holds the Aalen-Johansen cumulative incidence of every cause among its
#' subjects, and a prediction is the average over the trees of the curves of the leaves the
#' subject falls in. Like [random_survival_forest()], the forest finds nonlinear effects and
#' interactions without a model for the hazards (Ishwaran et al., 2014).
#'
#' The statistic of a split pools the causes: with \eqn{O_k - E_k} the observed minus expected
#' events of cause \eqn{k} in one child and \eqn{V_k} its hypergeometric variance, it's
#' \eqn{(\sum_k (O_k - E_k))^2 / \sum_k V_k}. A node is split while it has `min_samples_split`
#' subjects and at least one event, and every split leaves `min_samples_leaf` subjects on each
#' side.
#'
#' With bootstrap samples, each subject is left out of about a third of the trees. The out-of-bag
#' concordance (with `oob_score = TRUE`) scores each subject with only those trees, using the risk
#' score of the first cause and the other causes as censoring. [variable_importance()] measures
#' how much worse that score gets when a covariate's values are shuffled. The forest uses R's
#' random number generator, so call [set.seed()] first for results that can be repeated.
#'
#' @param formula A formula with a competing-risks response on the left, `Surv(time, event)` with
#'   `event` a factor whose first level is censoring, and the covariates on the right, such as
#'   `Surv(time, event) ~ age + sex`. Factors are coded with treatment contrasts.
#' @inheritParams random_survival_forest
#' @param bootstrap Whether to grow each tree on a bootstrap sample. The default is `TRUE`. The
#'   out-of-bag concordance and [variable_importance()] need it.
#'
#' @return A `competing_risk_forest` object. Use [predict()] for cumulative incidence curves and
#'   risk scores, and [variable_importance()] for the permutation importance of each covariate.
#'   The object also holds `times` (the event times the curves are given at), `states` (the
#'   causes), `oob_score` (the out-of-bag concordance for the first cause, or `NA`), `n`, and
#'   `n_event` (the number of events of any cause).
#'
#' @references Ishwaran, H., Gerds, T. A., Kogalur, U. B., Moore, R. D., Gange, S. J., and Lau,
#'   B. M. (2014). Random survival forests for competing risks. *Biostatistics*, 15(4), 757-773.
#'
#' @seealso [predict.competing_risk_forest()], [aalen_johansen()] for the curves without
#'   covariates, and [fine_gray()] for a regression model of one cause's cumulative incidence.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95,
#'     50, 100, 140, 170, 190, 230),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 0, 2, 1, 2),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61, 78, 57, 69,
#'     81, 65, 73),
#'   hgb = c(13, 12, 14, 11, 13, 12, 15, 12, 11, 10, 14, 13, 12, 11, 15, 12, 13, 12, 11, 14, 13,
#'     10, 12, 11)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' set.seed(1)
#' forest <- competing_risk_forest(Surv(time, event) ~ age + hgb,
#'   data = mgus, n_estimators = 50, oob_score = TRUE
#' )
#' forest
#'
#' # The cumulative incidence of death for two new subjects
#' newdata <- data.frame(age = c(60, 80), hgb = c(14, 11))
#' predict(forest, newdata = newdata, cause = "death", times = c(100, 200))
#' @export
competing_risk_forest <- function(formula,
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
    "competing_risk_forest()", n_estimators, max_depth, min_samples_split, min_samples_leaf,
    bootstrap, oob_score,
    read_inputs = cr_tree_inputs
  )
  x <- inputs$x
  time <- inputs$time
  status <- inputs$status
  # The tree fields keep the cause of each row as `event`.
  inputs$event <- status
  n <- length(time)
  n_causes <- length(inputs$states)
  settings <- tree_settings(
    max_depth, min_samples_split, min_samples_leaf, max_features, ncol(x), "competing"
  )
  grid <- sort(unique(time[status > 0L]))
  trees <- vector("list", n_estimators)
  oob <- matrix(FALSE, n, n_estimators)
  for (b in seq_len(n_estimators)) {
    rows <- if (bootstrap) sample.int(n, n, replace = TRUE) else seq_len(n)
    if (bootstrap) {
      oob[, b] <- tabulate(rows, n) == 0L
    }
    tree <- grow_logrank_tree(x[rows, , drop = FALSE], time[rows], status[rows], settings)
    curves <- cr_leaf_cif(tree, time[rows], status[rows], grid, n_causes)
    tree$leaves <- curves$leaves
    # Each leaf's risk score for each cause: its cumulative incidence summed over the event times.
    tree$risk <- matrix(unlist(lapply(curves$cif, rowSums)), ncol = n_causes)
    tree$rows <- if (bootstrap) rows else NULL
    trees[[b]] <- tree
  }

  fit <- structure(
    c(
      list(
        trees = trees,
        times = grid,
        states = inputs$states,
        n_estimators = as.integer(n_estimators),
        max_features = max_features,
        bootstrap = bootstrap,
        oob = oob,
        oob_score = NA_real_
      ),
      tree_fit_fields(inputs, settings, formula, match.call())
    ),
    class = "competing_risk_forest"
  )
  if (oob_score) {
    fit$oob_score <- forest_oob_concordance(fit, x, status == 1L, function(tree) tree$risk[, 1L])
  }
  fit
}

# The split of a node with the largest composite log-rank statistic over `features`, among the
# cut-points between two distinct values that leave `min_leaf` rows on each side. `event` is the
# cause of each row (0 for censored). For cause k, the risk set at each event time also keeps the
# rows that had another cause before it. NULL when there is no such split. A tie goes to the
# first covariate tried and the smallest cut-point.
logrank_cr_best_split <- function(x, time, event, features, min_leaf) {
  n <- length(time)
  tau <- sort(unique(time[event > 0L]))
  m <- length(tau)
  k <- findInterval(time, tau)
  n_risk <- rev(cumsum(rev(tabulate(k, m))))
  causes <- sort(unique(event[event > 0L]))
  d <- lapply(causes, function(cause) tabulate(k[event == cause], m))
  d_all <- Reduce(`+`, d)
  # For each cause, the risk set with the earlier events of the other causes kept in it, and the
  # hypergeometric variance factor d (n - d) / (n - 1) on it.
  inclusive <- lapply(d, function(d_k) n_risk + c(0, cumsum(d_all - d_k))[seq_len(m)])
  var_factor <- Map(function(d_k, r) {
    ifelse(n_risk >= 2, d_k * (r - d_k) / (r - 1), 0)
  }, d, inclusive)
  left_size <- seq_len(n - 1L)
  times <- seq_len(m)
  best <- NULL
  for (f in features) {
    col <- unname(x[, f])
    o <- order(col)
    sorted <- col[o]
    valid <- sorted[-n] < sorted[-1L] & left_size >= min_leaf & n - left_size >= min_leaf
    if (!any(valid)) {
      next
    }
    k_o <- k[o]
    event_o <- event[o]
    # The numbers at risk on the left, with the first p rows (in covariate order) on the left.
    n_left <- col_cumsum(outer(k_o, times, ">="))
    u <- numeric(n)
    v <- numeric(n)
    for (i in seq_along(causes)) {
      other <- event_o > 0L & event_o != causes[i]
      earlier <- col_cumsum(outer(ifelse(other, k_o, m + 1L), times, "<"))
      frac <- (n_left + earlier) / rep(inclusive[[i]], each = n)
      u <- u + cumsum(event_o == causes[i]) - drop(frac %*% d[[i]])
      v <- v + drop((frac * (1 - frac)) %*% var_factor[[i]])
    }
    chi <- ifelse(v > 0, u * u / v, 0)[-n]
    chi[!valid] <- -Inf
    p <- which.max(chi)
    if (is.null(best) || chi[p] > best$score) {
      best <- list(feature = f, threshold = 0.5 * (sorted[p] + sorted[p + 1L]), score = chi[p])
    }
  }
  best
}

# The Aalen-Johansen cumulative incidence of each cause in each leaf, at the times of `grid`, from
# the rows the tree was grown on. `cif` holds one matrix per cause, with a row per leaf.
cr_leaf_cif <- function(tree, time, status, grid, n_causes) {
  leaves <- which(tree$feature == 0L)
  n_leaves <- length(leaves)
  m <- length(grid)
  leaf <- match(tree$node_of, leaves)
  k <- findInterval(time, grid)
  exits <- matrix(tabulate(leaf + n_leaves * k, n_leaves * (m + 1L)), n_leaves, m + 1L)
  at_risk <- matrix(0, n_leaves, m)
  running <- numeric(n_leaves)
  for (i in rev(seq_len(m))) {
    running <- running + exits[, i + 1L]
    at_risk[, i] <- running
  }
  events <- lapply(seq_len(n_causes), function(cause) {
    hit <- status == cause
    matrix(tabulate(leaf[hit] + n_leaves * (k[hit] - 1L), n_leaves * m), n_leaves, m)
  })
  d_all <- Reduce(`+`, events)
  cif <- rep(list(matrix(0, n_leaves, m)), n_causes)
  total <- rep(list(numeric(n_leaves)), n_causes)
  s <- rep(1, n_leaves)
  for (i in seq_len(m)) {
    r <- at_risk[, i]
    for (cause in seq_len(n_causes)) {
      total[[cause]] <- total[[cause]] + ifelse(r > 0, s * events[[cause]][, i] / r, 0)
      cif[[cause]][, i] <- total[[cause]]
    }
    s <- s * ifelse(r > 0, 1 - d_all[, i] / r, 1)
  }
  list(leaves = leaves, cif = cif)
}

#' Predictions from a competing-risks forest
#'
#' `predict()` gives the cumulative incidence of one cause, averaged over the trees of a
#' [competing_risk_forest()], or a risk score for it: the average over the trees of the leaf's
#' cumulative incidence summed over the event times, larger for a higher incidence.
#'
#' @param object A [competing_risk_forest()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the forest was grown on. Rows with a missing covariate get `NA`.
#' @param type What to predict: `"cif"` (the default) or `"risk"`.
#' @param cause The cause, a level of `event` or its number among the causes. The default is the
#'   first cause.
#' @param times For `"cif"`, the times to evaluate at. The default is the event times of the fit.
#' @param ... Unused.
#'
#' @return For `"risk"`, a numeric vector with one value per row of `newdata`. For `"cif"`, a data
#'   frame with one row per subject and time: `subject` (the row of `newdata`), `time`, and
#'   `estimate`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95,
#'     50, 100, 140, 170, 190, 230),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1, 2, 1, 0, 2, 1, 2),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61, 78, 57, 69,
#'     81, 65, 73)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' set.seed(1)
#' forest <- competing_risk_forest(Surv(time, event) ~ age, data = mgus, n_estimators = 25)
#' newdata <- data.frame(age = c(60, 80))
#' predict(forest, newdata = newdata, cause = "pcm", times = c(100, 200))
#' predict(forest, newdata = newdata, type = "risk", cause = "death")
#' @export
predict.competing_risk_forest <- function(object,
                                          newdata,
                                          type = c("cif", "risk"),
                                          cause = 1,
                                          times,
                                          ...) {
  type <- match.arg(type)
  target <- cr_cause(cause, object$states)
  x <- tree_new_x(object, if (missing(newdata)) NULL else newdata)
  incomplete <- !stats::complete.cases(x)
  n_trees <- length(object$trees)
  if (type == "risk") {
    risk <- numeric(nrow(x))
    for (tree in object$trees) {
      risk <- risk + tree$risk[match(tree_route(tree, x), tree$leaves), target]
    }
    risk[incomplete] <- NA
    return(risk / n_trees)
  }
  times <- if (missing(times)) object$times else check_times(times)
  curves <- matrix(0, nrow(x), length(object$times))
  for (tree in object$trees) {
    rows <- if (is.null(tree$rows)) seq_len(object$n) else tree$rows
    leaf <- cr_leaf_cif(
      tree, object$time[rows], object$event[rows], object$times, length(object$states)
    )
    at <- match(tree_route(tree, x), leaf$leaves)
    curves <- curves + leaf$cif[[target]][at, , drop = FALSE]
  }
  curves[incomplete, ] <- NA
  curve_frame(curves / n_trees, object$times, times, 0)
}

#' @rdname variable_importance
#' @param cause For a [competing_risk_forest()], the cause whose out-of-bag concordance is used, a
#'   level of `event` or its number among the causes. The default is the first cause.
#' @export
variable_importance.competing_risk_forest <- function(object, cause = 1, n_repeats = 5, ...) {
  target <- cr_cause(cause, object$states)
  forest_permutation_importance(
    object, n_repeats, object$event == target, function(tree) tree$risk[, target]
  )
}

#' @export
nobs.competing_risk_forest <- function(object, ...) {
  object$n
}

#' @export
print.competing_risk_forest <- function(x, digits = getOption("digits"), ...) {
  cat("<competing_risk_forest> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    paste0(
      "%d competing-risks log-rank trees, %d of %d features tried per split%s.\n",
      "Causes: %s. n = %d, events = %d\n"
    ),
    x$n_estimators, x$n_features_split, ncol(x$x),
    if (x$bootstrap) ", bootstrap samples" else "", paste(x$states, collapse = ", "), x$n,
    x$n_event
  ))
  if (!is.na(x$oob_score)) {
    cat(sprintf(
      "Out-of-bag concordance for %s = %s\n", x$states[1L],
      format(x$oob_score, digits = max(3L, digits - 3L))
    ))
  }
  cat_dropped(x$n_dropped)
  invisible(x)
}
