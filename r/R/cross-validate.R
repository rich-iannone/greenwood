# K-fold cross-validation of the regression models. Tested by tests/testthat/test-cross-validate.R,
# which rebuilds each fold's score from the folds the result records.

#' Cross-validate a survival model
#'
#' `cross_validate()` estimates how well a model predicts for new subjects. It splits the rows into
#' `k` folds, fits the model to all but one fold, scores the predictions on the fold left out, and
#' repeats this for every fold. Scoring a model on the data it was fit to overstates its accuracy,
#' and the held-out folds avoid that.
#'
#' The metrics are Harrell's concordance (`"concordance"`, see [concordance_index()]), the
#' integrated Brier score over `times` (`"brier"`, see [integrated_brier_score()]), and the
#' integrated time-dependent AUC over `times` (`"auc"`, see [integrated_auc()]). Higher is better
#' for the concordance and the AUC, and lower is better for the Brier score. Every metric is
#' computed on the same folds from the same fits. The concordance and the AUC rank subjects by the
#' linear predictor of a [cox_ph()], [cox_net()], or [royston_parmar()] fit, by the negative
#' linear predictor of an [aft()] fit (where a larger value means longer survival), and by the risk
#' score of a [survival_tree()], [random_survival_forest()], [extra_survival_trees()], or
#' [gradient_boosting_survival_analysis()] fit.
#'
#' With `stratified = TRUE` (the default), the folds are drawn separately among the subjects with
#' each event status (censored, or each kind of event), so every fold gets about the same share of
#' events. Folds with few events give unstable fits and scores, and a warning is given when there
#' are fewer than two events per fold. Rows with a missing value in any variable of `formula` are
#' dropped before the folds are drawn.
#'
#' The folds are drawn from R's random number generator, so call [set.seed()] first for results
#' that can be repeated.
#'
#' @param model The model function: [cox_ph()], [cox_net()], [aft()], [royston_parmar()],
#'   [survival_tree()], [random_survival_forest()], [extra_survival_trees()], or
#'   [gradient_boosting_survival_analysis()].
#' @param formula The model formula, such as `Surv(time, status) ~ age + sex`. The response is
#'   right-censored data.
#' @param data A data frame holding the variables in `formula`.
#' @param ... Further arguments for `model`, such as `dist = "weibull"` for [aft()]. A `weights`
#'   argument must name a column of `data`, so it can be split along with the rows.
#' @param k The number of folds, at least 2 and at most the number of rows. The default is `5`.
#' @param metrics One or more of `"concordance"` (the default), `"brier"`, and `"auc"`.
#' @param times At least two times, for the `"brier"` and `"auc"` metrics.
#' @param stratified Draw the folds within each event status. The default is `TRUE`.
#'
#' @return A `greenwood_cv` object. `results` is a data frame with one row per metric: `metric`,
#'   `mean` (the average over the folds), and `sd` (the standard deviation over the folds).
#'   `scores` is a data frame with one row per fold and metric: `fold`, `metric`, and `score`.
#'   `folds` gives the fold of each row used, `k` the number of folds, and `n_dropped` the number
#'   of rows dropped for missing values. [as.data.frame()] gives `results`.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144,
#'     613, 707, 61, 88),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57, 67, 70, 63, 56, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1)
#' )
#'
#' set.seed(1)
#' cross_validate(cox_ph, Surv(time, status) ~ age + sex, data = lung, k = 4)
#'
#' # Several metrics from the same folds
#' cross_validate(aft, Surv(time, status) ~ age + sex,
#'   data = lung, dist = "weibull", k = 4, metrics = c("concordance", "brier"),
#'   times = c(150, 300, 450)
#' )
#' @export
cross_validate <- function(model,
                           formula,
                           data,
                           ...,
                           k = 5,
                           metrics = "concordance",
                           times = NULL,
                           stratified = TRUE) {
  model_name <- deparse1(substitute(model))
  if (!is.function(model)) {
    stop("`model` must be a model function, such as `cox_ph` or `aft`.", call. = FALSE)
  }
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided formula such as `Surv(time, status) ~ x`.", call. = FALSE)
  }
  if (missing(data) || !is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  check_number(k, "k", min = 2)
  if (k != round(k)) {
    stop("`k` must be a whole number.", call. = FALSE)
  }
  k <- as.integer(k)
  choices <- c("concordance", "brier", "auc")
  if (!is.character(metrics) || length(metrics) == 0L || !all(metrics %in% choices)) {
    stop("`metrics` must be one or more of \"concordance\", \"brier\", and \"auc\".", call. = FALSE)
  }
  metrics <- unique(metrics)
  if (any(metrics %in% c("brier", "auc"))) {
    if (is.null(times) || !is.numeric(times) || length(times) < 2L || anyNA(times)) {
      stop("The \"brier\" and \"auc\" metrics need at least two `times`.", call. = FALSE)
    }
  }
  check_flag(stratified, "stratified")

  vars <- intersect(all.vars(formula), names(data))
  keep <- stats::complete.cases(data[vars])
  data <- data[keep, , drop = FALSE]
  response <- eval(formula_response(formula[[2L]]), data, environment(formula))
  if (!inherits(response, "Surv")) {
    stop("The left-hand side of `formula` must be a `Surv()` response.", call. = FALSE)
  }
  n <- nrow(data)
  if (k > n) {
    stop(sprintf("`k` must be at most the number of rows used (%d).", n), call. = FALSE)
  }
  status <- unclass(response)[, "status"]
  folds <- cv_folds(status, k, stratified)
  n_events <- sum(status != 0)
  if (n_events < 2L * k) {
    warning(
      sprintf(
        paste(
          "There are %d events for %d folds, fewer than two per fold.",
          "Some folds may have too few events to fit or score."
        ),
        n_events, k
      ),
      call. = FALSE
    )
  }

  # Each fold's fit is called as `model(formula, data = <training rows>, ...)` in the caller's
  # frame, so the arguments in `...` (such as `weights = w`) are read the usual way.
  dots <- match.call(expand.dots = FALSE)$...
  env <- new.env(parent = parent.frame())
  assign(".cv_model", model, envir = env)
  fit_call <- as.call(c(list(quote(.cv_model), formula = formula, data = quote(.cv_train)), dots))

  scores <- vapply(seq_len(k), function(i) {
    assign(".cv_train", data[folds != i, , drop = FALSE], envir = env)
    fit <- eval(fit_call, env)
    test <- data[folds == i, , drop = FALSE]
    vapply(metrics, function(m) cv_score(fit, test, response[folds == i], m, times), numeric(1))
  }, numeric(length(metrics)))
  scores <- matrix(scores, nrow = length(metrics))

  structure(
    list(
      results = data.frame(
        metric = metrics,
        mean = rowMeans(scores),
        sd = apply(scores, 1L, stats::sd)
      ),
      scores = data.frame(
        fold = rep(seq_len(k), each = length(metrics)),
        metric = rep(metrics, times = k),
        score = as.vector(scores)
      ),
      folds = folds,
      k = k,
      model = model_name,
      n_dropped = sum(!keep),
      formula = formula,
      call = match.call()
    ),
    class = "greenwood_cv"
  )
}

# The fold of each row. Each group of rows is shuffled and dealt into k nearly equal parts (the
# first parts one larger when they don't divide evenly). With stratification, each event status
# is dealt separately, and each picks up at the fold after the last one the previous status
# filled, so the folds stay nearly equal in size.
cv_folds <- function(status, k, stratified) {
  fold <- integer(length(status))
  deal <- function(rows, start) {
    shuffled <- rows[sample.int(length(rows))]
    sizes <- length(rows) %/% k + (seq_len(k) <= length(rows) %% k)
    fold[shuffled] <<- (start + rep(seq_len(k), sizes) - 1L) %% k + 1L
  }
  if (!stratified) {
    deal(seq_along(status), 0L)
    return(fold)
  }
  start <- 0L
  for (s in sort(unique(status))) {
    rows <- which(status == s)
    deal(rows, start)
    start <- (start + length(rows)) %% k
  }
  fold
}

# One metric of a fit on the held-out rows.
cv_score <- function(fit, test, response, metric, times) {
  if (!inherits(fit, c("cox_ph", "cox_net", "aft", "royston_parmar", cv_risk_models)) ||
    inherits(fit, "cause_specific_cox")) {
    stop(
      sprintf(
        paste(
          "`cross_validate()` scores `cox_ph()`, `cox_net()`, `aft()`, `royston_parmar()`,",
          "survival tree, forest, and gradient boosting fits, not `%s` objects."
        ),
        class(fit)[1L]
      ),
      call. = FALSE
    )
  }
  if (metric == "brier") {
    p <- stats::predict(fit, newdata = test, type = "survival", times = times)
    probs <- matrix(p$estimate, ncol = length(times), byrow = TRUE)
    return(integrated_brier_score(response, survival_prob = probs, times = times))
  }
  risk <- cv_risk(fit, test)
  if (metric == "concordance") {
    concordance_index(response, risk = risk)
  } else {
    integrated_auc(response, marker = risk, times = times)
  }
}

# The models whose predict() gives a risk score with `type = "risk"`.
cv_risk_models <- c("survival_tree", "survival_forest", "gradient_boosting_survival_analysis")

# A risk score, larger for an earlier event.
cv_risk <- function(fit, newdata) {
  if (inherits(fit, cv_risk_models)) {
    return(stats::predict(fit, newdata = newdata, type = "risk"))
  }
  lp <- as.vector(stats::predict(fit, newdata = newdata, type = "lp"))
  if (inherits(fit, "aft")) -lp else lp
}

#' @export
as.data.frame.greenwood_cv <- function(x, ...) {
  x$results
}

#' @export
print.greenwood_cv <- function(x, digits = getOption("digits"), ...) {
  cat("<cross_validate> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf("%d-fold cross-validation of %s()\n\n", x$k, x$model))
  print(x$results, digits = digits, row.names = FALSE)
  cat_dropped(x$n_dropped)
  invisible(x)
}
