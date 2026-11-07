# The tidy(), glance(), and augment() methods, on the generics package's generics. The columns
# follow the Python adapters in greenwood/summaries/, and tests/testthat/test-tidiers.R checks
# them against each model's own tables and fields.

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' @importFrom generics augment
#' @export
generics::augment

#' Tidy a fitted model
#'
#' `tidy()` gives a fitted model's estimates as a data frame, one row per term, time, or
#' parameter. For most models it's the same table as [as.data.frame()], and any arguments in `...`
#' go to that method (such as `exponentiate` and `conf_level` for [cox_ph()]).
#'
#' What a row holds depends on the model:
#'
#' * Regression models ([cox_ph()], [cause_specific_cox()], [fine_gray()], [aft()],
#'   [royston_parmar()], [piecewise_exponential()], [aalen_additive()], [buckley_james()],
#'   [mixture_cure()]): one row per coefficient, with `term`, `estimate`, `std_error`,
#'   `statistic`, `p_value`, `conf_low`, and `conf_high`.
#' * Penalized models ([cox_net()], [penalized_fine_gray()], [ipc_ridge()]): one row per
#'   coefficient, with `term` and `estimate` (and `hazard_ratio` for the Cox models).
#' * Curves ([kaplan_meier()], [nelson_aalen()], [aalen_johansen()], [multi_state()],
#'   [turnbull()]): one row per time, with `strata` and the grouping variables first.
#' * [parametric()]: one row per parameter of each curve.
#' * Forests and boosting ([random_survival_forest()], [extra_survival_trees()],
#'   [competing_risk_forest()], [gradient_boosting_survival_analysis()]): one row per covariate,
#'   with `term` and `importance`, from [variable_importance()]. [survival_boost()] gives the
#'   covariate names in `term`.
#' * [survival_tree()]: one row per node, from its table of splits.
#' * Tests ([logrank_test()] and the other tests, [maxcombo_test()], [rmst_test()]) and
#'   [cox_zph()]: their result tables.
#'
#' @param x A fitted model or a test result.
#' @param ... Passed to the model's [as.data.frame()] method, or to [variable_importance()] for
#'   the forests and boosting models.
#'
#' @return A data frame.
#'
#' @seealso [glance()] for one row per model, and [augment()] for one row per observation.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' tidy(fit)
#' tidy(fit, exponentiate = TRUE)
#'
#' tidy(kaplan_meier(Surv(time, status) ~ sex, data = lung))
#' @name tidy_greenwood
NULL

#' Glance at a fitted model
#'
#' `glance()` gives a summary of a fitted model as a data frame. For regression models it has one
#' row, with the numbers of observations (`n`) and events (`nevent`) and model-level statistics
#' such as the log-likelihood, AIC, and the likelihood ratio test. For curves, it has one row per
#' curve, with `strata` and the grouping variables first, so the layout is the same for one curve
#' or many.
#'
#' The columns for each model are:
#'
#' * [cox_ph()] and [cause_specific_cox()]: `n`, `nevent`, `loglik`, `aic`, `lr_statistic`, `df`,
#'   `lr_p_value`, `frailty_theta`, `frailty_lrt_statistic`, and `frailty_lrt_p_value` (the
#'   frailty columns are `NA` without a frailty term).
#' * [kaplan_meier()]: `n_start`, `events`, `median`, `median_lower`, and `median_upper`.
#' * [nelson_aalen()]: `n_start`, `events`, and `max_cumhaz`.
#' * [turnbull()]: `n`, `n_atoms`, `n_ambiguous` (atoms that are intervals rather than points),
#'   `n_iter`, and `converged`.
#' * [aalen_johansen()]: `n_causes` and `causes`.
#' * [multi_state()]: `n_states`, `states`, and `n_times`.
#' * [aft()]: `dist`, `n`, `nevent`, `scale`, `threshold`, `loglik`, and `aic`.
#' * [parametric()]: `dist`, `n`, `nevent`, `loglik`, `aic`, and `bic`.
#' * [piecewise_exponential()]: `n_intervals`, `n`, `nevent`, `loglik`, `aic`, `bic`, `df`, and
#'   `lr_statistic`.
#' * [royston_parmar()]: `scale`, `n`, `nevent`, `loglik`, `aic`, `bic`, `df`, and `n_knots`.
#' * [aalen_additive()]: `n`, `nevent`, `n_times`, and `test`.
#' * [buckley_james()]: `n`, `nevent`, `n_iter`, `converged`, and `n_boot`.
#' * [mixture_cure()]: `n`, `nevent`, and `n_iter`.
#' * [ipc_ridge()]: `n`, `nevent`, `alpha`, and `n_features`.
#' * [cox_net()]: `n`, `nevent`, `loglik`, `aic`, `bic`, `effective_df`, `penalizer`, `l1_ratio`,
#'   and `n_nonzero`.
#' * [fine_gray()]: `n`, `nevent`, and `loglik`.
#' * [penalized_fine_gray()]: `n`, `nevent`, `penalizer`, `l1_ratio`, and `n_nonzero`.
#' * [survival_tree()]: `n`, `nevent`, `n_nodes`, `n_leaves`, and `n_features`.
#' * [random_survival_forest()] and [extra_survival_trees()]: `n`, `nevent`, `n_estimators`,
#'   `n_features`, and `oob_concordance`.
#' * [gradient_boosting_survival_analysis()]: `n`, `nevent`, `n_estimators`, `learning_rate`,
#'   `max_depth`, and `n_features`.
#' * [survival_boost()]: `n`, `nevent`, `n_causes`, `n_estimators`, `learning_rate`,
#'   `max_depth`, and `n_features`.
#' * [competing_risk_forest()]: `n`, `nevent`, `n_estimators`, `n_features`, `n_causes`,
#'   `causes`, and `oob_concordance`.
#'
#' The AIC is \eqn{-2 \ell + 2k} with \eqn{k} the number of estimated parameters, and the BIC
#' replaces \eqn{2} with the log of the sample size (the number of events for the Cox and
#' Royston-Parmar models).
#'
#' @param x A fitted model.
#' @param ... Unused.
#'
#' @return A data frame.
#'
#' @seealso [tidy()] for one row per term, and [augment()] for one row per observation.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' glance(cox_ph(Surv(time, status) ~ age + sex, data = lung))
#' glance(kaplan_meier(Surv(time, status) ~ sex, data = lung))
#' @name glance_greenwood
NULL

# tidy() ----------------------------------------------------------------------------------------

tidy_table <- function(x, ...) {
  as.data.frame(x, ...)
}

#' @rdname tidy_greenwood
#' @export
tidy.cox_ph <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.kaplan_meier <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.nelson_aalen <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.turnbull <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.aalen_johansen <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.multi_state <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.fine_gray <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.penalized_fine_gray <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.aft <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.parametric <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.piecewise_exponential <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.royston_parmar <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.aalen_additive <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.buckley_james <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.mixture_cure <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.ipc_ridge <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.cox_net <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.survival_tree <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.greenwood_test <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.maxcombo_test <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.rmst_test <- tidy_table

#' @rdname tidy_greenwood
#' @export
tidy.cox_zph <- tidy_table

tidy_importance <- function(x, ...) {
  variable_importance(x, ...)
}

#' @rdname tidy_greenwood
#' @export
tidy.survival_forest <- tidy_importance

#' @rdname tidy_greenwood
#' @export
tidy.competing_risk_forest <- tidy_importance

#' @rdname tidy_greenwood
#' @export
tidy.gradient_boosting_survival_analysis <- tidy_importance

#' @rdname tidy_greenwood
#' @export
tidy.survival_boost <- function(x, ...) {
  data.frame(term = colnames(x$x), stringsAsFactors = FALSE)
}

# glance() --------------------------------------------------------------------------------------

# One row of model-level values, with every value a length-one vector.
glance_row <- function(...) {
  data.frame(..., stringsAsFactors = FALSE, check.names = FALSE)
}

# One row per curve: the curve keys (`strata` and the grouping variables) and then `values`, a
# list of columns with one value per curve.
glance_curves <- function(groups, values) {
  out <- cbind(groups, as.data.frame(values, stringsAsFactors = FALSE))
  rownames(out) <- NULL
  out
}

# The log-likelihood, AIC, and BIC from a logLik object. `n` is the sample size for the BIC.
information_criteria <- function(ll, n = attr(ll, "nobs")) {
  k <- attr(ll, "df")
  list(
    loglik = as.numeric(ll),
    aic = -2 * as.numeric(ll) + 2 * k,
    bic = -2 * as.numeric(ll) + log(n) * k
  )
}

test_field <- function(test, field) {
  if (is.null(test)) NA_real_ else as.numeric(test[[field]])
}

#' @rdname glance_greenwood
#' @export
glance.cox_ph <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  frailty <- x$frailty
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    loglik = ll$loglik,
    aic = ll$aic,
    lr_statistic = test_field(x$lr_test, "statistic"),
    df = length(x$coefficients),
    lr_p_value = test_field(x$lr_test, "p_value"),
    frailty_theta = if (is.null(frailty)) NA_real_ else frailty$theta,
    frailty_lrt_statistic = test_field(frailty$test, "statistic"),
    frailty_lrt_p_value = test_field(frailty$test, "p_value")
  )
}

#' @rdname glance_greenwood
#' @export
glance.kaplan_meier <- function(x, ...) {
  med <- stats::median(x)
  glance_curves(x$groups, list(
    n_start = curve_n_start(x$curves),
    events = vapply(x$curves, function(curve) sum(curve$n_event), numeric(1)),
    median = med$time,
    median_lower = med$conf_low,
    median_upper = med$conf_high
  ))
}

#' @rdname glance_greenwood
#' @export
glance.nelson_aalen <- function(x, ...) {
  glance_curves(x$groups, list(
    n_start = curve_n_start(x$curves),
    events = vapply(x$curves, function(curve) sum(curve$n_event), numeric(1)),
    max_cumhaz = vapply(x$curves, function(curve) {
      if (nrow(curve) == 0L) NA_real_ else curve$estimate[nrow(curve)]
    }, numeric(1))
  ))
}

curve_n_start <- function(curves) {
  vapply(curves, function(curve) {
    if (nrow(curve) == 0L) NA_real_ else as.numeric(curve$n_risk[1L])
  }, numeric(1))
}

#' @rdname glance_greenwood
#' @export
glance.turnbull <- function(x, ...) {
  glance_curves(x$groups, list(
    n = vapply(x$curves, function(curve) as.numeric(curve$n), numeric(1)),
    n_atoms = vapply(x$curves, function(curve) nrow(curve$table), integer(1)),
    n_ambiguous = vapply(x$curves, function(curve) {
      sum(curve$table$interval_low != curve$table$interval_high)
    }, integer(1)),
    n_iter = vapply(x$curves, function(curve) as.integer(curve$n_iter), integer(1)),
    converged = vapply(x$curves, function(curve) isTRUE(curve$converged), logical(1))
  ))
}

#' @rdname glance_greenwood
#' @export
glance.aalen_johansen <- function(x, ...) {
  k <- nrow(x$groups)
  glance_curves(x$groups, list(
    n_causes = rep(length(x$states), k),
    causes = rep(paste(x$states, collapse = ", "), k)
  ))
}

#' @rdname glance_greenwood
#' @export
glance.multi_state <- function(x, ...) {
  k <- nrow(x$groups)
  glance_curves(x$groups, list(
    n_states = rep(length(x$states), k),
    states = rep(paste(x$states, collapse = ", "), k),
    n_times = vapply(x$curves, function(curve) length(curve$time), integer(1))
  ))
}

#' @rdname glance_greenwood
#' @export
glance.aft <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  glance_row(
    dist = x$dist,
    n = x$n,
    nevent = sum(x$n_event),
    scale = x$scale,
    threshold = x$threshold,
    loglik = ll$loglik,
    aic = ll$aic
  )
}

#' @rdname glance_greenwood
#' @export
glance.parametric <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  glance_row(
    dist = x$dist,
    n = x$n,
    nevent = sum(x$n_event),
    loglik = ll$loglik,
    aic = ll$aic,
    bic = ll$bic
  )
}

#' @rdname glance_greenwood
#' @export
glance.piecewise_exponential <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  glance_row(
    n_intervals = length(x$breaks) + 1L,
    n = x$n,
    nevent = sum(x$n_event),
    loglik = ll$loglik,
    aic = ll$aic,
    bic = ll$bic,
    df = attr(stats::logLik(x), "df"),
    lr_statistic = test_field(x$lr_test, "statistic")
  )
}

#' @rdname glance_greenwood
#' @export
glance.royston_parmar <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  glance_row(
    scale = x$scale,
    n = x$n,
    nevent = sum(x$n_event),
    loglik = ll$loglik,
    aic = ll$aic,
    bic = ll$bic,
    df = length(x$coefficients),
    n_knots = length(x$knots) - 1L
  )
}

#' @rdname glance_greenwood
#' @export
glance.aalen_additive <- function(x, ...) {
  glance_row(n = x$n, nevent = sum(x$n_event), n_times = x$n_times, test = x$test)
}

#' @rdname glance_greenwood
#' @export
glance.buckley_james <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_iter = x$iterations,
    converged = x$converged,
    n_boot = if (is.null(x$boot)) 0L else nrow(x$boot)
  )
}

#' @rdname glance_greenwood
#' @export
glance.mixture_cure <- function(x, ...) {
  glance_row(n = x$n, nevent = sum(x$n_event), n_iter = x$iterations)
}

#' @rdname glance_greenwood
#' @export
glance.ipc_ridge <- function(x, ...) {
  glance_row(n = x$n, nevent = sum(x$n_event), alpha = x$alpha, n_features = ncol(x$x))
}

#' @rdname glance_greenwood
#' @export
glance.cox_net <- function(x, ...) {
  ll <- information_criteria(stats::logLik(x))
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    loglik = ll$loglik,
    aic = ll$aic,
    bic = ll$bic,
    effective_df = x$effective_df,
    penalizer = x$penalizer,
    l1_ratio = x$l1_ratio,
    n_nonzero = x$n_nonzero
  )
}

#' @rdname glance_greenwood
#' @export
glance.fine_gray <- function(x, ...) {
  glance_row(n = x$n, nevent = sum(x$n_event), loglik = x$loglik[2L])
}

#' @rdname glance_greenwood
#' @export
glance.penalized_fine_gray <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    penalizer = x$penalizer,
    l1_ratio = x$l1_ratio,
    n_nonzero = x$n_nonzero
  )
}

#' @rdname glance_greenwood
#' @export
glance.survival_tree <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_nodes = x$n_nodes,
    n_leaves = x$n_leaves,
    n_features = ncol(x$x)
  )
}

#' @rdname glance_greenwood
#' @export
glance.survival_forest <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_estimators = x$n_estimators,
    n_features = ncol(x$x),
    oob_concordance = as.numeric(x$oob_score)
  )
}

#' @rdname glance_greenwood
#' @export
glance.gradient_boosting_survival_analysis <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_estimators = x$n_estimators,
    learning_rate = x$learning_rate,
    max_depth = x$max_depth,
    n_features = ncol(x$x)
  )
}

#' @rdname glance_greenwood
#' @export
glance.survival_boost <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_causes = length(x$states),
    n_estimators = x$n_estimators,
    learning_rate = x$learning_rate,
    max_depth = x$max_depth,
    n_features = ncol(x$x)
  )
}

#' @rdname glance_greenwood
#' @export
glance.competing_risk_forest <- function(x, ...) {
  glance_row(
    n = x$n,
    nevent = sum(x$n_event),
    n_estimators = x$n_estimators,
    n_features = ncol(x$x),
    n_causes = length(x$states),
    causes = paste(x$states, collapse = ", "),
    oob_concordance = as.numeric(x$oob_score)
  )
}

# augment() -------------------------------------------------------------------------------------

#' Add a Cox model's predictions and residuals to the data
#'
#' `augment()` adds a column of predictions, `.fitted`, to a data frame, and for the data the
#' model was fit to a column of residuals, `.resid`.
#'
#' With `data`, the rows are those the model was fit to. Rows that `na.action` dropped from the
#' fit get `NA`, so `data` can be the frame that was passed to [cox_ph()]. With `newdata`, the
#' predictions are for its rows and there are no residuals.
#'
#' @param x A [cox_ph()] fit.
#' @param data The data frame the model was fit to.
#' @param newdata A data frame to predict for, instead of `data`.
#' @param type_predict The prediction: `"lp"` (the default, the linear predictor centered at the
#'   means of the covariates) or `"risk"` (its exponential).
#' @param type_residuals The residual: `"martingale"` (the default) or `"deviance"`.
#' @param ... Unused.
#'
#' @return `data` or `newdata` with `.fitted`, and for `data` also `.resid`.
#'
#' @seealso [residuals.cox_ph()] and [predict.cox_ph()] for the other residuals and predictions.
#'
#' @examples
#' lung <- data.frame(
#'   time = c(306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567),
#'   status = c(2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2),
#'   age = c(74, 68, 56, 57, 60, 74, 68, 71, 53, 61, 57, 68, 68, 60, 57),
#'   sex = c(1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1)
#' )
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' head(augment(fit, data = lung))
#' augment(fit, newdata = data.frame(age = c(55, 70), sex = 1), type_predict = "risk")
#' @export
augment.cox_ph <- function(x,
                           data = NULL,
                           newdata = NULL,
                           type_predict = c("lp", "risk"),
                           type_residuals = c("martingale", "deviance"),
                           ...) {
  type_predict <- match.arg(type_predict)
  type_residuals <- match.arg(type_residuals)
  if (!is.null(newdata)) {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    newdata$.fitted <- stats::predict(x, newdata = newdata, type = type_predict)
    return(newdata)
  }
  if (is.null(data)) {
    stop("`augment()` needs the data frame the model was fit to, as `data`.", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  fitted <- x$linear_predictors
  if (type_predict == "risk") {
    fitted <- exp(fitted)
  }
  resid <- stats::residuals(x, type = type_residuals)
  rows <- augment_rows(x, nrow(data))
  data$.fitted <- NA_real_
  data$.fitted[rows] <- unname(fitted)
  data$.resid <- NA_real_
  data$.resid[rows] <- unname(resid)
  data
}

# The rows of `data` that the fit used: all of them, or all but those `na.action` dropped.
augment_rows <- function(x, n_data) {
  if (n_data == x$n) {
    return(seq_len(n_data))
  }
  dropped <- x$na_action
  if (!is.null(dropped) && n_data == x$n + length(dropped)) {
    return(seq_len(n_data)[-dropped])
  }
  stop(
    sprintf(
      paste(
        "`data` has %d rows, but the model was fit to %d. Give the data frame that was passed",
        "to the fit, without `subset`."
      ),
      n_data, x$n
    ),
    call. = FALSE
  )
}
