# Elastic-net penalized Fine-Gray regression: penalized_fine_gray() and its methods. Tested by
# tests/testthat/test-penalized-fine-gray.R, against fine_gray() for an unpenalized fit, the
# optimality conditions of the penalized fit, and Greenwood's Python values.

#' Elastic-net penalized Fine-Gray regression
#'
#' `penalized_fine_gray()` fits the Fine-Gray model for the subdistribution hazard of one cause
#' with an elastic-net penalty, for when there are many covariates for the number of events of
#' that cause, or covariates that are strongly correlated. The coefficients minimize the negative
#' log pseudo partial likelihood of [fine_gray()], divided by the number of subjects \eqn{n}, plus
#' the penalty
#' \deqn{\lambda \left(\alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2\right),}
#' with \eqn{\lambda} the `penalizer` and \eqn{\alpha} the `l1_ratio`. `l1_ratio = 1` is the lasso,
#' which sets some coefficients to exactly 0 and so selects covariates. `l1_ratio = 0` is ridge
#' regression, which shrinks all of the coefficients toward 0 without removing any. Values in
#' between mix the two. `penalizer = 0` gives the unpenalized [fine_gray()] fit.
#'
#' Subjects with a competing event stay in the risk sets after it, with the weight
#' \eqn{G(t-) / G(T-)}, where \eqn{G} is the Kaplan-Meier curve of the censoring times and
#' \eqn{T} the time of the competing event, as in [fine_gray()].
#'
#' With `standardize = TRUE` (the default), each covariate is penalized on the scale of its
#' standard deviation, so the penalty treats all of them alike whatever their units. The
#' coefficients are reported on the scale of the covariates either way. The fit uses accelerated
#' proximal gradient descent (FISTA) from \eqn{\beta = 0}, with a backtracking line search, and
#' stops once the relative change in the coefficients is below `tol`. Penalized coefficients are
#' biased toward 0 by design, so no standard errors or p-values are given.
#'
#' @param formula A formula with a competing-risks response on the left, `Surv(time, event)` with
#'   `event` a factor whose first level is censoring, and the covariates on the right, such as
#'   `Surv(time, event) ~ age + sex`. Factors are coded with treatment contrasts. Collinear
#'   covariates are allowed.
#' @param data A data frame holding the variables in `formula`.
#' @param cause The cause to model, a level of `event` or its number among the causes. The
#'   default is the first cause.
#' @param subset An optional expression selecting the rows of `data` to use.
#' @param na.action What to do with rows that have missing values. The default drops them.
#' @param penalizer The penalty strength \eqn{\lambda}, a non-negative number. The default is
#'   `0.1`.
#' @param l1_ratio The elastic-net mixing \eqn{\alpha}, between 0 (ridge) and 1 (lasso). The
#'   default is `0.5`.
#' @param standardize Whether to penalize each covariate on the scale of its standard deviation.
#'   The default is `TRUE`.
#' @param max_iter The largest number of iterations. The default is `1000`.
#' @param tol The convergence tolerance for the relative change in the coefficients. The default
#'   is `1e-7`.
#'
#' @return A `penalized_fine_gray` object. Use [coef()] or [as.data.frame()] for the coefficients
#'   and [predict()] for linear predictors, risk scores, and cumulative incidence curves. The
#'   object also holds `loglik` (the log pseudo partial likelihood at the estimate), `n_nonzero`,
#'   `center` and `scale` (each covariate's mean and the scale it was penalized on), `cause`,
#'   `states`, `n`, `n_event` (the number of events of the cause), `iterations`, and `converged`.
#'
#' @seealso [fine_gray()] for the unpenalized model, and [cox_net()] for the penalized Cox model.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61),
#'   hgb = c(13, 12, 14, 11, 13, 12, 15, 12, 11, 10, 14, 13, 12, 11, 15, 12, 13, 12)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#'
#' # The lasso, which sets some coefficients to 0
#' fit <- penalized_fine_gray(Surv(time, event) ~ age + hgb,
#'   data = mgus, cause = "pcm", penalizer = 0.1, l1_ratio = 1
#' )
#' fit
#'
#' # Cumulative incidence of the cause for new subjects
#' predict(fit, newdata = data.frame(age = c(60, 75), hgb = c(14, 11)), type = "cif",
#'   times = c(100, 200)
#' )
#' @export
penalized_fine_gray <- function(formula,
                                data,
                                cause = 1,
                                subset,
                                na.action = stats::na.omit,
                                penalizer = 0.1,
                                l1_ratio = 0.5,
                                standardize = TRUE,
                                max_iter = 1000,
                                tol = 1e-7) {
  check_coxnet_args(l1_ratio, standardize, max_iter, tol)
  check_number(penalizer, "penalizer", min = 0)
  fn <- "penalized_fine_gray()"
  frame <- gw_model_frame(
    match.call(expand.dots = FALSE), parent.frame(), formula, if (missing(data)) NULL else data
  )
  cr_no_specials(frame, fn)
  if (!is.null(split_formula(formula, if (missing(data)) NULL else data)$offset)) {
    stop(sprintf("`%s` doesn't support `offset()` terms.", fn), call. = FALSE)
  }
  cr <- cr_response(frame, fn)
  target <- cr_cause(cause, cr$states)
  design <- covariate_design(frame, check_rank = FALSE)
  if (ncol(design$x) == 0L) {
    stop(sprintf("`%s` needs at least one covariate.", fn), call. = FALSE)
  }
  if (!any(cr$status == target)) {
    stop(sprintf("There are no events of cause \"%s\".", cr$states[target]), call. = FALSE)
  }
  x <- design$x
  n <- nrow(x)
  standardized <- coxnet_standardize(x, rep(1, n), standardize)
  xs <- standardized$x
  setup <- fg_setup(cr$time, cr$status, target)
  ridge <- penalizer * (1 - l1_ratio)
  event_sums <- colSums(xs[setup$is_event, , drop = FALSE])

  smooth <- function(b, gradient = FALSE) {
    lp <- drop(xs %*% b)
    r <- exp(lp)
    s0 <- drop(fg_risk_sums(r, setup))
    loglik <- sum(lp[setup$is_event]) - sum(setup$d * log(s0))
    out <- list(value = -loglik / n + 0.5 * ridge * sum(b * b))
    if (gradient) {
      score <- event_sums - colSums(setup$d * fg_risk_sums(xs * r, setup) / s0)
      out$gradient <- -score / n + ridge * b
    }
    out
  }

  fit <- fista(smooth, ncol(x), penalizer * l1_ratio, max_iter, tol)
  if (!fit$converged) {
    warning(sprintf("`%s` didn't converge in %d iterations.", fn, as.integer(max_iter)),
      call. = FALSE
    )
  }
  beta <- stats::setNames(fit$beta / standardized$scale, colnames(x))
  lp <- drop(xs %*% fit$beta)
  s0 <- drop(fg_risk_sums(exp(lp), setup))

  structure(
    list(
      coefficients = beta,
      penalizer = penalizer,
      l1_ratio = l1_ratio,
      standardize = standardize,
      loglik = sum(lp[setup$is_event]) - sum(setup$d * log(s0)),
      n_nonzero = sum(beta != 0),
      center = stats::setNames(standardized$center, colnames(x)),
      scale = stats::setNames(standardized$scale, colnames(x)),
      baseline = data.frame(time = setup$times, cumhaz = cumsum(setup$d / s0)),
      x = x,
      n = n,
      n_event = sum(setup$d),
      cause = cr$states[target],
      states = cr$states,
      iterations = fit$iterations,
      converged = fit$converged,
      terms = design$terms,
      xlevels = design$xlevels,
      contrasts = design$contrasts,
      n_dropped = frame$n_dropped,
      formula = formula,
      call = match.call()
    ),
    class = "penalized_fine_gray"
  )
}

#' @export
coef.penalized_fine_gray <- function(object, ...) {
  object$coefficients
}

#' @export
nobs.penalized_fine_gray <- function(object, ...) {
  object$n
}

#' Predictions and coefficients of a penalized Fine-Gray model
#'
#' `predict()` gives the linear predictor, the relative subdistribution hazard, or the cumulative
#' incidence of the modeled cause for each row of `newdata` (or for the subjects the model was fit
#' to). The linear predictor is \eqn{(x - \bar{x})^\top \beta}, centered at the covariate means
#' of the fit, and the cumulative incidence is \eqn{1 - \exp\{-H_0(t) \exp(\eta)\}}, with
#' \eqn{H_0} the Breslow estimate of the baseline cumulative subdistribution hazard at the
#' penalized coefficients. `as.data.frame()` gives the coefficients and subdistribution hazard
#' ratios, with the coefficients set to 0 by the lasso kept in the table.
#'
#' @param object,x A [penalized_fine_gray()] fit.
#' @param newdata A data frame with the covariates of the subjects to predict for. The default is
#'   the data the model was fit to.
#' @param type What to predict: `"lp"` (the default), `"risk"` (\eqn{\exp(\eta)}), or `"cif"`.
#' @param times For `"cif"`, the times to evaluate at. The default is the event times of the
#'   cause.
#' @param row.names,optional Unused. They're part of the [as.data.frame()] generic.
#' @param ... Unused.
#'
#' @return For `"lp"` and `"risk"`, a numeric vector with one value per row of `newdata`. For
#'   `"cif"`, a data frame with one row per subject and time: `subject` (the row of `newdata`),
#'   `time`, and `estimate`. `as.data.frame()` gives a data frame with the columns `term`,
#'   `estimate`, and `hazard_ratio`.
#'
#' @examples
#' mgus <- data.frame(
#'   time = c(30, 45, 60, 72, 80, 95, 110, 120, 130, 150, 160, 180, 200, 210, 220, 240, 45, 95),
#'   status = c(1, 2, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2, 0, 1),
#'   age = c(62, 75, 58, 80, 66, 71, 59, 64, 77, 82, 60, 68, 55, 79, 63, 74, 70, 61),
#'   hgb = c(13, 12, 14, 11, 13, 12, 15, 12, 11, 10, 14, 13, 12, 11, 15, 12, 13, 12)
#' )
#' mgus$event <- factor(mgus$status, levels = 0:2, labels = c("censor", "pcm", "death"))
#' fit <- penalized_fine_gray(Surv(time, event) ~ age + hgb,
#'   data = mgus, cause = "pcm", penalizer = 0.02, l1_ratio = 0
#' )
#' as.data.frame(fit)
#' predict(fit, newdata = data.frame(age = c(60, 75), hgb = c(14, 11)), type = "risk")
#' @name penalized_fine_gray_predict
#' @export
predict.penalized_fine_gray <- function(object,
                                        newdata,
                                        type = c("lp", "risk", "cif"),
                                        times,
                                        ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    x <- object$x
  } else {
    if (!is.data.frame(newdata)) {
      stop("`newdata` must be a data frame.", call. = FALSE)
    }
    x <- cox_new_design(object, newdata)
  }
  lp <- unname(linear_predictor(sweep(x, 2L, object$center), object$coefficients))
  if (type == "lp") {
    return(lp)
  }
  if (type == "risk") {
    return(exp(lp))
  }
  base <- object$baseline
  times <- if (missing(times)) base$time else check_times(times)
  cumhaz <- step_at(base$time, base$cumhaz, times, 0)
  data.frame(
    subject = rep(seq_along(lp), each = length(times)),
    time = rep(times, times = length(lp)),
    estimate = 1 - exp(-rep(cumhaz, times = length(lp)) * rep(exp(lp), each = length(times)))
  )
}

#' @rdname penalized_fine_gray_predict
#' @export
as.data.frame.penalized_fine_gray <- function(x, row.names = NULL, optional = FALSE, ...) {
  as.data.frame.cox_net(x)
}

#' @export
print.penalized_fine_gray <- function(x, digits = getOption("digits"), ...) {
  cat("<penalized_fine_gray> ", deparse_formula(x$formula), "\n", sep = "")
  cat(sprintf(
    paste0(
      "Elastic net, penalizer = %s, l1_ratio = %s. Subdistribution hazards for cause \"%s\".\n",
      "n = %d, events = %s\n\n"
    ),
    format(x$penalizer), format(x$l1_ratio), x$cause, x$n, format(x$n_event)
  ))
  table <- as.data.frame(x)
  coefs <- data.frame(estimate = table$estimate, hazard_ratio = table$hazard_ratio,
    row.names = table$term
  )
  print(coefs, digits = max(3L, digits - 3L))
  cat(sprintf(
    "\nNonzero coefficients: %d of %d. Log pseudo partial likelihood = %s\n",
    x$n_nonzero, length(x$coefficients), format(x$loglik, digits = max(3L, digits - 2L))
  ))
  cat_dropped(x$n_dropped)
  invisible(x)
}
