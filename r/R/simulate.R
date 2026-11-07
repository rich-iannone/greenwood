# Synthetic competing-risks data. Tested by tests/testthat/test-simulate.R, which checks the
# layout and recovers the simulation's parameters with the package's own estimators.

#' Simulate competing-risks data
#'
#' `simulate_competing_risks()` draws competing-risks data from latent Weibull event times. Each
#' subject gets an independent event time for every cause, from that cause's Weibull distribution,
#' and the shortest one is the event observed. An independent exponential censoring time censors
#' the subject when it comes first.
#'
#' With `n_covariates` above 0, the covariates are standard normal and act on each cause's hazard
#' through proportional hazards: for cause \eqn{k} with shape \eqn{a_k} and scale \eqn{b_k}, a
#' subject with covariates \eqn{x} has the scale \eqn{b_k \exp(-x^\top \beta_k / a_k)}, which
#' multiplies the cause's hazard by \eqn{\exp(x^\top \beta_k)}.
#'
#' The draws come from R's random number generator, so call [set.seed()] first for data that can
#' be repeated.
#'
#' @param n The number of subjects. The default is `500`.
#' @param n_causes The number of competing causes. The default is `2`.
#' @param shapes The Weibull shape of each cause, one per cause. A shape of 1 gives a constant
#'   hazard, above 1 an increasing one, and below 1 a decreasing one. The default cycles through
#'   1, 1.5, and 0.8.
#' @param scales The Weibull scale of each cause, one per cause. The default is 10 for every cause.
#' @param censoring_scale How much censoring there is. The censoring times are exponential with a
#'   mean of `censoring_scale` times the mean event time (the mean over subjects of the shortest
#'   latent time), so larger values censor less. `NULL` or `0` gives no censoring. The default,
#'   `1.5`, censors about 40% of the subjects with the default shapes and scales.
#' @param n_covariates The number of covariates. The default is `0`.
#' @param coef The effects of the covariates, a matrix with one row per covariate and one column
#'   per cause, on the log-hazard scale. By default the effects are drawn uniformly from -0.5 to
#'   0.5.
#' @param states The names of the causes, one per cause. The default is `"cause1"`, `"cause2"`,
#'   and so on.
#'
#' @return A data frame with one row per subject: `time`, `event` (a factor whose first level,
#'   `"censor"`, marks censoring and whose other levels are the `states`), and the covariates `x1`,
#'   `x2`, and so on. `Surv(time, event)` is the competing-risks response. The data frame has
#'   three attributes for checking estimates against the truth: `latent_times`, a matrix of each
#'   subject's event time for every cause (one column per cause), `censoring_time` (`Inf` without
#'   censoring), and `coef`, the effects used (`NULL` without covariates).
#'
#' @examples
#' set.seed(42)
#' sim <- simulate_competing_risks(n = 300)
#' head(sim)
#' aalen_johansen(Surv(time, event) ~ 1, data = sim)
#'
#' # Covariates for a Fine-Gray model
#' sim <- simulate_competing_risks(n = 500, n_covariates = 2, states = c("relapse", "death"))
#' fine_gray(Surv(time, event) ~ x1 + x2, data = sim, cause = "relapse")
#' @export
simulate_competing_risks <- function(n = 500,
                                     n_causes = 2,
                                     shapes = NULL,
                                     scales = NULL,
                                     censoring_scale = 1.5,
                                     n_covariates = 0,
                                     coef = NULL,
                                     states = NULL) {
  check_count(n, "n", min = 1)
  check_count(n_causes, "n_causes", min = 1)
  check_count(n_covariates, "n_covariates", min = 0)
  if (is.null(shapes)) {
    shapes <- rep_len(c(1, 1.5, 0.8), n_causes)
  }
  if (is.null(scales)) {
    scales <- rep(10, n_causes)
  }
  check_per_cause(shapes, "shapes", n_causes)
  check_per_cause(scales, "scales", n_causes)
  if (is.null(states)) {
    states <- paste0("cause", seq_len(n_causes))
  }
  if (!is.character(states) || length(states) != n_causes || anyNA(states)) {
    stop(sprintf("`states` must be %d names, one per cause.", n_causes), call. = FALSE)
  }
  if (anyDuplicated(c("censor", states))) {
    stop("`states` must be distinct names other than \"censor\".", call. = FALSE)
  }
  if (!is.null(censoring_scale)) {
    check_number(censoring_scale, "censoring_scale", min = 0)
  }

  x <- NULL
  if (n_covariates > 0) {
    x <- matrix(stats::rnorm(n * n_covariates), n, n_covariates)
    colnames(x) <- paste0("x", seq_len(n_covariates))
    if (is.null(coef)) {
      coef <- matrix(stats::runif(n_covariates * n_causes, -0.5, 0.5), n_covariates, n_causes)
    } else if (!is.numeric(coef) || anyNA(coef) ||
      !identical(dim(as.matrix(coef)), as.integer(c(n_covariates, n_causes)))) {
      stop(
        sprintf(
          "`coef` must be a %d by %d matrix, one row per covariate and one column per cause.",
          n_covariates, n_causes
        ),
        call. = FALSE
      )
    }
    coef <- matrix(as.numeric(coef), n_covariates, n_causes, dimnames = list(colnames(x), states))
  } else {
    coef <- NULL
  }

  latent <- vapply(seq_len(n_causes), function(k) {
    scale <- rep(scales[k], n)
    if (!is.null(x)) {
      scale <- scale * exp(-linear_predictor(x, coef[, k]) / shapes[k])
    }
    stats::rweibull(n, shape = shapes[k], scale = scale)
  }, numeric(n))
  latent <- matrix(latent, n, n_causes, dimnames = list(NULL, states))
  cause <- max.col(-latent, ties.method = "first")
  event_time <- latent[cbind(seq_len(n), cause)]

  if (!is.null(censoring_scale) && censoring_scale > 0) {
    censoring_time <- stats::rexp(n, rate = 1 / (censoring_scale * mean(event_time)))
    censored <- censoring_time < event_time
  } else {
    censoring_time <- rep(Inf, n)
    censored <- rep(FALSE, n)
  }

  out <- data.frame(
    time = ifelse(censored, censoring_time, event_time),
    event = factor(ifelse(censored, 0L, cause), levels = 0:n_causes, labels = c("censor", states))
  )
  if (!is.null(x)) {
    out <- cbind(out, as.data.frame(x))
  }
  attr(out, "latent_times") <- latent
  attr(out, "censoring_time") <- censoring_time
  attr(out, "coef") <- coef
  out
}

check_count <- function(x, name, min) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x != round(x) || x < min) {
    stop(sprintf("`%s` must be a whole number no less than %d.", name, min), call. = FALSE)
  }
  invisible(x)
}

check_per_cause <- function(x, name, n_causes) {
  if (!is.numeric(x) || length(x) != n_causes || anyNA(x) || any(!is.finite(x)) || any(x <= 0)) {
    stop(sprintf("`%s` must be %d positive numbers, one per cause.", name, n_causes), call. = FALSE)
  }
  invisible(x)
}
