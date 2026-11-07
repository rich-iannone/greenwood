# Diagnostics of a Cox fit built on its residuals and refits: influence_diagnostics() and
# smooth_hr(). They're tested by tests/testthat/test-cox-diagnostics.R.

#' Influence diagnostics of a Cox model
#'
#' `influence_diagnostics()` gathers the per-observation measures that flag influential rows of a
#' [cox_ph()] fit into one table: the leverage, the martingale and deviance residuals, the
#' approximate change in each coefficient when the row is left out (`dfbeta`, and `dfbetas` on the
#' scale of the standard errors), and the likelihood displacement.
#'
#' The likelihood displacement of row \eqn{i} is
#' \eqn{\Delta\hat\beta_i^\top V^{-1} \Delta\hat\beta_i}, where \eqn{\Delta\hat\beta_i} is its
#' `dfbeta` row and \eqn{V} is the model-based variance of the coefficients. A large value means
#' that leaving the row out would move the coefficients a long way.
#'
#' @param fit A [cox_ph()] fit.
#'
#' @return A data frame with one row per observation used in the fit, and the columns `leverage`,
#'   `martingale`, `deviance`, `dfbeta_<term>` and `dfbetas_<term>` for each coefficient, and
#'   `ld`.
#'
#' @seealso [residuals.cox_ph()] for each measure on its own, and [plot_influence()] to plot them.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' diagnostics <- influence_diagnostics(fit)
#' head(diagnostics[order(diagnostics$ld, decreasing = TRUE), ])
#' @export
influence_diagnostics <- function(fit) {
  if (!inherits(fit, "cox_ph")) {
    stop("`fit` must be a `cox_ph()` fit.", call. = FALSE)
  }
  terms <- names(fit$coefficients)
  v <- if (is.null(fit$naive_var)) fit$var else fit$naive_var
  dfbeta <- stats::residuals(fit, type = "dfbeta")
  dfbetas <- sweep(dfbeta, 2L, sqrt(diag(v)), "/")
  out <- data.frame(
    leverage = stats::residuals(fit, type = "leverage"),
    martingale = stats::residuals(fit, type = "martingale"),
    deviance = stats::residuals(fit, type = "deviance")
  )
  for (j in seq_along(terms)) {
    out[[paste0("dfbeta_", terms[j])]] <- dfbeta[, j]
  }
  for (j in seq_along(terms)) {
    out[[paste0("dfbetas_", terms[j])]] <- dfbetas[, j]
  }
  out$ld <- rowSums((dfbeta %*% cox_solve(v, diag(length(terms)))) * dfbeta)
  rownames(out) <- NULL
  out
}

#' A smooth hazard ratio curve for a continuous covariate
#'
#' `smooth_hr()` shows how the effect of a continuous covariate on the hazard varies over its
#' range, which a single linear coefficient can't. It refits the [cox_ph()] model with the
#' covariate's linear term replaced by a cubic B-spline, keeping the other covariates, the strata,
#' the weights, the ties method, and the variance type. The curve is the log hazard ratio of each
#' value of the covariate against `reference`, with pointwise confidence limits at the fit's
#' `conf_level`.
#'
#' The spline has `df - 3` interior knots, at evenly spaced quantiles of the covariate, and
#' boundary knots at its range.
#'
#' @param fit A [cox_ph()] fit without frailty.
#' @param term The name of the covariate's coefficient, which must enter the model linearly.
#' @param df The degrees of freedom of the spline, at least 3. The default of 4 gives one interior
#'   knot, at the median.
#' @param n_grid The number of evenly spaced values at which the curve is evaluated, over the
#'   covariate's range.
#' @param reference The value of the covariate at which the log hazard ratio is zero. The default
#'   is its (weighted) mean.
#'
#' @return A `smooth_hr` object holding `term`, `reference`, `grid` (the covariate values),
#'   `log_hr` with its limits `log_hr_lower` and `log_hr_upper`, the same on the hazard ratio scale
#'   (`hr`, `hr_lower`, and `hr_upper`), `df`, the interior `knots`, `conf_level`, and `adjustment`
#'   (the mean of each other covariate). [as.data.frame()] gives the curve as a table, with
#'   `scale = "log_hr"` (the default) or `"hr"`.
#'
#' @seealso [plot_smooth_hr()] to plot the curve.
#'
#' @examples
#' fit <- cox_ph(Surv(time, status) ~ age + sex, data = lung)
#' curve <- smooth_hr(fit, term = "age")
#' curve
#' head(as.data.frame(curve, scale = "hr"))
#' @export
smooth_hr <- function(fit, term, df = 4, n_grid = 200, reference = NULL) {
  if (!inherits(fit, "cox_ph")) {
    stop("`fit` must be a `cox_ph()` fit.", call. = FALSE)
  }
  if (!is.null(fit$frailty)) {
    stop("`smooth_hr()` doesn't support a fit with frailty.", call. = FALSE)
  }
  terms <- names(fit$coefficients)
  if (!is.character(term) || length(term) != 1L || !term %in% terms) {
    stop(
      sprintf(
        "`term` must be one of the fit's coefficients: %s.",
        paste0("\"", terms, "\"", collapse = ", ")
      ),
      call. = FALSE
    )
  }
  check_number(df, "df", min = 3)
  check_number(n_grid, "n_grid", min = 2)
  df <- as.integer(df)
  n_grid <- as.integer(n_grid)

  setup <- fit$setup
  w <- fit$weights
  x <- sweep(setup$x, 2L, fit$means, "+")
  colnames(x) <- terms
  value <- x[, term]
  if (is.null(reference)) {
    reference <- sum(w * value) / sum(w)
  }
  check_number(reference, "reference")
  lower <- min(value)
  upper <- max(value)
  if (lower == upper) {
    stop(sprintf("`%s` takes a single value, so it has no curve.", term), call. = FALSE)
  }

  n_interior <- df - 3L
  knots <- if (n_interior > 0L) {
    unname(stats::quantile(value, seq(0, 1, length.out = n_interior + 2L)[-c(1L, n_interior + 2L)]))
  } else {
    numeric()
  }
  all_knots <- c(rep(lower, 4L), knots, rep(upper, 4L))
  grid <- seq(lower, upper, length.out = n_grid)
  # The full basis sums to one, which the model's baseline hazard absorbs, so the first column
  # is dropped. The curve and its limits are the same.
  basis <- bspline_basis(value, all_knots)[, -1L, drop = FALSE]
  diff <- bspline_basis(grid, all_knots)[, -1L, drop = FALSE] -
    rep(bspline_basis(min(max(reference, lower), upper), all_knots)[, -1L], each = n_grid)

  refit <- smooth_hr_refit(fit, basis, x[, terms != term, drop = FALSE])
  index <- seq_len(ncol(basis))
  beta <- refit$coefficients[index]
  v <- refit$var[index, index, drop = FALSE]
  log_hr <- drop(diff %*% beta)
  se <- sqrt(pmax(rowSums((diff %*% v) * diff), 0))
  z <- stats::qnorm(1 - (1 - fit$conf_level) / 2)
  others <- x[, terms != term, drop = FALSE]

  structure(
    list(
      term = term,
      reference = reference,
      grid = grid,
      log_hr = log_hr,
      log_hr_lower = log_hr - z * se,
      log_hr_upper = log_hr + z * se,
      hr = exp(log_hr),
      hr_lower = exp(log_hr - z * se),
      hr_upper = exp(log_hr + z * se),
      df = df,
      knots = knots,
      conf_level = fit$conf_level,
      adjustment = colSums(others * w) / sum(w)
    ),
    class = "smooth_hr"
  )
}

# The Cox fit with the spline basis in place of the covariate, through cox_ph() so the variance
# (model-based, robust, or clustered) is worked out as for the original fit.
smooth_hr_refit <- function(fit, basis, others) {
  setup <- fit$setup
  data <- data.frame(.start = setup$start, .stop = setup$stop, .event = setup$event)
  data$.weights <- fit$weights
  data$.basis <- basis
  rhs <- ".basis"
  if (ncol(others) > 0L) {
    data$.others <- others
    rhs <- c(rhs, ".others")
  }
  if (!is.null(fit$strata)) {
    data$.stratum <- fit$stratum
    rhs <- c(rhs, "strata(.stratum)")
  }
  lhs <- if (setup$counting) "Surv(.start, .stop, .event)" else "Surv(.stop, .event)"
  args <- list(
    formula = stats::as.formula(paste(lhs, "~", paste(rhs, collapse = " + ")), env = environment()),
    data = data,
    ties = fit$ties,
    robust = fit$robust,
    conf_level = fit$conf_level
  )
  if (isTRUE(fit$has_weights)) {
    args$weights <- quote(.weights)
  }
  if (!is.null(fit$cluster)) {
    data$.cluster <- fit$cluster
    args$data <- data
    args$cluster <- quote(.cluster)
  }
  suppressWarnings(do.call(cox_ph, args))
}

# The cubic B-spline basis at x for the full knot sequence, by the Cox-de Boor recursion. A value
# at the upper boundary belongs to the last interval.
bspline_basis <- function(x, knots, degree = 3L) {
  m <- length(knots) - 1L
  basis <- matrix(0, length(x), m)
  for (i in seq_len(m)) {
    basis[, i] <- knots[i] <= x & x < knots[i + 1L]
  }
  last <- max(which(knots[-1L] > knots[-length(knots)]))
  basis[x == knots[length(knots)], last] <- 1
  for (d in seq_len(degree)) {
    widened <- matrix(0, length(x), m - d)
    for (i in seq_len(m - d)) {
      left_width <- knots[i + d] - knots[i]
      right_width <- knots[i + d + 1L] - knots[i + 1L]
      left <- if (left_width > 0) (x - knots[i]) / left_width * basis[, i] else 0
      right <- if (right_width > 0) {
        (knots[i + d + 1L] - x) / right_width * basis[, i + 1L]
      } else {
        0
      }
      widened[, i] <- left + right
    }
    basis <- widened
  }
  basis
}

#' @export
print.smooth_hr <- function(x, ...) {
  cat(sprintf(
    "Smooth hazard ratio for %s: a cubic spline with %d df, against %s = %s\n",
    x$term, x$df, x$term, format(signif(x$reference, 6))
  ))
  if (length(x$knots) > 0L) {
    cat(sprintf("Interior knots: %s\n", paste(format(signif(x$knots, 6)), collapse = ", ")))
  }
  if (length(x$adjustment) > 0L) {
    cat(sprintf(
      "Other covariates held at their means: %s\n",
      paste(sprintf("%s = %s", names(x$adjustment), format(signif(x$adjustment, 4))),
        collapse = ", "
      )
    ))
  }
  cat("\n")
  table <- as.data.frame(x)
  shown <- unique(round(seq(1, nrow(table), length.out = min(nrow(table), 6L))))
  print(table[shown, ], row.names = FALSE, digits = 4)
  if (length(shown) < nrow(table)) {
    cat(sprintf("(%d of %d grid points shown)\n", length(shown), nrow(table)))
  }
  invisible(x)
}

#' @rdname smooth_hr
#' @param x A `smooth_hr` object.
#' @param row.names,optional Unused, for compatibility with the generic.
#' @param scale The scale of the table: `"log_hr"` (the default) or `"hr"`.
#' @param ... Unused.
#' @export
as.data.frame.smooth_hr <- function(x, row.names = NULL, optional = FALSE,
                                    scale = c("log_hr", "hr"), ...) {
  scale <- match.arg(scale)
  out <- data.frame(grid = x$grid)
  names(out) <- x$term
  if (scale == "hr") {
    out$hr <- x$hr
    out$conf_low <- x$hr_lower
    out$conf_high <- x$hr_upper
  } else {
    out$log_hr <- x$log_hr
    out$conf_low <- x$log_hr_lower
    out$conf_high <- x$log_hr_upper
  }
  out
}
